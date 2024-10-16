use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, get_device_info, Device, CL_DEVICE_MAX_WORK_GROUP_SIZE, CL_DEVICE_MAX_WORK_GROUP_SIZE_AMD, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MAP_WRITE, CL_MEM_READ_ONLY};
use opencl3::program::Program;
use opencl3::svm::SvmVec;
use opencl3::types::{cl_int, cl_uchar, CL_BLOCKING};
use std::ptr;

const PROGRAM_SOURCE: &str = include_str!("program.cl");

const KERNEL_NAME: &str = "load_team";

const BLOCK_SIZE: usize = 256;

fn main() -> anyhow::Result<()> {
    // Find a usable device for this application
    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
        .first()
        .expect("no device found in platform");
    let size = {
        match get_device_info(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE_AMD) {
            Ok(size) => {
                size.to_size()
            },
            Err(err) => {
                println!("警告: get_device_info failed: {}\n也许是你没有一张AMD显卡,让我们试试非AMD", err);
                match get_device_info(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE) {
                    Ok(size) => {
                        size.to_size()
                    },
                    Err(err) => {
                        println!("错误: get_device_info failed: {}\n", err);
                        panic!();
                    }
                }
            }
        }
    };
    let device = Device::new(device_id);

    let worker_count: cl_int = size as cl_int;
    println!("device max work group size: {} real count: {}", size, worker_count);

    // Create a Context on an OpenCL device
    let context = Context::from_device(&device).expect("Context::from_device failed");

    let queue = CommandQueue::create_default_with_properties(
        &context,
        CL_QUEUE_PROFILING_ENABLE,
        worker_count as u32,
    )
    .expect("create_command_queue_with_properties failed");

    // Build the OpenCL program source and create the kernel.
    let program = match Program::create_and_build_from_source(&context, PROGRAM_SOURCE, "") {
        Ok(p) => p,
        Err(err) => {
            println!(
                "OpenCL Program::create_and_build_from_source failed: {}",
                err
            );
            panic!();
        }
    };
    let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");

    let team_raw_vec = vec!["x"; worker_count as usize];
    let name_raw_vec = vec!["x"; worker_count as usize];
    let team_bytes_vec = team_raw_vec.iter().map(|s| s.as_bytes()).collect::<Vec<&[u8]>>();
    let name_bytes_vec = name_raw_vec.iter().map(|s| s.as_bytes()).collect::<Vec<&[u8]>>();
    let t_len_vec = team_bytes_vec.iter().map(|s| s.len() as cl_int + 1).collect::<Vec<i32>>();
    let n_len_vec = name_bytes_vec.iter().map(|s| s.len() as cl_int).collect::<Vec<i32>>();

    let work_count = team_bytes_vec.len();

    // Create OpenCL device buffers
    let mut team = unsafe {
        Buffer::<cl_uchar>::create(
            &context,
            CL_MEM_READ_ONLY,
            BLOCK_SIZE * work_count,
            ptr::null_mut(),
        )?
    };
    let mut name = unsafe {
        Buffer::<cl_uchar>::create(
            &context,
            CL_MEM_READ_ONLY,
            BLOCK_SIZE * work_count,
            ptr::null_mut(),
        )?
    };
    let mut t_len = unsafe {
        Buffer::<cl_int>::create(&context, CL_MEM_READ_ONLY, work_count, ptr::null_mut())?
    };
    let mut n_len = unsafe {
        Buffer::<cl_int>::create(&context, CL_MEM_READ_ONLY, work_count, ptr::null_mut())?
    };
    let mut output = SvmVec::<cl_uchar>::allocate(&context, BLOCK_SIZE * work_count)?;
    // 准备一下数据, 都给拼成一维数组
    // 填充成 256 * len
    let team_data_vec = {
        let mut vec = Vec::new();
        for data in team_bytes_vec {
            let left_over = BLOCK_SIZE - data.len();
            vec.extend_from_slice(data);
            vec.extend_from_slice(&vec![0; left_over]);
        }
        vec
    };
    let name_data_vec = {
        let mut vec = Vec::new();
        for data in name_bytes_vec {
            let left_over = BLOCK_SIZE - data.len();
            vec.extend_from_slice(data);
            vec.extend_from_slice(&vec![0; left_over]);
        }
        vec
    };

    // 阻塞写
    let _team_write_event =
        unsafe { queue.enqueue_write_buffer(&mut team, CL_BLOCKING, 0, &team_data_vec, &[]) }?;
    let _name_write_event =
        unsafe { queue.enqueue_write_buffer(&mut name, CL_BLOCKING, 0, &name_data_vec, &[]) }?;
    let _t_len_write_event =
        unsafe { queue.enqueue_write_buffer(&mut t_len, CL_BLOCKING, 0, &t_len_vec, &[]) }?;
    let _n_len_write_event =
        unsafe { queue.enqueue_write_buffer(&mut n_len, CL_BLOCKING, 0, &n_len_vec, &[]) }?;

    // println!("output: {:?} {}", output, output.len());
    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&team)
            .set_arg(&t_len)
            .set_arg(&name)
            .set_arg(&n_len)
            .set_arg_svm(output.as_mut_ptr())
            .set_arg(&worker_count)
            .set_global_work_size(worker_count as usize)
            .enqueue_nd_range(&queue)?
    };
    kernel_event.wait()?;
    queue.finish()?;

    if !output.is_fine_grained() {
        unsafe { queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_WRITE, &mut output, &[]) }?;
    }

    let start_time = kernel_event.profiling_command_start()?;
    let end_time = kernel_event.profiling_command_end()?;
    let duration = end_time - start_time;
    let time = std::time::Duration::from_nanos(duration as u64);
    println!("kernel execution duration: {:?}", time);
    let pre_sec = 1_000_000_000 as f32 / duration as f32;
    println!("kernel execution speed (pre/sec): {:?}", pre_sec * worker_count as f32);
    // println!("output: {:?} {}", output, output.len());

    if !output.is_fine_grained() {
        let unmap_event = unsafe { queue.enqueue_svm_unmap(&output, &[]) }?;
        unmap_event.wait()?;
    }

    Ok(())
}
