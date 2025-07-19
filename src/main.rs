use opencl3::command_queue::{
    CommandQueue, CL_QUEUE_ON_DEVICE, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
    CL_QUEUE_PROFILING_ENABLE,
};
use opencl3::context::Context;
use opencl3::device::{
    get_all_devices, get_device_info, Device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
    CL_DEVICE_MAX_WORK_GROUP_SIZE_AMD, CL_DEVICE_TYPE_GPU,
};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MAP_WRITE, CL_MEM_READ_ONLY};
use opencl3::program::Program;
use opencl3::svm::SvmVec;
use opencl3::types::{cl_int, cl_uchar, CL_BLOCKING, CL_NON_BLOCKING};
use std::ptr;

const PROGRAM_SOURCE: &str = include_str!("program.cl");

const KERNEL_NAME: &str = "load_team";

const BLOCK_SIZE: usize = 256;

// 运行次数
const RUN_TIMES: usize = 1000;

fn main() -> anyhow::Result<()> {
    // Find a usable device for this application
    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
        .first()
        .expect("no device found in platform");
    let max_size = {
        match get_device_info(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE_AMD) {
            Ok(size) => size.to_size(),
            Err(err) => {
                println!(
                    "警告: get_device_info failed: {err}\n也许是你没有一张AMD显卡,让我们试试非AMD"
                );
                match get_device_info(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE) {
                    Ok(size) => {
                        println!("非 amd size 获取成功");
                        size.to_size()
                    }
                    Err(err) => {
                        println!("错误: get_device_info failed: {err}\n");
                        panic!();
                    }
                }
            }
        }
    };
    let device = Device::new(device_id);

    let max_worker_count = max_size;
    println!("设备最大队列长度: {max_worker_count}");

    // Create a Context on an OpenCL device
    let context = Context::from_device(&device).expect("Context::from_device failed");

    let mut property =
        CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    // 如果命令行参数包含 -d, 则 | 一个 CL_QUEUE_ON_DEVICE
    let args = std::env::args().collect::<Vec<String>>();
    if args.contains(&"-d".to_string()) {
        property |= CL_QUEUE_ON_DEVICE;
    }
    let queue = match CommandQueue::create_default_with_properties(
        &context, property, 2
    ) {
        Ok(q) => q,
        Err(err) => {
            println!("创建命令队列失败: {err}, 属性: {property}");
            panic!();
        }
    };

    println!("队列创建完成");

    // Build the OpenCL program source and create the kernel.
    let program = match Program::create_and_build_from_source(&context, PROGRAM_SOURCE, "") {
        Ok(p) => {
            println!("程序构建成功");
            p
        }
        Err(err) => {
            println!(
                "OpenCL Program::create_and_build_from_source failed: {err}"
            );
            panic!();
        }
    };
    let kernel = match Kernel::create(&program, KERNEL_NAME) {
        Ok(k) => {
            println!("内核创建成功");
            k
        }
        Err(err) => {
            println!("OpenCL Kernel::create failed: {err}");
            panic!();
        }
    };

    let team_raw = "1234567";
    let team_bytes = team_raw.as_bytes();
    let t_len = team_bytes.len() as cl_int;
    let mut team = unsafe {
        Buffer::<cl_uchar>::create(
            &context,
            CL_MEM_READ_ONLY,
            BLOCK_SIZE,
            ptr::null_mut(),
        )?
    };
    let team_write_event =
        unsafe { queue.enqueue_write_buffer(&mut team, CL_NON_BLOCKING, 0, team_bytes, &[]) }?;
    team_write_event.wait()?;

    // 存储每个worker count的平均速度
    let mut results: Vec<(usize, f32)> = Vec::with_capacity(max_worker_count);

    println!("开始执行测试, worker_count 将从 1 到 {max_worker_count}");

    for current_worker_count in 1..=max_worker_count {
        println!("--> 正在测试 worker_count = {current_worker_count}");

        let worker_count_cl = current_worker_count as cl_int;
        let name_raw_vec: Vec<String> = (1..=current_worker_count).map(|i| i.to_string()).collect();

        let name_bytes_vec = name_raw_vec
            .iter()
            .map(|s| s.as_bytes())
            .collect::<Vec<&[u8]>>();
        let n_len_vec = name_bytes_vec
            .iter()
            .map(|s| s.len() as cl_int)
            .collect::<Vec<i32>>();

        // Create OpenCL device buffers
        let mut name = unsafe {
            Buffer::<cl_uchar>::create(
                &context,
                CL_MEM_READ_ONLY,
                BLOCK_SIZE * current_worker_count,
                ptr::null_mut(),
            )?
        };
        let mut n_len = unsafe {
            Buffer::<cl_int>::create(&context, CL_MEM_READ_ONLY, current_worker_count, ptr::null_mut())?
        };
        let mut output = SvmVec::<cl_uchar>::allocate(&context, BLOCK_SIZE * current_worker_count)?;

        let name_data_vec = {
            let mut vec = Vec::with_capacity(BLOCK_SIZE * current_worker_count);
            for data in name_bytes_vec {
                let left_over = BLOCK_SIZE - data.len();
                vec.extend_from_slice(data);
                vec.extend_from_slice(&vec![0; left_over]);
            }
            vec
        };

        let name_write_event =
            unsafe { queue.enqueue_write_buffer(&mut name, CL_NON_BLOCKING, 0, &name_data_vec, &[]) }?;
        let n_len_write_event =
            unsafe { queue.enqueue_write_buffer(&mut n_len, CL_NON_BLOCKING, 0, &n_len_vec, &[]) }?;
        name_write_event.wait()?;
        n_len_write_event.wait()?;

        let mut speeds = Vec::with_capacity(RUN_TIMES);

        for _ in 0..RUN_TIMES {
            let kernel_event = unsafe {
                ExecuteKernel::new(&kernel)
                    .set_arg(&team)
                    .set_arg(&t_len)
                    .set_arg(&name)
                    .set_arg(&n_len)
                    .set_arg_svm(output.as_mut_ptr())
                    .set_arg(&worker_count_cl)
                    .set_global_work_size(current_worker_count)
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

            let pre_sec = 1_000_000_000 as f32 / duration as f32;
            let speed = pre_sec * current_worker_count as f32;
            speeds.push(speed);

            if !output.is_fine_grained() {
                let unmap_event = unsafe { queue.enqueue_svm_unmap(&output, &[]) }?;
                unmap_event.wait()?;
            }
        }

        let total_speed: f32 = speeds.iter().sum();
        let avg_speed = total_speed / RUN_TIMES as f32;
        results.push((current_worker_count, avg_speed));
    }

    // 计算并输出统计结果
    println!("\n=============== 统计结果 ===============");
    println!("按 worker count 从小到大排序:");
    println!("Worker Count\t平均速度 (pre/sec)");

    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    for (wc, speed) in results {
        println!("{wc}\t\t{speed:.2}");
    }

    Ok(())
}
