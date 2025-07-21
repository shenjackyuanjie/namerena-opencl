use opencl3::{
    command_queue::CommandQueue,
    context::Context,
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, CL_MAP_WRITE, CL_MEM_READ_ONLY},
    svm::SvmVec,
    types::{cl_int, cl_uchar, CL_BLOCKING, CL_NON_BLOCKING},
};
use std::ptr;

use crate::{BLOCK_SIZE, RUN_TIMES};

pub fn run(
    context: &Context,
    queue: &CommandQueue,
    kernel: &Kernel,
    max_worker_count: usize,
    team: String,
) -> anyhow::Result<()> {
    let team_bytes = team.as_bytes();
    let t_len = team_bytes.len() as cl_int;
    let mut team_buffer = unsafe {
        Buffer::<cl_uchar>::create(context, CL_MEM_READ_ONLY, BLOCK_SIZE, ptr::null_mut())?
    };
    let team_write_event = unsafe {
        queue.enqueue_write_buffer(&mut team_buffer, CL_NON_BLOCKING, 0, team_bytes, &[])
    }?;
    team_write_event.wait()?;

    // 存储每个worker count的平均速度
    // let mut results: Vec<(usize, f32)> = Vec::with_capacity(max_worker_count);

    let worker_count_cl = max_worker_count as cl_int;
    let name_raw_vec: Vec<String> = (1..=max_worker_count).map(|i| i.to_string()).collect();

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
            context,
            CL_MEM_READ_ONLY,
            BLOCK_SIZE * max_worker_count,
            ptr::null_mut(),
        )?
    };
    let mut n_len = unsafe {
        Buffer::<cl_int>::create(context, CL_MEM_READ_ONLY, max_worker_count, ptr::null_mut())?
    };
    let mut output = SvmVec::<cl_uchar>::allocate(context, BLOCK_SIZE * max_worker_count)?;

    let name_data_vec = {
        let mut vec = Vec::with_capacity(BLOCK_SIZE * max_worker_count);
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
            ExecuteKernel::new(kernel)
                .set_arg(&team_buffer)
                .set_arg(&t_len)
                .set_arg(&name)
                .set_arg(&n_len)
                .set_arg_svm(output.as_mut_ptr())
                .set_arg(&worker_count_cl)
                .set_global_work_size(max_worker_count)
                // .set_local_work_size(128)
                .enqueue_nd_range(queue)?
        };

        kernel_event.wait()?;
        queue.finish()?;

        if !output.is_fine_grained() {
            unsafe { queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_WRITE, &mut output, &[]) }?;
        }

        let start_time = kernel_event.profiling_command_start()?;
        let end_time = kernel_event.profiling_command_end()?;
        let duration = end_time - start_time;

        let pre_sec = 1_000_000_000_f64 / duration as f64;
        let speed = pre_sec * max_worker_count as f64;
        speeds.push(speed);

        if !output.is_fine_grained() {
            let unmap_event = unsafe { queue.enqueue_svm_unmap(&output, &[]) }?;
            unmap_event.wait()?;
        }
    }

    let total_speed: f64 = speeds.iter().sum();
    let avg_speed = total_speed / RUN_TIMES as f64;
    println!("worker_count: {max_worker_count}, avg_speed: {avg_speed}");

    Ok(())
}


pub fn run_name()
