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
use std::time::Duration;

const PROGRAM_SOURCE: &str = include_str!("program.cl");

const KERNEL_NAME: &str = "load_team";

const BLOCK_SIZE: usize = 256;

// 运行次数
const RUN_TIMES: usize = 10;

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

    let worker_count: cl_int = max_size as cl_int;
    println!("设备最大队列长度: {max_size} real count: {worker_count}");

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

    let team_raw = "x";
    let team_bytes = team_raw.as_bytes();
    let name_raw_vec = vec!["x"; worker_count as usize];
    let name_bytes_vec = name_raw_vec
        .iter()
        .map(|s| s.as_bytes())
        .collect::<Vec<&[u8]>>();
    let t_len = team_bytes.len() as cl_int;
    let n_len_vec = name_bytes_vec
        .iter()
        .map(|s| s.len() as cl_int)
        .collect::<Vec<i32>>();

    let work_count = name_raw_vec.len();

    println!("开始准备buffer");

    // Create OpenCL device buffers
    let mut team = unsafe {
        Buffer::<cl_uchar>::create(
            &context,
            CL_MEM_READ_ONLY,
            BLOCK_SIZE,
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
    let mut n_len = unsafe {
        Buffer::<cl_int>::create(&context, CL_MEM_READ_ONLY, work_count, ptr::null_mut())?
    };
    let mut output = SvmVec::<cl_uchar>::allocate(&context, BLOCK_SIZE * work_count)?;
    // 准备一下数据, 都给拼成一维数组
    // 填充成 256 * len
    let name_data_vec = {
        let mut vec = Vec::new();
        for data in name_bytes_vec {
            let left_over = BLOCK_SIZE - data.len();
            vec.extend_from_slice(data);
            vec.extend_from_slice(&vec![0; left_over]);
        }
        vec
    };

    println!("开始写入buffer");

    // 阻塞写
    let _team_write_event =
        unsafe { queue.enqueue_write_buffer(&mut team, CL_NON_BLOCKING, 0, team_bytes, &[]) }?;
    _team_write_event.wait()?;
    let _name_write_event =
        unsafe { queue.enqueue_write_buffer(&mut name, CL_NON_BLOCKING, 0, &name_data_vec, &[]) }?;
    _name_write_event.wait()?;
    let _n_len_write_event =
        unsafe { queue.enqueue_write_buffer(&mut n_len, CL_NON_BLOCKING, 0, &n_len_vec, &[]) }?;
    _n_len_write_event.wait()?;

    println!("开始执行kernel，将运行 {} 次", RUN_TIMES);

    // 存储每次运行的时间（内部计时）
    let mut kernel_times = Vec::with_capacity(RUN_TIMES);
    // 存储每次运行的时间（外部计时）
    let mut external_times = Vec::with_capacity(RUN_TIMES);
    // 存储每次运行的速度
    let mut speeds = Vec::with_capacity(RUN_TIMES);


    for i in 0..RUN_TIMES {
        println!("\n--------- 第 {} 次运行 ---------", i + 1);

        // 执行kernel
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

        println!("等待kernel执行完成");
        let start_tick = std::time::Instant::now();
        kernel_event.wait()?;
        let end_tick = std::time::Instant::now();
        queue.finish()?;

        let external_duration = end_tick - start_tick;
        println!("外置计时: {external_duration:?}");
        external_times.push(external_duration);

        if !output.is_fine_grained() {
            unsafe { queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_WRITE, &mut output, &[]) }?;
        }

        let start_time = kernel_event.profiling_command_start()?;
        let end_time = kernel_event.profiling_command_end()?;
        let duration = end_time - start_time;
        let time = std::time::Duration::from_nanos(duration as u64);
        println!("kernel execution duration: {time:?}");
        kernel_times.push(time);

        let pre_sec = 1_000_000_000 as f32 / duration as f32;
        let speed = pre_sec * worker_count as f32;
        println!("kernel execution speed (pre/sec): {speed:?}");
        speeds.push(speed);

        if !output.is_fine_grained() {
            let unmap_event = unsafe { queue.enqueue_svm_unmap(&output, &[]) }?;
            unmap_event.wait()?;
        }
    }

    // 计算并输出统计结果
    println!("\n=============== 统计结果 ===============");

    // 计算内部计时平均值
    let total_kernel_time: Duration = kernel_times.iter().sum();
    let avg_kernel_time = total_kernel_time / RUN_TIMES as u32;

    // 计算外部计时平均值
    let total_external_time: Duration = external_times.iter().sum();
    let avg_external_time = total_external_time / RUN_TIMES as u32;

    // 计算平均速度
    let total_speed: f32 = speeds.iter().sum();
    let avg_speed = total_speed / RUN_TIMES as f32;

    println!("每次运行内部计时:");
    for (i, time) in kernel_times.iter().enumerate() {
        println!("  第 {} 次: {:?}", i + 1, time);
    }

    println!("每次运行外部计时:");
    for (i, time) in external_times.iter().enumerate() {
        println!("  第 {} 次: {:?}", i + 1, time);
    }

    println!("每次运行速度 (pre/sec):");
    for (i, speed) in speeds.iter().enumerate() {
        println!("  第 {} 次: {:.2}", i + 1, speed);
    }

    println!("\n平均内部计时: {avg_kernel_time:?}");
    println!("平均外部计时: {avg_external_time:?}");
    println!("平均速度 (pre/sec): {avg_speed:.2}");

    Ok(())
}

/*
 *
 ❯ cargo run --release
     Finished `release` profile [optimized] target(s) in 0.07s
      Running `target\release\opencl-test.exe`
 设备最大队列长度: 1024 real count: 1024
 队列创建完成
 程序构建成功
 内核创建成功
 开始准备buffer
 开始写入buffer
 开始执行kernel
 等待kernel执行完成
 外置计时: 502µs
 kernel execution duration: 274.24µs
 kernel execution speed (pre/sec): 3733955.8
 */
