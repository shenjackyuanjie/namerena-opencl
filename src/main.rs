use opencl3::{
    command_queue::{
        CommandQueue, CL_QUEUE_ON_DEVICE, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
        CL_QUEUE_PROFILING_ENABLE,
    },
    context::Context,
    device::{
        get_all_devices, get_device_info, Device, CL_DEVICE_LOCAL_MEM_SIZE,
        CL_DEVICE_MAX_WORK_GROUP_SIZE, CL_DEVICE_MAX_WORK_GROUP_SIZE_AMD, CL_DEVICE_TYPE_ALL,
    },
    kernel::Kernel,
    program::Program,
};

pub mod run;

const PROGRAM_SOURCE: &str = include_str!("program.cl");

const KERNEL_NAME: &str = "load_team";

const BLOCK_SIZE: usize = 256;

// 运行次数
const RUN_TIMES: usize = 10000;

pub fn team_rc4(team: String) -> [u8; 256] {
    if team.len() > 256 {
        panic!("team len < 256")
    }
    let mut val: [u8; 256] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
        71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93,
        94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
        113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
        131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148,
        149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166,
        167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184,
        185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202,
        203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
        221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238,
        239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
    ];
    let team_bytes = team.as_bytes();
    let mut s = 0_u8;
    let t_len = team.len() + 1;
    for i in 0..256 {
        if (i % t_len) != 0 {
            s = s.wrapping_add(team_bytes[(i % t_len) - 1]);
        }
        s = s.wrapping_add(val[i]);
        val.swap(i, s as usize);
    }
    val
}

fn main() -> anyhow::Result<()> {
    let device_id = {
        let all_devices = get_all_devices(CL_DEVICE_TYPE_ALL).expect("can't get any device here");
        if all_devices.len() == 1 {
            let device_id = all_devices[0];
            let name = get_device_info(device_id, opencl3::device::CL_DEVICE_NAME)
                .expect("Failed to get device name")
                .to_string();
            println!("Using device: {name}");
            device_id
        } else {
            println!("Available devices:");
            for (i, &device_id) in all_devices.iter().enumerate() {
                let name = get_device_info(device_id, opencl3::device::CL_DEVICE_NAME)
                    .expect("Failed to get device name")
                    .to_string();
                println!("[{i}] {name}");
            }
            println!("Select device index: ");
            let mut input = String::new();
            std::io::stdin()
                .read_line(&mut input)
                .expect("Failed to read input");
            let index: usize = input.trim().parse().expect("Invalid index");
            all_devices[index]
        }
    };
    let max_worker_count = match get_device_info(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE_AMD) {
        Ok(size) => size.to_size(),
        Err(_) => match get_device_info(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE) {
            Ok(size) => {
                println!("非 amd size 获取成功");
                size.to_size()
            }
            Err(err) => {
                println!("错误: get_device_info failed: {err}\n");
                panic!();
            }
        },
    };
    let max_local_mem = get_device_info(device_id, CL_DEVICE_LOCAL_MEM_SIZE)
        .expect("faild to get max local mem size")
        .to_ulong();
    println!("设备 local mem 大小: {max_local_mem}");
    println!("设备最大队列长度: {max_worker_count}");

    let device = Device::new(device_id);

    // Create a Context on an OpenCL device
    let context = Context::from_device(&device).expect("Context::from_device failed");

    let mut property = CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    // 如果命令行参数包含 -d, 则 | 一个 CL_QUEUE_ON_DEVICE
    let args = std::env::args().collect::<Vec<String>>();
    if args.contains(&"-d".to_string()) {
        property |= CL_QUEUE_ON_DEVICE;
    }
    let queue = match CommandQueue::create_default_with_properties(&context, property, 0) {
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
            println!("OpenCL Program::create_and_build_from_source failed: {err}");
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

    let team_raw = "1234567".to_string();
    run(&context, &queue, &kernel, max_worker_count, team_raw)?;

    Ok(())
}
