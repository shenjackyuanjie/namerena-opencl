// 取中值
uchar median(uchar a, uchar b, uchar c) {
    return max(min(a, b), min(max(a, b), c));
}

// 输入: 1~255 长度的 u8 数组
// 输出: 255 长度的 u8 数组
kernel void load_team(
    global const uchar* g_team_bytes,
    const int t_len,
    global const uchar* all_name_bytes,
    global const int* all_n_len,
    // 一个 svm 的 [u8; 256] * worker_count
    global uchar* all_val,
    const int worker_count
) {
    int gid = get_global_id(0);
    if (gid >= worker_count) {
        return;
    }

    // 使用局部内存以加速重复访问
    local uchar val[256];
    local uchar team_bytes[256];
    local uchar name_bytes[256];

    // 初始化val数组
    for (int i = 0; i < 64; i++) {
        uchar4 values = (uchar4)(i*4, i*4+1, i*4+2, i*4+3);
        vstore4(values, i, val);
    }
    // for (int i = 0; i < 256; i += 4) {
    //     vstore4(vload4(0, &g_team_bytes[i]), i, team_bytes);
    //     vstore4(vload4(0, &all_name_bytes[256 * gid + i]), i, name_bytes);
    // }
    // 一次性加载更大的块到本地内存
    // 使用异步预取提高内存访问效率
    event_t team_event, name_event;

    // 异步预取team_bytes
    team_event = async_work_group_copy(
        team_bytes,
        g_team_bytes,
        t_len > 256 ? 256 : t_len,
        0);

    // 异步预取name_bytes
    name_event = async_work_group_copy(
        name_bytes,
        all_name_bytes + 256 * gid,
        256,
        0);

    // 等待数据加载完成
    wait_group_events(2, (event_t[]){team_event, name_event});

    int n_len = all_n_len[gid];

    // 第一个置换循环
    uchar s = 0;
    uchar team_idx = 0;
    for (int i = 0; i < 256; ++i) {
        s += val[i];
        if (team_idx != 0) {
            s += team_bytes[team_idx - 1];
        }

        uchar tmp = val[i];
        val[i] = val[s];
        val[s] = tmp;

        team_idx++;
        if (team_idx >= t_len) {
            team_idx = 0;
        }
    }

    // 第二个置换循环 (重复两次)
    for (int _ = 0; _ < 2; _++) {
        uchar s_inner = 0;
        uchar k = 0;
        for (int i = 0; i < 256; i++) {
            // 使用三元运算符避免分支
            s_inner += (k != 0) ? name_bytes[k - 1] : 0;
            s_inner += val[i];

            uchar tmp = val[i];
            val[i] = val[s_inner];
            val[s_inner] = tmp;

            // 使用三元运算符避免分支预测失败
            k = (k >= n_len) ? 0 : (k + 1);
        }
    }

    local uchar val_2[256];

    for (int i = 0; i < 64; i++) { // 256 bytes / 4 bytes/vector = 64 iterations
        uchar4 temp_val = vload4(i, val);
        temp_val = temp_val * (uchar)181 + (uchar)160;
        vstore4(temp_val, i, val_2);
    }

    local uchar name_nase[40];
    local int b_counter;
    b_counter = 0;

    for (int i = 0; i < 256 && b_counter < 40; i++) {
        if (val_2[i] >= 89 && val_2[i] < 217) {
            name_nase[b_counter] = val_2[i] & 63;
            b_counter++;
        }
    }

    // 将结果从局部内存拷贝回全局内存
    // 这里的注释是原始代码保留的，用于确保编译器不会优化掉name_nase的计算
    // 如果需要最终结果，应该将相关数据写回 all_val
    all_val[256 * gid] = name_nase[0];
}

// 输入: u8[256]
// 输出: 256 长度的 u8 数组
kernel void load_name(
    global const uchar* global_val,
    global const uchar* all_name_bytes,
    global const int* all_n_len,
    // 一个 svm 的 [u8; 256] * worker_count
    global uchar* all_val,
    const int worker_count
) {
    int gid = get_global_id(0);
    if (gid >= worker_count) {
        return;
    }

    // 使用局部内存以加速重复访问
    local uchar val[256];
    local uchar name_bytes[256];

    // 初始化val数组
    for (int i = 0; i < 64; i++) {
        uchar4 values = (uchar4)(i*4, i*4+1, i*4+2, i*4+3);
        vstore4(values, i, val);
    }
    // 一次性加载更大的块到本地内存
    // 使用异步预取提高内存访问效率
    event_t name_event;

    // 异步预取name_bytes
    name_event = async_work_group_copy(
        name_bytes,
        all_name_bytes + 256 * gid,
        256,
        0);

    // 等待数据加载完成
    wait_group_events(2, (event_t[]){name_event});

    int n_len = all_n_len[gid];

    // 第二个置换循环 (重复两次)
    for (int _ = 0; _ < 2; _++) {
        uchar s_inner = 0;
        uchar k = 0;
        for (int i = 0; i < 256; i++) {
            // 使用三元运算符避免分支
            s_inner += (k != 0) ? name_bytes[k - 1] : 0;
            s_inner += val[i];

            uchar tmp = val[i];
            val[i] = val[s_inner];
            val[s_inner] = tmp;

            // 使用三元运算符避免分支预测失败
            k = (k >= n_len) ? 0 : (k + 1);
        }
    }

    local uchar val_2[256];

    for (int i = 0; i < 64; i++) { // 256 bytes / 4 bytes/vector = 64 iterations
        uchar4 temp_val = vload4(i, val);
        temp_val = temp_val * (uchar)181 + (uchar)160;
        vstore4(temp_val, i, val_2);
    }

    local uchar name_nase[40];
    local int b_counter;
    b_counter = 0;

    for (int i = 0; i < 256 && b_counter < 40; i++) {
        if (val_2[i] >= 89 && val_2[i] < 217) {
            name_nase[b_counter] = val_2[i] & 63;
            b_counter++;
        }
    }

    // 将结果从局部内存拷贝回全局内存
    // 这里的注释是原始代码保留的，用于确保编译器不会优化掉name_nase的计算
    // 如果需要最终结果，应该将相关数据写回 all_val
    all_val[256 * gid] = name_nase[0];
}
