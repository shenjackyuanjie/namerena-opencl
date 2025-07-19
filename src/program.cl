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
    for (int i = 0; i < 256; i++) {
        val[i] = i;
    }
    for (int i = 0; i < 256; i += 4) {
        vstore4(vload4(0, &g_team_bytes[i]), i, team_bytes);
        vstore4(vload4(0, &all_name_bytes[256 * gid + i]), i, name_bytes);
    }

    int n_len = all_n_len[gid];

    // --- 优化点 1: 消除模运算和相关的分支 ---
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

    // --- 优化点 2: 向量化计算 (无溢出保护) ---
    // 使用向量类型 (uchar4) 进行计算以提高吞吐量
    // 用户确认不需要防止乘法溢出，直接在 uchar4 上进行计算
    for (int i = 0; i < 64; i++) { // 256 bytes / 4 bytes/vector = 64 iterations
        uchar4 temp_val = vload4(i, val);
        temp_val = temp_val * (uchar)181 + (uchar)160;
        vstore4(temp_val, i, val_2);
    }

    local uchar name_nase[40];
    local int b_counter;
    b_counter = 0;

    // --- 优化点 3: 消除不必要的内层分支 (break) ---
    // 将 b_counter 检查合并到循环条件中
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
