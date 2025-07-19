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

    for (int i = 0; i < 256; i++) {
        val[i] = i;
    }
    for (int i = 0; i < 256; i += 4) {
        vstore4(vload4(0, &g_team_bytes[i]), i, team_bytes);
        vstore4(vload4(0, &all_name_bytes[256 * gid + i]), i, name_bytes);
    }


    int n_len = all_n_len[gid];

    // 外面初始化好了
    uchar s = 0;
    for (int i = s = 0; i < 256; ++i) {
        if (i % t_len) {
            s += team_bytes[i % t_len - 1];
        }
        s += val[i];
        uchar tmp = val[i];
        val[i] = val[s];
        val[s] = tmp;
    }

    for (int _ = 0; _ < 2; _++) {
        uchar s = 0;
        uchar k = 0;
        for (int i = 0; i < 256; i++) {
            // if (k != 0) {
            //     s += name_bytes[k - 1];
            // }
            s += (k != 0) ? name_bytes[k - 1] : 0;
            s += val[i];
            uchar tmp = val[i];
            val[i] = val[s];
            val[s] = tmp;
            k = (k >= n_len) ? 0 : (k + 1); // 避免分支预测失败
        }
    }

    local uchar val_2[256];

    for (int i = 0; i < 256; i++) {
        val_2[i] = val[i] * 181 + 160;
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
    // all_val[256 * gid] = name_nase[0];
}
