
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


    // int base = 256 * gid;
    // for (int i = 0; i < 256; i += 4) {
    //     uchar4 team_data = vload4(i, g_team_bytes);
    //     uchar4 name_data = vload4(base + i, all_name_bytes);
    //     vstore4(team_data, i, team_bytes);
    //     vstore4(name_data, i, name_bytes);
    //     val[i] = i;
    //     val[i+1] = i+1;
    //     val[i+2] = i+2;
    //     val[i+3] = i+3;
    // }

    // // 修正优化：分离val初始化和向量加载
    // for (int i = 0; i < 256; i++) {
    //     val[i] = i; // 保持连续写入模式
    // }

    // const int base = 256 * gid;
    // for (int i = 0; i < 256; i += 4) {
    //     const uchar4 team_data = vload4(i/4, g_team_bytes); // 修正索引计算
    //     const uchar4 name_data = vload4((base + i)/4, all_name_bytes);
    //     vstore4(team_data, i/4, team_bytes);
    //     vstore4(name_data, i/4, name_bytes);
    // }

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
    for (int i = 0; i < 256; i += 1) {
        if (val_2[i] >= 89 && val_2[i] < 217) {
            name_nase[b_counter] = val_2[i] & 63;
            b_counter++;
            if (b_counter >= 40) {
                break;
            }
        }
    }

    // 将结果从局部内存拷贝回全局内存

    // for (int i = 0; i < 256; i += 4) {
    //     vstore4(vload4(0, &val_2[i]), i, &all_val[256 * gid + i]);
    // }
    // 这里这么整一下, 防止他优化掉最后的这点东西
    // all_val[256 * gid] = name_nase[0];
}
