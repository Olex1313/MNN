#ifndef D_HEAD
#define D_HEAD 128
#endif
#define BLOCK_SIZE_M 32
#define BLOCK_SIZE_N 32
#define FLT_MAX 3.402823466e+38F

// Load a tile for fixed (batch, head) from [B, S, H, D] layout.
// base_offset = b * S * H * D_HEAD + h * D_HEAD
// stride_seq  = H * D_HEAD  (floats between consecutive tokens for this head)
// start_row   = first token index of this tile
void load_global_tile_mnn(
    __global const float *src,
    __local float *dst,
    const int base_offset,
    const int stride_seq,
    const int start_row,
    const int num_rows,
    const int max_rows,
    const int local_id,
    const int wg_size
) {
    const int d_head_vec = D_HEAD / 4;
    const int total_elements = num_rows * d_head_vec;
    for (int i = local_id; i < total_elements; i += wg_size) {
        int r = i / d_head_vec;
        int c = i % d_head_vec;
        int global_row = start_row + r;

        int dst_offset = (r*D_HEAD) + c*4;
        int src_offset = base_offset + global_row * stride_seq + c*4;

        if (global_row < max_rows) {
            float4 val = vload4(0, src + src_offset);
            vstore4(val, 0, dst+dst_offset);
        } else {
            vstore4((float4)(0.0f), 0, dst+dst_offset);
        }
    }
}

// Q, K, V, O: float32[B, S, H, D]  (MNN layout: seq before heads)
__kernel void flash_attention_v2_mnn_fwd(
    __global const float *Q,
    __global const float *K,
    __global const float *V,
    __global float *O,
    const int B,
    const int H,
    const int L,
    const int S,
    const float scale,
    const int is_causal
) {
    const int batch_idx   = get_global_id(1);
    const int head_idx    = get_global_id(2);
    const int q_block_idx = get_group_id(0);

    if (batch_idx >= B || head_idx >= H) return;

    // [B, S, H, D]: element(b, s, h, d) = b*S*H*D + s*H*D + h*D + d
    // for fixed (b, h): base = b*S*H*D + h*D, token stride = H*D
    const int stride_Q = H * D_HEAD;
    const int stride_K = H * D_HEAD;
    const int base_Q   = batch_idx * L * H * D_HEAD + head_idx * D_HEAD;
    const int base_K   = batch_idx * S * H * D_HEAD + head_idx * D_HEAD;

    const int tid     = get_local_id(0);
    const int wg_size = get_local_size(0);

    __local float Q_tile[BLOCK_SIZE_M * D_HEAD];
    __local float K_tile[BLOCK_SIZE_N * D_HEAD];
    __local float V_tile[BLOCK_SIZE_N * D_HEAD];

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float4 O_acc4[D_HEAD / 4];
    for (int d = 0; d < D_HEAD / 4; ++d) O_acc4[d] = (float4)(0.0f);

    const int q_start_row = q_block_idx * BLOCK_SIZE_M;
    const int num_q_rows  = (q_start_row + BLOCK_SIZE_M > L) ? (L - q_start_row) : BLOCK_SIZE_M;

    load_global_tile_mnn(Q, Q_tile, base_Q, stride_Q, q_start_row, BLOCK_SIZE_M, L, tid, wg_size);
    barrier(CLK_LOCAL_MEM_FENCE);

    const int num_kv_blocks = (S + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;

    for (int k_block = 0; k_block < num_kv_blocks; ++k_block) {
        const int k_start_row = k_block * BLOCK_SIZE_N;
        const int num_k_rows  = (k_start_row + BLOCK_SIZE_N > S) ? (S - k_start_row) : BLOCK_SIZE_N;

        if (is_causal && k_start_row > (q_start_row + BLOCK_SIZE_M - 1)) break;

        load_global_tile_mnn(K, K_tile, base_K, stride_K, k_start_row, BLOCK_SIZE_N, S, tid, wg_size);
        load_global_tile_mnn(V, V_tile, base_K, stride_K, k_start_row, BLOCK_SIZE_N, S, tid, wg_size);
        barrier(CLK_LOCAL_MEM_FENCE);

        if (tid < BLOCK_SIZE_M && tid < num_q_rows) {
            float scores[BLOCK_SIZE_N];
            float m_block = -FLT_MAX;
            const int q_off = tid * D_HEAD;
            for (int j = 0; j < num_k_rows; ++j) {
                if (is_causal && (k_start_row + j) > (q_start_row + tid)) {
                    scores[j] = -FLT_MAX;
                    continue;
                }
                float s = 0.0f;
                const int k_off = j * D_HEAD;
                for (int d = 0; d < D_HEAD / 4; ++d) {
                    s += dot(vload4(d, Q_tile + q_off),
                             vload4(d, K_tile + k_off));
                }
                scores[j] = s * scale;
                m_block = fmax(m_block, scores[j]);
            }

            float m_new = fmax(m_i, m_block);
            float alpha = exp(m_i - m_new);

            for (int d = 0; d < D_HEAD / 4; ++d) O_acc4[d] *= alpha;

            float block_l = 0.0f;
            for (int j = 0; j < num_k_rows; ++j) {
                float e = exp(scores[j] - m_new);
                const int v_off = j * D_HEAD;
                for (int d = 0; d < D_HEAD / 4; ++d) {
                    O_acc4[d] += e * vload4(d, V_tile + v_off);
                }
                block_l += e;
            }

            l_i = l_i * alpha + block_l;
            m_i = m_new;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid < BLOCK_SIZE_M && tid < num_q_rows) {
        // write [B, S, H, D]: base_Q + token * stride_Q + d
        const int global_out_base = base_Q + (q_start_row + tid) * stride_Q;
        const float inv_l = 1.0f / l_i;
        for (int d = 0; d < D_HEAD / 4; ++d) {
            vstore4(O_acc4[d] * inv_l, d, O + global_out_base);
        }
    }
}
