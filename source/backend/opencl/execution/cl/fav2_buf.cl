#ifndef D_HEAD
#define D_HEAD 128
#endif
#ifndef BLOCK_SIZE_M
#define BLOCK_SIZE_M 32
#endif
#ifndef BLOCK_SIZE_N
#define BLOCK_SIZE_N 32
#endif
#ifndef THREADS_PER_ROW
#define THREADS_PER_ROW 4
#endif

#define WG_SIZE (BLOCK_SIZE_M * THREADS_PER_ROW)
#define D_SLICE (D_HEAD / THREADS_PER_ROW)
#define D_SLICE_VEC (D_SLICE / 4)
#define FLT_MAX 3.402823466e+38F

void load_global_tile(
    __global const float *src,
    __local float *dst,
    const int base_offset,
    const int stride_seq,
    const int start_row,
    const int num_rows,
    const int max_rows,
    const int local_id
) {
    const int d_head_vec = D_HEAD / 4;
    const int total_elements = num_rows * d_head_vec;
    for (int i = local_id; i < total_elements; i += WG_SIZE) {
        int r = i / d_head_vec;
        int c = i % d_head_vec;
        int global_row = start_row + r;

        int dst_offset = r * D_HEAD + c * 4;
        int src_offset = base_offset + global_row * stride_seq + c * 4;

        if (global_row < max_rows) {
            vstore4(vload4(0, src + src_offset), 0, dst + dst_offset);
        } else {
            vstore4((float4)(0.0f), 0, dst + dst_offset);
        }
    }
}

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

    const int tid    = get_local_id(0);
    const int row_id = tid / THREADS_PER_ROW;
    const int d_id   = tid % THREADS_PER_ROW;
    const int d_start     = d_id * D_SLICE;
    const int d_start_vec = d_id * D_SLICE_VEC;

    const int stride_seq = H * D_HEAD;
    const int base_Q = batch_idx * L * stride_seq + head_idx * D_HEAD;
    const int base_K = batch_idx * S * stride_seq + head_idx * D_HEAD;

    __local float Q_tile[BLOCK_SIZE_M * D_HEAD];
    __local float KV_tile[BLOCK_SIZE_N * D_HEAD];

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float4 O_acc4[D_SLICE_VEC];
    for (int d = 0; d < D_SLICE_VEC; ++d) O_acc4[d] = (float4)(0.0f);

    const int q_start_row = q_block_idx * BLOCK_SIZE_M;
    const int num_q_rows  = min(BLOCK_SIZE_M, L - q_start_row);

    load_global_tile(Q, Q_tile, base_Q, stride_seq, q_start_row, BLOCK_SIZE_M, L, tid);
    barrier(CLK_LOCAL_MEM_FENCE);

    const int num_kv_blocks = (S + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;

    for (int k_block = 0; k_block < num_kv_blocks; ++k_block) {
        const int k_start_row = k_block * BLOCK_SIZE_N;
        const int num_k_rows  = min(BLOCK_SIZE_N, S - k_start_row);

        if (is_causal && k_start_row > (q_start_row + BLOCK_SIZE_M - 1)) break;

        load_global_tile(K, KV_tile, base_K, stride_seq, k_start_row, BLOCK_SIZE_N, S, tid);
        barrier(CLK_LOCAL_MEM_FENCE);

        float scores[BLOCK_SIZE_N];
        float m_block = -FLT_MAX;

        if (row_id < num_q_rows) {
            const int q_off = row_id * D_HEAD;
            for (int j = 0; j < num_k_rows; ++j) {
                if (is_causal && (k_start_row + j) > (q_start_row + row_id)) {
                    scores[j] = -FLT_MAX;
                    continue;
                }
                float s = 0.0f;
                const int k_off = j * D_HEAD;
                for (int d = 0; d < D_HEAD / 4; ++d) {
                    s += dot(vload4(d, Q_tile + q_off),
                             vload4(d, KV_tile + k_off));
                }
                scores[j] = s * scale;
                m_block = fmax(m_block, scores[j]);
            }

            float m_new = fmax(m_i, m_block);
            float alpha = exp(m_i - m_new);
            for (int d = 0; d < D_SLICE_VEC; ++d) O_acc4[d] *= alpha;

            float block_l = 0.0f;
            for (int j = 0; j < num_k_rows; ++j) {
                scores[j] = exp(scores[j] - m_new);
                block_l += scores[j];
            }
            l_i = l_i * alpha + block_l;
            m_i = m_new;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        load_global_tile(V, KV_tile, base_K, stride_seq, k_start_row, BLOCK_SIZE_N, S, tid);
        barrier(CLK_LOCAL_MEM_FENCE);

        if (row_id < num_q_rows) {
            for (int j = 0; j < num_k_rows; ++j) {
                const float e = scores[j];
                const int v_off = j * D_HEAD + d_start;
#pragma unroll
                for (int d = 0; d < D_SLICE_VEC; ++d) {
                    O_acc4[d] += e * vload4(d, KV_tile + v_off);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row_id < num_q_rows) {
        const int global_out_base = base_Q + (q_start_row + row_id) * stride_seq;
        const float inv_l = 1.0f / l_i;
#pragma unroll
        for (int d = 0; d < D_SLICE_VEC; ++d) {
            vstore4(O_acc4[d] * inv_l, d_start_vec + d, O + global_out_base);
        }
    }
}
