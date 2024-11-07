import triton
import triton.language as tl
import torch
from .utils import block_n_config


@triton.autotune(configs=block_n_config(), key=[])
@triton.jit
def dx_u8_kernel(
    q_ptr, k_ptr, o_ptr, 
    stride_qb,
    stride_kb,
    stride_kn,
    stride_ob,
    B, N, D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b_idx, n_idx = tl.program_id(0), tl.program_id(1)

    n_range = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    d_range = tl.arange(0, D)

    bit_1 = tl.full((1,), value=1, dtype=tl.uint8)

    # load query
    q_offs = b_idx * stride_qb + d_range[None, :]
    q_data = tl.load(q_ptr + q_offs)

    # load key
    k_offs = b_idx * stride_kb + n_range[None, :, None] * stride_kn + d_range[None, None, :]
    k_data = tl.load(k_ptr + k_offs, mask=k_offs < B * N * D)

    # lsh attention
    xor_result = ~(q_data[:, None, :] ^ k_data)
    accum = tl.zeros((1, BLOCK_N), dtype=tl.uint8)
        
    for _ in range(8):
        accum += tl.sum(xor_result & bit_1, axis=2)
        xor_result = xor_result >> bit_1

    o_offs = b_idx * stride_ob + n_range[None, :]
    tl.store(o_ptr + o_offs, accum, mask=o_offs < B * N)


def lsh_attn_dx_u8(q_hash, k_hash):

    batch_size, num_heads, q_len, dim = q_hash.shape
    _, _, k_len, _ = k_hash.shape

    assert q_len == 1, f"only support single query"
    
    q_hash = q_hash.view(-1, dim)  
    k_hash = k_hash.view(-1, k_len, dim) 

    result = torch.zeros((q_hash.shape[0], k_len), dtype=torch.uint8, device=q_hash.device)
    
    q_hash = q_hash.contiguous()
    k_hash = k_hash.contiguous()

    grid = lambda META: (
        batch_size * num_heads,
        triton.cdiv(k_len, META['BLOCK_N']))

    with torch.cuda.device(q_hash.device):
        dx_u8_kernel[grid](
            q_hash, k_hash, result,
            q_hash.stride(0),
            k_hash.stride(0),
            k_hash.stride(1),
            result.stride(0),
            k_hash.shape[0], k_hash.shape[1], k_hash.shape[2])
    
    return result.view(batch_size, num_heads, k_len)