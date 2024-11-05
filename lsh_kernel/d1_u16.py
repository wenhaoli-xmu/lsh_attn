import triton
import triton.language as tl
import torch
from .utils import block_n_config


@triton.autotune(configs=block_n_config(), key=[])
@triton.jit
def d1_u16_kernel(
    q_ptr, k_ptr, o_ptr,
    stride_kb,
    stride_ob,
    B, N,
    BLOCK_N: tl.constexpr,
):
    b_idx, n_idx = tl.program_id(0), tl.program_id(1)
    n_range = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)

    bit_1 = tl.full((1,), value=1, dtype=tl.uint16)

    # load query hash
    q_data = tl.load(q_ptr + b_idx)

    # load key hash
    k_offs = b_idx * stride_kb + n_range
    k_data = tl.load(k_ptr + k_offs, mask=k_offs < B * N)

    # attention
    xor_result = ~(q_data ^ k_data)
    accum = tl.zeros((BLOCK_N,), dtype=tl.int64)
    for _ in range(16):
        accum += xor_result & bit_1
        xor_result = xor_result >> bit_1

    # store
    o_offs = b_idx + n_range
    tl.store(o_ptr + o_offs, accum, mask=o_offs < B * N)


def lsh_attn_d1_u16(q_hash, k_hash):
    batch_size, num_heads, q_len, dim = q_hash.shape
    _, _, k_len, _ = k_hash.shape

    assert q_len == 1, f"only support single query."
    assert dim == 1, f"dedicated for last dim = 1"
    
    q_hash = q_hash.ravel()
    k_hash = k_hash.view(-1, k_len)

    result = torch.zeros((q_hash.shape[0], k_len), dtype=torch.uint8, device=q_hash.device)
    
    q_hash = q_hash.contiguous()
    k_hash = k_hash.contiguous()

    grid = lambda META: (
        batch_size * num_heads,
        triton.cdiv(k_len, META['BLOCK_N']))

    d1_u8_kernel[grid](
        q_hash, k_hash, result,
        k_hash.stride(0),
        result.stride(0),
        k_hash.shape[0], k_hash.shape[1])
    
    return result.view(batch_size, num_heads, k_len)
