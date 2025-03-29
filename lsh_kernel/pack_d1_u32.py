import triton 
import triton.language as tl
import torch


TORCH_DTYPE = torch.int32
TRITON_DTYPE = tl.uint32
NUM_BITS = 32


@triton.jit
def pack_d1_u32_kernel(
    x_ptr, o_ptr, # (n, 32), (n,)
    stride_xn,
    N, BLOCK_N: tl.constexpr,
):
    last_range = tl.arange(0, NUM_BITS)
    n_range = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)

    shift = tl.arange(0, NUM_BITS).to(TRITON_DTYPE)
    shift = tl.view(shift, (1, NUM_BITS))

    # manipulate x
    x_offs = n_range[:, None] * stride_xn + last_range
    x_data = tl.load(x_ptr + x_offs, mask=x_offs < N * NUM_BITS)
    x_data = x_data.to(TRITON_DTYPE) << shift
    x_data = tl.sum(x_data, axis=1)

    # write back to HBM
    tl.store(o_ptr + n_range, value=x_data, mask=n_range < N)


def pack_d1_u32(x: torch.BoolTensor):
    last_dim = x.shape[-1]
    other_dims = x.shape[:-1]
    num_threads = 512

    assert last_dim == NUM_BITS, f"suppose `last_dim == {NUM_BITS}`"
    assert x.dtype == torch.bool

    BLOCK_N = num_threads // NUM_BITS
    x = x.view(-1, NUM_BITS).contiguous()
    o = torch.zeros(x.shape[:-1], dtype=TORCH_DTYPE, device=x.device)

    grid = (triton.cdiv(x.shape[0], BLOCK_N),)

    with torch.cuda.device(x.device):
        pack_d1_u32_kernel[grid](
            x, o,
            x.stride(0),
            x.shape[0], BLOCK_N)

    return o.view(*other_dims).contiguous()


if __name__ == '__main__':
    x = torch.randn((4096, 32), device='cuda') > 0
    result = pack_d1_u32(x)
    print(result)