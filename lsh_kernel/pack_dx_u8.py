import triton 
import triton.language as tl
import torch


@triton.jit
def pack_dx_u8_kernel(
    x_ptr, o_ptr,
    stride_xn,
    stride_xd,
    stride_on,
    N, D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):

    last_range = tl.arange(0, 8)
    d_range = tl.arange(0, D)
    n_range = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)

    shift = tl.arange(0, 8).to(tl.uint8)
    shift = tl.view(shift, (1, 1, 8))

    # manipulate x
    x_offs = n_range[:, None, None] * stride_xn + d_range[None, :, None] * stride_xd + last_range
    x_data = tl.load(x_ptr + x_offs, mask=x_offs < N * D * 8)
    x_data = x_data.to(tl.uint8) << shift
    x_data = tl.sum(x_data, axis=2)

    # write back to HBM
    o_offs = n_range[:, None] * stride_on + d_range
    tl.store(o_ptr + o_offs, value=x_data, mask=o_offs < N * D)


def pack_dx_u8(x):
    last_dim = x.shape[-1]
    other_dims = x.shape[:-1]
    num_threads = 512

    assert last_dim % 8 == 0, f"suppose `last_dim % 8 == 0`"
    assert num_threads % last_dim == 0, f"suppose `512 % last_dim == 0`"
    assert x.dtype == torch.bool

    BLOCK_N = num_threads // last_dim

    x = x.view(-1, last_dim // 8, 8).contiguous()
    o = torch.zeros(x.shape[:-1], dtype=torch.uint8, device=x.device)

    grid = (triton.cdiv(x.shape[0], BLOCK_N),)

    with torch.cuda.device(x.device):
        pack_dx_u8_kernel[grid](
            x, o,
            x.stride(0),
            x.stride(1),
            o.stride(0),
            x.shape[0], x.shape[1],
            BLOCK_N)

    return o.view(*other_dims, last_dim // 8).contiguous()


if __name__ == '__main__':
    x = torch.randn((4096, 128), device='cuda') > 0
    result = pack_dx_u8(x)
    print(result)