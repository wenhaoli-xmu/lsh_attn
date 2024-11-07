from lsh_kernel import pack_dx_u8
from profiler import WallTime
import torch


if __name__ == '__main__':


    for prompt_length in [1024 * (2 ** i) for i in range(12)]:

        wt_triton = WallTime("triton", cuda=0)
        wt_torch = WallTime("torch", cuda=0)

        x = torch.randn((prompt_length, 128), device='cuda') > 0

        for _ in range(10):
            
            with wt_triton:
                result1 = pack_dx_u8(x)
            
            with wt_torch:
                x = x.type(torch.uint8).unflatten(-1, (-1, 8))
                shift = torch.arange(0, 8, dtype=x.dtype, device=x.device)[None, None, :]
                x <<= shift
                result2 = x.sum(-1)

        wt_triton.result(detail=True, postfix=f"/prompt-{prompt_length}")
        wt_torch.result(detail=True, postfix=f"/prompt-{prompt_length}")
        print('\n')