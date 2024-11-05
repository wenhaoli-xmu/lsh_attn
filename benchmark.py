from profiler import WallTime
import triton
import triton.language as tl
import torch
from lsh_kernel import lsh_attn_d1_u8


def lsh_attn(q_hash, k_hash):
    meta = {"dtype": k_hash.dtype, "device": k_hash.device}
    sim = torch.bitwise_not(torch.bitwise_xor(q_hash, k_hash)).int()
    bit_count_table = torch.tensor([bin(i).count('1') for i in range(256)], **meta)
    count_of_ones = bit_count_table[sim]
    return count_of_ones.sum(dim=-1)


if __name__ == '__main__':

    WallTime("lsh-attn(triton)", cuda=0)
    WallTime("lsh-attn(torch)", cuda=0)
    WallTime("std-attn(sdpa)", cuda=0)
    WallTime("sparse-attn(sdpa)", cuda=0)

    # 计算 LSH 相似度
    for ctx_length in [1024 * (2 ** i) for i in range(11)]:
        
        q_hash = torch.randint(0, 256, (1, 32, 1, 1), dtype=torch.uint8, device='cuda')
        k_hash = torch.randint(0, 256, (1, 32, ctx_length, 1), dtype=torch.uint8, device='cuda')

        query = torch.rand((1,32,1,128), dtype=torch.bfloat16, device='cuda')
        key = torch.rand((1,32,ctx_length,128), dtype=torch.bfloat16, device='cuda')
        value = torch.rand((1,32,ctx_length,128), dtype=torch.bfloat16, device='cuda')

        for _ in range(10):
            with WallTime.get("lsh-attn(triton)"):
                attn = lsh_attn_d1_u8(q_hash, k_hash)

            indx = attn.topk(k=ctx_length // 50, dim=-1, sorted=False).indices
            key_topk = torch.gather(key, dim=-2, index=indx.unsqueeze(-1).expand(-1,-1,-1,128))
            val_topk = torch.gather(value, dim=-2, index=indx.unsqueeze(-1).expand(-1,-1,-1,128))

            with WallTime.get("sparse-attn(sdpa)"):
                torch.nn.functional.scaled_dot_product_attention(
                    query=query,
                    key=key_topk,
                    value=val_topk,
                    is_causal=False)

            with WallTime.get("lsh-attn(torch)"):
                lsh_attn(q_hash, k_hash)

            with WallTime.get("std-attn(sdpa)"):
                torch.nn.functional.scaled_dot_product_attention(
                    query=query,
                    key=key,
                    value=value,
                    is_causal=False)

        WallTime.get("lsh-attn(triton)").result(postfix=f'-{ctx_length}')
        WallTime.get("sparse-attn(sdpa)").result(postfix=f'-{ctx_length}')
        WallTime.get("lsh-attn(torch)").result(postfix=f'-{ctx_length}')
        WallTime.get("std-attn(sdpa)").result(postfix=f'-{ctx_length}')

        WallTime.get("lsh-attn(triton)").reset()
        WallTime.get("sparse-attn(sdpa)").reset()
        WallTime.get("lsh-attn(torch)").reset()
        WallTime.get("std-attn(sdpa)").reset()

        print("\n")