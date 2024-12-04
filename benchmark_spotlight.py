from profiler import WallTime
import triton
import triton.language as tl
import torch
from lsh_kernel import lsh_attn_dx_u8
from lsh_kernel import pack_dx_u8
import argparse


def slow_lsh_attn(q_hash, k_hash):
    meta = {"dtype": k_hash.dtype, "device": k_hash.device}
    sim = torch.bitwise_not(torch.bitwise_xor(q_hash, k_hash)).int()
    bit_count_table = torch.tensor([bin(i).count('1') for i in range(256)], **meta)
    count_of_ones = bit_count_table[sim]
    return count_of_ones.sum(dim=-1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    benchmark_pack = WallTime("pack bits(triton)", cuda=0)
    benchmark_lsh = WallTime("lsh-attn(triton)", cuda=0)
    benchmark_topk = WallTime("topk", cuda=0)
    benchmark_sparse = WallTime("sparse-attn(sdpa)", cuda=0)

    # 计算 LSH 相似度
    for ctx_length in [1024 * (2 ** i) for i in range(11)]:
        
        q_hash = torch.randint(0, 256, (args.batch_size, 32, 1, 1), dtype=torch.uint8, device='cuda')
        k_hash = torch.randint(0, 256, (args.batch_size, 32, ctx_length, 1), dtype=torch.uint8, device='cuda')

        query = torch.rand((args.batch_size,32,1,128), dtype=torch.bfloat16, device='cuda')
        fake = torch.rand((args.batch_size, 32, 1, 128), dtype=torch.bfloat16, device='cuda')
        key_or_value = torch.rand((args.batch_size,32,ctx_length,128), dtype=torch.bfloat16, device='cuda')

        for _ in range(10):

            # Step 1: Pack bits
            with benchmark_pack:
                pack_dx_u8(fake > 0) # pack query
                pack_dx_u8(fake > 0) # pack key

            # Step 2: LSH attention
            with benchmark_lsh:
                attn = lsh_attn_dx_u8(q_hash, k_hash)

            # Step 3: Top-K retrieve
            with benchmark_topk:
                indx = attn.topk(k=ctx_length // 50, dim=-1, sorted=False).indices.unsqueeze(-1).expand(-1,-1,-1,128)
                key_topk = torch.gather(key_or_value, dim=-2, index=indx)
                val_topk = torch.gather(key_or_value, dim=-2, index=indx)

            # Step 4: Sparse Attention
            with benchmark_sparse:
                torch.nn.functional.scaled_dot_product_attention(
                    query=query,
                    key=key_topk,
                    value=val_topk,
                    is_causal=False)

        benchmark_pack.result(postfix=f'-{ctx_length}')
        benchmark_lsh.result(postfix=f'-{ctx_length}')
        benchmark_topk.result(postfix=f'-{ctx_length}')
        benchmark_sparse.result(postfix=f'-{ctx_length}')

        benchmark_pack.reset()
        benchmark_lsh.reset()
        benchmark_topk.reset()
        benchmark_sparse.reset()

        print("\n")