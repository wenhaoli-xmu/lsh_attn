from profiler import WallTime
import torch
import argparse
from flash_attn import flash_attn_func


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

    benchmark = WallTime("std-attn(sdpa)", cuda=0)

    # 计算 LSH 相似度
    for ctx_length in [1024 * (2 ** i) for i in range(11)]:

        query = torch.rand((args.batch_size,1,32,128), dtype=torch.bfloat16, device='cuda')
        key = torch.rand((args.batch_size,ctx_length,32,128), dtype=torch.bfloat16, device='cuda')
        value = torch.rand((args.batch_size,ctx_length,32,128), dtype=torch.bfloat16, device='cuda')

        for _ in range(10):
            # Baseline
            with benchmark:
                flash_attn_func(
                    q=query,
                    k=key,
                    v=value,
                    causal=True)

        benchmark.result(postfix=f'-{ctx_length}', detail=True)
        benchmark.reset()