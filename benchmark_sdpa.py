from profiler import WallTime
import torch
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

    benchmark = WallTime("std-attn(sdpa)", cuda=0)

    # 计算 LSH 相似度
    for ctx_length in [1024 * (2 ** i) for i in range(11)]:
        
        q_hash = torch.randint(0, 256, (args.batch_size, 32, 1, 1), dtype=torch.uint8, device='cuda')
        k_hash = torch.randint(0, 256, (args.batch_size, 32, ctx_length, 1), dtype=torch.uint8, device='cuda')

        query = torch.rand((args.batch_size,32,1,128), dtype=torch.bfloat16, device='cuda')
        key = torch.rand((args.batch_size,32,ctx_length,128), dtype=torch.bfloat16, device='cuda')
        value = torch.rand((args.batch_size,32,ctx_length,128), dtype=torch.bfloat16, device='cuda')

        for _ in range(10):
            # Baseline
            with benchmark:
                torch.nn.functional.scaled_dot_product_attention(
                    query=query,
                    key=key,
                    value=value,
                    is_causal=False)

        benchmark.result(postfix=f'-{ctx_length}', detail=True)
        benchmark.reset()