from quest.quest.models import QuestAttention
from transformers import AutoConfig
from profiler import WallTime
import argparse
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    benchmark_sketch = WallTime("step-1", cuda=0)
    benchmark_topk = WallTime("step-2", cuda=0)
    benchmark_sparse = WallTime("step-3", cuda=0)

    config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
    attn = QuestAttention(config, 0)

    # 计算 LSH 相似度
    for ctx_length in [1024 * (2 ** i) for i in range(11)]:

        keys = torch.rand((args.batch_size,32,ctx_length,128), dtype=torch.bloat16, device='cuda')
        vals = torch.rand((args.batch_size,32,ctx_length,128), dtype=torch.bfloat16, device='cuda')
        hidden_states = torch.rand((args.batch_size,ctx_length,4096), dtype=torch.bfloat16, device='cuda')

        for _ in range(10):
            attn(hidden_states, past_key_value=(keys, vals), use_cache=True)

        benchmark_sketch.result(postfix=f'-{ctx_length}')
        benchmark_topk.result(postfix=f'-{ctx_length}')
        benchmark_sparse.result(postfix=f'-{ctx_length}')

        benchmark_sketch.reset()
        benchmark_topk.reset()
        benchmark_sparse.reset()

        print("\n")