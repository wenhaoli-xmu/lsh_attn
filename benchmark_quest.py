from quest.quest.models.QuestAttention import QuestAttention
from quest.quest.utils.controller import InferenceController
from quest.quest.utils import append_kv
from transformers import AutoConfig
from profiler import WallTime
import argparse
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--page_size", type=int, default=16)
    parser.add_argument("--token_budget", type=int, default=128)
    parser.add_argument("--num_repeats", type=int, default=3)
    args = parser.parse_args()

    benchmark_sketch = WallTime("step-1", cuda=0)
    benchmark_topk = WallTime("step-2", cuda=0)
    benchmark_sparse = WallTime("step-3", cuda=0)

    page_budget = args.token_budget // args.page_size

    config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
    attn = QuestAttention(config, 0).cuda().half()
    iController = InferenceController(
        num_layers=1, 
        num_heads=args.num_heads, 
        head_dim=args.head_dim, 
        page_size=args.page_size, 
        page_budget=page_budget,
        max_seq_len=1024 * 512,
        dtype=torch.float16,
        device='cuda:0')

    # 计算 LSH 相似度
    for ctx_length in [1024 * (2 ** i) for i in range(10)]:        

        for _ in range(args.num_repeats):
            iController.prepare_metadata(ctx_length)

            # 1.pre-filling stage
            iController.begin_forward(ctx_length)
            hidden_states = torch.rand((args.batch_size, ctx_length, args.num_heads * args.head_dim), dtype=torch.float16).cuda()
            attn(hidden_states, use_cache=True, iController=iController)
            iController.end_forward()

            # 2.decoding stage
            iController.begin_forward(1)
            hidden_states = torch.rand((args.batch_size, 1, args.num_heads * args.head_dim), dtype=torch.float16).cuda()
            attn(hidden_states, use_cache=True, iController=iController)
            iController.end_forward()
            iController.clean_states()

        benchmark_sketch.result(postfix=f'-{ctx_length}')
        benchmark_topk.result(postfix=f'-{ctx_length}')
        benchmark_sparse.result(postfix=f'-{ctx_length}')

        benchmark_sketch.reset()
        benchmark_topk.reset()
        benchmark_sparse.reset()

        print("\n")