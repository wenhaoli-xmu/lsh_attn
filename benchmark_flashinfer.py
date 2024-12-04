import torch
import flashinfer
from profiler import WallTime
import argparse

num_repeat = 10
num_qo_heads = 32
num_kv_heads = 32
head_dim = 128
page_size = 16

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    benchmark = WallTime("flashinfer(page)", cuda=0)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device='cuda:0')
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")

    for prompt_length in [1024 * (2 ** i) for i in range(12)]:
        kv_cache_pages = prompt_length // page_size
        max_num_pages = kv_cache_pages * args.batch_size
        
        kv_page_indices = torch.arange(max_num_pages).int().to('cuda:0')
        kv_page_indptr = torch.tensor(list(range(0, max_num_pages+1, kv_cache_pages)), dtype=torch.int32, device="cuda:0")
        kv_last_page_len = torch.tensor([page_size] * args.batch_size, dtype=torch.int32, device="cuda:0")
        kv_cache = torch.randn(max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda:0")

        decode_wrapper.plan(
            kv_page_indptr,
            kv_page_indices,
            kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            pos_encoding_mode="NONE",
            data_type=torch.float16
        )
        outputs = []
        q = torch.randn(args.batch_size, num_qo_heads, head_dim).bfloat16().to('cuda:0')
        
        for i in range(num_repeat):
            with benchmark:
                o = decode_wrapper.run(q, kv_cache)

        benchmark.result(postfix=f'-{prompt_length}', detail=True)
        benchmark.reset()

        