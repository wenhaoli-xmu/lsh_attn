import triton


def block_n_config():
    return [
        triton.Config(kwargs={"BLOCK_N": 8}),
        triton.Config(kwargs={"BLOCK_N": 16}),
        triton.Config(kwargs={"BLOCK_N": 32}),
        triton.Config(kwargs={"BLOCK_N": 64}),
        triton.Config(kwargs={"BLOCK_N": 128}),
        triton.Config(kwargs={"BLOCK_N": 256}),
        triton.Config(kwargs={"BLOCK_N": 512}),
        triton.Config(kwargs={"BLOCK_N": 1024}),
        triton.Config(kwargs={"BLOCK_N": 2048}),
        triton.Config(kwargs={"BLOCK_N": 4096})
    ]