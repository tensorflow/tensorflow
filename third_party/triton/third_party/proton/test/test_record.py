import torch
import pathlib

import triton
import triton.language as tl
import triton.profiler.language as pl


def test_proton_record(tmp_path: pathlib.Path):

    @triton.jit
    def add_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        pl.record(True, 0)
        y = tl.load(y_ptr + offsets, mask=mask)
        pl.record(False, 0)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    torch.manual_seed(0)
    size = 2**12
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = (1, 1, 1)
    pgm = add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    ttir = pgm.asm['ttir']
    assert "proton.record() {isStart = true, regionId = 0 : i32}" in ttir
    assert "proton.record() {isStart = false, regionId = 0 : i32}" in ttir
