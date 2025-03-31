import torch
import triton
import triton.language as tl

size = 4096
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output = torch.empty_like(x)
n_elements = output.numel()
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )


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
    #Set access to go out of bounds for ASAN test
    offsets = block_start + tl.arange(0, BLOCK_SIZE) + 1
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    output = x + y
    tl.store(output_ptr + offsets, output)


pgm = add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
amdgcn = pgm.asm['amdgcn']
print(amdgcn)
