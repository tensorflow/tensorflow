import triton.language as tl
import triton


@triton.jit
def custom_add(a_ptr):
    tl.store(a_ptr, 1.0)
