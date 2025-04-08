import pytest
import torch

import triton

from triton._internal_testing import is_hip

num_ctas_list = [1]

GPU_DIALECT = "ttg"

if is_hip():
    THREADS_PER_WARP = triton.runtime.driver.active.get_current_target().warp_size
else:
    THREADS_PER_WARP = 32


class BlockedLayout:

    def __init__(self, size_per_thread, threads_per_warp, warps_per_cta, order, ctas_per_cga, cta_split_num, cta_order):
        self.sz_per_thread = size_per_thread
        self.threads_per_warp = threads_per_warp
        self.warps_per_cta = warps_per_cta
        self.order = order
        self.ctas_per_cga = ctas_per_cga
        self.cta_split_num = cta_split_num
        self.cta_order = cta_order

    def __str__(self):
        return f"#{GPU_DIALECT}.blocked<{{sizePerThread={self.sz_per_thread}, threadsPerWarp={self.threads_per_warp}, warpsPerCTA={self.warps_per_cta}, order={self.order}, CTAsPerCGA={self.ctas_per_cga}, CTASplitNum={self.cta_split_num}, CTAOrder={self.cta_order}}}>"


# -----------------------
# test extract slice
# -----------------------

extract_layout = [
    BlockedLayout([1, 8], [16, 4], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([2, 2], [64, 1], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([2, 2], [16, 4], [4, 1], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 8], [16, 4], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    # FIXME(Lezcano): This layout errors out
    #BlockedLayout([1, 8], [16, 4], [4, 1], [0, 1], [1, 1], [1, 1], [0, 1]),
]
blocked_layout = [
    BlockedLayout([1, 8], [16, 4], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([2, 2], [16, 4], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([2, 2], [16, 4], [2, 2], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 8], [16, 4], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 8], [16, 4], [4, 1], [0, 1], [1, 1], [1, 1], [0, 1]),
]


@pytest.mark.parametrize("M, N, M_tile_size, N_tile_size, M_tile_offset, N_tile_offset",
                         [[256, 256, 256, 32, 0, 32], [128, 128, 128, 64, 0, 64]])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("extract_layout", extract_layout)
@pytest.mark.parametrize("blocked_layout", blocked_layout)
def test_extract_slice(dtype, M, N, M_tile_size, N_tile_size, M_tile_offset, N_tile_offset, blocked_layout,
                       extract_layout, device='cuda'):
    if not is_hip():
        pytest.skip("extract_slice is AMD specific instruction.")

    ir = f"""
    #blocked = {blocked_layout}
    #extract_layout = {extract_layout}
    module attributes {{"ttg.num-ctas" = 1, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = {str(64)} : i32}} {{
    tt.func public @kernel(%arg0: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}) {{
        %cst = arith.constant dense<{N}> : tensor<{M}x1xi32, #blocked>
        %cst_n = arith.constant dense<{N_tile_size}> : tensor<{M_tile_size}x1xi32, #blocked>
        %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>>
        %42 = tt.make_range {{end = {M_tile_size} : i32, start = 0 : i32}} : tensor<{M_tile_size}xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>>
        %1 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>>
        %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<{M}x{N}x!tt.ptr<f16>, #blocked>
        %4 = tt.expand_dims %0 {{axis = 1 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>> -> tensor<{M}x1xi32, #blocked>
        %43 = tt.expand_dims %42 {{axis = 1 : i32}} : tensor<{M_tile_size}xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>> -> tensor<{M_tile_size}x1xi32, #blocked>
        %5 = arith.muli %4, %cst : tensor<{M}x1xi32, #blocked>
        %44 = arith.muli %43, %cst_n : tensor<{M_tile_size}x1xi32, #blocked>
        %6 = tt.expand_dims %1 {{axis = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>> -> tensor<1x{M}xi32, #blocked>
        %7 = tt.broadcast %6 : tensor<1x{M}xi32, #blocked> -> tensor<{M}x{N}xi32, #blocked>
        %8 = tt.broadcast %5 : tensor<{M}x1xi32, #blocked> -> tensor<{M}x{N}xi32, #blocked>
        %9 = arith.addi %8, %7 : tensor<{M}x{N}xi32, #blocked>
        %33 = tt.make_range {{end = {N_tile_size} : i32, start = 0 : i32}} : tensor<{N_tile_size}xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>>
        %34 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<{M_tile_size}x{N_tile_size}x!tt.ptr<f16>, #blocked>
        %37 = tt.expand_dims %33 {{axis = 0 : i32}} : tensor<{N_tile_size}xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>> -> tensor<1x{N_tile_size}xi32, #blocked>
        %38 = tt.broadcast %37 : tensor<1x{N_tile_size}xi32, #blocked> -> tensor<{M_tile_size}x{N_tile_size}xi32, #blocked>
        %39 = tt.broadcast %44 : tensor<{M_tile_size}x1xi32, #blocked> -> tensor<{M_tile_size}x{N_tile_size}xi32, #blocked>
        %40 = arith.addi %38, %39 : tensor<{M_tile_size}x{N_tile_size}xi32, #blocked>
        %10 = tt.addptr %2, %9 : tensor<{M}x{N}x!tt.ptr<f16>, #blocked>, tensor<{M}x{N}xi32, #blocked>
        %11 = tt.load %10 {{cache = 1 : i32, evict = 1 : i32, isVolatile = false}} : tensor<{M}x{N}x!tt.ptr<f16>, #blocked>
        %12 = ttg.convert_layout %11 : tensor<{M}x{N}xf16, #blocked> -> tensor<{M}x{N}xf16, #extract_layout>
        %13 = amdgpu.extract_slice %12 [{M_tile_offset}, {N_tile_offset}] : tensor<{M}x{N}xf16, #extract_layout> to tensor<{M_tile_size}x{N_tile_size}xf16, #extract_layout>
        %14 = ttg.convert_layout %13 : tensor<{M_tile_size}x{N_tile_size}xf16, #extract_layout> -> tensor<{M_tile_size}x{N_tile_size}xf16, #blocked>
        %15 = tt.addptr %34, %40 : tensor<{M_tile_size}x{N_tile_size}x!tt.ptr<f16>, #blocked>, tensor<{M_tile_size}x{N_tile_size}xi32, #blocked>
        tt.store %15, %14 : tensor<{M_tile_size}x{N_tile_size}x!tt.ptr<f16>, #blocked>
        tt.return
    }}
    }}
    """
    x = torch.randn((M, N), device=device, dtype=torch.float16)
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        kernel = triton.compile(f.name)

    extract_slice = torch.empty((M_tile_size, N_tile_size), device=device, dtype=torch.float16)

    kernel[(1, 1, 1)](x.data_ptr(), extract_slice)
    test_result = torch.equal(x[M_tile_offset:M_tile_size + M_tile_offset, N_tile_offset:N_tile_offset + N_tile_size],
                              extract_slice)
    assert test_result
