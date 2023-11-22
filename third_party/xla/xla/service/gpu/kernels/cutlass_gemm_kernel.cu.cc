/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/kernels/cutlass_gemm_kernel.h"

#include "third_party/gpus/cutlass/include/cutlass/gemm/device/gemm.h"
#include "xla/stream_executor/kernel.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::kernel {

// The most basic CUTLASS f32 gemm kernel.
using CutlassGemm =
    cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float,
                                cutlass::layout::RowMajor, float,
                                cutlass::layout::RowMajor>;

StatusOr<CustomKernel> GetCutlassGemmKernel(PrimitiveType dtype, int32_t m,
                                            int32_t n, int32_t k) {
  if (dtype != PrimitiveType::F32)
    return absl::InvalidArgumentError(
        "Currently cutlass gemm kernel supports only F32 data type");

  // Underlying CUDA kernel implementing gemm operation.
  using GemmKernel = typename CutlassGemm::GemmKernel;

  cutlass::gemm::GemmCoord problem_size = {m, n, k};

  using ThreadblockShape = typename CutlassGemm::ThreadblockShape;
  cutlass::gemm::GemmCoord tile_size = {
      ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK};

  typename CutlassGemm::ThreadblockSwizzle threadblock_swizzle;
  cutlass::gemm::GemmCoord tiled_shape =
      threadblock_swizzle.get_tiled_shape(problem_size, tile_size,
                                          /*split_k_slices=*/1);

  // Compute kernel launch grid size and shared memory requirement.
  dim3 grid = threadblock_swizzle.get_grid_shape(tiled_shape);
  se::BlockDim block_dims(grid.x, grid.y, grid.z);
  se::ThreadDim thread_dims(GemmKernel::kThreadCount, 1, 1);
  size_t shared_memory_bytes = sizeof(typename GemmKernel::SharedStorage);

  // Packs device memory arguments into CUTLASS kernel parameters struct.
  using PackedArgs = StatusOr<std::unique_ptr<se::KernelArgsPackedArrayBase>>;
  auto pack = [problem_size,
               tiled_shape](const se::KernelArgs &args) -> PackedArgs {
    auto *mem_args = Cast<se::KernelArgsDeviceMemoryArray>(&args);

    // Converts DeviceMemoryBase to an opaque `void *` device pointer.
    //
    // TODO(ezhulenev): Add more checks for the number and types of device
    // memory arguments. Right now we unsafely cast and extract buffers.
    auto device_ptr = [&](size_t index) {
      const void *opaque = mem_args->device_memory_ptr(index);
      return static_cast<float *>(const_cast<void *>(opaque));
    };

    // Strides for a row major layout.
    int32_t lda = problem_size.k();
    int32_t ldb = problem_size.n();
    int32_t ldc = problem_size.n();

    // Check if GemmKernel can implement the given problem size.
    cutlass::Status can_implement = GemmKernel::can_implement(
        problem_size,          // problem size
        {device_ptr(0), lda},  // Tensor-ref for source matrix A
        {device_ptr(1), ldb},  // Tensor-ref for source matrix B
        {device_ptr(2), ldc},  // Tensor-ref for source matrix C
        {device_ptr(2), ldc}   // Tensor-ref for destination matrix D
    );

    if (can_implement != cutlass::Status::kSuccess) {
      return absl::InternalError(
          "CUTLASS GemmKernel can not implement gemm for a given problem size");
    }

    // Sanity check that we do not accidentally get a giant parameters struct.
    static_assert(sizeof(GemmKernel::Params) < 512,
                  "GemmKernel::Params struct size is unexpectedly large");

    float alpha = 1.0, beta = 0.0;
    GemmKernel::Params params{
        problem_size,
        tiled_shape,
        {device_ptr(0), lda},  // Tensor-ref for source matrix A
        {device_ptr(1), ldb},  // Tensor-ref for source matrix B
        {device_ptr(2), ldc},  // Tensor-ref for source matrix C
        {device_ptr(2), ldc},  // Tensor-ref for destination matrix D
        {alpha, beta},         // Scalars used in the Epilogue
        /*workspace=*/nullptr,
        /*gather_A_indices=*/nullptr,
        /*gather_B_indices=*/nullptr,
        /*gather_D_indices=*/nullptr};

    return se::PackKernelArgs<GemmKernel::Params>(args.number_of_shared_bytes(),
                                                  params);
  };

  // TODO(ezhulenev): We should generate a more descriptive names for custom
  // kernels, i.e. include tile and dimensions sizes, dtypes, etc.
  se::MultiKernelLoaderSpec kernel_spec(/*arity=*/1, std::move(pack));
  kernel_spec.AddInProcessSymbol(
      reinterpret_cast<void *>(cutlass::Kernel<GemmKernel>), "cutlass_gemm");

  return CustomKernel("cutlass_gemm:f32<-f32xf32", std::move(kernel_spec),
                      block_dims, thread_dims, shared_memory_bytes);
}

}  // namespace xla::gpu::kernel
