/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// Specialization of SpaceToBatchFunctor for a GPUDevice.

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/spacetobatch_functor.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// Shape and padding parameters for space-to-batch and batch-to-space conversion
// GPU kernel.
template <int NUM_BLOCK_DIMS>
struct S2BParameters {
  int32 space_tensor_batch;
  int32 batch_tensor_shape[NUM_BLOCK_DIMS + 2];
  int32 space_tensor_spatial_shape[NUM_BLOCK_DIMS];
  int32 pad_start[NUM_BLOCK_DIMS];
  int32 block_shape[NUM_BLOCK_DIMS];
};

// GPU kernel for space-to-batch (if B2S = false) and batch-to-space conversion
// (if B2S = true).
//
// To simplify template implementation given lack of constexpr if, both the
// input and output pointers are non-const.
template <typename T, int NUM_BLOCK_DIMS, bool B2S>
__global__ void S2B(const int32 nthreads, T* space_tensor_ptr,
                    S2BParameters<NUM_BLOCK_DIMS> args, T* batch_tensor_ptr) {
  CUDA_1D_KERNEL_LOOP(batch_tensor_idx, nthreads) {
    int32 remaining_batch_tensor_idx = batch_tensor_idx;

    int32 batch_tensor_pos[NUM_BLOCK_DIMS + 2];

    for (int dim = NUM_BLOCK_DIMS + 1; dim >= 1; --dim) {
      batch_tensor_pos[dim] =
          remaining_batch_tensor_idx % args.batch_tensor_shape[dim];
      remaining_batch_tensor_idx /= args.batch_tensor_shape[dim];
    }
    batch_tensor_pos[0] = remaining_batch_tensor_idx;

    int32 remaining_block_idx = batch_tensor_pos[0] / args.space_tensor_batch;
    int32 space_tensor_idx = batch_tensor_pos[NUM_BLOCK_DIMS + 1];
    int32 space_tensor_stride = args.batch_tensor_shape[NUM_BLOCK_DIMS + 1];
    const int32 space_tensor_batch_pos =
        batch_tensor_pos[0] % args.space_tensor_batch;
    for (int block_dim = NUM_BLOCK_DIMS - 1; block_dim >= 0; --block_dim) {
      int32 offset = remaining_block_idx;
      if (block_dim > 0) {
        offset %= args.block_shape[block_dim];
      }
      int32 space_tensor_pos =
          batch_tensor_pos[block_dim + 1] * args.block_shape[block_dim] +
          offset - args.pad_start[block_dim];
      if (space_tensor_pos < 0 ||
          space_tensor_pos >= args.space_tensor_spatial_shape[block_dim]) {
        if (B2S == false) {
          // In the space-to-batch case, write zero padding.
          batch_tensor_ptr[batch_tensor_idx] = static_cast<T>(0);
        }
        break;
      }
      space_tensor_idx += space_tensor_stride * space_tensor_pos;
      space_tensor_stride *= args.space_tensor_spatial_shape[block_dim];
      if (block_dim == 0) {
        space_tensor_idx += space_tensor_stride * space_tensor_batch_pos;
        if (B2S == false) {
          batch_tensor_ptr[batch_tensor_idx] =
              ldg(space_tensor_ptr + space_tensor_idx);
        } else {
          space_tensor_ptr[space_tensor_idx] =
              ldg(batch_tensor_ptr + batch_tensor_idx);
        }
      }
      remaining_block_idx /= args.block_shape[block_dim];
    }
  }
}

namespace functor {
template <typename T, int NUM_BLOCK_DIMS, bool B2S>
struct SpaceToBatchFunctor<GPUDevice, T, NUM_BLOCK_DIMS, B2S> {
  using SpaceT = typename std::conditional<B2S, T, const T>::type;
  using BatchT = typename std::conditional<B2S, const T, T>::type;
  Status operator()(
      const GPUDevice& d,
      typename TTypes<SpaceT, NUM_BLOCK_DIMS + 2>::Tensor space_tensor,
      const int64 block_shape[NUM_BLOCK_DIMS],
      const int64 paddings[NUM_BLOCK_DIMS * 2],
      typename TTypes<BatchT, NUM_BLOCK_DIMS + 2>::Tensor batch_tensor) {
    // Kernel execution fails if number of elements is zero.
    if (batch_tensor.size() == 0) {
      return Status::OK();
    }
    S2BParameters<NUM_BLOCK_DIMS> args;
    args.space_tensor_batch = space_tensor.dimension(0);
    for (int block_dim = 0; block_dim < NUM_BLOCK_DIMS; ++block_dim) {
      if (block_shape[block_dim] > std::numeric_limits<int32>::max()) {
        return errors::InvalidArgument("block_shape value exceeds 2^32-1");
      }
      args.block_shape[block_dim] = block_shape[block_dim];
      if (space_tensor.dimension(block_dim + 1) >
          std::numeric_limits<int32>::max()) {
        return errors::InvalidArgument("space_tensor dimension exceeds 2^32-1");
      }
      args.space_tensor_spatial_shape[block_dim] =
          space_tensor.dimension(block_dim + 1);
      if (paddings[block_dim * 2] > std::numeric_limits<int32>::max()) {
        return errors::InvalidArgument("paddings/crops value exceeds 2^32-1");
      }
      args.pad_start[block_dim] = paddings[block_dim * 2];
    }
    int64 total_count = 1;
    for (int dim = 0; dim < NUM_BLOCK_DIMS + 2; ++dim) {
      args.batch_tensor_shape[dim] = batch_tensor.dimension(dim);
      total_count *= args.batch_tensor_shape[dim];
    }
    if (total_count > std::numeric_limits<int32>::max()) {
      return errors::InvalidArgument(
          "number of batch_tensor elements exceeds 2^32-1");
    }
    CudaLaunchConfig config =
        GetCudaLaunchConfig(static_cast<int32>(total_count), d);
    return CudaLaunchKernel(S2B<T, NUM_BLOCK_DIMS, B2S>, config.block_count,
                            config.thread_per_block, 0, d.stream(),
                            config.virtual_thread_count,
                            const_cast<T*>(space_tensor.data()), args,
                            const_cast<T*>(batch_tensor.data()));
  }
};

// Instantiate.
#define INSTANTIATE(NUM_BLOCK_DIMS, T)                                      \
  template struct SpaceToBatchFunctor<GPUDevice, T, NUM_BLOCK_DIMS, false>; \
  template struct SpaceToBatchFunctor<GPUDevice, T, NUM_BLOCK_DIMS, true>;  \
  /**/

#define INSTANTIATE_FOR_T(T) \
  TF_SPACETOBATCH_FOR_EACH_NUM_BLOCK_DIMS(INSTANTIATE, T)

TF_CALL_GPU_NUMBER_TYPES(INSTANTIATE_FOR_T)

#undef INSTANTIATE_FOR_T
#undef INSTANTIATE

}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
