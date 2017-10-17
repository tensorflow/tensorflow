/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/contrib/seq2seq/kernels/beam_search_ops.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
__global__ void GatherTreeOpKernel(const int32 batch_size, const int32 max_time,
                                   const int32 beam_width, const T* step_ids,
                                   const T* parent_ids,
                                   const int32* max_sequence_lengths,
                                   const T end_token, T* beams) {
  CUDA_1D_KERNEL_LOOP(i, batch_size * beam_width) {
    const int32 batch = i / beam_width;
    const int32 beam = i % beam_width;

    const int32 max_seq_len_b =
        Eigen::numext::mini(max_time, ldg(max_sequence_lengths + batch));
    if (max_seq_len_b <= 0) {
      continue;
    }

#define GET_IX(time_ix, beam_ix) \
  (batch_size * beam_width * (time_ix) + beam_width * batch + (beam_ix))
    const int32 initial_beam_ix = GET_IX(max_seq_len_b - 1, beam);
    beams[initial_beam_ix] = ldg(step_ids + initial_beam_ix);
    int32 parent = ldg(parent_ids + initial_beam_ix);
    for (int32 level = max_seq_len_b - 2; level >= 0; --level) {
      const int32 level_beam_ix = GET_IX(level, beam);
      const int32 level_parent_ix = GET_IX(level, parent);
      if (parent < 0 || parent > beam_width) {
        beams[level_beam_ix] = -1;
        parent = -1;
      } else {
        beams[level_beam_ix] = ldg(step_ids + level_parent_ix);
        parent = ldg(parent_ids + level_parent_ix);
      }
    }
    bool finished = false;
    for (int32 time = 0; time < max_seq_len_b; ++time) {
      const int32 level_beam_ix = GET_IX(time, beam);
      if (finished) {
        beams[level_beam_ix] = -1;
      } else if (beams[level_beam_ix] == end_token) {
        finished = true;
      }
    }
#undef GET_IX
  }
}

template <typename T>
struct GatherTree<GPUDevice, T> {
  void operator()(OpKernelContext* ctx, const GPUDevice& d,
                  typename TTypes<T, 3>::ConstTensor step_ids,
                  typename TTypes<T, 3>::ConstTensor parent_ids,
                  TTypes<int32>::ConstVec max_sequence_length,
                  const T end_token, typename TTypes<T, 3>::Tensor beams) {
    const int32 max_time = parent_ids.dimension(0);
    const int32 batch_size = parent_ids.dimension(1);
    const int32 beam_width = parent_ids.dimension(2);
    // First kernel launch to zero things out
    beams.device(d) = beams.constant(T(-1));

    CudaLaunchConfig config = GetCudaLaunchConfig(batch_size * beam_width, d);
    // clang-format off
    GatherTreeOpKernel<T>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            batch_size, max_time, beam_width,
            step_ids.data(),
            parent_ids.data(),
            max_sequence_length.data(),
            end_token,
            beams.data());
    // clang-format on
  }
};

#define DEFINE_GPU_SPECS(T) template struct GatherTree<GPUDevice, T>;

DEFINE_GPU_SPECS(int32);
#undef DEFINE_GPU_SPECS

}  // end namespace functor
}  // end namespace tensorflow
#endif  // GOOGLE_CUDA
