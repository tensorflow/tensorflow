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

#ifndef TENSORFLOW_CONTRIB_SEQ2SEQ_KERNELS_BEAM_SEARCH_OPS_H_
#define TENSORFLOW_CONTRIB_SEQ2SEQ_KERNELS_BEAM_SEARCH_OPS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/eigen_activations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
class OpKernelContext;

namespace functor {

template <typename Device, typename T>
struct GatherTree {
  void operator()(OpKernelContext* ctx, const Device& d,
                  typename TTypes<T, 3>::ConstTensor step_ids,
                  typename TTypes<T, 3>::ConstTensor parent_ids,
                  TTypes<int32>::ConstVec max_sequence_lengths,
                  const T end_token, typename TTypes<T, 3>::Tensor beams);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_SEQ2SEQ_KERNELS_BEAM_SEARCH_OPS_H_
