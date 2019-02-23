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

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/service/cpu/custom_call_target_registry.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/dynamic_annotations.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

EIGEN_STRONG_INLINE void argmax_float_1d_xla_impl(void* out, void** data) {
  float* input = static_cast<float*>(data[0]);
  int64 input_size = *static_cast<int64*>(data[1]);

  Eigen::DSizes<Eigen::DenseIndex, 1> in_eig_sizes(input_size);
  TTypes<float, 1>::ConstTensor in_eig(input, in_eig_sizes);

  Eigen::DSizes<Eigen::DenseIndex, 0> out_eig_sizes;
  int64* out_t = static_cast<int64*>(out);
  TTypes<int64, 0>::Tensor out_eig(out_t, out_eig_sizes);

  out_eig = in_eig.argmax(0).cast<int64>();
}

}  // namespace tensorflow

// Implements argmax on CPU. This is called by an XLA custom call, set up by
// index_ops.cc.
extern "C" void TF_EXPORT argmax_float_1d_xla_impl(void* out, void** data) {
  tensorflow::argmax_float_1d_xla_impl(out, data);
}

REGISTER_CUSTOM_CALL_TARGET(argmax_float_1d_xla_impl);
