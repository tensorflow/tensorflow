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
#include "tensorflow/compiler/tf2xla/xla_local_runtime_context.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/gather_functor.h"
#include "tensorflow/core/platform/dynamic_annotations.h"

namespace tensorflow {

EIGEN_STRONG_INLINE void gather_float_int64_xla_impl(float* out, void** data) {
  // data is managed by the JIT code so msan can't tell it's initialized.
  TF_ANNOTATE_MEMORY_IS_INITIALIZED(data, 6 * sizeof(void*));

  int64 indices_size = *static_cast<int64*>(data[1]);
  int64 params_x = *static_cast<int64*>(data[2]);
  int64 params_y = *static_cast<int64*>(data[3]);

  float* in = static_cast<float*>(data[4]);

  int64* indices = static_cast<int64*>(data[5]);
  Eigen::DSizes<Eigen::DenseIndex, 2> in_eig_sizes;
  in_eig_sizes[0] = params_x;
  in_eig_sizes[1] = params_y;
  tensorflow::TTypes<float, 2>::ConstMatrix in_eig(in, in_eig_sizes);

  Eigen::DSizes<Eigen::DenseIndex, 1> indices_eig_sizes;
  indices_eig_sizes[0] = indices_size;
  tensorflow::TTypes<int64>::ConstFlat indices_eig(indices, indices_eig_sizes);

  Eigen::DSizes<Eigen::DenseIndex, 2> out_eig_sizes;
  out_eig_sizes[0] = indices_size;
  out_eig_sizes[1] = params_y;
  tensorflow::TTypes<float>::Matrix out_eig(out, out_eig_sizes);

  tensorflow::functor::GatherFunctorCPU<float, int64> f;
  const int64 bad_i = f(in_eig, indices_eig, out_eig);
  if (bad_i != -1) {
    tensorflow::XlaLocalRuntimeContext* runtime_context =
        static_cast<tensorflow::XlaLocalRuntimeContext*>(data[0]);
    runtime_context->error = true;
    runtime_context->error_msg = "Invalid index for gather";
    for (int i = 0; i < out_eig.size(); ++i) out[i] = 0;
  }
}

}  // namespace tensorflow

// Implements gather on CPU. This is called by an XLA custom call, set up by
// gather_op.cc.
extern "C" void __attribute__((visibility("default")))
gather_float_int64_xla_impl(float* out, void** data) {
  tensorflow::gather_float_int64_xla_impl(out, data);
}
