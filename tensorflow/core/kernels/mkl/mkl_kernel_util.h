/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_MKL_MKL_KERNEL_UTIL
#define TENSORFLOW_CORE_KERNELS_MKL_MKL_KERNEL_UTIL

#ifdef INTEL_MKL

#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {

class MklTestingUtil {
 public:
  static void RunMklQuantizeOp(const Tensor& input, const float input_min,
                               const float input_max, DataType type,
                               string mode, Tensor* output);
  static void RunDequantizeOp(const Tensor& input, const Tensor& input_min,
                              const Tensor& input_max, string mode,
                              Tensor* output);

  static void RunGraph(const tensorflow::GraphDef graph_def,
                       const string& fetch, Tensor* output);
  template <typename T>
  static void ComputeMinMax(const Tensor& tf_tensor, T* tensor_min,
                            T* tensor_max) {
    auto eigen_tensor = tf_tensor.flat<T>();
    Eigen::Tensor<T, 0, Eigen::RowMajor> min = eigen_tensor.minimum();
    Eigen::Tensor<T, 0, Eigen::RowMajor> max = eigen_tensor.maximum();
    *tensor_min = min();
    *tensor_max = max();
  }
};

}  // namespace tensorflow

#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_KERNELS_MKL_MKL_KERNEL_UTIL
