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

#include "tensorflow/core/kernels/cwise_ops_common.h"

#include "tensorflow/core/platform/dynamic_annotations.h"

namespace tensorflow {

template <typename Device, typename Functor>
class LgammaOp : public UnaryOp<Device, Functor> {
 public:
  explicit LgammaOp(OpKernelConstruction* ctx) : UnaryOp<Device, Functor>(ctx) {
    TF_ANNOTATE_BENIGN_RACE(&signgam, "signgam output from lgamma is unused");
  }
};

#if EIGEN_HAS_C99_MATH
REGISTER3(LgammaOp, CPU, "Lgamma", functor::lgamma, float, Eigen::half, double);
#if GOOGLE_CUDA
REGISTER3(LgammaOp, GPU, "Lgamma", functor::lgamma, float, Eigen::half, double);
#endif
#endif  // EIGEN_HAS_C99_MATH

}  // namespace tensorflow
