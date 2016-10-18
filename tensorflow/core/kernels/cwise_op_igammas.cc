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

namespace tensorflow {
#if EIGEN_HAS_C99_MATH

template <typename Device, typename Functor>
class IGammaOp : public BinaryOp<Device, Functor> {
 public:
  explicit IGammaOp(OpKernelConstruction* ctx)
      : BinaryOp<Device, Functor>(ctx) {
    TF_ANNOTATE_BENIGN_RACE(&signgam, "signgam output from lgamma is unused");
  }
};

REGISTER2(IGammaOp, CPU, "Igamma", functor::igamma, float, double);
REGISTER2(IGammaOp, CPU, "Igammac", functor::igammac, float, double);
#endif  // EIGEN_HAS_C99_MATH
}  // namespace tensorflow
