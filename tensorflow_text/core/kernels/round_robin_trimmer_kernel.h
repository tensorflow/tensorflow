// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_ROUND_ROBIN_TRIMMER_KERNEL_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_ROUND_ROBIN_TRIMMER_KERNEL_H_

#include "tensorflow/lite/kernels/shim/tf_op_shim.h"
#include "tensorflow_text/core/kernels/round_robin_trimmer_kernel_template.h"

namespace tensorflow {
namespace text {

template <typename T, typename Tsplits>
class RoundRobinGenerateMasksOpKernel
    : public tflite::shim::TfOpKernel<RoundRobinGenerateMasksOp, T, Tsplits> {
 public:
  using tflite::shim::TfOpKernel<RoundRobinGenerateMasksOp, T,
                                 Tsplits>::TfOpKernel;
};

template <typename T, typename Tsplits>
class RoundRobinTrimOpKernel
    : public tflite::shim::TfOpKernel<RoundRobinTrimOp, T, Tsplits> {
 public:
  using tflite::shim::TfOpKernel<RoundRobinTrimOp, T, Tsplits>::TfOpKernel;
};

}  // namespace text
}  // namespace tensorflow


#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_ROUND_ROBIN_TRIMMER_KERNEL_H_
