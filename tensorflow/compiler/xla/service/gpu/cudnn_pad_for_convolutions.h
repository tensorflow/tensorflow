/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_PAD_FOR_CONVOLUTIONS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_PAD_FOR_CONVOLUTIONS_H_

#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace gpu {

// Two zero-paddings for CuDNN thunking are done in this transform: padding for
// tensor cores and padding for integer convolutions.  This transform also
// add slice instruction to remove unnecessary output features.
class CudnnPadForConvolutions : public HloModulePass {
 public:
  explicit CudnnPadForConvolutions(se::CudaComputeCapability compute_capability)
      : compute_capability_(compute_capability) {}

  absl::string_view name() const override {
    return "cudnn_pad_for_convolutions";
  }
  // Run PadForConvolutions on the given module and return if any change is made
  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const se::CudaComputeCapability compute_capability_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_PAD_FOR_CONVOLUTIONS_H_
