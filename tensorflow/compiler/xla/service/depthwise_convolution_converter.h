/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_DEPTHWISE_CONVOLUTION_CONVERTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_DEPTHWISE_CONVOLUTION_CONVERTER_H_

#include <functional>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {

class DepthwiseConvolutionConverter : public HloModulePass {
 public:
  explicit DepthwiseConvolutionConverter(
      std::function<bool(HloInstruction*)> is_cost_viable)
      : is_cost_viable_(is_cost_viable) {}

  absl::string_view name() const override {
    return "depthwise-convolution-converter";
  }

  // Run convolution rewriting on the given computation. Returns whether the
  // computation was changed.
  StatusOr<bool> Run(HloModule* module) override;

  // Lambda containing cost model that decides whether to expand
  // batch_group_count.
  std::function<bool(HloInstruction*)> is_cost_viable_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_DEPTHWISE_CONVOLUTION_CONVERTER_H_
