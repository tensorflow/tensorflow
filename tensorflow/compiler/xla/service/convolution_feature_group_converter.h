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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CONVOLUTION_FEATURE_GROUP_CONVERTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CONVOLUTION_FEATURE_GROUP_CONVERTER_H_

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {

// A pass which rewrites convolutions with feature_group_count > 1 into
// convolutions with feature_group_count = 1.
class ConvolutionFeatureGroupConverter : public HloModulePass {
 public:
  ConvolutionFeatureGroupConverter(bool canonicalize_depthwise_filter = false)
      : filter_expansion_(canonicalize_depthwise_filter) {}

  absl::string_view name() const override {
    return "convolution-feature-group-converter";
  }

  // Run convolution rewriting on the given computation. Returns whether the
  // computation was changed.
  StatusOr<bool> Run(HloModule* module) override;

  // Tells whether filter expansion is required.
  bool filter_expansion_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CONVOLUTION_FEATURE_GROUP_CONVERTER_H_
