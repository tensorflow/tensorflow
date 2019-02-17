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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_CONVOLUTION_CLASSIFIER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_CONVOLUTION_CLASSIFIER_H_

#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

#include <string>
#include <vector>

namespace xla {

class HloModule;
class HloInstruction;

namespace poplarplugin {

struct CompilerAnnotations;

enum ConvClassificationType {
  FORWARD,
  BACKPROP_INPUT,
  BACKPROP_FILTER,
  INFERENCE,
};

ConvClassificationType GetConvClassificationType(const HloInstruction*,
                                                 const CompilerAnnotations&);
std::string ConvClassificationTypeToString(const ConvClassificationType&);
std::string ConvClassificationTypeToString(const HloInstruction*,
                                           const CompilerAnnotations&);

using ConvClassification =
    std::map<const HloInstruction*, ConvClassificationType>;

/**
 * This class marks each convolution as either a forward pass, a backprop input
 * (gradient), a backprop filter (weight update), or a standalone inference only
 * convolution.
 */
class ConvolutionClassifier : public HloModulePass {
 public:
  ConvolutionClassifier(CompilerAnnotations& annotations);

  ~ConvolutionClassifier() = default;

  absl::string_view name() const override { return "convolution-classifier"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  ConvClassification& classification_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
