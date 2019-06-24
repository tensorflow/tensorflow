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
#include "tensorflow/compiler/plugin/poplar/driver/tools/convolution_preplanning.h"
#include <poplar/Target.hpp>
#include "tensorflow/compiler/plugin/poplar/driver/ops/conv_graph_caching.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

namespace xla {
namespace poplarplugin {

/*
 * Visit all non-fused operations in the whole module looking for convolutions,
 * and add the parameters and the options for that convolution to the set
 *  of things to pass to the poplibs convolution pre-planner.
 */

Status ConvolutionPreplanning::Plan(const HloModule* module,
                                    CompilerResources& resources) {
  preplan_convs.clear();
  option_flags_store.clear();

  for (auto* comp : module->computations()) {
    if (!IsPopOpsFusion(comp)) {
      for (HloInstruction* inst : comp->instructions()) {
        if (inst->opcode() == HloOpcode::kConvolution) {
          TF_RETURN_IF_ERROR(StorePreplanConv(inst, resources, 0, 1));
        } else if (IsPopOpsConvolution(inst)) {
          TF_RETURN_IF_ERROR(StorePreplanConv(inst, resources, 0, 1));
        } else if (IsPopOpsFusion(inst, "conv_scaled_inplace")) {
          TF_RETURN_IF_ERROR(StorePreplanConv(inst, resources, 1, 2));
        }
      }
    }
  }

  poplin::preplanConvolutions(preplan_convs, resources.convolution_cache);

  return Status::OK();
}

Status ConvolutionPreplanning::StorePreplanConv(
    const HloInstruction* inst, const CompilerResources& resources,
    int64 input_index, int64 kernel_index) {
  auto& target = resources.main_graph.getTarget();
  TF_ASSIGN_OR_RETURN(
      poplin::ConvParams conv_params,
      GetConvolutionParameters(inst, input_index, kernel_index));
  auto conv_type = ConvClassificationTypeToString(
      GetConvClassificationType(inst, resources.annotations));

  poplar::OptionFlags option_flags = resources.default_conv_options;
  option_flags.set("pass", conv_type);
  auto ins = option_flags_store.insert(std::make_pair(conv_type, option_flags));
  preplan_convs.insert(
      std::make_tuple(&target, conv_params, &(ins.first->second)));
  return Status::OK();
}

}  // namespace poplarplugin
}  // namespace xla
