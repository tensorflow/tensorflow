/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/pass/hlo_pipeline_editor.h"

#include <cstdint>

#include "absl/strings/string_view.h"
#include "xla/hlo/pass/hlo_pass_options.h"
#include "xla/hlo/pass/hlo_pipeline.pb.h"

namespace xla {
namespace {

HloPipelineNodeProto& AppendNode(HloPipelineNodeSequenceProto& sequence,
                                 absl::string_view name,
                                 int32_t max_fixed_point_iterations) {
  HloPipelineNodeProto& node = *sequence.add_nodes();
  node.set_name(name);
  node.set_max_fixed_point_iterations(max_fixed_point_iterations);
  return node;
}

}  // namespace

void HloPipelineEditor::AppendPass(absl::string_view name,
                                   const HloPassOptions& options,
                                   int32_t max_fixed_point_iterations) {
  *AppendNode(*sequence_, name, max_fixed_point_iterations).mutable_options() =
      options.ToProto();
}

HloPipelineEditor HloPipelineEditor::AppendSequence(
    absl::string_view name, int32_t max_fixed_point_iterations) {
  // `mutable_sub_nodes()` marks the node as a sequence even while it is empty.
  return HloPipelineEditor(
      *AppendNode(*sequence_, name, max_fixed_point_iterations)
           .mutable_sub_nodes());
}

}  // namespace xla
