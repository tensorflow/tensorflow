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

#ifndef XLA_HLO_PASS_HLO_PIPELINE_EDITOR_H_
#define XLA_HLO_PASS_HLO_PIPELINE_EDITOR_H_

#include <cstdint>

#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/pass/hlo_pass_options.h"
#include "xla/hlo/pass/hlo_pipeline.pb.h"

namespace xla {

// Non-owning, cheap-to-copy view that edits an externally owned
// `HloPipelineNodeSequenceProto` in place.
//
// The proto is the only state, so there is nothing to sync back or "build".
// Editors of nested sequences stay valid while their parents keep growing, as
// repeated message fields never relocate their elements on append.
//
//   HloPipelineProto proto;
//   HloPipelineEditor root(proto);
//   root.AppendPass("flatten-call-graph");
//   root.AppendPass("algsimp", {{"limit", 2}},
//                   /*max_fixed_point_iterations=*/25);
//   HloPipelineEditor simplification =
//       root.AppendSequence("simplification",
//                           /*max_fixed_point_iterations=*/10);
//   simplification.AppendPass("hlo-dce");
//
// The proto must outlive the editor and all editors derived from it.
// `max_fixed_point_iterations` follows `HloPipelineNodeProto`: 0 or 1 runs the
// node once; N > 1 reruns it until the module stops changing, at most N times.
class HloPipelineEditor {
 public:
  explicit HloPipelineEditor(
      HloPipelineNodeSequenceProto& sequence ABSL_ATTRIBUTE_LIFETIME_BOUND)
      : sequence_(&sequence) {}
  explicit HloPipelineEditor(
      HloPipelineProto& pipeline ABSL_ATTRIBUTE_LIFETIME_BOUND)
      : HloPipelineEditor(*pipeline.mutable_root()) {}

  // Appends the pass registered as `name`.
  void AppendPass(absl::string_view name, const HloPassOptions& options = {},
                  int32_t max_fixed_point_iterations = 0);

  // Appends a node sequence and returns its editor.
  [[nodiscard]] HloPipelineEditor AppendSequence(
      absl::string_view name, int32_t max_fixed_point_iterations = 0);

 private:
  HloPipelineNodeSequenceProto* sequence_;
};

}  // namespace xla

#endif  // XLA_HLO_PASS_HLO_PIPELINE_EDITOR_H_
