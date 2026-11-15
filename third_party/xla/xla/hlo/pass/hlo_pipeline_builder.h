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

#ifndef XLA_HLO_PASS_HLO_PIPELINE_BUILDER_H_
#define XLA_HLO_PASS_HLO_PIPELINE_BUILDER_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/pass/hlo_pass_context.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/pass/hlo_pass_registry.h"
#include "xla/hlo/pass/hlo_pipeline.pb.h"
#include "xla/service/compilation_stats.h"

namespace xla {

// Instantiates the pass pipeline described by `proto` as an `HloPassPipeline`
// named `name`, creating its passes with `registry`. Nested pipelines share
// `compilation_stats`.
//
// The caller runs the pipeline and adds invariant checkers. The objects in
// `context` and `compilation_stats` must outlive the pipeline.
absl::StatusOr<std::unique_ptr<HloPassPipeline>> BuildHloPassPipeline(
    const HloPipelineProto& proto, absl::string_view name,
    const HloPassContext& context,
    CompilationStats* compilation_stats = nullptr,
    const HloPassRegistry& registry = HloPassRegistry::Global());

}  // namespace xla

#endif  // XLA_HLO_PASS_HLO_PIPELINE_BUILDER_H_
