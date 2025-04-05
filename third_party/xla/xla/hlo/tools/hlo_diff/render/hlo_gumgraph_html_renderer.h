/*
 * Copyright 2025 The OpenXLA Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XLA_HLO_TOOLS_HLO_DIFF_RENDER_HLO_GUMGRAPH_HTML_RENDERER_H_
#define XLA_HLO_TOOLS_HLO_DIFF_RENDER_HLO_GUMGRAPH_HTML_RENDERER_H_

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>

#include "absl/functional/function_ref.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_summary.h"

namespace xla {
namespace hlo_diff {

// A function that returns a visualization url for the given instruction pair.
using UrlGenerator = absl::FunctionRef<std::string(const HloInstruction*,
                                                   const HloInstruction*)>;

// A function that returns the op metric for the given op name.
using GetOpMetricFn =
    absl::FunctionRef<std::optional<uint64_t>(absl::string_view)>;

// Renders the diff result in HTML format, and writes the result to the given
// output stream.

// url_generator can be specified which is used to link an url to each generated
// diff result.
void RenderHtml(const DiffResult& diff_result, const DiffSummary& diff_summary,
                UrlGenerator url_generator, GetOpMetricFn left_op_metrics,
                GetOpMetricFn right_op_metrics, std::ostringstream& out);

}  // namespace hlo_diff
}  // namespace xla

#endif  // XLA_HLO_TOOLS_HLO_DIFF_RENDER_HLO_GUMGRAPH_HTML_RENDERER_H_
