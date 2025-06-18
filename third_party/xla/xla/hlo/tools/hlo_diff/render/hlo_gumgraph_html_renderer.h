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

#include <sstream>

#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_summary.h"
#include "xla/hlo/tools/hlo_diff/render/graph_url_generator.h"
#include "xla/hlo/tools/hlo_diff/render/op_metric_getter.h"

namespace xla {
namespace hlo_diff {

// Renders the diff result in HTML format, and writes the result to the given
// output stream. url_generator can be specified which is used to link an url to
// each generated diff result.
void RenderHtml(const DiffResult& diff_result, const DiffSummary& diff_summary,
                GraphUrlGenerator* url_generator,
                OpMetricGetter* left_op_metric_getter,
                OpMetricGetter* right_op_metric_getter,
                std::ostringstream& out);
inline void RenderHtml(const DiffResult& diff_result,
                       const DiffSummary& diff_summary,
                       GraphUrlGenerator* url_generator,
                       std::ostringstream& out) {
  RenderHtml(diff_result, diff_summary, url_generator,
             /*left_op_metric_getter=*/nullptr,
             /*right_op_metric_getter=*/nullptr, out);
}
inline void RenderHtml(const DiffResult& diff_result,
                       const DiffSummary& diff_summary,
                       std::ostringstream& out) {
  RenderHtml(diff_result, diff_summary, /*url_generator=*/nullptr,
             /*left_op_metric_getter=*/nullptr,
             /*right_op_metric_getter=*/nullptr, out);
}

}  // namespace hlo_diff
}  // namespace xla

#endif  // XLA_HLO_TOOLS_HLO_DIFF_RENDER_HLO_GUMGRAPH_HTML_RENDERER_H_
