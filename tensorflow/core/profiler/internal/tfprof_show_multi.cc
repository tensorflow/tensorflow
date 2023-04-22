/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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

#include "tensorflow/core/profiler/internal/tfprof_show_multi.h"

#include <memory>
#include <set>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/profiler/internal/tfprof_scope.h"

namespace tensorflow {
namespace tfprof {

const MultiGraphNodeProto& TFMultiShow::Show(const string& prefix,
                                             const Options& opts) {
  if (opts.output_type == kOutput[0]) {
    Timeline timeline(opts.step, opts.output_options.at(kTimelineOpts[0]));
    return ShowInternal(opts, &timeline)->proto();
  } else {
    const ShowMultiNode* ret = ShowInternal(opts, nullptr);
    if (opts.output_type == kOutput[1]) {
      absl::PrintF("%s%s", prefix, ret->formatted_str);
      fflush(stdout);
    } else if (opts.output_type == kOutput[2]) {
      Status s = WriteStringToFile(Env::Default(),
                                   opts.output_options.at(kFileOpts[0]),
                                   prefix + ret->formatted_str);
      if (!s.ok()) {
        absl::FPrintF(stderr, "%s\n", s.ToString());
      }
    } else if (opts.output_type == kOutput[3] ||
               opts.output_type == kOutput[4]) {
    } else {
      absl::FPrintF(stderr, "Unknown output type: %s\n", opts.output_type);
    }
    return ret->proto();
  }
}

bool TFMultiShow::ShouldShow(const ShowMultiNode* node, const Options& opts,
                             int depth) const {
  // Always show kTFProfRoot.
  if (node->name() == kTFProfRoot) return true;

  // TODO(xpan): Think more carefully about node filtering in code view.
  // Unlike graph/scope view, which users want to see the exact leaf op.
  // In code view, users want to see the middle code traces they wrote.
  //
  // This is a subtle difference from scope/graph view. Usually mostly
  // want to see the middle code traces (i.e. their own codes.), instead
  // of the TensorFlow internal codes traces.
  if (node->proto().total_requested_bytes() < opts.min_bytes ||
      node->proto().total_peak_bytes() < opts.min_peak_bytes ||
      node->proto().total_residual_bytes() < opts.min_residual_bytes ||
      node->proto().total_output_bytes() < opts.min_output_bytes ||
      node->proto().total_exec_micros() < opts.min_micros ||
      node->proto().total_accelerator_exec_micros() <
          opts.min_accelerator_micros ||
      node->proto().total_cpu_exec_micros() < opts.min_cpu_micros ||
      node->proto().total_parameters() < opts.min_params ||
      node->proto().total_float_ops() < opts.min_float_ops ||
      depth > opts.max_depth || !ShouldShowIfExtra(node, opts, depth)) {
    return false;
  }

  bool show = false;
  if (opts.show_name_regexes.size() == 1 && opts.show_name_regexes[0] == ".*") {
    show = true;
  } else {
    for (const string& regex : opts.show_name_regexes) {
      if (RE2::FullMatch(node->name(), regex)) {
        show = true;
        break;
      }
    }
  }
  // Don't show if show_name_regexes don't cover it.
  if (!show) return false;
  // Don't show if hide_name_regexes cover it.
  for (const string& regex : opts.hide_name_regexes) {
    if (RE2::FullMatch(node->name(), regex)) return false;
  }
  return true;
}

bool TFMultiShow::ShouldTrim(const ShowMultiNode* node,
                             const std::vector<string>& regexes) const {
  for (const string& regex : regexes) {
    if (RE2::FullMatch(node->name(), regex)) {
      return true;
    }
  }
  return false;
}

bool TFMultiShow::ReAccount(ShowMultiNode* node, const Options& opts) {
  return node->ReInit(opts.step, opts.account_type_regexes);
}

string TFMultiShow::FormatLegend(const Options& opts) const {
  std::vector<string> legends;
  if (opts.select.find(kShown[0]) != opts.select.end()) {
    legends.push_back("requested bytes");
  }
  if (opts.select.find(kShown[11]) != opts.select.end()) {
    legends.push_back("peak bytes");
  }
  if (opts.select.find(kShown[12]) != opts.select.end()) {
    legends.push_back("residual bytes");
  }
  if (opts.select.find(kShown[13]) != opts.select.end()) {
    legends.push_back("output bytes");
  }
  if (opts.select.find(kShown[1]) != opts.select.end()) {
    legends.push_back("total execution time");
    legends.push_back("accelerator execution time");
    legends.push_back("cpu execution time");
  }
  if (opts.select.find(kShown[9]) != opts.select.end() &&
      opts.select.find(kShown[1]) == opts.select.end()) {
    legends.push_back("accelerator execution time");
  }
  if (opts.select.find(kShown[10]) != opts.select.end() &&
      opts.select.find(kShown[1]) == opts.select.end()) {
    legends.push_back("cpu execution time");
  }
  if (opts.select.find(kShown[2]) != opts.select.end()) {
    legends.push_back("# parameters");
  }
  if (opts.select.find(kShown[3]) != opts.select.end()) {
    legends.push_back("# float_ops");
  }
  if (opts.select.find(kShown[5]) != opts.select.end()) {
    legends.push_back("assigned devices");
  }
  if (opts.select.find(kShown[6]) != opts.select.end()) {
    legends.push_back("op types");
  }
  if (opts.select.find(kShown[7]) != opts.select.end()) {
    legends.push_back("op occurrence (run|defined)");
  }
  if (opts.select.find(kShown[8]) != opts.select.end()) {
    legends.push_back("input shapes");
  }
  return absl::StrFormat("node name | %s\n", absl::StrJoin(legends, " | "));
}

string TFMultiShow::FormatInputShapes(const MultiGraphNodeProto& proto) const {
  // input_shape string -> (static defined count, run count, run_micros)
  std::map<string, std::tuple<int64, int64, int64>> input_shapes_attr;
  for (int i = 0; i < proto.graph_nodes_size(); ++i) {
    const GraphNodeProto& gnode = proto.graph_nodes(i);
    // Convert and sort by input_idx.
    std::map<int, std::vector<int64>> input_shapes;
    for (const auto& inp : gnode.input_shapes()) {
      input_shapes[inp.first] = ShapeProtoToVec(inp.second);
    }

    std::vector<string> input_vec;
    for (const auto& s : input_shapes) {
      if (s.second.empty()) {
        input_vec.push_back(absl::StrFormat("%d:unknown", s.first));
      } else {
        input_vec.push_back(
            absl::StrFormat("%d:%s", s.first, absl::StrJoin(s.second, "x")));
      }
    }
    string shape_type_str =
        absl::StrFormat("input_type: %s", absl::StrJoin(input_vec, ",\t"));
    auto t = input_shapes_attr.find(shape_type_str);
    if (t == input_shapes_attr.end()) {
      input_shapes_attr.insert(
          std::make_pair(shape_type_str, std::make_tuple(0, 0, 0)));
      t = input_shapes_attr.find(shape_type_str);
    }
    input_shapes_attr[shape_type_str] = std::make_tuple(
        std::get<0>(t->second) + 1, std::get<1>(t->second) + gnode.run_count(),
        std::get<2>(t->second) + gnode.exec_micros());
  }
  if (input_shapes_attr.empty()) {
    return "";
  }

  std::vector<std::pair<string, std::tuple<int64, int64, int64>>>
      shape_count_vec(input_shapes_attr.begin(), input_shapes_attr.end());
  std::sort(
      shape_count_vec.begin(), shape_count_vec.end(),
      [](const std::pair<const string, std::tuple<int64, int64, int64>>& a,
         const std::pair<const string, std::tuple<int64, int64, int64>>& b) {
        return std::get<1>(a.second) > std::get<1>(b.second);
      });

  std::vector<string> input_types;
  input_types.reserve(shape_count_vec.size());
  for (const auto& s : shape_count_vec) {
    std::tuple<int64, int64, int64> t = s.second;
    input_types.push_back(absl::StrFormat(
        "%s\t(run*%d|defined*%d)\texec_time: %s", s.first, std::get<1>(t),
        std::get<0>(t), FormatTime(std::get<2>(t))));
  }
  return absl::StrJoin(input_types, "\n");
}

std::vector<string> TFMultiShow::FormatTimes(const ShowMultiNode* node,
                                             const Options& opts) const {
  std::vector<string> attrs;
  if (opts.select.find(kShown[1]) != opts.select.end()) {
    attrs.push_back(FormatTotalExecTime(node, opts));
    attrs.push_back(FormatAcceleratorExecTime(node, opts));
    attrs.push_back(FormatCPUExecTime(node, opts));
  }
  if (opts.select.find(kShown[9]) != opts.select.end() &&
      opts.select.find(kShown[1]) == opts.select.end()) {
    attrs.push_back(FormatAcceleratorExecTime(node, opts));
  }
  if (opts.select.find(kShown[10]) != opts.select.end() &&
      opts.select.find(kShown[1]) == opts.select.end()) {
    attrs.push_back(FormatCPUExecTime(node, opts));
  }
  return attrs;
}

}  // namespace tfprof
}  // namespace tensorflow
