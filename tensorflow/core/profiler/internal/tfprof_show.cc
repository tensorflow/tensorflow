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

#include "tensorflow/core/profiler/internal/tfprof_show.h"

#include <memory>
#include <set>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/regexp.h"

namespace tensorflow {
namespace tfprof {

const GraphNodeProto& TFShow::Show(const string& prefix, const Options& opts) {
  if (opts.output_type == kOutput[0]) {
    Timeline timeline(opts.step, opts.output_options.at(kTimelineOpts[0]));
    return ShowInternal(opts, &timeline)->proto();
  } else {
    const ShowNode* ret = ShowInternal(opts, nullptr);
    if (opts.output_type == kOutput[1]) {
      absl::PrintF("%s", (prefix + ret->formatted_str));
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

bool TFShow::LookUpCheckPoint(const string& name,
                              std::unique_ptr<TFProfTensor>* tensor) {
  if (name == kTFProfRoot || !ckpt_reader_ || !tensor) {
    return false;
  }
  std::unique_ptr<Tensor> out_tensor;
  TF_Status* status = TF_NewStatus();
  ckpt_reader_->GetTensor(name, &out_tensor, status);
  if (TF_GetCode(status) != TF_OK) {
    absl::FPrintF(stderr, "%s\n", TF_Message(status));
    TF_DeleteStatus(status);
    return false;
  }
  tensor->reset(new TFProfTensor(std::move(out_tensor)));
  TF_DeleteStatus(status);
  return true;
}

bool TFShow::ShouldShow(const ShowNode* node, const Options& opts,
                        int depth) const {
  // Always show kTFProfRoot.
  if (node->name() == kTFProfRoot) return true;

  if (node->proto().total_requested_bytes() < opts.min_bytes ||
      node->proto().total_peak_bytes() < opts.min_peak_bytes ||
      node->proto().total_residual_bytes() < opts.min_residual_bytes ||
      node->proto().total_output_bytes() < opts.min_output_bytes ||
      node->proto().total_exec_micros() < opts.min_micros ||
      node->proto().total_accelerator_exec_micros() <
          opts.min_accelerator_micros ||
      node->proto().total_cpu_exec_micros() < opts.min_cpu_micros ||
      node->proto().parameters() < opts.min_params ||
      node->proto().float_ops() < opts.min_float_ops ||
      node->proto().run_count() < opts.min_occurrence ||
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

bool TFShow::ShouldTrim(const ShowNode* node,
                        const std::vector<string>& regexes) const {
  for (const string& regex : regexes) {
    if (RE2::FullMatch(node->name(), regex)) {
      return true;
    }
  }
  return false;
}

bool TFShow::ReAccount(ShowNode* node, const Options& opts) {
  node->ReInit(opts.step);
  if (opts.account_type_regexes.size() == 1 &&
      opts.account_type_regexes[0] == ".*") {
    return true;
  }
  for (const string& regex : opts.account_type_regexes) {
    for (const string& type : node->node->op_types()) {
      if (RE2::FullMatch(type, regex)) {
        return true;
      }
    }
  }
  return false;
}

string TFShow::FormatNodeMemory(ShowNode* node, int64 bytes,
                                int64 total_bytes) const {
  string memory = FormatMemory(total_bytes);
  if (node->account) {
    memory = FormatMemory(bytes) + "/" + memory;
  } else {
    memory = "--/" + memory;
  }
  return memory;
}

string TFShow::FormatNode(ShowNode* node, const Options& opts) const {
  std::vector<string> info;
  if (opts.select.find(kShown[2]) != opts.select.end()) {
    const string shape = FormatShapes(node->node->shape());
    if (!shape.empty()) {
      info.push_back(shape);
    }
    string params = FormatNumber(node->proto().total_parameters()) + " params";
    if (node->account) {
      params = FormatNumber(node->proto().parameters()) + "/" + params;
    } else {
      params = "--/" + params;
    }
    info.push_back(params);
  }
  if (opts.select.find(kShown[3]) != opts.select.end()) {
    string fops = FormatNumber(node->proto().total_float_ops()) + " flops";
    if (node->account) {
      fops = FormatNumber(node->proto().float_ops()) + "/" + fops;
    } else {
      fops = "--/" + fops;
    }
    info.push_back(fops);
  }
  if (opts.select.find(kShown[0]) != opts.select.end()) {
    info.push_back(FormatNodeMemory(node, node->proto().requested_bytes(),
                                    node->proto().total_requested_bytes()));
  }
  if (opts.select.find(kShown[11]) != opts.select.end()) {
    info.push_back(FormatNodeMemory(node, node->proto().peak_bytes(),
                                    node->proto().total_peak_bytes()));
  }
  if (opts.select.find(kShown[12]) != opts.select.end()) {
    info.push_back(FormatNodeMemory(node, node->proto().residual_bytes(),
                                    node->proto().total_residual_bytes()));
  }
  if (opts.select.find(kShown[13]) != opts.select.end()) {
    info.push_back(FormatNodeMemory(node, node->proto().output_bytes(),
                                    node->proto().total_output_bytes()));
  }
  if (opts.select.find(kShown[1]) != opts.select.end()) {
    info.push_back(FormatTotalExecTime(node, opts));
    info.push_back(FormatAcceleratorExecTime(node, opts));
    info.push_back(FormatCPUExecTime(node, opts));
  }
  if (opts.select.find(kShown[9]) != opts.select.end() &&
      opts.select.find(kShown[1]) == opts.select.end()) {
    info.push_back(FormatAcceleratorExecTime(node, opts));
  }
  if (opts.select.find(kShown[10]) != opts.select.end() &&
      opts.select.find(kShown[1]) == opts.select.end()) {
    info.push_back(FormatCPUExecTime(node, opts));
  }
  if (opts.select.find(kShown[5]) != opts.select.end()) {
    if (node->proto().devices_size() > 0) {
      info.push_back(absl::StrJoin(node->proto().devices(), "|"));
    }
  }
  if (opts.select.find(kShown[6]) != opts.select.end()) {
    const std::set<string>& op_types = node->node->op_types();
    info.push_back(absl::StrJoin(op_types, "|"));
  }
  if (opts.select.find(kShown[7]) != opts.select.end()) {
    string run = FormatNumber(node->proto().total_run_count());
    if (node->account) {
      run = FormatNumber(node->proto().run_count()) + "/" + run;
    } else {
      run = "--/" + run;
    }
    string definition = FormatNumber(node->proto().total_definition_count());
    if (node->account) {
      definition = "1/" + definition;
    } else {
      definition = "--/" + definition;
    }
    info.push_back(run + "|" + definition);
  }
  if (opts.select.find(kShown[8]) != opts.select.end()) {
    std::vector<string> shape_vec;
    for (const auto& s : node->node->input_shapes()) {
      if (s.second.empty()) {
        shape_vec.push_back(absl::StrFormat("%d:unknown", s.first));
      } else {
        shape_vec.push_back(
            absl::StrFormat("%d:%s", s.first, absl::StrJoin(s.second, "x")));
      }
    }
    info.push_back(absl::StrJoin(shape_vec, "|"));
  }

  return absl::StrFormat("%s (%s)", node->name(), absl::StrJoin(info, ", "));
}

string TFShow::FormatLegend(const Options& opts) const {
  std::vector<string> legends;
  if (opts.select.find(kShown[2]) != opts.select.end()) {
    legends.push_back("# parameters");
  }
  if (opts.select.find(kShown[3]) != opts.select.end()) {
    legends.push_back("# float_ops");
  }
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
  if (opts.select.find(kShown[5]) != opts.select.end()) {
    legends.push_back("assigned devices");
  }
  if (opts.select.find(kShown[6]) != opts.select.end()) {
    legends.push_back("op types");
  }
  if (opts.select.find(kShown[7]) != opts.select.end()) {
    legends.push_back("op count (run|defined)");
  }
  if (opts.select.find(kShown[8]) != opts.select.end()) {
    legends.push_back("input shapes");
  }
  return absl::StrFormat("node name | %s\n", absl::StrJoin(legends, " | "));
}

}  // namespace tfprof
}  // namespace tensorflow
