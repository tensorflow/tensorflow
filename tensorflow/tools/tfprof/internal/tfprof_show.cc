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

#include "tensorflow/tools/tfprof/internal/tfprof_show.h"

#include <memory>
#include <set>

#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/regexp.h"

namespace tensorflow {
namespace tfprof {
ShowNode::ShowNode(TFNode* node) : node(node), account(true) {
  mutable_proto()->set_name(name());
  if (!node->device().empty()) {
    mutable_proto()->set_device(node->device());
  }
  mutable_proto()->set_exec_micros(node->kernel_compute_micros());
  mutable_proto()->set_requested_bytes(node->requested_byptes());
  mutable_proto()->set_float_ops(node->float_ops());

  if (!node->shape().empty()) {
    int64 params = 1;
    bool complete_shape = true;
    for (int64 d : node->shape()) {
      // Sometimes parameters could be <0 when a dim is unknown.
      if (d < 0) {
        complete_shape = false;
        break;
      }
      params *= d;
    }
    if (complete_shape) {
      mutable_proto()->set_parameters(proto_.parameters() + params);
    } else {
      fprintf(stderr, "Incomplete shape.");
    }
  }
}

string ShowNode::Format(const Options& opts) {
  if (opts.select.empty()) {
    return name();
  }
  return strings::Printf("%s (%s)", name().c_str(), FormatMeta(opts).c_str());
}

string ShowNode::FormatMeta(const Options& opts) {
  std::vector<string> info;
  if (opts.select.find(kShown[2]) != opts.select.end()) {
    const string shape = FormatShapes(node->shape());
    if (!shape.empty()) {
      info.push_back(shape);
    }
    string params = FormatNumber(proto().total_parameters()) + " params";
    if (account) {
      params = FormatNumber(proto().parameters()) + "/" + params;
    } else {
      params = "--/" + params;
    }
    info.push_back(params);
  }
  if (opts.select.find(kShown[3]) != opts.select.end()) {
    string fops = FormatNumber(proto().total_float_ops()) + " flops";
    if (account) {
      fops = FormatNumber(proto().float_ops()) + "/" + fops;
    } else {
      fops = "--/" + fops;
    }
    info.push_back(fops);
  }
  if (opts.select.find(kShown[0]) != opts.select.end()) {
    string memory = FormatMemory(proto().total_requested_bytes());
    if (account) {
      memory = FormatMemory(proto().requested_bytes()) + "/" + memory;

    } else {
      memory = "--/" + memory;
    }
    info.push_back(memory);
  }
  if (opts.select.find(kShown[1]) != opts.select.end()) {
    string time = FormatTime(proto().total_exec_micros());
    if (account) {
      time = FormatTime(proto().exec_micros()) + "/" + time;
    } else {
      time = "--/" + time;
    }
    info.push_back(time);
  }
  if (opts.select.find(kShown[6]) != opts.select.end()) {
    if (!proto().device().empty()) {
      info.push_back(proto().device());
    }
  }
  if (opts.select.find(kShown[7]) != opts.select.end()) {
    std::set<string> op_types = node->op_types();
    // Device is considered a type.
    if (!proto().device().empty()) {
      op_types.insert(proto().device());
    }
    info.push_back(str_util::Join(op_types, "|"));
  }
  return str_util::Join(info, ", ");
}

TFProfNode* ShowNode::mutable_proto() { return &proto_; }

const TFProfNode& ShowNode::proto() const { return proto_; }

void ShowNode::AggregateTotalStats(ShowNode* node) {
  TFProfNode* node_pb = node->mutable_proto();
  mutable_proto()->set_total_exec_micros(proto().total_exec_micros() +
                                         node_pb->total_exec_micros());
  mutable_proto()->set_total_requested_bytes(proto().total_requested_bytes() +
                                             node_pb->total_requested_bytes());
  mutable_proto()->set_total_parameters(proto().total_parameters() +
                                        node_pb->total_parameters());
  mutable_proto()->set_total_float_ops(proto().total_float_ops() +
                                       node_pb->total_float_ops());
}

void ShowNode::AddSelfToTotalStats() {
  mutable_proto()->set_total_exec_micros(proto().total_exec_micros() +
                                         proto().exec_micros());
  mutable_proto()->set_total_requested_bytes(proto().total_requested_bytes() +
                                             proto().requested_bytes());
  mutable_proto()->set_total_parameters(proto().total_parameters() +
                                        proto().parameters());
  mutable_proto()->set_total_float_ops(proto().total_float_ops() +
                                       proto().float_ops());
}

void ShowNode::ResetTotalStats() {
  mutable_proto()->set_total_exec_micros(0);
  mutable_proto()->set_total_requested_bytes(0);
  mutable_proto()->set_total_parameters(0);
  mutable_proto()->set_total_float_ops(0);
}

const TFProfNode& TFShow::Show(const Options& opts) {
  const ShowNode* root = ShowInternal(opts);
  if (opts.dump_to_file.empty()) {
    printf("%s", root->formatted_str.c_str());
    fflush(stdout);
  } else {
    Status s = WriteStringToFile(Env::Default(), opts.dump_to_file,
                                 root->formatted_str);
    if (!s.ok()) {
      fprintf(stderr, "%s\n", s.ToString().c_str());
    }
  }
  return root->proto();
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
    fprintf(stderr, "%s\n", TF_Message(status));
    TF_DeleteStatus(status);
    return false;
  }
  tensor->reset(new TFProfTensor(std::move(out_tensor)));
  TF_DeleteStatus(status);
  return true;
}

bool TFShow::ShouldShow(ShowNode* node, const Options& opts, int depth) {
  // Always show kTFProfRoot.
  if (node->name() == kTFProfRoot) return true;

  if (!node->account) return false;

  if (node->proto().requested_bytes() < opts.min_bytes ||
      node->proto().exec_micros() < opts.min_micros ||
      node->proto().parameters() < opts.min_params ||
      node->proto().float_ops() < opts.min_float_ops ||
      depth > opts.max_depth || !ShouldShowIfExtra(node, opts, depth)) {
    return false;
  }

  bool show = false;
  if (opts.device_regexes.size() == 1 && opts.device_regexes[0] == ".*") {
    show = true;
  } else {
    for (const string& regex : opts.device_regexes) {
      if (RE2::FullMatch(node->proto().device(), regex)) {
        show = true;
        break;
      }
    }
  }
  // Don't show if device_regexes don't cover it.
  if (!show) return false;

  show = false;
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

bool TFShow::ShouldTrim(ShowNode* node, const std::vector<string>& regexes) {
  for (const string& regex : regexes) {
    if (RE2::FullMatch(node->name(), regex)) {
      return true;
    }
  }
  return false;
}

bool TFShow::ShouldAccount(ShowNode* node, const Options& opts) {
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
    if (RE2::FullMatch(node->proto().device(), regex)) {
      return true;
    }
  }

  return false;
}

}  // namespace tfprof
}  // namespace tensorflow
