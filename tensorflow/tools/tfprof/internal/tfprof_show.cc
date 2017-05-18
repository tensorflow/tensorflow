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

#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/regexp.h"

namespace tensorflow {
namespace tfprof {

const TFGraphNodeProto& TFShow::Show(const Options& opts) {
  if (opts.output_type == kOutput[0]) {
    Timeline timeline(opts.output_options.at(kTimelineOpts[0]));
    return ShowInternal(opts, &timeline)->proto();
  } else if (opts.output_type == kOutput[2]) {
    const ShowNode* root = ShowInternal(opts, nullptr);
    Status s =
        WriteStringToFile(Env::Default(), opts.output_options.at(kFileOpts[0]),
                          root->formatted_str);
    if (!s.ok()) {
      fprintf(stderr, "%s\n", s.ToString().c_str());
    }
    return root->proto();
  } else {
    const ShowNode* root = ShowInternal(opts, nullptr);
    printf("%s", root->formatted_str.c_str());
    fflush(stdout);
    return root->proto();
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
      for (const string& device : node->proto().devices()) {
        if (RE2::FullMatch(device, regex)) {
          show = true;
          break;
        }
      }
      if (show) break;
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
    for (const string& device : node->proto().devices())
      if (RE2::FullMatch(device, regex)) {
        return true;
      }
  }
  return false;
}

}  // namespace tfprof
}  // namespace tensorflow
