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

#include "tensorflow/tools/tfprof/internal/tfprof_code.h"

#include <stdio.h>
#include <utility>

#include "tensorflow/c/c_api.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/tools/tfprof/internal/tfprof_constants.h"
#include "tensorflow/tools/tfprof/internal/tfprof_tensor.h"

namespace tensorflow {
namespace tfprof {
namespace {
// Convert to Trace proto into a short readable string.
string GetTraceString(const CodeDef::Trace& trace) {
  string ntrace = "";
  if (trace.file().find_last_of('/') != trace.file().npos) {
    ntrace += trace.file().substr(trace.file().find_last_of('/') + 1);
  } else {
    ntrace += trace.file();
  }
  ntrace += strings::StrCat(":", trace.lineno());
  if (trace.function().length() < 20) {
    ntrace += ":" + trace.function();
  } else {
    ntrace += ":" + trace.function().substr(0, 17) + "...";
  }
  if (trace.line().length() < 20) {
    ntrace += ":" + trace.line();
  } else {
    ntrace += ":" + trace.line().substr(0, 17) + "...";
  }
  return ntrace;
}
}  // namespace

void TFCode::AddNode(TFGraphNode* node) {
  if (!node->code()) {
    return;
  }
  TFCodeNode* pre_trace_node = nullptr;
  for (int i = 0; i < node->code()->traces_size(); ++i) {
    // Unlike op name, which is globally unique, trace name is only unique
    // w.r.t. it's parent.
    const string& trace = GetTraceString(node->code()->traces(i));
    if (i == 0) {
      if (!trace_root_) {
        trace_root_.reset(new TFCodeNode(trace));
      }
      CHECK(trace_root_->name() == trace) << "Different trace root";
      pre_trace_node = trace_root_.get();
      continue;
    }
    pre_trace_node->AddChildren(trace);
    TFCodeNode* trace_node = pre_trace_node->children()[trace].get();

    if (i == node->code()->traces_size()-1) {
      trace_node->AddGraphNode(node);
    }
    pre_trace_node = trace_node;
  }
}

void TFCode::Build() {
  if (!trace_root_) {
    return;
  }
  code_root_ = BuildCodeNodes(trace_root_.get());
}

CodeNode* TFCode::BuildCodeNodes(TFCodeNode* root) {
  auto code_root = std::unique_ptr<CodeNode>(new CodeNode(root));
  CodeNode* code_root_ptr = code_root.get();
  code_nodes_.insert(std::move(code_root));

  for (auto it = root->children().cbegin();
       it != root->children().cend(); ++it) {
    code_root_ptr->children.push_back(BuildCodeNodes(it->second.get()));
  }
  return code_root_ptr;
}

const ShowCodeNode* TFCode::ShowInternal(const Options& opts) {
  // Search from roots recursively to find start node, if start_name_regexes
  // is specified.
  tfprof_trace_root_.reset(new TFCodeNode(kTFProfRoot));
  tfprof_code_root_.reset(new CodeNode(tfprof_trace_root_.get()));
  if (!code_root_) {
    return tfprof_code_root_.get();
  }

  std::vector<CodeNode*> roots = {code_root_};
  if (opts.start_name_regexes.size() != 1 ||
      opts.start_name_regexes[0] != ".*") {
    roots = SearchRoot(roots, opts.start_name_regexes);
  }

  tfprof_code_root_->children.assign(roots.begin(), roots.end());
  Account({tfprof_code_root_.get()}, opts);

  return PrintScope({tfprof_code_root_.get()}, opts, 1, 0)[0];
}

std::vector<CodeNode*> TFCode::SearchRoot(
    std::vector<CodeNode*> roots, const std::vector<string>& regexes) {
  std::vector<CodeNode*> res;
  if (roots.empty()) {
    return res;
  }
  for (CodeNode* root : roots) {
    bool match_start_node = false;
    for (const string& regex : regexes) {
      if (RE2::FullMatch(root->name(), regex)) {
        res.push_back(root);
        match_start_node = true;
        break;
      }
    }
    if (match_start_node) {
      // Found a start node at this branch, no need to continue.
      continue;
    }
    std::vector<CodeNode*> nroots = SearchRoot(root->children, regexes);
    res.insert(res.end(), nroots.begin(), nroots.end());
  }
  return res;
}

std::vector<CodeNode*> TFCode::PrintScope(const std::vector<CodeNode*> roots,
                                            const Options& opts, int depth,
                                            int last_ident) {
  std::vector<CodeNode*> show_nodes;

  for (CodeNode* node : roots) {
    int nlast_ident = last_ident;
    bool show = ShouldShow(node, opts, depth);
    if (show) {
      node->formatted_str.clear();
      if (opts.account_displayed_op_only) {
        node->ResetTotalStats();
        node->AddSelfToTotalStats();
      }
      nlast_ident += 2;
    }

    std::vector<CodeNode*> show_cnodes;
    if (!ShouldTrim(node, opts.trim_name_regexes)) {
      show_cnodes = PrintScope(node->children, opts, depth + 1, nlast_ident);
    }
    if (show) {
      show_cnodes = SortNodes(show_cnodes, opts);
      string children_str;
      for (CodeNode* sc : show_cnodes) {
        children_str += sc->formatted_str;
        node->mutable_proto()->add_children()->MergeFrom(sc->proto());
        if (opts.account_displayed_op_only) {
          node->AggregateTotalStats(sc);
        }
      }

      node->formatted_str =
          strings::Printf("%s%s\n", string(last_ident, ' ').c_str(),
                          node->Format(opts).c_str());

      if (opts.select.find(kShown[5]) != opts.select.end()) {
        fprintf(stderr, "code view has no tensor value to show\n");
      }

      node->formatted_str += children_str;
      show_nodes.push_back(node);
    } else {
      show_nodes.insert(show_nodes.end(), show_cnodes.begin(),
                        show_cnodes.end());
    }
  }
  return show_nodes;
}

void TFCode::Account(const std::vector<CodeNode*>& roots,
                      const Options& opts) {
  if (roots.empty()) return;

  for (CodeNode* node : roots) {
    node->ResetTotalStats();
    Account(node->children, opts);

    node->account = ShouldAccount(node, opts);
    if (node->account) {
      node->AddSelfToTotalStats();
    }
    for (CodeNode* c : node->children) {
      node->AggregateTotalStats(c);
    }
  }
}
}  // namespace tfprof
}  // namespace tensorflow
