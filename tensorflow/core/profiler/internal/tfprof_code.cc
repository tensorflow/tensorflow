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

#include "tensorflow/core/profiler/internal/tfprof_code.h"

#include <stdio.h>
#include <utility>

#include "tensorflow/c/c_api.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/profiler/internal/tfprof_constants.h"

namespace tensorflow {
namespace tfprof {
namespace {
// Convert to Trace proto into a short readable string.
string GetTraceString(const CodeDef::Trace& trace) {
  string ntrace = io::Basename(trace.file()).ToString();
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
  if (node->code().traces_size() == 0) {
    return;
  }
  TFMultiGraphNode* pre_trace_node = nullptr;
  // TODO(xpan): Consider to release CodeDef after TFCode is built. It
  // takes a lot of memory.
  for (int i = 0; i < node->code().traces_size(); ++i) {
    // Unlike op name, which is globally unique, trace name is only unique
    // w.r.t. it's parent.
    const string& trace = GetTraceString(node->code().traces(i));
    if (i == 0) {
      if (!trace_root_) {
        trace_root_.reset(new TFMultiGraphNode(trace));
      }
      CHECK(trace_root_->name() == trace) << "Different trace root";
      pre_trace_node = trace_root_.get();
      continue;
    }
    pre_trace_node->AddChildren(trace);
    TFMultiGraphNode* trace_node = pre_trace_node->children().at(trace).get();

    if (i == node->code().traces_size() - 1) {
      trace_node->AddGraphNode(node);
    }
    pre_trace_node = trace_node;
  }
}

void TFCode::Build() {
  if (root_) {
    return;
  }
  tfprof_trace_root_.reset(new TFMultiGraphNode(kTFProfRoot));
  root_.reset(new CodeNode(tfprof_trace_root_.get()));

  if (trace_root_) {
    code_root_ = BuildCodeNodes(trace_root_.get());
    root_->children.push_back(code_root_);
  }
}

CodeNode* TFCode::BuildCodeNodes(TFMultiGraphNode* root) {
  auto code_root = std::unique_ptr<CodeNode>(new CodeNode(root));
  CodeNode* code_root_ptr = code_root.get();
  code_nodes_.insert(std::move(code_root));

  for (auto it = root->children().cbegin(); it != root->children().cend();
       ++it) {
    code_root_ptr->children.push_back(BuildCodeNodes(it->second.get()));
  }
  return code_root_ptr;
}

const ShowMultiNode* TFCode::ShowInternal(const Options& opts,
                                          Timeline* timeline) {
  std::vector<CodeNode*> roots = Account(root_->children, opts);
  root_->ResetTotalStats();
  root_->show_children.clear();
  for (CodeNode* n : roots) {
    root_->AggregateTotalStats(n);
  }

  if (opts.start_name_regexes.size() != 1 ||
      opts.start_name_regexes[0] != ".*") {
    roots = SearchRoot(roots, opts.start_name_regexes);
  }

  root_->show_children.assign(roots.begin(), roots.end());

  CodeNode* root = PrintScope({root_.get()}, opts, 1, 0)[0];

  root->formatted_str = FormatLegend(opts) + root->formatted_str;
  Format(root->show_children, &root->formatted_str, root->mutable_proto());

  if (timeline) {
    timeline->GenerateCodeTimeline(root);
  }
  return root;
}

void TFCode::Format(const std::vector<CodeNode*> roots, string* display_str,
                    TFMultiGraphNodeProto* proto) {
  for (CodeNode* node : roots) {
    display_str->append(node->formatted_str);
    TFMultiGraphNodeProto* child = proto->add_children();
    child->MergeFrom(node->proto());
    Format(node->show_children, display_str, child);
  }
}

std::vector<CodeNode*> TFCode::SearchRoot(std::vector<CodeNode*> roots,
                                          const std::vector<string>& regexes) {
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
    std::vector<CodeNode*> nroots = SearchRoot(root->show_children, regexes);
    res.insert(res.end(), nroots.begin(), nroots.end());
  }
  return res;
}

std::vector<CodeNode*> TFCode::PrintScope(const std::vector<CodeNode*> roots,
                                          const Options& opts, int depth,
                                          int last_ident) {
  std::vector<CodeNode*> show_nodes;

  for (CodeNode* node : roots) {
    int ident = last_ident;
    bool show = ShouldShow(node, opts, depth);
    if (show) ident += 2;

    std::vector<CodeNode*> show_cnodes;
    if (!ShouldTrim(node, opts.trim_name_regexes) && depth <= opts.max_depth) {
      show_cnodes = PrintScope(node->show_children, opts, depth + 1, ident);
    }
    if (show) {
      node->show_children.clear();
      if (opts.account_displayed_op_only) {
        node->ResetTotalStats();
        node->AddSelfToTotalStats();
      }

      show_cnodes = SortNodes(show_cnodes, opts);
      for (CodeNode* sc : show_cnodes) {
        node->show_children.push_back(sc);
        if (opts.account_displayed_op_only) {
          node->AggregateTotalStats(sc);
        }
      }

      node->formatted_str = FormatNode(node, opts, last_ident);

      if (opts.select.find(kShown[4]) != opts.select.end()) {
        fprintf(stderr, "code view has no tensor value to show\n");
      }
      show_nodes.push_back(node);
    } else {
      show_nodes.insert(show_nodes.end(), show_cnodes.begin(),
                        show_cnodes.end());
    }
  }
  return show_nodes;
}

std::vector<CodeNode*> TFCode::Account(const std::vector<CodeNode*>& roots,
                                       const Options& opts) {
  std::vector<CodeNode*> act_nodes;

  for (CodeNode* node : roots) {
    node->ResetTotalStats();
    std::vector<CodeNode*> act_cnodes = Account(node->children, opts);
    node->account = ReAccount(node, opts);
    if (node->account || !act_cnodes.empty()) {
      node->show_children.clear();
      node->ResetTotalStats();
      node->AddSelfToTotalStats();
      for (CodeNode* c : act_cnodes) {
        node->AggregateTotalStats(c);
        node->show_children.push_back(c);
      }
      act_nodes.push_back(node);
    }
  }
  return act_nodes;
}

string TFCode::FormatNode(CodeNode* node, const Options& opts, int64 indent) {
  std::vector<string> attrs;
  if (opts.select.find(kShown[0]) != opts.select.end()) {
    string memory = FormatMemory(node->proto().total_requested_bytes());
    if (node->account) {
      memory = FormatMemory(node->proto().requested_bytes()) + "/" + memory;
    } else {
      memory = "--/" + memory;
    }
    attrs.push_back(memory);
  }
  std::vector<string> time_attrs = FormatTimes(node, opts);
  attrs.insert(attrs.end(), time_attrs.begin(), time_attrs.end());

  if (opts.select.find(kShown[2]) != opts.select.end()) {
    string params = FormatNumber(node->proto().total_parameters()) + " params";
    if (node->account) {
      params = FormatNumber(node->proto().parameters()) + "/" + params;
    } else {
      params = "--/" + params;
    }
    attrs.push_back(params);
  }

  if (opts.select.find(kShown[3]) != opts.select.end()) {
    string fops = FormatNumber(node->proto().total_float_ops()) + " flops";
    if (node->account) {
      fops = FormatNumber(node->proto().float_ops()) + "/" + fops;
    } else {
      fops = "--/" + fops;
    }
    attrs.push_back(fops);
  }

  if (opts.select.find(kShown[5]) != opts.select.end() &&
      !node->node->devices().empty()) {
    attrs.push_back(str_util::Join(node->node->devices(), "|"));
  }
  if (opts.select.find(kShown[6]) != opts.select.end()) {
    std::set<string> op_types = node->node->op_types();
    attrs.push_back(str_util::Join(op_types, "|"));
  }
  if (opts.select.find(kShown[7]) != opts.select.end()) {
    // TODO(xpan): Make op count available in code view?
    attrs.push_back(strings::Printf("%s N/A in code view", kShown[7]));
  }
  if (opts.select.find(kShown[8]) != opts.select.end()) {
    attrs.push_back(strings::Printf("%s N/A in code view", kShown[8]));
  }

  return strings::Printf("%s%s (%s)\n", string(indent, ' ').c_str(),
                         node->name().c_str(),
                         str_util::Join(attrs, ", ").c_str());
}
}  // namespace tfprof
}  // namespace tensorflow
