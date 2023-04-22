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
// This checker checks the most expensive operations.
#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_ADVISOR_EXPENSIVE_OPERATION_CHECKER_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_ADVISOR_EXPENSIVE_OPERATION_CHECKER_H_

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/profiler/internal/advisor/checker.h"

namespace tensorflow {
namespace tfprof {

class ExpensiveOperationChecker : public Checker {
 public:
  string name() const override { return kCheckers[2]; }

 private:
  AdviceProto::Checker Check(const AdvisorOptionsProto::CheckerOption& options,
                             const TFStats* stats) override {
    if (!stats) {
      absl::FPrintF(
          stderr, "Missing profiles (e.g. graph, run_meta). Skip %s\n", name());
      return reports_;
    }
    if (stats->steps().empty()) {
      absl::FPrintF(stderr, "Missing RunMetadata info. Skip %s\n", name());
    }
    CheckOpView(stats);
    CheckScopeView(stats);
    CheckCodeView(stats);
    return reports_;
  }

  void CheckOpView(const TFStats* stats) {
    if (stats->steps().empty()) {
      absl::FPrintF(stderr, "Missing run_meta for %s\n", name());
      return;
    }
    Options opts(3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, "micros", {".*"}, {".*"},
                 {}, {".*"}, {}, false, {"micros", "occurrence"}, "none", {});
    const MultiGraphNodeProto root = stats->ShowMultiGraphNode("op", opts);
    if (root.children_size() == 0) {
      return;
    }
    const MultiGraphNodeProto* node = &root;
    std::vector<string> outputs;
    for (int i = 0; i < 3 && node->children_size() > 0; ++i) {
      node = &node->children(0);
      outputs.push_back(absl::StrFormat(
          "top %d operation type: %s, "
          "cpu: %s, accelerator: %s, total: %s (%.2f%%)",
          i + 1, node->name(), FormatTime(node->cpu_exec_micros()),
          FormatTime(node->accelerator_exec_micros()),
          FormatTime(node->exec_micros()),
          100.0 * node->exec_micros() / (root.total_exec_micros() + 1e-10)));
    }
    reports_.add_reports(absl::StrJoin(outputs, "\n"));
  }

  void CheckCodeView(const TFStats* stats) {
    if (!stats->has_code_traces()) {
      absl::FPrintF(stderr, "Missing op_log (code traces) for %s\n", name());
      return;
    }
    Options opts(100, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, "micros", {".*"},
                 {".*"}, {}, {".*"}, {}, false, {"micros"}, "none", {});
    const MultiGraphNodeProto root = stats->ShowMultiGraphNode("code", opts);
    const MultiGraphNodeProto* node = &root;
    // A trick here is: Usually, codes in library file are usually referenced
    // only once, while user's own code are referenced multiple times.
    while (node->children_size() == 1) {
      node = &node->children(0);
    }
    if (node->children_size() == 0) {
      return;
    }

    std::vector<string> outputs;
    CodeViewHelper(node, 0, &outputs);
    reports_.add_reports(absl::StrJoin(outputs, "\n"));
  }

  void CheckScopeView(const TFStats* stats) {
    Options opts(100, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, -1, "micros", {".*"},
                 {".*"}, {}, {".*"}, {}, false, {"micros"}, "none", {});
    const GraphNodeProto root = stats->ShowGraphNode("scope", opts);
    if (root.children_size() == 0) {
      return;
    }
    std::vector<string> outputs;
    for (int i = 0; i < 3 && i < root.children_size(); ++i) {
      const GraphNodeProto& node = root.children(i);
      outputs.push_back(absl::StrFormat(
          "top %d graph node: %s, cpu: %s, accelerator: %s, total: %s", i + 1,
          node.name(), FormatTime(node.cpu_exec_micros()),
          FormatTime(node.accelerator_exec_micros()),
          FormatTime(node.exec_micros())));
    }
    reports_.add_reports(absl::StrJoin(outputs, "\n"));
  }

  void CodeViewHelper(const MultiGraphNodeProto* node, int depth,
                      std::vector<string>* outputs) {
    if (node->children_size() <= 1 || depth > 3) {
      return;
    }
    for (int j = 0; j < 3 && j < node->children_size(); ++j) {
      const MultiGraphNodeProto* c = &node->children(j);
      if (c->total_exec_micros() < 1000) {
        continue;
      }
      outputs->push_back(
          absl::StrFormat("%s%s, cpu: %s, accelerator: %s, total: %s",
                          std::string(depth * 2, ' '), c->name(),
                          FormatTime(c->total_cpu_exec_micros()),
                          FormatTime(c->total_accelerator_exec_micros()),
                          FormatTime(c->total_exec_micros())));
      CodeViewHelper(c, depth + 1, outputs);
    }
  }

  AdviceProto::Checker reports_;
};

}  // namespace tfprof
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_ADVISOR_EXPENSIVE_OPERATION_CHECKER_H_
