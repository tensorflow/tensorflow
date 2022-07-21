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

// Parent class and utilities for tfprof_graph and tfprof_scope.

#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_SHOW_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_SHOW_H_

#include <algorithm>
#include <string>
#include <vector>

#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/profiler/internal/tfprof_constants.h"
#include "tensorflow/core/profiler/internal/tfprof_node.h"
#include "tensorflow/core/profiler/internal/tfprof_node_show.h"
#include "tensorflow/core/profiler/internal/tfprof_tensor.h"
#include "tensorflow/core/profiler/internal/tfprof_timeline.h"
#include "tensorflow/core/profiler/internal/tfprof_utils.h"
#include "tensorflow/core/profiler/tfprof_options.h"
#include "tensorflow/core/profiler/tfprof_output.pb.h"

namespace tensorflow {
namespace tfprof {
class TFShow {
 public:
  explicit TFShow(checkpoint::CheckpointReader* ckpt_reader)
      : ckpt_reader_(ckpt_reader) {}
  virtual ~TFShow() {}
  virtual void AddNode(TFGraphNode* node) = 0;
  virtual void Build() = 0;
  virtual const GraphNodeProto& Show(const string& prefix,
                                     const Options& opts) final;

 protected:
  virtual const ShowNode* ShowInternal(const Options& opts,
                                       Timeline* timeline) = 0;

  bool LookUpCheckPoint(const string& name,
                        std::unique_ptr<TFProfTensor>* tensor);

  // Overridden by subclass if extra requirements need to be met.
  virtual bool ShouldShowIfExtra(const ShowNode* node, const Options& opts,
                                 int depth) const {
    return true;
  }

  bool ShouldShow(const ShowNode* node, const Options& opts, int depth) const;

  bool ShouldTrim(const ShowNode* node,
                  const std::vector<string>& regexes) const;

  bool ReAccount(ShowNode* node, const Options& opts);

  string FormatNode(ShowNode* node, const Options& opts) const;
  string FormatNodeMemory(ShowNode* node, int64_t bytes,
                          int64_t total_bytes) const;

  string FormatLegend(const Options& opts) const;

  template <typename T>
  std::vector<T*> SortNodes(const std::vector<T*>& nodes, const Options& opts) {
    if (opts.order_by.empty() || nodes.empty()) {
      return nodes;
    }
    std::vector<T*> sorted_nodes = nodes;
    std::stable_sort(sorted_nodes.begin(), sorted_nodes.end(),
                     [&opts](const T* n1, const T* n2) {
                       if (n1->name() == kTFProfRoot) return true;
                       if (n2->name() == kTFProfRoot) return false;
                       bool name_cmp = n1->name() < n2->name();
                       if (opts.order_by == kOrderBy[0]) {
                         return name_cmp;
                       } else if (opts.order_by == kOrderBy[1]) {
                         return n1->proto().total_requested_bytes() >
                                n2->proto().total_requested_bytes();
                       } else if (opts.order_by == kOrderBy[2]) {
                         return n1->proto().total_peak_bytes() >
                                n2->proto().total_peak_bytes();
                       } else if (opts.order_by == kOrderBy[3]) {
                         return n1->proto().total_residual_bytes() >
                                n2->proto().total_residual_bytes();
                       } else if (opts.order_by == kOrderBy[4]) {
                         return n1->proto().total_output_bytes() >
                                n2->proto().total_output_bytes();
                       } else if (opts.order_by == kOrderBy[5]) {
                         return n1->proto().total_exec_micros() >
                                n2->proto().total_exec_micros();
                       } else if (opts.order_by == kOrderBy[6]) {
                         return n1->proto().total_accelerator_exec_micros() >
                                n2->proto().total_accelerator_exec_micros();
                       } else if (opts.order_by == kOrderBy[7]) {
                         return n1->proto().total_cpu_exec_micros() >
                                n2->proto().total_cpu_exec_micros();
                       } else if (opts.order_by == kOrderBy[8]) {
                         return n1->proto().total_parameters() >
                                n2->proto().total_parameters();
                       } else if (opts.order_by == kOrderBy[9]) {
                         return n1->proto().total_float_ops() >
                                n2->proto().total_float_ops();
                       }
                       return name_cmp;
                     });
    return sorted_nodes;
  }

  checkpoint::CheckpointReader* ckpt_reader_;
};

template <typename T>
string FormatTotalExecTime(const T* node, const Options& opts) {
  string time = FormatTime(node->proto().total_exec_micros());
  if (node->account) {
    time = FormatTime(node->proto().exec_micros()) + "/" + time;
  } else {
    time = "--/" + time;
  }
  return time;
}
template <typename T>
string FormatCPUExecTime(const T* node, const Options& opts) {
  string time = FormatTime(node->proto().total_cpu_exec_micros());
  if (node->account) {
    time = FormatTime(node->proto().cpu_exec_micros()) + "/" + time;
  } else {
    time = "--/" + time;
  }
  return time;
}
template <typename T>
string FormatAcceleratorExecTime(const T* node, const Options& opts) {
  string time = FormatTime(node->proto().total_accelerator_exec_micros());
  if (node->account) {
    time = FormatTime(node->proto().accelerator_exec_micros()) + "/" + time;
  } else {
    time = "--/" + time;
  }
  return time;
}
}  // namespace tfprof
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_SHOW_H_
