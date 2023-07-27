/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

// This file contains heuristics data and methods that are used to
// decide whether to rewrite node to use oneDNN kernels

#ifndef TENSORFLOW_CORE_UTIL_MKL_HEURISTICS_H_
#define TENSORFLOW_CORE_UTIL_MKL_HEURISTICS_H_
#ifdef INTEL_MKL

#include <vector>

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/tsl/platform/cpu_info.h"

namespace tensorflow {

struct RewriteThreshold {
  std::string op;
  int cpu_family;
  int cpu_model_num;
  // The model that is used to decide whether it is worth
  // accelerating operations using oneDNN is:
  //
  // threshold = thread_synchronisation * thread_num + framework_tax
  //
  // This finds threshold when framework overhead and thread synchronisations
  // are amortized with amount of computation that has to be performed.
  // If we are below this threshold then we will not rewrite the operation to
  // to be run using oneDNN primitive.
  struct PerformanceParameters {
    double thread_sync_cost;
    double framework_cost;
  } params;
};

// Table storing thread synchronization and framework overhead costs on each CPU
// architecture for each oneNN-eligible operation. Our heuristics use these
// costs to determine whether we should rewrite the operation to use oneDNN.
static const RewriteThreshold rewrite_thresholds[] = {
#ifdef DNNL_AARCH64_USE_ACL
    {"Conv2D", 0x41, 0xd40, {0.9349, 22.603}},
    {"_FusedConv2D", 0x41, 0xd40, {0.9349, 22.603}},
    {"FusedBatchNormV3", 0x41, 0xd40, {0.3223, -0.8822}},
    {"Sigmoid", 0x41, 0xd40, {0.0, 0.064736}},
#endif  // DNNL_AARCH64_USE_ACL
    {"", 0x0, 0x0, {0, 0}}};

static double FindRewriteThreshold(const string node_name, int threads) {
  int cpu_family_ = tsl::port::CPUFamily();
  int cpu_model_num_ = tsl::port::CPUModelNum();

  if (threads == 0) {
    // if we do not have information how many threads are used
    // to parallelise operation we revert to the old behaviour
    return 0;
  }

  for (const RewriteThreshold* i = rewrite_thresholds;
       i->op != "" && threads > 0; i++) {
    if (node_name == i->op && cpu_family_ == i->cpu_family &&
        cpu_model_num_ == i->cpu_model_num) {
      return i->params.thread_sync_cost * threads + i->params.framework_cost;
    }
  }

  return 0;
}

static double CalculateNodeMFlops(const AttrSlice& attrs,
                                  const string node_name) {
  // Check if we can obtained dimensions for this node.
  std::vector<const TensorShapeProto*> shape_attrs;
  if (!TryGetNodeAttr(attrs, "_input_shapes", &shape_attrs)) {
    // We can't obtain shape so we will revert to default behaviour
    // to rewrite node.
    return -1;
  }

  if ((node_name == "Conv2D" || node_name == "_FusedConv2D") &&
      shape_attrs.size() == 2) {
    TensorShape input_shape, filter_shape;
    if (TensorShape::BuildTensorShape(*shape_attrs[0], &input_shape) !=
        tsl::OkStatus()) {
      return -1;
    }
    if (TensorShape::BuildTensorShape(*shape_attrs[1], &filter_shape) !=
        tsl::OkStatus()) {
      return -1;
    }

    // MFLOPS = N * H * W * C * FH * FW * FC / 1e6.
    return input_shape.dim_size(0) * input_shape.dim_size(1) *
           input_shape.dim_size(2) * input_shape.dim_size(3) *
           filter_shape.dim_size(0) * filter_shape.dim_size(1) *
           filter_shape.dim_size(3) / (double)1e6;
  } else if ((node_name == "FusedBatchNormV3" || node_name == "Sigmoid") &&
             shape_attrs.size() >= 1) {
    TensorShape input_shape;
    if (TensorShape::BuildTensorShape(*shape_attrs[0], &input_shape) !=
        tsl::OkStatus()) {
      return -1;
    }
    return input_shape.dim_size(0) * input_shape.dim_size(1) *
           input_shape.dim_size(2) * input_shape.dim_size(3) / (double)1e6;
  }

  return -1;
}

}  // namespace tensorflow

#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_UTIL_MKL_HEURISTICS_H_
