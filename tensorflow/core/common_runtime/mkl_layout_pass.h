/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// A graph pass that rewrites graph for propagating MKL layout as a tensor

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_MKL_LAYOUT_PASS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_MKL_LAYOUT_PASS_H_

#ifdef INTEL_MKL

#include <sys/types.h>
#include <memory>
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

struct RewriteThreshold {
  string op;
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

// Interface to invoke the pass for unit test
//
// Returns true if and only if 'g' is mutated.
extern bool RunMklLayoutRewritePass(std::unique_ptr<Graph>* g);
}  // namespace tensorflow

#endif

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_MKL_LAYOUT_PASS_H_
