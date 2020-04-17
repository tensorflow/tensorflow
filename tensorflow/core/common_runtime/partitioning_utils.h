/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_PARTITIONING_UTILS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_PARTITIONING_UTILS_H_

#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Given a `device_set` and a `graph`, partitions the `graph` into
// `subgraphs`. `subgraphs` maps device names to the graph assigned to that
// device. `graph` must have been placed (e.g. by running Placer),
// i.e. all nodes must have an assigned_device set.
// `graph` is non-const because the underlying Partition() function transforms
// the graph to correctly partition distributed control flow.
Status PartitionFunctionGraph(
    const DeviceSet& device_set, std::unique_ptr<Graph> graph,
    std::unordered_map<string, std::unique_ptr<Graph>>* subgraphs);

// Each subgraph produced by partitioning the function body contains a subset
// of the original `Arg` and `Retval` nodes. This function performs
// bookkeeping to track which `Arg` and `Retval` nodes were placed on a
// particular device / subgraph.
//
// More specifically, this function
//  (1) rewrites the indices of the `Arg` and `Retval` nodes placed
//      on a particular device.  When a function is partitioned, each
//      partition `subgraph` gets a subset of the arguments and
//      return values. The `index` attributes of these _Arg and _Retval
//      nodes reflect the indices of these parameters in the original
//      function. To convert `subgraph` to a function, we need to replace
//      there original indices with 0, 1, 2, ... .
//
//      The argument and return value order in the partitioned function is
//      determined by the argument and return value order in the original
//      function. This stability is important because it enables us to treat
//      a single-partition function as having the same signature as the
//      subgraph.
//  (2) records the subsets of `Arg` and `Retval` nodes assigned to the
//      device in `*_indices`, and
//  (3) records which `Arg` and `Retval` nodes live in host memory in
//      `*_alloc_attrs`.
Status UpdateArgAndRetvalMetadata(
    Graph* subgraph, const string& device_type, std::vector<int>* arg_indices,
    std::vector<int>* ret_indices,
    std::vector<AllocatorAttributes>* arg_alloc_attrs,
    std::vector<AllocatorAttributes>* ret_alloc_attrs);

// Extracts tensors at `indices` from `arguments`.
std::vector<Tensor> GetArgsForIndices(const std::vector<int>& indices,
                                      gtl::ArraySlice<Tensor> arguments);

// Utility for generating function names not present in `flib_def`, using
// given `name` as the base for the name.
class FunctionNameGenerator {
 public:
  // `flib_def` must outlive this.
  FunctionNameGenerator(const FunctionLibraryDefinition* flib_def,
                        const string& name)
      : flib_def_(flib_def), name_(name), counter_(0) {}

  // Returns a function name not present in `flib_def` using `name` as
  // the base and appending a numeric suffix.
  string GetName();

 private:
  const FunctionLibraryDefinition* flib_def_;
  const string name_;
  uint32 counter_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_PARTITIONING_UTILS_H_
