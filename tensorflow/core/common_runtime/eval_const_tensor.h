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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EVAL_CONST_TENSOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EVAL_CONST_TENSOR_H_

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"

// TODO(skyewm): can this be combined with ConstantFold?

namespace tensorflow {

class GraphRunner;
class OpRegistryInterface;
class ShapeRefiner;
class Tensor;

// Attempts to evaluate `tensor`. This will only be possible if `tensor` doesn't
// depend on any graph inputs (this function is safe to call if this isn't the
// case though).
//
// If the evaluation is successful, `evaluated` will be set to true and
// `tensor`s value returned in `result`. Otherwise `evaluated` will be set to
// false. An error status is returned if something is wrong with the graph or
// input. Note that `evaluated` may set to false if OkStatus() is returned.
//
// Params:
//   tensor - the tensor to be evaluated.
//   refiner - used to fetch the InferenceContexts for nodes in the graph.
//   ops - the OpRegistryInterface for the graph.
//   graph_def_version - the producer version of the graph.
//   evaluated - output param indicating whether evaluation was successful.
//   result - output param containing the result if evaluated is true.
//   graph_runner - optional. If not set, a GraphRunner will be created for
//     evaluating tensor. This can be set to avoid creating a new GraphRunner
//     for every call.
//   cached_values - optional. This can be used to cache evaluated results
//     across calls, to avoid evaluating the same parts of the graph multiple
//     times.
//   max_cached_value_size - optional. If `cached_values` is set, the maximum
//     result size to cache.
//   disable_constant_propagation - if true, only Const node values will be
//     returned.
//   outer_context - optional. The InferenceContext for the call node if inside
//     a nested function. This is useful for doing constant propagation across
//     Arg nodes.
Status EvaluateConstantTensor(
    OutputTensor tensor, const ShapeRefiner& refiner,
    const OpRegistryInterface& ops, int32_t graph_def_version, bool* evaluated,
    Tensor* result, GraphRunner* graph_runner = nullptr,
    std::unordered_map<string, Tensor>* cached_values = nullptr,
    int64_t max_cached_value_size = 1024,
    bool disable_constant_propagation = false,
    shape_inference::InferenceContext* outer_context = nullptr);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EVAL_CONST_TENSOR_H_
