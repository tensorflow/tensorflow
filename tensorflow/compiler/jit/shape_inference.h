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

#ifndef TENSORFLOW_COMPILER_JIT_SHAPE_INFERENCE_H_
#define TENSORFLOW_COMPILER_JIT_SHAPE_INFERENCE_H_

#include <map>
#include <vector>

#include "absl/status/statusor.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

struct InferredShape {
  // Shape of the argument tensor.
  PartialTensorShape shape;

  // If the argument is a resource variable, the type and shape of the
  // variable's value.
  DataType handle_type = DT_INVALID;
  PartialTensorShape handle_shape;
};
typedef std::unordered_map<string, std::vector<InferredShape>> GraphShapeInfo;

// Infer shapes for all Tensors in a graph, and save them in a map.  The vector
// for a Node contains the information about each of its outputs.
// TODO(phawkins): this code does not infer accurate shapes for cyclic graphs.
// `arg_shapes`: user given map from the `index` to shapes of this
// node, where `index` is the `index` attribute of `_Arg` op or `_index`
// attribute of `Placeholder` op.
Status InferShapes(Graph* graph, const std::map<int, InferredShape>& arg_shapes,
                   const tensorflow::FunctionLibraryDefinition* fnlib_def,
                   GraphShapeInfo* shape_info);

// Merges two InferredShapes. Return an error if the two shapes cannot be
// merged.
absl::StatusOr<InferredShape> MergeInferredShapes(const InferredShape& a,
                                                  const InferredShape& b);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_SHAPE_INFERENCE_H_
