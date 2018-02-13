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
#ifndef TENSORFLOW_CONTRIB_LITE_TOCO_GRAPH_TRANSFORMATIONS_REMOVE_TRIVIAL_PASSTHROUGH_H_
#define TENSORFLOW_CONTRIB_LITE_TOCO_GRAPH_TRANSFORMATIONS_REMOVE_TRIVIAL_PASSTHROUGH_H_

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"

namespace toco {

// A "passthrough op" is an op that satisfies the following conditions:
//   1. One of its inputs is (per the semantics of that op) its "main input"
//      for some notion of "main input" that is operator-specific; for example,
//      for a Reshape op, the main input is the array being reshaped, not the
//      other input which gives the new shape.
//   2. It has exactly one output.
//   3. It forwards exactly its main input to its single output.
//
// Examples include:
//   1. TensorFlow Identity ops. (Have one input).
//   2. TensorFlow Reshape ops when the input and output shapes agree.
//   3. Any binary operator, one of whose two inputs is a constant and is the
//      neutral value for that operation. For example, a binary Add operator
//      where one of its inputs is a constant array filled with zeros.
//
// A passthrough op is "trivial" and can be removed when it is possible to
// discard either its main input or output array, rerouting any
// edge involving it to the other of these two arrays.
//
// It is only possible to discard such an array if it is not explicitly
// designated as a global input/output array of the graph, e.g. the model's
// input arrays, output arrays, and any array involved in a RNN back-edge
// specified by the model.
//
// This function does not check that the given operator is a passthrough op:
// that's the responsibility of the caller.
// Given that it is a passthrough op, this function checks whether it is trivial
// and then discards it and returns true, or, if it's not trivial (if neither
// the input nor the output may be discarded), returns false.
bool RemoveTrivialPassthroughOp(GraphTransformation* transformation,
                                Model* model, std::size_t op_index);

}  // namespace toco

#endif  // TENSORFLOW_CONTRIB_LITE_TOCO_GRAPH_TRANSFORMATIONS_REMOVE_TRIVIAL_PASSTHROUGH_H_
