/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_PYTHON_FRAMEWORK_CPP_SHAPE_INFERENCE_H_
#define THIRD_PARTY_TENSORFLOW_PYTHON_FRAMEWORK_CPP_SHAPE_INFERENCE_H_

// Must be included first
#include "tensorflow/python/lib/core/numpy.h"

#include <vector>
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace swig {

// Calls the registered C++ shape inference function for <node> (a serialized
// NodeDef).
// Should not be called for shape functions that access input tensors; constant
// input tensor values are not made available, and so the inferred shapes will
// be less precise than they could be.
//
// Returns an error, or OK, in <out_status> according to whether the shape
// inference was successful.
//
// On success, returns a vector populated with the inferred output shapes (as
// serialized CppShapeInferenceResult protos) followed by a serialized
// CppShapeInferenceInputsNeeded proto.
//
// This is temporary code to be used during the migration
// from python shape inference functions to C++ shape inference functions.
std::vector<string> RunCppShapeInference(
    int graph_def_version, const string& serialized_node_def,
    const std::vector<string>& input_serialized_shapes,
    PyObject* input_constant_tensor_values,
    const std::vector<string>& input_constant_tensor_as_shape_values,
    TF_Status* out_status);

}  // namespace swig
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_PYTHON_FRAMEWORK_CPP_SHAPE_INFERENCE_H_
