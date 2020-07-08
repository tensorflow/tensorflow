/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_SAVED_MODEL_UTILS_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_SAVED_MODEL_UTILS_H_

// Some internal utility functions for the SavedModelAPI, factored out into a
// separately unit-testable header.

#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/constant.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/variable.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"

namespace tensorflow {
namespace internal {

// Load a TensorProto into a tensorflow::Constant. This is similar to the
// constant loading logic in python:
// https://github.com/tensorflow/tensorflow/blob/516608035f85cec8b126712b0ff8407220206b22/tensorflow/python/saved_model/load.py#L437
Status TensorProtoToConstant(ImmediateExecutionContext* ctx,
                             const TensorProto& proto,
                             std::unique_ptr<Constant>* output);

// Creates a tensorflow::Variable from a SavedVariable. This is similar to the
// logic in:
// https://github.com/tensorflow/tensorflow/blob/516608035f85cec8b126712b0ff8407220206b22/tensorflow/python/saved_model/load.py#L407
// Note that the caller **must assign a value** to the loaded variable.
Status LoadSavedVariable(ImmediateExecutionContext* ctx,
                         const SavedVariable& variable,
                         std::unique_ptr<Variable>* output);

}  // namespace internal
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_SAVED_MODEL_UTILS_H_
