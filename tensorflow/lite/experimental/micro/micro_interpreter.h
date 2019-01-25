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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_MICRO_INTERPRETER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_MICRO_INTERPRETER_H_

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/experimental/micro/simple_tensor_allocator.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

class MicroInterpreter {
 public:
  // The lifetime of the model, op resolver, allocator, and error reporter must
  // be at least as long as that of the interpreter object, since the
  // interpreter may need to access them at any time. This means that you should
  // usually create them with the same scope as each other, for example having
  // them all allocated on the stack as local variables through a top-level
  // function.
  // The interpreter doesn't do any deallocation of any of the pointed-to
  // objects, ownership remains with the caller.
  MicroInterpreter(const Model* model, const OpResolver& op_resolver,
                   SimpleTensorAllocator* tensor_allocator,
                   ErrorReporter* error_reporter);

  TfLiteStatus Invoke();

  size_t tensors_size() const { return context_.tensors_size; }
  TfLiteTensor* tensor(int tensor_index);

  TfLiteTensor* input(int index);
  size_t inputs_size() const { return subgraph_->inputs()->Length(); }

  TfLiteTensor* output(int index);
  size_t outputs_size() const { return subgraph_->outputs()->Length(); }

  TfLiteStatus initialization_status() const { return initialization_status_; }

  ErrorReporter* error_reporter() { return error_reporter_; }

 private:
  const Model* model_;
  const OpResolver& op_resolver_;
  SimpleTensorAllocator* tensor_allocator_;
  ErrorReporter* error_reporter_;

  TfLiteStatus initialization_status_;
  const flatbuffers::Vector<flatbuffers::Offset<Tensor>>* tensors_;
  const flatbuffers::Vector<flatbuffers::Offset<Operator>>* operators_;
  TfLiteContext context_;

  const SubGraph* subgraph_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_MICRO_INTERPRETER_H_
