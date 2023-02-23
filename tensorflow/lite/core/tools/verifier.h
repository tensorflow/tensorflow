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
/// WARNING: Users of TensorFlow Lite should not include this file directly,
/// but should instead include "third_party/tensorflow/lite/tools/verifier.h".
/// Only the TensorFlow Lite implementation itself should include this
/// file directly.
#ifndef TENSORFLOW_LITE_CORE_TOOLS_VERIFIER_H_
#define TENSORFLOW_LITE_CORE_TOOLS_VERIFIER_H_

#include <stdio.h>

#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/model.h"      // Legacy.
#include "tensorflow/lite/error_reporter.h"  // Legacy.

namespace tflite {

class AlwaysTrueResolver : public OpResolver {
 public:
  AlwaysTrueResolver() {}
  const TfLiteRegistration* FindOp(tflite::BuiltinOperator op,
                                   int version) const override {
    static TfLiteRegistration null_registration = {nullptr, nullptr, nullptr,
                                                   nullptr};
    return &null_registration;
  }
  const TfLiteRegistration* FindOp(const char* op, int version) const override {
    static TfLiteRegistration null_registration = {nullptr, nullptr, nullptr,
                                                   nullptr};
    return &null_registration;
  }
};

// Verifies the integrity of a Tensorflow Lite flatbuffer model file.
// Currently, it verifies:
// * The file is following a legit flatbuffer schema.
// * The model is in supported version.
// * All ops used in the model are supported by OpResolver.
// DEPRECATED:
//   This function is deprecated, because it doesn't take delegates into
//   account, and as a result may report errors if the model contains
//   operators that are not supported by the OpResolver but that would be
//   rewritten by any TfLiteDelegate that you are using.
// Suggested replacement:
//   Use the version below that doesn't takes an OpResolver (and
//   doesn't check the validity of the ops) instead of this function,
//   and delay verification of the ops until after you have constructed
//   the Interpreter.  To verify that the operators in the model are supported
//   by the delegate(s) and/or by the OpResolver, construct the Interpreter,
//   applying the TfLiteDelegate(s) using InterpreterBuilder::AddDelegate,
//   and then just check the return value from Interpreter::AllocateTensors().
bool Verify(const void* buf, size_t len, const OpResolver& resolver,
            ErrorReporter* error_reporter);

// Verifies the integrity of a Tensorflow Lite flatbuffer model file.
// Currently, it verifies:
// * The file is following a legit flatbuffer schema.
// * The model is in supported version.
// * Some basic consistency checks on the graph.
// * Some validity checks on the tensors.
bool Verify(const void* buf, size_t len, ErrorReporter* error_reporter);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_TOOLS_VERIFIER_H_
