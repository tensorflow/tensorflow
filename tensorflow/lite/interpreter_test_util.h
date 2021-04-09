#ifndef TENSORFLOW_LITE_INTERPRETER_TEST_UTIL_H_
#define TENSORFLOW_LITE_INTERPRETER_TEST_UTIL_H_

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <stdint.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {

// InterpreterTest is a friend of Interpreter, so it can access context_.
class InterpreterTest : public ::testing::Test {
 public:
  template <typename Delegate>
  static TfLiteStatus ModifyGraphWithDelegate(
      Interpreter* interpreter, std::unique_ptr<Delegate> delegate) {
    return interpreter->ModifyGraphWithDelegate(std::move(delegate));
  }

 protected:
  TfLiteContext* GetInterpreterContext() { return interpreter_.context_; }

  std::vector<Interpreter::TfLiteDelegatePtr>*
  mutable_lazy_delegate_providers() {
    return &interpreter_.lazy_delegate_providers_;
  }

  bool HasDelegates() { return interpreter_.HasDelegates(); }

  void BuildSignature(const std::string& method_name, const std::string& key,
                      const std::map<std::string, uint32_t>& inputs,
                      const std::map<std::string, uint32_t>& outputs) {
    Interpreter::SignatureDef signature;
    signature.inputs = inputs;
    signature.outputs = outputs;
    signature.method_name = method_name;
    signature.signature_def_key = key;
    interpreter_.SetSignatureDef({signature});
  }

  Interpreter interpreter_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_INTERPRETER_TEST_UTIL_H_
