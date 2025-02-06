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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_FUNCTION_TESTLIB_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_FUNCTION_TESTLIB_H_

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/function.h"

namespace tensorflow {
namespace test {
namespace function {

// {} -> y:DT_STRING (device where this op runs).
FunctionDef FindDevice();
FunctionDef FindDeviceWithUuid();

class BlockingOpState {
 public:
  void AwaitState(int awaiting_state);

  void MoveToState(int expected_current, int next);

 private:
  mutex mu_;
  condition_variable cv_;
  int state_ = 0;
};

extern BlockingOpState* blocking_op_state;

FunctionDef BlockingOpFn();

// Adds a function call to the given scope and returns the output for the node.
// TODO(phawkins): replace with C++ API for calling functions, when that exists.
Output Call(Scope* scope, const string& op_name, const string& fn_name,
            absl::Span<const Input> inputs);

}  // namespace function
}  // namespace test
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_FUNCTION_TESTLIB_H_
