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

#ifndef TENSORFLOW_COMPILER_JIT_MARK_FOR_COMPILATION_PASS_TEST_HELPER_H_
#define TENSORFLOW_COMPILER_JIT_MARK_FOR_COMPILATION_PASS_TEST_HELPER_H_

#include "tensorflow/compiler/jit/mark_for_compilation_pass.h"

namespace tensorflow {
class MarkForCompilationPassTestHelper {
 public:
  // Runs the MarkForCompilation pass on `graph` after assigning all nodes in
  // `graph` to the CPU device.  To make testing easier, ignores device
  // registration, _XlaCompile attributes, input deadness and global jit level.
  static Status MarkForCompilation(std::unique_ptr<Graph>* graph,
                                   FunctionLibraryDefinition* flib_def,
                                   SessionOptions* session_options);

  // Like `MarkForCompilation` but creates a default SessionOptions.
  static Status MarkForCompilation(std::unique_ptr<Graph>* graph,
                                   FunctionLibraryDefinition* flib_def);

  // Like `MarkForCompilation` but creates `flib_def` from the op registry.
  static Status MarkForCompilation(std::unique_ptr<Graph>* graph);
};
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_MARK_FOR_COMPILATION_PASS_TEST_HELPER_H_
