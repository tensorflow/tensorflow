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

#include "tensorflow/compiler/jit/mark_for_compilation_pass_test_helper.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
/*static*/ Status MarkForCompilationPassTestHelper::MarkForCompilation(
    std::unique_ptr<Graph>* graph, FunctionLibraryDefinition* flib_def,
    SessionOptions* session_options) {
  // Assign all nodes to the CPU device.
  static const char* kCpuDevice = "/job:localhost/replica:0/task:0/cpu:0";
  for (Node* n : (*graph)->nodes()) {
    n->set_assigned_device_name(kCpuDevice);
  }

  GraphOptimizationPassOptions opt_options;
  opt_options.graph = graph;
  opt_options.session_options = session_options;
  opt_options.flib_def = flib_def;
  MarkForCompilationPass pass;
  return pass.RunImpl(opt_options);
}

/*static*/ Status MarkForCompilationPassTestHelper::MarkForCompilation(
    std::unique_ptr<Graph>* graph, FunctionLibraryDefinition* flib_def) {
  SessionOptions session_options;
  return MarkForCompilation(graph, flib_def, &session_options);
}

/*static*/ Status MarkForCompilationPassTestHelper::MarkForCompilation(
    std::unique_ptr<Graph>* graph) {
  FunctionDefLibrary flib;
  FunctionLibraryDefinition flib_def((*graph)->op_registry(), flib);
  return MarkForCompilation(graph, &flib_def);
}
}  // namespace tensorflow
