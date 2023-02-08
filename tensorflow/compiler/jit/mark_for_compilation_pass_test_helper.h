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

#include <memory>
#include <string>
#include <utility>

#include "tensorflow/compiler/jit/mark_for_compilation_pass.h"

namespace tensorflow {
class MarkForCompilationPassTestHelper {
 public:
  struct Options {
    bool enable_global_jit;
    bool disable_deadness_analysis;
    bool enable_cluster_scoping;
    bool deterministic_cluster_names;
    std::string friendly_name;  // TF ConfigProto.Experimental.friendly_name

    Options()
        : enable_global_jit(true),
          disable_deadness_analysis(true),
          enable_cluster_scoping(true),
          deterministic_cluster_names(false) {}

    Options WithNoGlobalJit() {
      Options copy = *this;
      copy.enable_global_jit = false;
      return copy;
    }

    Options WithDeadnessAnalysis() {
      Options copy = *this;
      copy.disable_deadness_analysis = false;
      return copy;
    }

    Options WithNoClusterScoping() {
      Options copy = *this;
      copy.enable_cluster_scoping = false;
      return copy;
    }

    Options WithDeterministicClusterNames() {
      Options copy = *this;
      copy.deterministic_cluster_names = true;
      return copy;
    }

    Options WithFriendlyName(std::string name) {
      Options copy = *this;
      copy.friendly_name = std::move(name);
      return copy;
    }
  };

  // Runs the MarkForCompilation pass on `graph` after assigning all nodes in
  // `graph` to the CPU device.  To make testing easier, ignores device
  // registration and  _XlaCompile attributes.
  static Status MarkForCompilation(std::unique_ptr<Graph>* graph,
                                   FunctionLibraryDefinition* flib_def,
                                   Options options = Options());

  // Like `MarkForCompilation` but creates `flib_def` from the op registry.
  static Status MarkForCompilation(std::unique_ptr<Graph>* graph,
                                   Options options = Options());
};
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_MARK_FOR_COMPILATION_PASS_TEST_HELPER_H_
