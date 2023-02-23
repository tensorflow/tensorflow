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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_JIT_PYTHON_BINDING_TF_JITRT_EXECUTOR_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_JIT_PYTHON_BINDING_TF_JITRT_EXECUTOR_H_

#include <cstdint>
#include <string>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/runtime/jit_executable.h"
#include "tfrt/host_context/host_context.h"  // from @tf_runtime

namespace tensorflow {

// PyBind integration to compile and execute tf_jitrt MLIR modules. The only
// intended use case is testing tests for tf_jitrt in python, as an alternative
// of writing them directly in MLIR and executing with bef_executor.
class TfJitRtExecutor {
 public:
  using Handle = int64_t;
  using Specialization = xla::runtime::JitExecutable::Specialization;

  TfJitRtExecutor();

  // Compiles mlir module and caches it. Returns a handle, that can be passed to
  // execute function.
  Handle Compile(const std::string& mlir_module, const std::string& entrypoint,
                 Specialization specialization, bool vectorize,
                 bool legalize_i1_tensors);

  // Executes compiled mlir module with Python array arguments. Converts
  // returned memrefs into Python arrays.
  std::vector<pybind11::array> Execute(
      Handle handle, const std::vector<pybind11::array>& arguments);

  // Returns true if the binary was built with the given CPU feature.
  // The list of supported CPU features is purposedly incomplete; we
  // will only add features if JitRt relies on them.
  bool BuiltWith(const std::string& cpu_feature);

 private:
  tfrt::HostContext host_context_;
  llvm::DenseMap<Handle, xla::runtime::JitExecutable> jit_executables_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_JIT_PYTHON_BINDING_TF_JITRT_EXECUTOR_H_
