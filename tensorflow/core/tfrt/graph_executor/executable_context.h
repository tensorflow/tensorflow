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
#ifndef TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_EXECUTABLE_CONTEXT_H_
#define TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_EXECUTABLE_CONTEXT_H_

#include <memory>
#include <utility>

#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/context.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime
#include "tfrt/bef_executor/bef_file.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {

// Stores executable-related data.
struct ExecutableContext {
  ExecutableContext(mlrt::bc::Buffer bytecode_buffer,
                    std::unique_ptr<mlrt::LoadedExecutable> bytecode_executable)
      : bytecode_buffer(std::move(bytecode_buffer)),
        bytecode_executable(std::move(bytecode_executable)) {}

  ExecutableContext(tfrt::BefBuffer bef,
                    tfrt::RCReference<tfrt::BEFFile> bef_file)
      : bef(std::move(bef)), bef_file(std::move(bef_file)) {}

  bool IsForMlrt() const { return bytecode_executable != nullptr; }

  // Only one set of values will be filled.

  // For the MLRT path.
  mlrt::bc::Buffer bytecode_buffer;
  std::unique_ptr<mlrt::LoadedExecutable> bytecode_executable;

  // For the TFRT path.
  tfrt::BefBuffer bef;
  tfrt::RCReference<tfrt::BEFFile> bef_file;

  // There are some resources that need re-creating when the executable is
  // re-created, so a resource context is stored along with the executable.
  // This resource context is meant to be passed to the op kernels for their
  // references. See the comment above `GraphExecutor::resource_context_`
  // about the todo to merge that resource context with this one.
  tfrt::ResourceContext resource_context;
};

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_EXECUTABLE_CONTEXT_H_
