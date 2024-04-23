/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_PLUGIN_PROGRAM_H_
#define XLA_PYTHON_IFRT_PLUGIN_PROGRAM_H_

#include <string>

#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/program.h"

namespace xla {
namespace ifrt {

// `PluginProgram` is a subclass of `xla::ifrt::Program` used mainly with
// the IFRT proxy as of Apr 2024, and facilitates generic RPCs from the IFRT
// frontend (on the proxy-client) to the IFRT backend (on the proxy-server). A
// `PluginProgram` and its compiled executable need not be associated with a
// particular `xla::ifrt::Device`; instead, IFRT backends are expected to
// intercept and act on the compilation and subsequent executions of
// PluginProgram without passing them to particular devices.
//
// Another way to think of `PluginProgram` is that it is associated with a
// 'controller device', as opposed to CPU or GPU devices, where the term
// 'controller' means the same as in 'JAX uses a multi-controller programming
// model'.
struct PluginProgram
    : public llvm::RTTIExtends<PluginProgram, xla::ifrt::Program> {
  std::string data;
  static char ID;  // NOLINT
};

struct PluginCompileOptions
    : llvm::RTTIExtends<PluginCompileOptions, CompileOptions> {
  PluginCompileOptions() = default;
  ~PluginCompileOptions() override = default;

  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PLUGIN_PROGRAM_H_
