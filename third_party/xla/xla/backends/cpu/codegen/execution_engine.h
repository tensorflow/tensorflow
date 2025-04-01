/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_CODEGEN_EXECUTION_ENGINE_H_
#define XLA_BACKENDS_CPU_CODEGEN_EXECUTION_ENGINE_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/IR/DataLayout.h"

namespace xla::cpu {

class ExecutionEngine {
 public:
  // A callback that returns a definition generator that will be added to all
  // dynamic libraries created by the engine. Definition generator enables
  // linking host runtime symbols into the jit-compiled function library.
  using DefinitionGenerator =
      std::function<std::unique_ptr<llvm::orc::DefinitionGenerator>(
          const llvm::DataLayout&)>;

  // Specifying a data layout adds a runtime symbol generator to each dylib.
  explicit ExecutionEngine(
      std::unique_ptr<llvm::orc::ExecutionSession> execution_session,
      const llvm::DataLayout& data_layout,
      DefinitionGenerator definition_generator = nullptr);

  void AllocateDylibs(size_t num_dylibs);

  void RegisterJITEventListeners();

  // Implementation from LLJIT, required to find symbols on Windows.
  void SetObjectLayerFlags();

  llvm::orc::RTDyldObjectLinkingLayer* object_layer() {
    return object_layer_.get();
  }

  llvm::orc::ExecutionSession* execution_session() {
    return execution_session_.get();
  }

  llvm::DataLayout data_layout() const { return data_layout_; }

  size_t num_dylibs() const { return dylibs_.size(); }

  absl::StatusOr<llvm::orc::JITDylib*> dylib(size_t dylib_index) {
    if (dylib_index >= num_dylibs()) {
      return absl::Status(
          absl::StatusCode::kInvalidArgument,
          absl::StrFormat("Invalid dylib index %d (num dylibs: %d))",
                          dylib_index, num_dylibs()));
    }
    return dylibs_[dylib_index];
  }

  ~ExecutionEngine();

 private:
  // LLVM execution session that holds jit-compiled functions.
  std::unique_ptr<llvm::orc::ExecutionSession> execution_session_;
  // Owns resources required for the execution session.
  std::unique_ptr<llvm::orc::RTDyldObjectLinkingLayer> object_layer_;
  // Non-owning pointers to dynamic libraries created for the execution session.
  std::vector<llvm::orc::JITDylib*> dylibs_;

  llvm::DataLayout data_layout_;
  DefinitionGenerator definition_generator_;

  /// GDB notification listener.
  llvm::JITEventListener* gdb_listener_ = nullptr;  // not owned
  /// Perf notification listener.
  llvm::JITEventListener* perf_listener_ = nullptr;  // not owned
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_EXECUTION_ENGINE_H_
