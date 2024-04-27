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

#ifndef XLA_STREAM_EXECUTOR_SCOPED_MODULE_HANDLE_H_
#define XLA_STREAM_EXECUTOR_SCOPED_MODULE_HANDLE_H_

#include "absl/log/check.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/stream_executor_interface.h"

namespace stream_executor {

// A wrapper around ModuleHandle that uses RAII to manage its lifetime.
class ScopedModuleHandle {
 public:
  ScopedModuleHandle(StreamExecutorInterface* executor,
                     ModuleHandle module_handle)
      : executor_(executor), module_handle_(module_handle) {}

  ScopedModuleHandle(ScopedModuleHandle&& other) {
    executor_ = other.executor_;
    module_handle_ = other.module_handle_;
    other.executor_ = nullptr;
    other.module_handle_ = ModuleHandle();
  }

  ScopedModuleHandle& operator=(ScopedModuleHandle&& other) {
    executor_ = other.executor_;
    module_handle_ = other.module_handle_;
    other.executor_ = nullptr;
    other.module_handle_ = ModuleHandle();
    return *this;
  }

  ~ScopedModuleHandle() {
    if (static_cast<bool>(module_handle_)) {
      CHECK(executor_->UnloadModule(module_handle_));
    }
  }

 private:
  StreamExecutorInterface* executor_;
  ModuleHandle module_handle_;

  ScopedModuleHandle(const ScopedModuleHandle&) = delete;
  void operator=(const ScopedModuleHandle&) = delete;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_SCOPED_MODULE_HANDLE_H_
