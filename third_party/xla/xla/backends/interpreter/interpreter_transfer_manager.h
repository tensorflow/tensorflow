/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_INTERPRETER_INTERPRETER_TRANSFER_MANAGER_H_
#define XLA_BACKENDS_INTERPRETER_INTERPRETER_TRANSFER_MANAGER_H_

#include "xla/service/generic_transfer_manager.h"
#include "xla/service/shaped_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {

// An implementation of the XLA GenericTransferManager for interpreter backend.
class InterpreterTransferManager : public GenericTransferManager {
 public:
  InterpreterTransferManager();
  ~InterpreterTransferManager() override = default;

  bool CanShapedBufferBeAccessedNow(
      se::StreamExecutor* executor,
      const ShapedBuffer& device_buffer) const override {
    return true;
  }

  bool CanBufferBeAccessedNow(
      se::StreamExecutor* executor,
      const se::DeviceMemoryBase& device_buffer) const override {
    return true;
  }

 private:
  InterpreterTransferManager(const InterpreterTransferManager&) = delete;
  InterpreterTransferManager& operator=(const InterpreterTransferManager&) =
      delete;
};

}  // namespace xla

#endif  // XLA_BACKENDS_INTERPRETER_INTERPRETER_TRANSFER_MANAGER_H_
