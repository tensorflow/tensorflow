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

#ifndef XLA_SERVICE_CPU_RUNTIME_THUNK_TESTLIB_H_
#define XLA_SERVICE_CPU_RUNTIME_THUNK_TESTLIB_H_

#include "absl/status/status.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

// A test-only thunk to create a Thunk with a specific buffer use.
class BufferUseThunk : public Thunk {
 public:
  explicit BufferUseThunk(BufferUse buffer_use)
      : Thunk(Kind::kKernel, {"buffer-use"}), buffer_use_(buffer_use) {}

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams&) final {
    return absl::UnimplementedError("Unimplemented");
  }

  BufferUses buffer_uses() const final { return {buffer_use_}; }

 private:
  BufferUse buffer_use_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_RUNTIME_THUNK_TESTLIB_H_
