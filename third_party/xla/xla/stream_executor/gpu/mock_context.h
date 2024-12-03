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

#ifndef XLA_STREAM_EXECUTOR_GPU_MOCK_CONTEXT_H_
#define XLA_STREAM_EXECUTOR_GPU_MOCK_CONTEXT_H_

#include "xla/stream_executor/gpu/context.h"
#include "xla/test.h"

namespace stream_executor::gpu {

// Implements the Context interface for testing.
class MockContext : public Context {
 public:
  MockContext() = default;
  MOCK_METHOD(void, SetActive, (), (override));
  MOCK_METHOD(bool, IsActive, (), (const, override));
  MOCK_METHOD(int, device_ordinal, (), (const, override));
  MOCK_METHOD(absl::Status, Synchronize, (), (override));
};
}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_MOCK_CONTEXT_H_
