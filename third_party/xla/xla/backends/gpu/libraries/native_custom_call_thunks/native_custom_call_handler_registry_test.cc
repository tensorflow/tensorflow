/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_registry.h"

#include <cstdint>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_emitter_context.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_registration.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/ffi/attribute_map.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu_topology.h"
#include "xla/shape_util.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;

absl::StatusOr<ThunkSequence> DummyHandler(
    const HloCustomCallInstruction&, const NativeCustomCallEmitterContext&) {
  return ThunkSequence::Empty();
}

// Registered via the macro at static-init time; used to verify end-to-end
// registration through the public macro path.
XLA_GPU_REGISTER_NATIVE_CUSTOM_CALL_HANDLER(
    "xla.gpu.test_registry_macro_handler", DummyHandler);

TEST(NativeCustomCallHandlerRegistryTest, LookupUnknownReturnsNullopt) {
  EXPECT_EQ(NativeCustomCallHandlerRegistry::GetGlobal().Lookup(
                "xla.gpu.this_target_does_not_exist"),
            std::nullopt);
}

TEST(NativeCustomCallHandlerRegistryTest, MacroRegistersHandler) {
  EXPECT_TRUE(NativeCustomCallHandlerRegistry::GetGlobal()
                  .Lookup("xla.gpu.test_registry_macro_handler")
                  .has_value());
}

TEST(NativeCustomCallHandlerRegistryTest, RegisterAndLookup) {
  NativeCustomCallHandlerRegistry registry;
  EXPECT_EQ(registry.Lookup("target"), std::nullopt);
  EXPECT_OK(registry.Register("target", DummyHandler));
  EXPECT_TRUE(registry.Lookup("target").has_value());
}

TEST(NativeCustomCallHandlerRegistryTest,
     DuplicateRegistrationReturnsAlreadyExists) {
  NativeCustomCallHandlerRegistry registry;
  EXPECT_OK(registry.Register("target", DummyHandler));
  EXPECT_THAT(registry.Register("target", DummyHandler),
              StatusIs(absl::StatusCode::kAlreadyExists));
}

TEST(NativeCustomCallHandlerRegistryTest,
     RegisterNullHandlerReturnsInvalidArgument) {
  NativeCustomCallHandlerRegistry registry;
  EXPECT_THAT(registry.Register("target", nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

class MockNativeCustomCallEmitterContext
    : public NativeCustomCallEmitterContext {
 public:
  const GpuTopology& GetTargetTopology() const override { return *topology_; }
  const DebugOptions& GetDebugOptions() const override {
    return debug_options_;
  }
  Thunk::ThunkInfo GenerateThunkInfo() const override {
    Thunk::ThunkInfo info;
    info.thunk_id = ThunkId(42);
    return info;
  }
  absl::StatusOr<BufferAllocation::Slice> GetResultAllocationSlice(
      const ShapeIndex& index) const override {
    return slice_;
  }
  absl::StatusOr<BufferAllocation::Slice> GetOperandAllocationSlice(
      int64_t operand_index, const ShapeIndex& index) const override {
    return slice_;
  }
  absl::StatusOr<xla::ffi::AttributesMap> GetFfiAttributes() const override {
    return xla::ffi::AttributesMap();
  }

  const GpuTopology* topology_ = nullptr;
  DebugOptions debug_options_;
  BufferAllocation::Slice slice_;
};

TEST(NativeCustomCallEmitterContextTest, VirtualDispatchWorks) {
  MockNativeCustomCallEmitterContext mock_ctx;
  const NativeCustomCallEmitterContext& ctx = mock_ctx;

  EXPECT_EQ(ctx.GenerateThunkInfo().thunk_id, ThunkId(42));
  EXPECT_OK(ctx.GetResultAllocationSlice({}));
  EXPECT_OK(ctx.GetOperandAllocationSlice(0, {}));
}

}  // namespace
}  // namespace xla::gpu
