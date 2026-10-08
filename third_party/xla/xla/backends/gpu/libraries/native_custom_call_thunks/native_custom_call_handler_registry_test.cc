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

#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_emitter_context.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_registration.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_scratch_context.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/shape.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;

absl::StatusOr<ThunkSequence> DummyHandler(
    const HloCustomCallInstruction&, const NativeCustomCallEmitterContext&) {
  return ThunkSequence::Empty();
}

absl::StatusOr<std::vector<Shape>> DummyScratchHandler(
    const HloCustomCallInstruction&, const NativeCustomCallScratchContext&) {
  return std::vector<Shape>{};
}

// Registered via the macro at static-init time; used to verify end-to-end
// registration through the public macro path.
XLA_GPU_REGISTER_NATIVE_CUSTOM_CALL_HANDLER(
    "xla.gpu.test_registry_macro_handler", DummyHandler);

// Same, but registering a bundle with a scratch handler. The designated
// initializer list contains a comma, which is why the macro is variadic.
XLA_GPU_REGISTER_NATIVE_CUSTOM_CALL_HANDLER(
    "xla.gpu.test_registry_macro_bundle",
    NativeCustomCallHandlerBundle{
        /*emit_thunks=*/DummyHandler,
        /*request_scratch_buffers=*/DummyScratchHandler});

TEST(NativeCustomCallHandlerRegistryTest, LookupUnknownReturnsNullopt) {
  EXPECT_EQ(NativeCustomCallHandlerRegistry::GetGlobal().Lookup(
                "xla.gpu.this_target_does_not_exist"),
            std::nullopt);
  EXPECT_EQ(NativeCustomCallHandlerRegistry::GetGlobal().LookupScratchHandler(
                "xla.gpu.this_target_does_not_exist"),
            std::nullopt);
}

TEST(NativeCustomCallHandlerRegistryTest, MacroRegistersHandler) {
  EXPECT_TRUE(NativeCustomCallHandlerRegistry::GetGlobal()
                  .Lookup("xla.gpu.test_registry_macro_handler")
                  .has_value());
  EXPECT_FALSE(NativeCustomCallHandlerRegistry::GetGlobal()
                   .LookupScratchHandler("xla.gpu.test_registry_macro_handler")
                   .has_value());
}

TEST(NativeCustomCallHandlerRegistryTest, MacroRegistersBundle) {
  EXPECT_TRUE(NativeCustomCallHandlerRegistry::GetGlobal()
                  .Lookup("xla.gpu.test_registry_macro_bundle")
                  .has_value());
  EXPECT_TRUE(NativeCustomCallHandlerRegistry::GetGlobal()
                  .LookupScratchHandler("xla.gpu.test_registry_macro_bundle")
                  .has_value());
}

TEST(NativeCustomCallHandlerRegistryTest, RegisterAndLookup) {
  NativeCustomCallHandlerRegistry registry;
  EXPECT_EQ(registry.Lookup("target"), std::nullopt);
  EXPECT_OK(registry.Register("target", DummyHandler));
  EXPECT_TRUE(registry.Lookup("target").has_value());
  EXPECT_EQ(registry.LookupScratchHandler("target"), std::nullopt);
}

TEST(NativeCustomCallHandlerRegistryTest, RegisterBundleAndLookup) {
  NativeCustomCallHandlerRegistry registry;
  EXPECT_OK(registry.Register(
      "target", NativeCustomCallHandlerBundle{
                    /*emit_thunks=*/DummyHandler,
                    /*request_scratch_buffers=*/DummyScratchHandler}));
  EXPECT_TRUE(registry.Lookup("target").has_value());
  EXPECT_TRUE(registry.LookupScratchHandler("target").has_value());
}

TEST(NativeCustomCallHandlerRegistryTest,
     DuplicateRegistrationReturnsAlreadyExists) {
  NativeCustomCallHandlerRegistry registry;
  EXPECT_OK(registry.Register("target", DummyHandler));
  EXPECT_THAT(registry.Register("target", DummyHandler),
              StatusIs(absl::StatusCode::kAlreadyExists));
  EXPECT_THAT(
      registry.Register("target",
                        NativeCustomCallHandlerBundle{
                            /*emit_thunks=*/DummyHandler,
                            /*request_scratch_buffers=*/DummyScratchHandler}),
      StatusIs(absl::StatusCode::kAlreadyExists));
}

TEST(NativeCustomCallHandlerRegistryTest,
     RegisterNullHandlerReturnsInvalidArgument) {
  NativeCustomCallHandlerRegistry registry;
  EXPECT_THAT(registry.Register("target", nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(NativeCustomCallHandlerRegistryTest,
     RegisterBundleWithoutThunkHandlerReturnsInvalidArgument) {
  NativeCustomCallHandlerRegistry registry;
  EXPECT_THAT(
      registry.Register("target",
                        NativeCustomCallHandlerBundle{
                            /*emit_thunks=*/nullptr,
                            /*request_scratch_buffers=*/DummyScratchHandler}),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(registry.Lookup("target"), std::nullopt);
  EXPECT_EQ(registry.LookupScratchHandler("target"), std::nullopt);
}

}  // namespace
}  // namespace xla::gpu
