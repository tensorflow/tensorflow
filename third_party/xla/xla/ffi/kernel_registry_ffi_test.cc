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

#include "xla/ffi/kernel_registry_ffi.h"

#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/string_view.h"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/kernel_api.h"
#include "xla/ffi/api/kernel_c_api.h"
#include "xla/ffi/api/kernel_registry_api.h"
#include "xla/ffi/api/kernel_registry_c_api.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/invoke.h"

namespace xla::ffi {
namespace {

static const XLA_FFI_Api* Api() { return GetXlaFfiApi(); }

template <typename T, typename = void>
struct CanRegister : std::false_type {};

template <typename T>
struct CanRegister<T, std::void_t<decltype(std::declval<T>().Register(
                          "k", nullptr, 0, SourceFormat::kPtx))>>
    : std::true_type {};

template <typename T, typename = void>
struct CanLaunch : std::false_type {};

template <typename T>
struct CanLaunch<T, std::void_t<decltype(std::declval<T>().Launch(
                        nullptr, XLA_FFI_LaunchDims{}, 0, false))>>
    : std::true_type {};

static_assert(CanRegister<PrepareKernelRegistry>::value);
static_assert(CanRegister<InitializeKernelRegistry>::value);
static_assert(!CanRegister<ExecuteKernelRegistry>::value);

static_assert(!CanLaunch<PrepareKernelRegistry>::value);
static_assert(!CanLaunch<InitializeKernelRegistry>::value);
static_assert(CanLaunch<ExecuteKernelRegistry>::value);

static_assert(internal::IsStageCompatibleCtx<PrepareKernelRegistry,
                                             ExecutionStage::kPrepare>::value);
static_assert(!internal::IsStageCompatibleCtx<
              PrepareKernelRegistry, ExecutionStage::kInitialize>::value);
static_assert(!internal::IsStageCompatibleCtx<PrepareKernelRegistry,
                                              ExecutionStage::kExecute>::value);
static_assert(internal::IsStageCompatibleCtx<
              InitializeKernelRegistry, ExecutionStage::kInitialize>::value);
static_assert(!internal::IsStageCompatibleCtx<InitializeKernelRegistry,
                                              ExecutionStage::kExecute>::value);
static_assert(internal::IsStageCompatibleCtx<ExecuteKernelRegistry,
                                             ExecutionStage::kExecute>::value);
static_assert(!internal::IsStageCompatibleCtx<
              KernelRegistry<ExecutionStage::kInstantiate>,
              ExecutionStage::kInstantiate>::value);
static_assert(
    !internal::IsStageCompatibleCtx<KernelRegistry<ExecutionStage::kRecord>,
                                    ExecutionStage::kRecord>::value);

TEST(KernelRegistryFfiTest, KernelRegistry) {
  static const auto* kExpectedKernel =
      reinterpret_cast<const XLA_FFI_Kernel*>(0x1234);

  struct TestKernelRegistryState {
    int32_t device_ordinal = 1;
    bool registered = false;
    bool launched = false;
  };

  auto register_cb =
      +[](XLA_FFI_KernelRegistry_Register_Args* args) -> XLA_FFI_Error* {
    auto* state = reinterpret_cast<TestKernelRegistryState*>(args->state);
    EXPECT_EQ(state->device_ordinal, 1);
    EXPECT_NE(args->kernel_spec, nullptr);
    if (args->kernel_spec == nullptr) {
      return nullptr;
    }
    EXPECT_EQ(absl::string_view(args->kernel_spec->name.ptr,
                                args->kernel_spec->name.len),
              "internal_kernel");
    EXPECT_EQ(args->kernel_spec->size, 6u);
    EXPECT_EQ(args->kernel_spec->format, XLA_FFI_SourceFormat_CUBIN);
    state->registered = true;
    args->kernel = kExpectedKernel;
    return nullptr;
  };

  auto launch_cb =
      +[](XLA_FFI_KernelRegistry_Launch_Args* args) -> XLA_FFI_Error* {
    auto* state = reinterpret_cast<TestKernelRegistryState*>(args->state);
    EXPECT_EQ(state->device_ordinal, 1);
    EXPECT_EQ(args->kernel, kExpectedKernel);
    EXPECT_NE(args->launch_params, nullptr);
    if (args->launch_params == nullptr) {
      return nullptr;
    }
    EXPECT_EQ(args->launch_params->launch_dims.grid.x, 2);
    EXPECT_EQ(args->launch_params->launch_dims.block.x, 64);
    EXPECT_EQ(args->launch_params->shared_mem_bytes, 256u);
    EXPECT_FALSE(args->launch_params->uses_pdl);
    EXPECT_EQ(args->launch_params->args->num_args, 1);
    state->launched = true;
    return nullptr;
  };

  TestKernelRegistryState state;
  XLA_FFI_KernelRegistryApi registry_api = {register_cb, launch_cb};
  XLA_FFI_KernelRegistry_Extension extension = {
      MakeExtensionHeader<internal::KernelRegistryExtensionBase<void>>(),
      reinterpret_cast<XLA_FFI_KernelRegistryState*>(&state),
      &registry_api,
  };
  void* dev_ptr = reinterpret_cast<void*>(0x5678);

  const XLA_FFI_Kernel* registered_kernel = nullptr;
  auto init_fn = [&](InitializeKernelRegistry kernel_registry) -> absl::Status {
    ABSL_ASSIGN_OR_RETURN(const XLA_FFI_Kernel* kernel,
                     kernel_registry.Register("internal_kernel", "cubin!", 6,
                                              SourceFormat::kCubin));
    EXPECT_EQ(kernel, kExpectedKernel);
    registered_kernel = kernel;
    return absl::OkStatus();
  };

  auto exec_fn = [&](ExecuteKernelRegistry kernel_registry) -> absl::Status {
    std::vector<void*> ptr_args = {dev_ptr};
    return kernel_registry.Launch(registered_kernel, {{2, 1, 1}, {64, 1, 1}},
                                  256, false, ptr_args);
  };

  auto prepare_fn = [&](PrepareKernelRegistry kernel_registry) -> absl::Status {
    ABSL_ASSIGN_OR_RETURN(const XLA_FFI_Kernel* kernel,
                     kernel_registry.Register("internal_kernel", "cubin!", 6,
                                              SourceFormat::kCubin));
    EXPECT_EQ(kernel, kExpectedKernel);
    return absl::OkStatus();
  };

  auto prepare_handler =
      Ffi::BindPrepare().Ctx<KernelRegistry>().To(prepare_fn);
  auto init_handler = Ffi::BindInitialize().Ctx<KernelRegistry>().To(init_fn);
  auto exec_handler = Ffi::BindExecute().Ctx<KernelRegistry>().To(exec_fn);
  CallFrame call_frame =
      CallFrameBuilder(/*num_args=*/0, /*num_rets=*/0).Build();

  InvokeContext context;
  context.device_ordinal = 1;
  context.extension_start = &extension.extension_base;

  ASSERT_OK(Invoke(Api(), *prepare_handler, call_frame, context,
                   ExecutionStage::kPrepare));
  EXPECT_TRUE(state.registered);
  state.registered = false;

  ASSERT_OK(Invoke(Api(), *init_handler, call_frame, context,
                   ExecutionStage::kInitialize));
  EXPECT_TRUE(state.registered);
  EXPECT_FALSE(state.launched);

  ASSERT_OK(Invoke(Api(), *exec_handler, call_frame, context,
                   ExecutionStage::kExecute));
  EXPECT_TRUE(state.launched);
}

}  // namespace
}  // namespace xla::ffi
