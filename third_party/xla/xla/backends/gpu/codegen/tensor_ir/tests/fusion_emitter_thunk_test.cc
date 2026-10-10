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

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/tests/gpu_pjrt_codegen_test.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

using ::testing::ElementsAre;

constexpr absl::string_view kAddF32 = R"(
fused_computation {
  p0 = f32[8,16] parameter(0)
  p1 = f32[8,16] parameter(1)
  ROOT add = f32[8,16] add(p0, p1)
}

ENTRY main {
  p0 = f32[8,16] parameter(0)
  p1 = f32[8,16] parameter(1)
  ROOT fusion = f32[8,16] fusion(p0, p1), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";

constexpr absl::string_view kTwoOutputsF32 = R"(
fused_computation {
  p0 = f32[8,16] parameter(0)
  p1 = f32[8,16] parameter(1)
  add = f32[8,16] add(p0, p1)
  mul = f32[8,16] multiply(p0, p1)
  ROOT tuple = (f32[8,16], f32[8,16]) tuple(add, mul)
}

ENTRY main {
  p0 = f32[8,16] parameter(0)
  p1 = f32[8,16] parameter(1)
  ROOT fusion = (f32[8,16], f32[8,16]) fusion(p0, p1), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";

// Checks the shape of what the TensorIR emitter hands to the runtime. Unlike
// the numerics tests next door these only compile and never launch, so they
// need a GPU to be visible but not one whose driver can load the kernel.
class TensorIrThunkTest : public GpuPjRtCodegenTest {
 protected:
  // Compiles without the HLO optimization passes, which would otherwise
  // rewrite the hand-written fusion and route it to a different emitter.
  absl::StatusOr<std::unique_ptr<Executable>> Compile(
      absl::string_view hlo_text) {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                     ParseAndReturnVerifiedModule(hlo_text));
    return CompileToExecutable(std::move(module),
                               /*run_optimization_passes=*/false);
  }

  // Compiles `hlo_text` and returns its only thunk.
  absl::StatusOr<const Thunk*> CompileToSingleThunk(
      absl::string_view hlo_text, std::unique_ptr<Executable>* executable) {
    ABSL_ASSIGN_OR_RETURN(*executable, Compile(hlo_text));
    auto* gpu_executable = absl::down_cast<GpuExecutable*>(executable->get());
    const ThunkSequence& thunks = gpu_executable->thunk_executor().thunks();
    if (thunks.size() != 1) {
      return absl::InternalError(
          absl::StrCat("expected exactly one thunk, got ", thunks.size()));
    }
    return thunks.front().get();
  }

  absl::StatusOr<const CustomKernelThunk*> CompileToSingleCustomKernelThunk(
      absl::string_view hlo_text, std::unique_ptr<Executable>* executable) {
    ABSL_ASSIGN_OR_RETURN(const Thunk* thunk,
                     CompileToSingleThunk(hlo_text, executable));
    const auto* custom_kernel_thunk =
        dynamic_cast<const CustomKernelThunk*>(thunk);
    if (custom_kernel_thunk == nullptr) {
      return absl::InternalError(
          absl::StrCat("expected a CustomKernelThunk, got ",
                       Thunk::KindToString(thunk->kind())));
    }
    return custom_kernel_thunk;
  }
};

TEST_F(TensorIrThunkTest, EmitsACustomKernelThunk) {
  std::unique_ptr<Executable> executable;
  ASSERT_OK_AND_ASSIGN(const CustomKernelThunk* thunk,
                       CompileToSingleCustomKernelThunk(kAddF32, &executable));
  EXPECT_EQ(thunk->kind(), Thunk::Kind::kCustomKernel);
}

// CudaTile's launch ABI: the grid comes from the compiler's tiling, every
// block is a single thread, and nothing uses shared memory.
TEST_F(TensorIrThunkTest, UsesTheCudaTileLaunchAbi) {
  std::unique_ptr<Executable> executable;
  ASSERT_OK_AND_ASSIGN(const CustomKernelThunk* thunk,
                       CompileToSingleCustomKernelThunk(kAddF32, &executable));

  LaunchDimensions launch = thunk->launch_dimensions();
  EXPECT_EQ(launch.thread_counts_per_block().x, 1);
  EXPECT_EQ(launch.thread_counts_per_block().y, 1);
  EXPECT_EQ(launch.thread_counts_per_block().z, 1);
  EXPECT_GE(launch.block_counts().x, 1);
  EXPECT_EQ(thunk->shmem_bytes(), 0);
}

// The kernel takes one device pointer per buffer: the two inputs, then the
// output.
TEST_F(TensorIrThunkTest, PassesInputsThenOutput) {
  std::unique_ptr<Executable> executable;
  ASSERT_OK_AND_ASSIGN(const CustomKernelThunk* thunk,
                       CompileToSingleCustomKernelThunk(kAddF32, &executable));

  EXPECT_EQ(thunk->arguments().size(), 3);
  EXPECT_THAT(thunk->written(), ElementsAre(false, false, true));
}

// A fusion with several outputs gets one pointer per output after the inputs.
TEST_F(TensorIrThunkTest, PassesInputsThenAllOutputs) {
  std::unique_ptr<Executable> executable;
  ASSERT_OK_AND_ASSIGN(
      const CustomKernelThunk* thunk,
      CompileToSingleCustomKernelThunk(kTwoOutputsF32, &executable));

  EXPECT_EQ(thunk->arguments().size(), 4);
  EXPECT_THAT(thunk->written(), ElementsAre(false, false, true, true));
}

// The reason for using `CustomKernelThunk` rather than a bespoke thunk: the
// kernel, its device code and its launch dimensions all round-trip through a
// proto, which ahead-of-time compilation requires. A bespoke thunk would have
// to reimplement all of this.
TEST_F(TensorIrThunkTest, RoundTripsThroughAProto) {
  std::unique_ptr<Executable> executable;
  ASSERT_OK_AND_ASSIGN(const CustomKernelThunk* thunk,
                       CompileToSingleCustomKernelThunk(kAddF32, &executable));

  ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk->ToProto());
  ASSERT_TRUE(proto.has_custom_kernel_thunk());

  auto* gpu_executable = absl::down_cast<GpuExecutable*>(executable.get());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CustomKernelThunk> restored,
                       CustomKernelThunk::FromProto(
                           thunk->thunk_info(), proto.custom_kernel_thunk(),
                           gpu_executable->allocations(),
                           /*devices_in_process=*/1));

  EXPECT_EQ(restored->custom_kernel_name(), thunk->custom_kernel_name());
  EXPECT_EQ(restored->launch_dimensions(), thunk->launch_dimensions());
  EXPECT_EQ(restored->shmem_bytes(), thunk->shmem_bytes());
  EXPECT_EQ(restored->arguments().size(), thunk->arguments().size());
  EXPECT_EQ(restored->written(), thunk->written());
}

// Using `CustomKernelThunk` also makes the kernel eligible for command
// buffers. The bespoke thunk it replaced was not: the conversion pass only
// knows the thunk kinds it enumerates.
class TensorIrCommandBufferTest : public TensorIrThunkTest {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = TensorIrThunkTest::GetDebugOptionsForTest();
    debug_options.clear_xla_gpu_enable_command_buffer();
    debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
    // Otherwise a single-kernel graph is below the size threshold.
    debug_options.set_xla_gpu_graph_min_graph_size(1);
    return debug_options;
  }
};

TEST_F(TensorIrCommandBufferTest, RecordsIntoACommandBuffer) {
  std::unique_ptr<Executable> executable;
  ASSERT_OK_AND_ASSIGN(const Thunk* thunk,
                       CompileToSingleThunk(kAddF32, &executable));
  EXPECT_EQ(thunk->kind(), Thunk::Kind::kCommandBuffer);
}

}  // namespace
}  // namespace xla::gpu
