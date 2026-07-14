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
#include "xla/backends/gpu/codegen/cubin_custom_kernel_compiler.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/backends/gpu/codegen/triton/triton_kernel_source.h"
#include "xla/backends/gpu/codegen/triton/xtile_compiler.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/runtime/object_pool.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/service/gpu/target_constants.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

TEST(CubinCustomKernelCompilerTest, CallbackInvoked) {
  int compiler_invoked = 0;
  auto llvm_compiler =
      [&](llvm::Module& llvm_module, const se::DeviceDescription& descr,
          const DebugOptions& opts) -> absl::StatusOr<std::vector<uint8_t>> {
    compiler_invoked++;
    return std::vector<uint8_t>{1};
  };

  DebugOptions debug_options;
  CubinCustomKernelCompiler kernel_compiler(
      llvm_compiler, TestGpuDeviceInfo::H100SXMDeviceInfo(), debug_options);

  int hook_invoked = 0;
  kernel_compiler.SetPreOptimizationHook(
      [&](const llvm::Module&) { hook_invoked++; });

  constexpr int kIterations = 3;
  for (int i = 0; i < kIterations; i++) {
    auto llvm_context = std::make_unique<llvm::LLVMContext>();
    auto llvm_module = std::make_unique<llvm::Module>("Test", *llvm_context);
    LlvmKernelSource kernel_source(std::move(llvm_context),
                                   std::move(llvm_module));
    emitters::KernelArguments kernel_arguments({});

    Thunk::ThunkInfo thunk_info;
    LaunchDimensions dimensions;
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Thunk> thunk,
        kernel_compiler
            .Compile(thunk_info, std::move(kernel_source),
                     absl::StrCat("kernel", i), kernel_arguments, dimensions)
            .Await());
  }

  EXPECT_EQ(kIterations, compiler_invoked);
  EXPECT_EQ(kIterations, hook_invoked);
}

TEST_F(HloHardwareIndependentTestBase, TritonCompile) {
  ObjectPool<std::unique_ptr<mlir::MLIRContext>> mlir_context_pool(
      []() { return CreateMlirContext(); });
  TF_ASSERT_OK_AND_ASSIGN(BorrowedMlirContext borrowed_context,
                          mlir_context_pool.GetOrCreate());
  LoadMlirDialectsForTriton(**borrowed_context);

  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          ParseMlirModuleString(R"(
module {
  xtile.entry_func @random_name(%arg0: memref<125x127xf32>, %arg1: memref<125x127xf32>, %arg2: index) attributes {num_opaque_args = 0 : i32} {
    %c0 = arith.constant 0 : index
    %0 = xla.apply_indexing #xla.indexing_map<"(pid_0) -> (pid_0 * 4), domain: pid_0 in [0, 31]">(%arg2)
    %1 = xtile.extract %arg0[%0, %c0] [4, 128] [1, 1] : memref<125x127xf32> -> tensor<4x128xf32>
    %2 = math.absf %1 : tensor<4x128xf32>
    xtile.insert %2 into %arg1[%0, %c0] [4, 128] [1, 1] : tensor<4x128xf32> -> memref<125x127xf32>
    xtile.return
  }
})",
                                                **borrowed_context));
  TritonKernelSource triton_source(std::move(module));

  auto llvm_compiler =
      [&](llvm::Module& llvm_module, const se::DeviceDescription& descr,
          const DebugOptions& opts) -> absl::StatusOr<std::vector<uint8_t>> {
    return std::vector<uint8_t>{1};
  };
  DebugOptions debug_options;
  CubinCustomKernelCompiler kernel_compiler(
      llvm_compiler, TestGpuDeviceInfo::H100SXMDeviceInfo(), debug_options);

  llvm::Triple triple(nvptx::TargetTriple());
  std::string data_layout = nvptx::DataLayout();

  BlockLevelParameters block_level_parameters =
      BlockLevelParameters::FromBlockLevelFusionConfig(
          tsl::proto_testing::ParseTextProtoOrDie<BlockLevelFusionConfig>(R"pb(
            output_tiles:
            [ { sizes: [ 4, 127 ] }],
            num_warps: 4
            num_ctas: 1
            num_stages: 1
          )pb"));
  TF_ASSERT_OK_AND_ASSIGN(
      TritonWrapperResult result,
      kernel_compiler
          .CompileTritonToLlvm("random_name", HloModule{"test_module", {}},
                               TestGpuDeviceInfo::H100SXMDeviceInfo(),
                               block_level_parameters, triple, data_layout,
                               std::move(triton_source),
                               std::move(borrowed_context), true)
          .Await());
  EXPECT_THAT(RunFileCheck(result.kernel_source.ToString(),
                           "// CHECK: call float @llvm.fabs.f32(float "),
              absl_testing::IsOkAndHolds(true));
}

}  // namespace
}  // namespace xla::gpu
