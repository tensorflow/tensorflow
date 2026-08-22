/* Copyright 2022 The OpenXLA Authors.

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
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "third_party/py/jax/jaxlib/gpu/triton.pb.h"
#include "third_party/py/jax/jaxlib/gpu/triton_kernels.h"
#include "third_party/py/jax/jaxlib/gpu/triton_utils.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/ffi/ffi.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/compiled_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/restricted/hlo_test_base_legacy.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
using ::testing::IsEmpty;
using ::testing::Not;

class GpuAotCompilationTest : public HloTestBaseLegacy {
 protected:
  void SetUp() override { debug_options_ = GetDebugOptionsForTest(); }

  DebugOptions debug_options_;
};

TEST_F(GpuAotCompilationTest, ExportAndLoadExecutable) {
  const absl::string_view hlo_string = R"hlo(
    HloModule Test

    ENTRY main {
      a = f32[100, 200]parameter(0)
      ROOT b = f32[100, 200] copy(a)
    }
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  auto compiler = backend().compiler();
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName(name));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * stream_exec,
                       platform->ExecutorForDevice(0));

  // Compile AOT.
  AotCompilationOptions aot_options(compiler->PlatformId());
  aot_options.set_executor(stream_exec);

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<CompiledModule>> aot_results,
      compiler->CompileAheadOfTime(std::move(module), aot_options));

  // Serialize-deserialize AOT compilation result.
  ASSERT_OK_AND_ASSIGN(std::string serialized_aot_result,
                       aot_results[0]->SerializeAsString());
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CompiledModule> aot_result,
      compiler->LoadAotCompilationResult(serialized_aot_result));

  // Load Executable from AOT compilation result.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Executable> executable,
      std::move(*aot_result)
          .LoadExecutable(compiler->PlatformId(),
                          stream_exec->GetDeviceDescription(), debug_options_));

  auto* gpu_executable = dynamic_cast<GpuExecutable*>(executable.get());
  ASSERT_NE(gpu_executable, nullptr);
  EXPECT_THAT(gpu_executable->buffer_allocations_debug_summary(),
              Not(IsEmpty()));
}

TEST_F(GpuAotCompilationTest, AotCompilationWithoutGpuDevice) {
  const absl::string_view hlo_string = R"hlo(
    HloModule Test

    ENTRY main {
      a = f32[100, 200] parameter(0)
      ROOT b = f32[100, 200] copy(a)
    }
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  auto compiler = backend().compiler();
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName(name));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * stream_exec,
                       platform->ExecutorForDevice(0));

  // Stream executor is not passed as an option.
  Compiler::GpuTargetConfig gpu_target_config(stream_exec);
  AotCompilationOptions aot_options(compiler->PlatformId());
  aot_options.set_gpu_topology(
      GetSingleDeviceGpuTopology("", gpu_target_config));

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<CompiledModule>> aot_results,
      compiler->CompileAheadOfTime(std::move(module), aot_options));

  // Serialize-deserialize AOT compilation result.
  ASSERT_OK_AND_ASSIGN(std::string serialized_aot_result,
                       aot_results[0]->SerializeAsString());
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CompiledModule> aot_result,
      compiler->LoadAotCompilationResult(serialized_aot_result));

  // Load Executable from AOT compilation result.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Executable> executable,
      std::move(*aot_result)
          .LoadExecutable(compiler->PlatformId(),
                          stream_exec->GetDeviceDescription(), debug_options_));
}

namespace {

using ::mlir::ArrayRef;
using ::mlir::NamedAttribute;

std::string CreateTritonCustomCallBackendConfig() {
  mlir::MLIRContext context_;
  mlir::Builder builder(&context_);

  // Create the backend_config for the triton custom call.
  const std::string kMLIRText = R"mlir(
    module {
      tt.func public @add_one(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}, %arg3: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}) {
        %0 = tt.get_program_id x : i32
        %1 = tt.load %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
        %2 = tt.load %arg1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
        %cst = arith.constant 1.000000e+00 : f32
        %3 = arith.addf %1, %cst : f32
        tt.store %arg2, %3 {cache = 1 : i32, evict = 1 : i32} : !tt.ptr<f32>
        tt.store %arg3, %2 {cache = 1 : i32, evict = 1 : i32} : !tt.ptr<f32>
        tt.return
      }
    }
  )mlir";

  NamedAttribute name =
      builder.getNamedAttr("name", builder.getStringAttr("add_one"));
  NamedAttribute ir =
      builder.getNamedAttr("ir", builder.getStringAttr(kMLIRText));
  NamedAttribute num_stages =
      builder.getNamedAttr("num_stages", builder.getI32IntegerAttr(3));
  NamedAttribute num_warps =
      builder.getNamedAttr("num_warps", builder.getI32IntegerAttr(4));
  NamedAttribute grid_x =
      builder.getNamedAttr("grid_x", builder.getI32IntegerAttr(1));
  NamedAttribute grid_y =
      builder.getNamedAttr("grid_y", builder.getI32IntegerAttr(1));
  NamedAttribute grid_z =
      builder.getNamedAttr("grid_z", builder.getI32IntegerAttr(1));
  NamedAttribute debug =
      builder.getNamedAttr("debug", builder.getBoolAttr(false));

  std::vector<NamedAttribute> attributes = {
      name, ir, num_stages, num_warps, grid_x, grid_y, grid_z, debug};
  ArrayRef<NamedAttribute> attributesRef(attributes);
  mlir::DictionaryAttr backend_config =
      mlir::DictionaryAttr::get(&context_, attributesRef);

  // Parse the backend_config into a string.
  std::string backend_config_str;
  llvm::raw_string_ostream(backend_config_str) << backend_config;

  return backend_config_str;
}

}  // namespace

TEST_F(GpuAotCompilationTest, ExportAndLoadExecutableWithTriton) {
  auto triton_support =
      EnsureTritonSupportsComputeCapability(backend()
                                                .default_stream_executor()
                                                ->GetDeviceDescription()
                                                .gpu_compute_capability());
  if (!triton_support.ok()) {
    GTEST_SKIP() << triton_support;
  }

  const absl::string_view hlo_string_template = R"hlo(
    HloModule Test

    ENTRY main {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT c = (f32[],f32[]) custom-call(a, b), custom_call_target="__gpu$xla.gpu.triton", backend_config="%s"
    }
    )hlo";

  std::string hlo_string =
      absl::StrFormat(hlo_string_template,
                      absl::CEscape(CreateTritonCustomCallBackendConfig()));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  auto compiler = backend().compiler();
  auto platform_name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName(platform_name));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * stream_exec,
                       platform->ExecutorForDevice(0));

  // Compile AOT.
  AotCompilationOptions aot_options(compiler->PlatformId());
  aot_options.set_executor(stream_exec);

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<CompiledModule>> aot_results,
      compiler->CompileAheadOfTime(std::move(module), aot_options));

  // Serialize-deserialize AOT compilation result.
  ASSERT_OK_AND_ASSIGN(std::string serialized_aot_result,
                       aot_results[0]->SerializeAsString());
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CompiledModule> aot_result,
      compiler->LoadAotCompilationResult(serialized_aot_result));

  // Load Executable from AOT compilation result.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Executable> executable,
      std::move(*aot_result)
          .LoadExecutable(compiler->PlatformId(),
                          stream_exec->GetDeviceDescription(), debug_options_));
  std::unique_ptr<OpaqueExecutable> wrapped_executable =
      test_runner_as_hlo_runner().WrapExecutable(std::move(executable));

  const xla::Literal literal_1 = xla::LiteralUtil::CreateR0<float>(1.0f);
  const xla::Literal literal_2 = xla::LiteralUtil::CreateR0<float>(2.0f);
  const xla::Literal literal_3 = xla::LiteralUtil::CreateR0<float>(3.0f);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       test_runner_as_hlo_runner().ExecuteWithExecutable(
                           wrapped_executable.get(), {&literal_1, &literal_3}));

  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::MakeTuple({&literal_2, &literal_3}), result));
}

namespace {

// Escapes non-printable binary bytes into 2-digit uppercase hex (\XX) and
// quotes/slashes as required by the MLIR string attribute lexer rules.
std::string MlirEscapeString(absl::string_view s) {
  std::string result;
  result.reserve(s.size() * 3);
  for (unsigned char c : s) {
    if (c == '\\') {
      result += "\\\\";
    } else if (c == '"') {
      result += "\\\"";
    } else if (c >= 0x20 && c <= 0x7e) {
      result += c;
    } else {
      absl::StrAppendFormat(&result, "\\%02X", c);
    }
  }
  return result;
}

std::string CreateTritonKernelPtxBackendConfig(
    se::StreamExecutor* stream_exec) {
  int cc_major = 8, cc_minor = 0;
  auto compute_capability =
      stream_exec->GetDeviceDescription().gpu_compute_capability();
  if (auto* cuda_cc = compute_capability.cuda_compute_capability()) {
    cc_major = cuda_cc->major;
    cc_minor = cuda_cc->minor;
  }
  int cc_int = cc_major * 10 + cc_minor;

  jax_triton::TritonAnyKernelCall any_call_proto;
  auto* kernel_call = any_call_proto.mutable_kernel_call();
  auto* kernel = kernel_call->mutable_kernel();
  kernel->set_kernel_name("add_kernel");
  kernel->set_num_warps(4);
  kernel->set_shared_mem_bytes(0);
  kernel->set_ptx(absl::StrFormat(R"(
    .version 7.0
    .target sm_%d
    .address_size 64

    .visible .entry add_kernel(
        .param .u64 p0,
        .param .u64 p1,
        .param .u64 p2
    ) {
        .reg .b64 %%rd<8>;
        .reg .f32 %%f<4>;
        .reg .b32 %%r<2>;

        ld.param.u64 %%rd1, [p0];
        ld.param.u64 %%rd2, [p1];
        ld.param.u64 %%rd3, [p2];

        mov.u32 %%r1, %%ctaid.x;
        mul.wide.u32 %%rd4, %%r1, 4;

        add.s64 %%rd5, %%rd1, %%rd4;
        add.s64 %%rd6, %%rd2, %%rd4;
        add.s64 %%rd7, %%rd3, %%rd4;

        ld.global.f32 %%f1, [%%rd5];
        ld.global.f32 %%f2, [%%rd6];
        add.f32 %%f3, %%f1, %%f2;
        st.global.f32 [%%rd7], %%f3;

        ret;
}
)",
                                  cc_int));
  kernel->set_compute_capability(cc_int);

  kernel_call->set_grid_0(1024);
  kernel_call->set_grid_1(1);
  kernel_call->set_grid_2(1);

  // Parameter 0: array (p0)
  auto* p0 = kernel_call->add_parameters();
  p0->mutable_array()->set_bytes_to_zero(0);
  p0->mutable_array()->set_ptr_divisibility(16);

  // Parameter 1: array (p1)
  auto* p1 = kernel_call->add_parameters();
  p1->mutable_array()->set_bytes_to_zero(0);
  p1->mutable_array()->set_ptr_divisibility(16);

  // Parameter 2: array (result)
  auto* p2 = kernel_call->add_parameters();
  p2->mutable_array()->set_bytes_to_zero(0);
  p2->mutable_array()->set_ptr_divisibility(16);

  std::string compressed_opaque =
      jax::JAX_GPU_NAMESPACE::ZlibCompress(any_call_proto.SerializeAsString())
          .value();
  return absl::StrFormat(R"({opaque = "%s"})",
                         MlirEscapeString(compressed_opaque));
}
}  // namespace

TEST_F(GpuAotCompilationTest, TritonKernelCallFfiAot) {
  XLA_FFI_Handler_Bundle bundle = {
      /*instantiate=*/jax::JAX_GPU_NAMESPACE::kTritonKernelCallFfiInstantiate,
      /*prepare=*/nullptr,
      /*initialize=*/jax::JAX_GPU_NAMESPACE::kTritonKernelCallFfiInitialize,
      /*execute=*/jax::JAX_GPU_NAMESPACE::kTritonKernelCallFfi,
  };
  if (auto* error = xla::ffi::Ffi::RegisterStaticHandler(
          xla::ffi::GetXlaFfiApi(), "triton_kernel_call_ffi", "CUDA", bundle)) {
    FAIL() << "RegisterStaticHandler failed";
  }

  auto compiler = backend().compiler();
  auto platform_name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName(platform_name));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * stream_exec,
                       platform->ExecutorForDevice(0));

  std::string backend_config_str =
      CreateTritonKernelPtxBackendConfig(stream_exec);

  std::string hlo_string = absl::StrFormat(R"hlo(
    HloModule test

    ENTRY main {
      p0 = f32[1024]{0} parameter(0)
      p1 = f32[1024]{0} parameter(1)
      ROOT res = f32[1024]{0} custom-call(p0, p1),
        custom_call_target="triton_kernel_call_ffi",
        api_version=API_VERSION_TYPED_FFI,
        backend_config="%s"
    }
)hlo",
                                           absl::CEscape(backend_config_str));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  AotCompilationOptions aot_options(compiler->PlatformId());
  aot_options.set_executor(stream_exec);

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<CompiledModule>> aot_results,
      compiler->CompileAheadOfTime(std::move(module), aot_options));

  ASSERT_OK_AND_ASSIGN(std::string serialized_aot_result,
                       aot_results[0]->SerializeAsString());
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CompiledModule> aot_result,
      compiler->LoadAotCompilationResult(serialized_aot_result));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Executable> executable,
      std::move(*aot_result)
          .LoadExecutable(compiler->PlatformId(),
                          stream_exec->GetDeviceDescription(), debug_options_));
  std::unique_ptr<OpaqueExecutable> wrapped_executable =
      test_runner_as_hlo_runner().WrapExecutable(std::move(executable));

  std::vector<float> data_a(1024, 1.0f);
  std::vector<float> data_b(1024, 2.0f);
  const xla::Literal literal_a = xla::LiteralUtil::CreateR1<float>(data_a);
  const xla::Literal literal_b = xla::LiteralUtil::CreateR1<float>(data_b);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       test_runner_as_hlo_runner().ExecuteWithExecutable(
                           wrapped_executable.get(), {&literal_a, &literal_b}));

  std::vector<float> expected(1024, 3.0f);
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR1<float>(expected), result));
}

}  // namespace gpu
}  // namespace xla
