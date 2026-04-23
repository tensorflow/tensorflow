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
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::HasSubstr;

MATCHER_P2(StatusHasPayload, key, value_matcher,
           "Matches payloads for absl::Status") {
  auto payload = arg.GetPayload(key);
  if (!payload.has_value()) {
    *result_listener << "status has no payload with key " << key;
    return false;
  }
  return ExplainMatchResult(value_matcher, std::string(payload.value()),
                            result_listener);
}

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> CompileExecutable(
    absl::string_view program, xla::PjRtClient& client,
    xla::CompileOptions compile_options = xla::CompileOptions()) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      ParseAndReturnUnverifiedModule(program, {}));
  xla::XlaComputation xla_computation(hlo_module->ToProto());
  return client.CompileAndLoad(xla_computation, compile_options);
}

class PjRtGpuClientStreamErrorTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    bool const use_tfrt = GetParam();
    GpuClientOptions options;
    options.use_tfrt_gpu_client = use_tfrt;
    ASSERT_OK_AND_ASSIGN(client_, GetXlaPjrtGpuClient(options));
  }

  std::unique_ptr<PjRtClient> client_;
};

INSTANTIATE_TEST_SUITE_P(PjRtGpuClientStreamErrorTest,
                         PjRtGpuClientStreamErrorTest, testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "TfrtGpuClient"
                                             : "PjRtStreamExecutorClient";
                         });

struct KernelHolder {
  std::unique_ptr<se::Kernel> kernel;
};

absl::Status IllegalAccess(se::Stream* stream, KernelHolder* holder) {
  static constexpr absl::string_view kPtx = R"(
    .version 4.2
    .target sm_50
    .address_size 64
    .visible .entry IllegalAccess() {
      .reg .u64 %addr;
      mov.u64 %addr, 0x0;
      st.u64 [%addr], %addr;
      ret;
    }
  )";
  if (holder->kernel == nullptr) {
    TF_ASSIGN_OR_RETURN(
        holder->kernel,
        gpu::CreateKernel("IllegalAccess", 0, kPtx, stream->parent(), 0));
  }
  return gpu::ExecuteKernelOnStream(*holder->kernel, {},
                                    xla::gpu::LaunchDimensions(1, 1),
                                    std::nullopt, stream);
}

XLA_FFI_DEFINE_HANDLER(
    kIllegalAccess, IllegalAccess,
    ffi::Ffi::Bind().Ctx<ffi::Stream>().Ctx<ffi::UserData<KernelHolder>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "IllegalAccess", "CUDA",
                         kIllegalAccess);

TEST_P(PjRtGpuClientStreamErrorTest, AbortsOnStreamError) {
  static constexpr absl::string_view kIllegalProgram = R"(
    HloModule illegal_access
    ENTRY main {
      ROOT %custom-call = f32[4] custom-call(),
                          custom_call_target="IllegalAccess",
                          api_version=API_VERSION_TYPED_FFI
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtLoadedExecutable> executable,
                          CompileExecutable(kIllegalProgram, *client_));
  ExecuteContext context;
  TF_ASSERT_OK(context.ffi_context().Emplace<KernelHolder>());
  ExecuteOptions opts;
  opts.context = &context;
  using PjrtExecuteResult = std::vector<std::unique_ptr<PjRtBuffer>>;
  using PjrtExecuteAllResults = std::vector<PjrtExecuteResult>;
  absl::StatusOr<PjrtExecuteAllResults> async_result =
      executable->Execute(/*argument_handles=*/{{}}, opts);
  ASSERT_OK(async_result.status());
  ASSERT_EQ(async_result->size(), 1);
  const PjrtExecuteResult& result_buffers = (*async_result)[0];
  ASSERT_EQ(result_buffers.size(), 1);
  auto result = result_buffers[0]->ToLiteral().Await();
  // Execution should both exit with an error and not hang.
  EXPECT_THAT(
      result.status(),
      AllOf(StatusIs(absl::StatusCode::kInternal,
                     HasSubstr("CUDA_ERROR_ILLEGAL_ADDRESS")),
            StatusHasPayload("executable_name", HasSubstr("illegal_access"))));
}

TEST_P(PjRtGpuClientStreamErrorTest, OOMIncludesContext) {
  static constexpr absl::string_view kHugeProgram = R"(
    HloModule huge_alloc
    ENTRY main {
      ROOT %iota = f32[100000000000] iota(), iota_dimension=0
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtLoadedExecutable> executable,
                          CompileExecutable(kHugeProgram, *client_));
  ExecuteOptions opts;
  auto async_result = executable->Execute(/*argument_handles=*/{{}}, opts);
  ASSERT_OK(async_result.status());
  ASSERT_EQ(async_result->size(), 1);
  const auto& result_buffers = (*async_result)[0];
  ASSERT_EQ(result_buffers.size(), 1);
  auto result = result_buffers[0]->GetReadyFuture().Await();
  EXPECT_THAT(result, AllOf(StatusIs(absl::StatusCode::kResourceExhausted),
                            StatusHasPayload("executable_name",
                                             HasSubstr("huge_alloc"))));
}

}  // namespace
}  // namespace xla
