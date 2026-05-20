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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/ffi.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/execution_context.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/service/platform_util.h"
#include "xla/service/service.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

static const char* PLATFORM = "Host";

enum class BinaryOp : int8_t { kAdd, kMul };
enum class InitMethod : int { kZero, kOne };

XLA_FFI_REGISTER_ENUM_ATTR_DECODING(BinaryOp);
XLA_FFI_REGISTER_ENUM_ATTR_DECODING(InitMethod);

namespace xla {
namespace {

class CustomCallClientAPITest : public ::testing::Test {};

//===----------------------------------------------------------------------===//
// XLA:FFI handler with execution context
//===----------------------------------------------------------------------===//

// Arbitrary user-defined context passed via the execution context side channel
// to a custom call handlers.
struct SomeExtraContext {
  explicit SomeExtraContext(int32_t value) : value(value) {}
  int32_t value;
  bool executed = false;
};

template <ffi::ExecutionStage stage>
static absl::Status ExecutionContext(ffi::Result<ffi::AnyBuffer>,
                                     SomeExtraContext* ctx) {
  if (ctx->value != 42) {
    return absl::InternalError("Unexpected value");
  }
  if constexpr (stage == ffi::ExecutionStage::kExecute) {
    ctx->executed = true;
  }

  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kExecutionContextExecute,
                       ExecutionContext<ffi::ExecutionStage::kExecute>,
                       ffi::Ffi::BindExecute()
                           .Ret<ffi::AnyBuffer>()
                           .Ctx<ffi::UserData<SomeExtraContext>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla.cpu.ffi_execution_context",
                         PLATFORM,
                         {
                             /*instantiate=*/nullptr,
                             /*prepare=*/nullptr,
                             /*initialize=*/nullptr,
                             /*execute=*/kExecutionContextExecute,
                         });

absl::StatusOr<LocalClient*> CreateClient() {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform(PLATFORM));
  LocalClientOptions client_options(platform, 1, 1, std::nullopt);
  return ClientLibrary::GetOrCreateLocalClient(client_options);
}

TEST_F(CustomCallClientAPITest, FfiExecutionContext) {
  XlaBuilder b("FfiExecutionContext");
  const Shape shape = ShapeUtil::MakeShape(F32, {});
  CustomCall(&b, "xla.cpu.ffi_execution_context", /*operands=*/{}, shape,
             /*opaque=*/"",
             /*has_side_effect=*/false,
             /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
             /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
             /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);

  TF_ASSERT_OK_AND_ASSIGN(auto local_client, CreateClient());
  EXPECT_NE(local_client->device_count(), 0);

  TF_ASSERT_OK_AND_ASSIGN(auto computation, b.Build());
  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      local_client->Compile(computation, /*argument_layouts=*/{},
                            /*options=*/{}));

  ffi::ExecutionContext execution_context;
  TF_ASSERT_OK(execution_context.Emplace<SomeExtraContext>(42));

  ExecutableRunOptions run_options;
  run_options.set_allocator(local_client->backend().memory_allocator());
  run_options.set_ffi_execution_context(&execution_context);

  std::vector<const ShapedBuffer*> args;
  TF_ASSERT_OK_AND_ASSIGN(auto result, executable[0]->Run(args, run_options));
  TF_ASSERT_OK_AND_ASSIGN(auto* user_context,
                          execution_context.Lookup<SomeExtraContext>());
  EXPECT_TRUE(user_context->executed);
}

}  // namespace
}  // namespace xla
