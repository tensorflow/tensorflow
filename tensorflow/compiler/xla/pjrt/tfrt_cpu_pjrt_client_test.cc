/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/compiler/xla/service/custom_call_status.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"

namespace xla {
namespace {

void TestError(void* out, const void** in, XlaCustomCallStatus* status) {
  static constexpr char kError[] = "test error.";
  XlaCustomCallStatusSetFailure(status, kError, sizeof(kError));
}
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(TestError);

TEST(TfrtCpuClientTest, DonationWithExecutionError) {
  constexpr char kProgram[] =
      R"(HloModule DonationWithExecutionError, input_output_alias={ {}: (0, {}, must-alias) }
ENTRY DonationWithExecutionError() -> f32[2, 2] {
    %input = f32[2, 2] parameter(0)
    %custom-call = (f32[2, 2], u8[0]) custom-call(%input), custom_call_target="TestError", api_version=API_VERSION_STATUS_RETURNING, output_to_operand_aliasing={{0}: (0, {})}
    ROOT %result = f32[2, 2] get-tuple-element(%custom-call), index=0
})";

  TF_ASSERT_OK_AND_ASSIGN(auto client, GetTfrtCpuClient(/*asynchronous=*/true));

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(kProgram, {}));
  XlaComputation xla_computation(hlo_module->ToProto());
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_executable,
                          client->Compile(xla_computation, {}));

  std::vector<float> data(4, 0);
  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          client->addressable_devices()[0]));

  auto result = pjrt_executable->Execute(/*argument_handles=*/{{buffer.get()}},
                                         /*options=*/{});
  ASSERT_FALSE(result.ok());
  EXPECT_THAT(result.status().error_message(),
              ::testing::HasSubstr("test error."));

  result = pjrt_executable->Execute(/*argument_handles=*/{{buffer.get()}},
                                    /*options=*/{});
  ASSERT_FALSE(result.ok());
  EXPECT_THAT(result.status().error_message(),
              ::testing::HasSubstr("Donation requested for invalid buffer"));
}

}  // namespace
}  // namespace xla
