/* Copyright 2025 The OpenXLA Authors.

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
#include "absl/status/status.h"
#include "xla/debug_options_flags.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

absl::Status HostIOCallback(ffi::Token, ffi::Result<ffi::Token>,
                            ffi::Result<ffi::AnyBuffer>) {
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(
    kIOCallback, HostIOCallback,
    ffi::Ffi::Bind().Arg<ffi::Token>().Ret<ffi::Token>().Ret<ffi::AnyBuffer>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$io_callback", "Host",
                         kIOCallback);

class CpuFFITest : public HloPjRtTestBase {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GetDebugOptionsFromFlags();
    return debug_options;
  }
};

TEST_F(CpuFFITest, EmulateImpureCallbackWithTokens) {
  std::unique_ptr<HloModule> module = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  HloInstruction* p0 = builder.AddInstruction(HloInstruction::CreateToken());
  auto instr = Cast<HloCustomCallInstruction>(
      builder.AddInstruction(HloInstruction::CreateCustomCall(
          ShapeUtil::MakeTupleShape(
              {ShapeUtil::MakeTokenShape(), ShapeUtil::MakeShape(S32, {})}),
          {p0}, "__xla_test$$io_callback", "",
          CustomCallApiVersion::API_VERSION_TYPED_FFI)));

  instr->set_custom_call_has_side_effect(true);
  module->AddEntryComputation(builder.Build());

  EXPECT_OK(Execute(std::move(module), {}));
}

}  // namespace
}  // namespace xla
