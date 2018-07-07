/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_computation.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class ExecutionProfileTest : public ClientLibraryTestBase {};

XLA_TEST_F(ExecutionProfileTest, ExecuteWithExecutionProfile) {
  Shape shape = ShapeUtil::MakeShape(F32, {256, 256});

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GlobalData> input,
      client_->TransferToServer(
          *LiteralUtil::CreateR2F32Linspace(1e0, 1e5, 256, 256)));

  XlaBuilder b(TestName() + ".add");
  Dot(Parameter(&b, 0, shape, "param_0"), Parameter(&b, 1, shape, "param_1"));
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation dot_product, b.Build());

  ExecutionProfile execution_profile;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GlobalData> data,
      client_->Execute(dot_product, {input.get(), input.get()},
                       &execution_options_, &execution_profile));

  VLOG(3) << "execution_profile.compute_cycle_count() = "
          << execution_profile.compute_cycle_count();
  VLOG(3) << "execution_profile.compute_and_transfer_time_ns() = "
          << execution_profile.compute_and_transfer_time_ns();
  VLOG(3) << "execution_profile.compute_time_ns() = "
          << execution_profile.compute_time_ns();

  bool hlo_profiling_enabled =
      execution_options_.debug_options().xla_hlo_profile();

  // If HLO profiling is enabled we always expect cycle count to be populated.
  // If HLO profiling is disabled then depending on the backend the cycle count
  // may or may not be populated.
  if (hlo_profiling_enabled) {
    EXPECT_GT(execution_profile.compute_cycle_count(), 0);
  }

  EXPECT_GT(execution_profile.compute_and_transfer_time_ns(), 0);
  EXPECT_GT(execution_profile.compute_time_ns(), 0);

  TF_ASSERT_OK_AND_ASSIGN(auto computed, client_->Transfer(*data, &shape));
  (void)computed;
}

}  // namespace
}  // namespace xla
