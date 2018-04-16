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

#include "tensorflow/compiler/xla/tests/test_utils.h"

#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/local_client_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

// A test fixture is used because we need a client for our computation builder.
class TestUtilsTest : public LocalClientTestBase {};

XLA_TEST_F(TestUtilsTest, UnusedParam) {
  ComputationBuilder builder(local_client_, TestName());
  // Make the reduction lambda.
  Shape single_float = ShapeUtil::MakeShape(F32, {});
  builder.Parameter(0, single_float, "unused");
  builder.Parameter(1, single_float, "used");
  auto computation_status = builder.Build();
  TF_ASSERT_OK(computation_status.status());

  // Make the reduction.
  Shape pair_float = ShapeUtil::MakeShape(F32, {2});
  builder.Reduce(builder.Parameter(0, pair_float, "operand"),
                 builder.Parameter(1, single_float, "init"),
                 computation_status.ValueOrDie(), {0});
  computation_status = builder.Build();
  TF_ASSERT_OK(computation_status.status());

  auto executable_status = local_client_->Compile(
      computation_status.ValueOrDie(), {&pair_float, &single_float},
      ExecutableBuildOptions());
  TF_ASSERT_OK(executable_status.status());
  HloModule& module = const_cast<HloModule&>(
      executable_status.ValueOrDie()->executable()->module());
  TF_ASSERT_OK(MakeFakeArguments(&module).status());
}

}  // namespace
}  // namespace xla
