/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

using ArithmeticTest = ClientLibraryTestBase;

XLA_TEST_F(ArithmeticTest, ArgMinR2Axis0) {
  XlaBuilder builder(TestName());
  auto x = ConstantR2<int32>(&builder, {{1, 7, 4}, {6, 3, 5}, {8, 3, 3}});
  ArgMin(x, S32, /*axis=*/0);

  std::vector<int32> expected = {0, 2, 2};
  ComputeAndCompareR1<int32>(&builder, expected, {});
}

XLA_TEST_F(ArithmeticTest, ArgMinR2Axis1) {
  XlaBuilder builder(TestName());
  auto x = ConstantR2<int32>(&builder, {{1, 7, 4}, {6, 3, 5}, {8, 3, 3}});
  ArgMin(x, S32, /*axis=*/1);

  std::vector<int32> expected = {0, 1, 2};
  ComputeAndCompareR1<int32>(&builder, expected, {});
}

XLA_TEST_F(ArithmeticTest, ArgMaxR2Axis0) {
  XlaBuilder builder(TestName());
  auto x = ConstantR2<int32>(&builder, {{1, 7, 4}, {6, 3, 5}, {8, 3, 3}});
  ArgMax(x, S32, /*axis=*/0);

  std::vector<int32> expected = {2, 0, 1};
  ComputeAndCompareR1<int32>(&builder, expected, {});
}

XLA_TEST_F(ArithmeticTest, ArgMaxR2Axis1) {
  XlaBuilder builder(TestName());
  auto x = ConstantR2<int32>(&builder, {{1, 7, 4}, {6, 3, 5}, {8, 3, 3}});
  ArgMax(x, S32, /*axis=*/1);

  std::vector<int32> expected = {1, 0, 0};
  ComputeAndCompareR1<int32>(&builder, expected, {});
}

}  // namespace
}  // namespace xla
