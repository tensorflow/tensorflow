/* Copyright 2019 The OpenXLA Authors.

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

#include <vector>

#include "xla/error_spec.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/service/service.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/test_macros.h"

namespace xla {
namespace {

class FmaxSimpleTest : public ClientLibraryTestBase {};

XLA_TEST_F(FmaxSimpleTest, FmaxTenValues) {
  SetFastMathDisabled(true);
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(
      &builder, {-0.0, 1.0, 2.0, -3.0, -4.0, 5.0, 6.0, -7.0, -8.0, 9.0});
  auto y = ConstantR1<float>(
      &builder, {-0.0, -1.0, -2.0, 3.0, 4.0, -5.0, -6.0, 7.0, 8.0, -9.0});
  Max(x, y);

  std::vector<float> expected = {-0.0, 1.0, 2.0, 3.0, 4.0,
                                 5.0,  6.0, 7.0, 8.0, 9.0};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(FmaxSimpleTest, FmaxEdgeCases) {
  SetFastMathDisabled(true);
  XlaBuilder builder(TestName());
  XlaOp param0, param1;
  std::unique_ptr<GlobalData> param0_data = CreateR1Parameter<float>(
      {INFINITY, INFINITY, INFINITY, -INFINITY, INFINITY, -INFINITY, NAN,
       INFINITY, -INFINITY, NAN},
      /*parameter_number=*/0, /*name=*/"param0",
      /*builder=*/&builder, /*data_handle=*/&param0);
  std::unique_ptr<GlobalData> param1_data = CreateR1Parameter<float>(
      {INFINITY, -INFINITY, NAN, NAN, -4.0, -5.0, -6.0, 7.0, 8.0, 9.0},
      /*parameter_number=*/1, /*name=*/"param1",
      /*builder=*/&builder, /*data_handle=*/&param1);

  Max(param0, param1);
  std::vector<float> expected = {INFINITY, INFINITY, NAN,      NAN, INFINITY,
                                 -5,       NAN,      INFINITY, 8,   NAN};
  ComputeAndCompareR1<float>(&builder, expected,
                             {param0_data.get(), param1_data.get()},
                             ErrorSpec(0.0001));
}

XLA_TEST_F(FmaxSimpleTest, FminEdgeCases) {
  SetFastMathDisabled(true);
  XlaBuilder builder(TestName());
  XlaOp param0, param1;
  std::unique_ptr<GlobalData> param0_data = CreateR1Parameter<float>(
      {INFINITY, INFINITY, INFINITY, -INFINITY, INFINITY, -INFINITY, NAN,
       INFINITY, -INFINITY, NAN},
      /*parameter_number=*/0, /*name=*/"param0",
      /*builder=*/&builder, /*data_handle=*/&param0);
  std::unique_ptr<GlobalData> param1_data = CreateR1Parameter<float>(
      {INFINITY, -INFINITY, NAN, NAN, -4.0, -5.0, -6.0, 7.0, 8.0, 9.0},
      /*parameter_number=*/1, /*name=*/"param1",
      /*builder=*/&builder, /*data_handle=*/&param1);

  Min(param0, param1);
  std::vector<float> expected = {INFINITY,  -INFINITY, NAN, NAN,       -4,
                                 -INFINITY, NAN,       7,   -INFINITY, NAN};
  ComputeAndCompareR1<float>(&builder, expected,
                             {param0_data.get(), param1_data.get()},
                             ErrorSpec(0.0001));
}

}  // namespace
}  // namespace xla
