/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Miscellaneous tests with the PRED type that don't fit anywhere else.
#include <memory>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class PredTest : public ClientLibraryTestBase {
 protected:
  void TestCompare(bool lhs, bool rhs, bool expected,
                   std::function<XlaOp(const xla::XlaOp&, const xla::XlaOp&,
                                       absl::Span<const int64_t>)>
                       op) {
    XlaBuilder builder(TestName());
    XlaOp lhs_op = ConstantR0<bool>(&builder, lhs);
    XlaOp rhs_op = ConstantR0<bool>(&builder, rhs);
    op(lhs_op, rhs_op, {});
    ComputeAndCompareR0<bool>(&builder, expected, {});
  }
};

TEST_F(PredTest, ConstantR0PredTrue) {
  XlaBuilder builder(TestName());
  ConstantR0<bool>(&builder, true);
  ComputeAndCompareR0<bool>(&builder, true, {});
}

TEST_F(PredTest, ConstantR0PredFalse) {
  XlaBuilder builder(TestName());
  ConstantR0<bool>(&builder, false);
  ComputeAndCompareR0<bool>(&builder, false, {});
}

TEST_F(PredTest, ConstantR0PredCompareEq) {
  TestCompare(true, false, false, &Eq);
}

TEST_F(PredTest, ConstantR0PredCompareNe) {
  TestCompare(true, false, true, &Ne);
}

TEST_F(PredTest, ConstantR0PredCompareLe) {
  TestCompare(true, false, false, &Le);
}

TEST_F(PredTest, ConstantR0PredCompareLt) {
  TestCompare(true, false, false, &Lt);
}

TEST_F(PredTest, ConstantR0PredCompareGe) {
  TestCompare(true, false, true, &Ge);
}

TEST_F(PredTest, ConstantR0PredCompareGt) {
  TestCompare(true, false, true, &Gt);
}

TEST_F(PredTest, ConstantR1Pred) {
  XlaBuilder builder(TestName());
  ConstantR1<bool>(&builder, {true, false, false, true});
  ComputeAndCompareR1<bool>(&builder, {true, false, false, true}, {});
}

TEST_F(PredTest, ConstantR2Pred) {
  XlaBuilder builder(TestName());
  ConstantR2<bool>(&builder, {{false, true, true}, {true, false, false}});
  const std::string expected = R"(pred[2,3] {
  { 0, 1, 1 },
  { 1, 0, 0 }
})";
  EXPECT_EQ(expected, ExecuteToString(&builder, {}));
}

TEST_F(PredTest, AnyR1True) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<bool>(&builder, {true, false});
  Any(a);
  ComputeAndCompareR0<bool>(&builder, true, {});
}

TEST_F(PredTest, AnyR1False) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<bool>(&builder, {false, false});
  Any(a);
  ComputeAndCompareR0<bool>(&builder, false, {});
}

TEST_F(PredTest, AnyR1VacuouslyFalse) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<bool>(&builder, {});
  Any(a);
  ComputeAndCompareR0<bool>(&builder, false, {});
}

TEST_F(PredTest, AnyR2True) {
  XlaBuilder builder(TestName());
  auto a = ConstantR2<bool>(&builder, {
                                          {false, false, false},
                                          {false, false, false},
                                          {false, false, true},
                                      });
  Any(a);
  ComputeAndCompareR0<bool>(&builder, true, {});
}

TEST_F(PredTest, AnyR2False) {
  XlaBuilder builder(TestName());
  auto a = ConstantR2<bool>(&builder, {
                                          {false, false, false},
                                          {false, false, false},
                                          {false, false, false},
                                      });
  Any(a);
  ComputeAndCompareR0<bool>(&builder, false, {});
}

}  // namespace
}  // namespace xla
