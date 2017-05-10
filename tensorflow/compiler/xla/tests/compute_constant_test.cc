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

#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class ComputeConstantTest : public ClientLibraryTestBase {
 public:
  StatusOr<std::unique_ptr<Literal>> ComputeConstantLiteral(
      ComputationDataHandle operand, ComputationBuilder* builder,
      Layout* output_layout = nullptr) {
    TF_ASSIGN_OR_RETURN(auto remote_computed,
                        builder->ComputeConstant(operand, output_layout));
    TF_ASSIGN_OR_RETURN(auto computed, client_->Transfer(*remote_computed));
    return std::move(computed);
  }

  template <class Scalar>
  StatusOr<Scalar> ComputeConstantScalar(ComputationDataHandle operand,
                                         ComputationBuilder* builder) {
    TF_ASSIGN_OR_RETURN(auto literal, ComputeConstantLiteral(operand, builder));
    return LiteralUtil::Get<Scalar>(*literal, {});
  }

  bool IsConstant(const ComputationDataHandle& operand,
                  ComputationBuilder* builder) {
    StatusOr<bool> result = builder->IsConstant(operand);
    EXPECT_TRUE(result.ok()) << result.status();
    return result.ok() ? result.ValueOrDie() : false;
  }

  template <class Scalar>
  void ExpectConstantComputedScalar(ComputationDataHandle operand,
                                    Scalar expected,
                                    ComputationBuilder* builder) {
    Scalar computed = ComputeConstantScalar<Scalar>(operand, builder);
    ASSERT_TRUE(computed.ok()) << computed.status();
    std::unique_ptr<Literal> expected_literal = LiteralUtil::CreateR0(expected);
    LiteralTestUtil::ExpectEqual(*expected_literal, *computed);
  }
};

TEST_F(ComputeConstantTest, ScalarInt32Literal) {
  ComputationBuilder b(client_, TestName());
  auto computation = b.ConstantR0<int32>(42);
  EXPECT_TRUE(IsConstant(computation, &b));

  auto value = ComputeConstantScalar<int32>(computation, &b);
  ASSERT_TRUE(value.ok()) << value.status();
  EXPECT_EQ(value.ValueOrDie(), 42);
}

TEST_F(ComputeConstantTest, ScalarFloatAdd) {
  ComputationBuilder b(client_, TestName());
  auto computation =
      b.Add(b.ConstantR0<float>(42.5f), b.ConstantR0<float>(1.5f));
  EXPECT_TRUE(IsConstant(computation, &b));

  auto value = ComputeConstantScalar<float>(computation, &b);
  ASSERT_TRUE(value.ok()) << value.status();
  EXPECT_EQ(value.ValueOrDie(), 44.0f);
}

TEST_F(ComputeConstantTest, ScalarRng) {
  ComputationBuilder b(client_, TestName());
  auto computation =
      b.RngUniform(b.ConstantR0<float>(1.1f), b.ConstantR0<float>(2.1f),
                   ShapeUtil::MakeShape(F32, {}));
  EXPECT_FALSE(IsConstant(computation, &b));

  auto value = ComputeConstantScalar<float>(computation, &b);
  ASSERT_FALSE(value.ok())
      << "computing a RNG value should not be considered a constant";
}

TEST_F(ComputeConstantTest, DirectParam) {
  ComputationBuilder b(client_, TestName());
  auto computation = b.Parameter(0, ShapeUtil::MakeShape(F32, {}), "param");
  EXPECT_FALSE(IsConstant(computation, &b));

  auto value = ComputeConstantScalar<float>(computation, &b);
  EXPECT_TRUE(tensorflow::StringPiece(value.status().ToString())
                  .contains("depends on parameter"))
      << value.status();
}

TEST_F(ComputeConstantTest, IndirectParam) {
  ComputationBuilder b(client_, TestName());
  auto computation =
      b.Add(b.ConstantR0<float>(1.0f),
            b.Parameter(0, ShapeUtil::MakeShape(F32, {}), "param"));
  EXPECT_FALSE(IsConstant(computation, &b));

  auto value = ComputeConstantScalar<float>(computation, &b);
  EXPECT_TRUE(tensorflow::StringPiece(value.status().ToString())
                  .contains("depends on parameter"))
      << value.status();
}

// Test computation of an expression interspersed with param nodes but
// the expression does not depend on the param nodes.
TEST_F(ComputeConstantTest, UnrelatedParam) {
  ComputationBuilder b(client_, TestName());

  auto param_a = b.Parameter(10, ShapeUtil::MakeShape(F32, {}), "param0");
  auto constant_4 = b.Add(b.ConstantR0<float>(2.5f), b.ConstantR0<float>(1.5f));
  auto not_constant_a = b.Add(constant_4, param_a);

  auto param_b = b.Parameter(1, ShapeUtil::MakeShape(F32, {}), "param1");
  auto constant_9 = b.Mul(b.ConstantR0<float>(2.0f), b.ConstantR0<float>(4.5f));
  auto not_constant_b = b.Add(param_b, constant_9);

  auto constant_13 = b.Add(constant_4, constant_9);
  b.Add(not_constant_b, b.Add(constant_13, not_constant_a));

  EXPECT_TRUE(IsConstant(constant_13, &b));

  auto value = ComputeConstantScalar<float>(constant_13, &b);
  ASSERT_TRUE(value.ok()) << value.status();
  EXPECT_EQ(value.ValueOrDie(), 13.0f);
}

TEST_F(ComputeConstantTest, NonScalarAdd) {
  ComputationBuilder b(client_, TestName());

  auto computation =
      b.Add(b.ConstantR1<int32>({1, 2}), b.ConstantR1<int32>({3, 4}));
  EXPECT_TRUE(IsConstant(computation, &b));

  auto computed = ComputeConstantLiteral(computation, &b);
  ASSERT_TRUE(computed.ok()) << computed.status();
  std::unique_ptr<Literal> expected_literal =
      LiteralUtil::CreateR1<int32>({4, 6});
  LiteralTestUtil::ExpectEqual(*expected_literal, *computed.ValueOrDie());
}

TEST_F(ComputeConstantTest, IntegerDivide) {
  ComputationBuilder b(client_, TestName());
  auto computation = b.Div(b.ConstantR0<int32>(15), b.ConstantR0<int32>(3));
  EXPECT_TRUE(IsConstant(computation, &b));

  auto computed = ComputeConstantLiteral(computation, &b);
  ASSERT_TRUE(computed.ok()) << computed.status();
  std::unique_ptr<Literal> expected_literal = LiteralUtil::CreateR0<int32>(5);
  LiteralTestUtil::ExpectEqual(*expected_literal, *computed.ValueOrDie());
}

XLA_TEST_F(ComputeConstantTest, Layout) {
  ComputationBuilder b(client_, TestName());

  std::vector<std::vector<int64>> layouts = {{0, 1}, {1, 0}};
  for (const std::vector<int64>& layout : layouts) {
    auto layout_proto = LayoutUtil::MakeLayout(layout);
    auto computed =
        ComputeConstantLiteral(b.Add(b.ConstantR2<int32>({{1, 2}, {3, 4}}),
                                     b.ConstantR2<int32>({{10, 20}, {30, 40}})),
                               &b, &layout_proto);
    ASSERT_TRUE(computed.ok()) << computed.status();

    std::unique_ptr<Literal> expected_literal =
        test_utils::CreateR2LiteralWithLayout<int32>({{11, 22}, {33, 44}},
                                                     layout);
    LiteralTestUtil::AssertEqualShapesAndLayouts(
        expected_literal->shape(), computed.ValueOrDie()->shape());
    LiteralTestUtil::ExpectEqual(*expected_literal, *computed.ValueOrDie());
  }
}

// This test is permanently disabled on CPU because it requires that the
// backend used for execution is different than the backend used for
// ComputeConstant which is always cpu.
TEST_F(ComputeConstantTest, DISABLED_ON_CPU(ReuseComputedConstant)) {
  // Compute a trivial constant, then try to use the value in an Execute
  // call. This should fail because the constant resides on the CPU and the
  // Execute call is executed on a different backend.
  ComputationBuilder constant_b(client_, TestName());
  auto constant = constant_b.ConstantR0<int32>(42);
  auto handle = constant_b.ComputeConstant(constant).ConsumeValueOrDie();
  auto literal = client_->Transfer(*handle).ConsumeValueOrDie();
  LiteralTestUtil::ExpectR0Equal(42, *literal);

  // Build trivial computation which takes one parameter.
  ComputationBuilder b(client_, TestName());
  b.Neg(b.Parameter(0, ShapeUtil::MakeShape(S32, {}), "param0"));
  auto computation = b.Build().ConsumeValueOrDie();

  // Try to use value from ComputeConstant in Execute.
  auto execute_status = client_->Execute(computation, {handle.get()});
  EXPECT_FALSE(execute_status.ok());
  EXPECT_THAT(
      execute_status.status().error_message(),
      ::testing::ContainsRegex("argument 0 is on device Host:0 but computation "
                               "will be executed on device"));
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list;
  xla::legacy_flags::AppendCpuCompilerFlags(&flag_list);
  xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << "\n" << usage;
    return 2;
  }
  testing::InitGoogleTest(&argc, argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return 2;
  }
  return RUN_ALL_TESTS();
}
