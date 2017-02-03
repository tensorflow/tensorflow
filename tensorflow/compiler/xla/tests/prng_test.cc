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

#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class PrngTest : public ClientLibraryTestBase {
 protected:
  template <typename T>
  void UniformTest(T a, T b, tensorflow::gtl::ArraySlice<int64> dims);
  void BernoulliTest(float p, tensorflow::gtl::ArraySlice<int64> dims);
};

template <typename T>
void PrngTest::UniformTest(T a, T b, tensorflow::gtl::ArraySlice<int64> dims) {
  ComputationBuilder builder(client_, TestName());
  builder.RngUniform(
      builder.ConstantR0<T>(a), builder.ConstantR0<T>(b),
      ShapeUtil::MakeShape(primitive_util::NativeToPrimitiveType<T>(), dims));

  auto actual = ExecuteAndTransferOrDie(&builder, /*arguments=*/{});
  EXPECT_TRUE(ContainersEqual(dims, actual->shape().dimensions()));
  LiteralUtil::EachCell<T>(*actual,
                           [=](tensorflow::gtl::ArraySlice<int64>, T value) {
                             EXPECT_LE(a, value);
                             EXPECT_LT(value, b);
                           });
}

void PrngTest::BernoulliTest(float p, tensorflow::gtl::ArraySlice<int64> dims) {
  ComputationBuilder builder(client_, TestName());
  auto shape = ShapeUtil::MakeShape(U32, dims);
  builder.RngBernoulli(builder.ConstantR0<float>(p), shape);

  TF_ASSIGN_OR_ASSERT_OK(auto computation, builder.Build());
  ExecutionOptions execution_options;
  execution_options.set_seed(42);
  TF_ASSIGN_OR_ASSERT_OK(
      auto actual,
      client_->ExecuteAndTransfer(computation, /*arguments=*/{},
                                  &execution_options));
  EXPECT_TRUE(ContainersEqual(dims, actual->shape().dimensions()));
  int32 sum = 0;
  LiteralUtil::EachCell<uint32>(
      *actual, [&sum](tensorflow::gtl::ArraySlice<int64>, uint32 value) {
        EXPECT_TRUE(value == 0 || value == 1);
        sum += value;
      });
  int32 total = ShapeUtil::ElementsIn(shape);
  float p_tilde = sum / static_cast<float>(total);

  // Test within expected range using normal approximation. The test uses a
  // fixed seed and has a fixed output per p and backend. Using the normal
  // approximation as this test is invoked for different `p` and the different
  // backends could use different random number generators and produce different
  // values. Choose 95% confidence level, so that z_{1-\alpha/2} = 1.96.
  float normal_approximation_term = 1.96 * sqrt(p * (1 - p) / total);
  EXPECT_GE(p_tilde, p - normal_approximation_term);
  EXPECT_LE(p_tilde, p + normal_approximation_term);
}

// Uniform random number generation tests
XLA_TEST_F(PrngTest, ScalarU01) { UniformTest<float>(0, 1, {}); }
XLA_TEST_F(PrngTest, ZeroValuesU01) { UniformTest<float>(0, 1, {0}); }
XLA_TEST_F(PrngTest, TenValuesU01) { UniformTest<float>(0, 1, {10}); }
XLA_TEST_F(PrngTest, TenValuesU37) { UniformTest<float>(3, 7, {10}); }
XLA_TEST_F(PrngTest, ZeroValuesR2) { UniformTest<float>(0, 1, {0, 20}); }
XLA_TEST_F(PrngTest, LargeU01) { UniformTest<float>(0, 1, {0x100, 0x100}); }
XLA_TEST_F(PrngTest, TwelveValuesU524) { UniformTest<int32>(5, 24, {12}); }

XLA_TEST_F(PrngTest, MapUsingRng) {
  // Build a x -> (x + U[0,1)) computation.
  auto build_sum_rng = [this](ComputationBuilder& builder) {
    auto b = builder.CreateSubBuilder("sum_with_rng");
    auto x = b->Parameter(0, ShapeUtil::MakeShape(F32, {}), "input");
    b->Add(x,
           b->RngUniform(b->ConstantR0<float>(0), b->ConstantR0<float>(1),
                         ShapeUtil::MakeShape(F32, {})));
    return b->BuildAndNoteError();
  };

  ComputationBuilder builder(client_, TestName());
  std::unique_ptr<Literal> param0_literal =
      LiteralUtil::CreateR1<float>({2.2f, 5.3f, 4.4f, 5.5f});
  TF_ASSIGN_OR_ASSERT_OK(std::unique_ptr<GlobalData> param0_data,
                         client_->TransferToServer(*param0_literal));

  auto param0 = builder.Parameter(0, param0_literal->shape(), "param0");
  auto fn = build_sum_rng(builder);
  builder.Map({param0}, fn);

  TF_ASSIGN_OR_ASSERT_OK(auto computation, builder.Build());

  ExecutionOptions execution_options;
  execution_options.set_seed(125);
  TF_ASSIGN_OR_ASSERT_OK(
      auto actual,
      client_->ExecuteAndTransfer(computation,
                                  /*arguments=*/{param0_data.get()},
                                  &execution_options));

  EXPECT_EQ(actual->f32s_size(), param0_literal->f32s_size());
  for (int i = 0; i < param0_literal->f32s_size(); ++i) {
    EXPECT_GE(actual->f32s(i), param0_literal->f32s(i));
    EXPECT_LT(actual->f32s(i), param0_literal->f32s(i) + 1.0f);
  }
}

// This tests demonstrates the global seeding behaviour.
// * If a seed is passed in via Execute (ExecuteAndTransfer) then the output is
//   fixed (i.e., there is a single output for a given seed);
// * If no seed is passed in then the output of every call can be different;
XLA_TEST_F(PrngTest, PassInGlobalRngSeed) {
  // Build a U[0,1) computation.
  auto build_computation = [this]() {
    ComputationBuilder builder(client_, TestName());
    builder.RngUniform(builder.ConstantR0<float>(0),
                       builder.ConstantR0<float>(1),
                       ShapeUtil::MakeShape(F32, {10}));
    return builder.Build();
  };

  ExecutionOptions execution_options1;
  execution_options1.set_seed(42);

  ExecutionOptions execution_options2;
  execution_options2.set_seed(65);

  std::unique_ptr<Literal> result1;
  {
    TF_ASSIGN_OR_ASSERT_OK(auto computation, build_computation());
    TF_ASSIGN_OR_ASSERT_OK(
        result1,
        client_->ExecuteAndTransfer(computation, /*arguments=*/{},
                                    &execution_options1));
  }
  std::unique_ptr<Literal> result2;
  std::unique_ptr<Literal> result3;
  {
    TF_ASSIGN_OR_ASSERT_OK(auto computation, build_computation());
    TF_ASSIGN_OR_ASSERT_OK(
        result2,
        client_->ExecuteAndTransfer(computation, /*arguments=*/{},
                                    &execution_options1));
    TF_ASSIGN_OR_ASSERT_OK(
        result3,
        client_->ExecuteAndTransfer(computation, /*arguments=*/{},
                                    &execution_options1));
  }

  std::unique_ptr<Literal> result4;
  std::unique_ptr<Literal> result5;
  std::unique_ptr<Literal> result6;
  {
    TF_ASSIGN_OR_ASSERT_OK(auto computation, build_computation());
    TF_ASSIGN_OR_ASSERT_OK(
        result4,
        client_->ExecuteAndTransfer(computation, /*arguments=*/{},
                                    &execution_options2));
    TF_ASSIGN_OR_ASSERT_OK(
        result5, client_->ExecuteAndTransfer(computation, /*arguments=*/{}));
    TF_ASSIGN_OR_ASSERT_OK(
        result6, client_->ExecuteAndTransfer(computation, /*arguments=*/{}));
  }

  LiteralTestUtil::ExpectEqual(*result1, *result2);
  LiteralTestUtil::ExpectEqual(*result1, *result3);
  LiteralTestUtil::ExpectNotEqual(*result1, *result4);
  LiteralTestUtil::ExpectNotEqual(*result4, *result5);
  LiteralTestUtil::ExpectNotEqual(*result5, *result6);
}

// Bernoulli random number generation tests
XLA_TEST_F(PrngTest, HundredValuesB10p5) { BernoulliTest(0.5, {100}); }
XLA_TEST_F(PrngTest, HundredValuesB10p1) { BernoulliTest(0.1, {100}); }

XLA_TEST_F(PrngTest, TenValuesN01) {
  ComputationBuilder builder(client_, TestName());
  builder.RngNormal(builder.ConstantR0<float>(0), builder.ConstantR0<float>(1),
                    ShapeUtil::MakeShape(F32, {10}));

  ExecuteAndTransferOrDie(&builder, /*arguments=*/{});
  // TODO(b/25995601): Test that resultant values are reasonable
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
