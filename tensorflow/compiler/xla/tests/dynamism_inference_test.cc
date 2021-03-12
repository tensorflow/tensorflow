/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/strings/match.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/prng.h"
#include "tensorflow/compiler/xla/client/value_inference.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

// An enumerator for the client types that we want to iterate over in
// the various tests.
enum class ClientType { kLocal, kCompileOnly };

class DynamismInferenceTest : public ::testing::Test {
 public:
  explicit DynamismInferenceTest(se::Platform* platform = nullptr)
      : platform_(platform) {}

  string TestName() const {
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
  }

  Client* ClientOrDie(se::Platform* platform, ClientType client_type) {
    if (client_type == ClientType::kLocal) {
      StatusOr<Client*> result =
          ClientLibrary::GetOrCreateLocalClient(platform);
      TF_CHECK_OK(result.status())
          << "could not create LocalClient for testing";
      return result.ValueOrDie();
    } else if (client_type == ClientType::kCompileOnly) {
      StatusOr<Client*> result =
          ClientLibrary::GetOrCreateCompileOnlyClient(platform);
      TF_CHECK_OK(result.status())
          << "could not create CompileOnlyClient for testing";
      return result.ValueOrDie();
    }
    LOG(FATAL) << "invalid client_type value";
  }

  StatusOr<Literal> ComputeDynamismLiteral(XlaOp operand, XlaBuilder* builder,
                                           Layout* output_layout = nullptr) {
    ValueInference value_inference(builder);
    TF_ASSIGN_OR_RETURN(auto literal_slice,
                        value_inference.AnalyzeIsDynamic(operand));
    return literal_slice.Clone();
  }

  StatusOr<bool> ComputeDynamismScalar(XlaOp operand, XlaBuilder* builder,
                                       ShapeIndex index = {}) {
    TF_ASSIGN_OR_RETURN(auto literal,
                        ComputeDynamismLiteral(operand, builder, nullptr));
    return literal.Get<bool>({}, index);
  }

  se::Platform* platform_;
};

TEST_F(DynamismInferenceTest, ScalarInt32Literal) {
  XlaBuilder b(TestName());
  auto computation = ConstantR0<int32>(&b, 42);

  auto value = ComputeDynamismScalar(computation, &b);
  ASSERT_TRUE(value.ok()) << value.status();
  // A constant is not dynamic.
  EXPECT_EQ(value.ValueOrDie(), false);
}

TEST_F(DynamismInferenceTest, Iota) {
  // The output of iota are consistened static.
  XlaBuilder b(TestName());
  auto computation = Iota(&b, S32, 2);
  // Iota is not dynamic.
  EXPECT_FALSE(
      ComputeDynamismLiteral(computation, &b).ValueOrDie().Get<bool>({0}));
}

TEST_F(DynamismInferenceTest, TupleSimple) {
  XlaBuilder b(TestName());
  auto c = ConstantR0<int32>(&b, 42);
  auto p = Parameter(&b, 0, ShapeUtil::MakeScalarShape(S32), "p0");

  auto tuple = Tuple(&b, {c, p});
  EXPECT_EQ(ComputeDynamismScalar(tuple, &b, {0}).ValueOrDie(), false);
  EXPECT_EQ(ComputeDynamismScalar(tuple, &b, {1}).ValueOrDie(), true);
}

TEST_F(DynamismInferenceTest, TupleGteKeepsDynamism) {
  XlaBuilder b(TestName());
  auto c = ConstantR0<int32>(&b, 42);
  auto p = Parameter(&b, 0, ShapeUtil::MakeScalarShape(S32), "p0");

  auto tuple = Tuple(&b, {c, p});
  auto gte0 = GetTupleElement(tuple, 0);
  auto gte1 = GetTupleElement(tuple, 1);
  auto tuple_2 = Tuple(&b, {gte0, gte1});
  EXPECT_EQ(ComputeDynamismScalar(tuple_2, &b, {0}).ValueOrDie(), false);
  EXPECT_EQ(ComputeDynamismScalar(tuple_2, &b, {1}).ValueOrDie(), true);
}

TEST_F(DynamismInferenceTest, PredValueUsedTwice) {
  XlaBuilder b(TestName());
  auto c = ConstantR0<int32>(&b, 42);
  auto p = Parameter(&b, 0, ShapeUtil::MakeScalarShape(S32), "p0");
  auto pred = Eq(c, p);
  auto result = Select(pred, p, c);
  EXPECT_EQ(ComputeDynamismScalar(result, &b, {}).ValueOrDie(), true);
}

TEST_F(DynamismInferenceTest, ReduceUsedTwice) {
  XlaBuilder b(TestName());
  auto c = ConstantR0<int32>(&b, 42);
  auto p = Parameter(&b, 0, ShapeUtil::MakeShape(S32, {2}), "p0");
  auto zero = ConstantR0<int32>(&b, 0);
  XlaComputation add_s32 = CreateScalarAddComputation(S32, &b);
  auto reduce = Reduce(p, zero, add_s32, {0});
  auto pred = Eq(c, reduce);
  auto result = Select(pred, reduce, c);
  EXPECT_EQ(ComputeDynamismScalar(result, &b, {}).ValueOrDie(), true);
}

TEST_F(DynamismInferenceTest, DynamicSelectorWithMixedValues) {
  XlaBuilder b(TestName());
  auto constant_pred = ConstantR1<bool>(&b, {true});
  auto dynamic_pred = Parameter(&b, 0, ShapeUtil::MakeShape(PRED, {1}), "p0");
  auto concat = ConcatInDim(&b, {constant_pred, dynamic_pred}, 0);
  auto constant_values = ConstantR1<bool>(&b, {true, true});
  auto result = Select(concat, constant_values, constant_values);
  // First result is static (selector is constant, both values are constant).
  // Iota is not dynamic.
  EXPECT_FALSE(ComputeDynamismLiteral(result, &b).ValueOrDie().Get<bool>({0}));
  // Second result is dynamic (selector is dynamic).
  EXPECT_TRUE(ComputeDynamismLiteral(result, &b).ValueOrDie().Get<bool>({1}));
}

TEST_F(DynamismInferenceTest, ConcatSliceReshapeKeepsDynamism) {
  XlaBuilder b(TestName());
  auto c = ConstantR0<int32>(&b, 42);
  auto p = Parameter(&b, 0, ShapeUtil::MakeScalarShape(S32), "p0");

  auto concat = ConcatScalars(&b, {c, p});
  auto slice0 = SliceInDim(concat, 0, 1, 1, 0);
  auto reshape0 = Reshape(slice0, {});
  auto slice1 = SliceInDim(concat, 1, 2, 1, 0);
  auto reshape1 = Reshape(slice1, {});
  auto tuple_2 = Tuple(&b, {reshape0, reshape1});
  EXPECT_EQ(ComputeDynamismScalar(tuple_2, &b, {0}).ValueOrDie(), false);
  EXPECT_EQ(ComputeDynamismScalar(tuple_2, &b, {1}).ValueOrDie(), true);
}

TEST_F(DynamismInferenceTest, ParameterIsDynamic) {
  XlaBuilder b(TestName());
  auto computation = Parameter(&b, 0, ShapeUtil::MakeScalarShape(S32), "p0");

  auto value = ComputeDynamismScalar(computation, &b);
  ASSERT_TRUE(value.ok()) << value.status();
  // A parameter is considered dynamic.
  EXPECT_EQ(value.ValueOrDie(), true);
}

TEST_F(DynamismInferenceTest, UnaryOpKeepsDynamism) {
  XlaBuilder b(TestName());
  auto c = ConstantR0<int32>(&b, 42);
  auto p = Parameter(&b, 0, ShapeUtil::MakeScalarShape(S32), "p0");

  auto neg0 = Neg(c);
  auto neg1 = Neg(p);
  auto tuple_2 = Tuple(&b, {neg0, neg1});
  EXPECT_EQ(ComputeDynamismScalar(tuple_2, &b, {0}).ValueOrDie(), false);
  EXPECT_EQ(ComputeDynamismScalar(tuple_2, &b, {1}).ValueOrDie(), true);
}

TEST_F(DynamismInferenceTest, BinaryOpsOrsDynamism) {
  XlaBuilder b(TestName());
  auto c = ConstantR0<int32>(&b, 42);
  auto p = Parameter(&b, 0, ShapeUtil::MakeScalarShape(S32), "p0");

  // Static value + static value = static
  auto add1 = Add(c, c);
  // Dynamic value + dynamic value = dynamic
  auto add2 = Add(p, c);
  auto tuple_2 = Tuple(&b, {add1, add2});
  EXPECT_EQ(ComputeDynamismScalar(tuple_2, &b, {0}).ValueOrDie(), false);
  EXPECT_EQ(ComputeDynamismScalar(tuple_2, &b, {1}).ValueOrDie(), true);
}

TEST_F(DynamismInferenceTest, GetDimensionSize) {
  XlaBuilder b(TestName());
  // param = Param([<=2, 3])
  // get_dimension_size(param, 0) is dynamic
  // get_dimension_size(param, 1) is static
  auto p =
      Parameter(&b, 0, ShapeUtil::MakeShape(S32, {2, 3}, {true, false}), "p0");

  auto gds0 = GetDimensionSize(p, 0);
  auto gds1 = GetDimensionSize(p, 1);
  auto tuple_2 = Tuple(&b, {gds0, gds1});
  EXPECT_EQ(ComputeDynamismScalar(tuple_2, &b, {0}).ValueOrDie(), true);
  EXPECT_EQ(ComputeDynamismScalar(tuple_2, &b, {1}).ValueOrDie(), false);
}

TEST_F(DynamismInferenceTest, GatherWithCommonParent) {
  XlaBuilder b(TestName());
  // Test the analysis on a gather where first operand and second operand have
  // common parents.
  Shape indices_shape = ShapeUtil::MakeShape(S32, {2});

  auto operand1 = Parameter(&b, 0, indices_shape, "p1");
  auto operand2 = Parameter(&b, 1, indices_shape, "p2");
  auto indices = Sub(operand1, operand2);
  GatherDimensionNumbers dim_numbers;
  dim_numbers.add_offset_dims(1);
  dim_numbers.add_start_index_map(0);
  dim_numbers.set_index_vector_dim(1);
  auto gather = Gather(operand1, indices, dim_numbers, {1});
  ASSERT_TRUE(b.first_error().ok()) << b.first_error().error_message();
  EXPECT_TRUE(
      ComputeDynamismLiteral(gather, &b).ValueOrDie().Get<bool>({0, 0}));
}

TEST_F(DynamismInferenceTest, GatherWithConstantParent) {
  XlaBuilder b(TestName());
  // Test the analysis on a gather.
  Shape indices_shape = ShapeUtil::MakeShape(S32, {2});
  auto data_operand = ConstantR1<int32>(&b, {1, 2});
  auto indices = ConstantR1<int32>(&b, {1, 2});
  GatherDimensionNumbers dim_numbers;
  dim_numbers.add_offset_dims(1);
  dim_numbers.add_start_index_map(0);
  dim_numbers.set_index_vector_dim(1);
  auto gather = Gather(data_operand, indices, dim_numbers, {1});
  ASSERT_TRUE(b.first_error().ok()) << b.first_error().error_message();
  // Everything is constant, result is also contant.
  EXPECT_FALSE(
      ComputeDynamismLiteral(gather, &b).ValueOrDie().Get<bool>({0, 0}));
}

TEST_F(DynamismInferenceTest, GatherWithSharedConstantParent) {
  XlaBuilder b(TestName());
  // Test the analysis on a gather.
  Shape indices_shape = ShapeUtil::MakeShape(S32, {2});
  auto operand1 = ConstantR1<int32>(&b, {1, 2});
  auto operand2 = ConstantR1<int32>(&b, {1, 2});
  auto indices = Sub(operand1, operand2);
  GatherDimensionNumbers dim_numbers;
  dim_numbers.add_offset_dims(1);
  dim_numbers.add_start_index_map(0);
  dim_numbers.set_index_vector_dim(1);
  auto gather = Gather(operand1, indices, dim_numbers, {1});
  ASSERT_TRUE(b.first_error().ok()) << b.first_error().error_message();
  // Everything is constant, result is also contant.
  EXPECT_FALSE(
      ComputeDynamismLiteral(gather, &b).ValueOrDie().Get<bool>({0, 0}));
}

TEST_F(DynamismInferenceTest, InferThroughPad) {
  XlaBuilder b(TestName());
  // Test the analysis on a gather.
  auto operand1 = ConstantR1<int32>(&b, {1, 2});
  auto parameter = Parameter(&b, 0, ShapeUtil::MakeShape(S32, {}), "p0");
  PaddingConfig padding_config;
  padding_config.add_dimensions()->set_edge_padding_high(1);
  // After pad the value is [constant, constant, parameter].
  auto pad = Pad(operand1, parameter, padding_config);
  ASSERT_TRUE(b.first_error().ok()) << b.first_error().error_message();
  // Everything is constant, result is also contant.
  EXPECT_FALSE(ComputeDynamismLiteral(pad, &b).ValueOrDie().Get<bool>({0}));
  EXPECT_FALSE(ComputeDynamismLiteral(pad, &b).ValueOrDie().Get<bool>({1}));
  EXPECT_TRUE(ComputeDynamismLiteral(pad, &b).ValueOrDie().Get<bool>({2}));
}

}  // namespace
}  // namespace xla
