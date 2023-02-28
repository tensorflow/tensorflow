/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/eval_const_tensor.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/meta/type_traits.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {
class EvaluateConstantTensorTest : public ::testing::Test {
 public:
  EvaluateConstantTensorTest& WithRunner() {
    runner_ = EvaluateConstantTensorRunner{
        scope_.graph()->op_registry(),
        scope_.graph()->versions().producer(),
    };
    return *this;
  }

  StatusOr<std::optional<Tensor>> Run(const Output& output) {
    TF_RETURN_IF_ERROR(scope_.status());
    const auto& graph = *scope_.graph();
    ShapeRefiner refiner(graph.versions(), graph.op_registry());
    for (const auto* node : graph.nodes()) {
      TF_RETURN_IF_ERROR(refiner.AddNode(node));
    }
    auto lookup = [this](const Node& node, int index) -> std::optional<Tensor> {
      auto it = cache_.find(std::make_pair(&node, index));
      if (it == cache_.end()) {
        return std::nullopt;
      }
      return it->second;
    };
    auto runner = runner_;
    runner_ = std::nullopt;
    return EvaluateConstantTensor(*output.node(), output.index(), refiner,
                                  lookup, runner);
  }

  void ExpectTensor(const Output& output, const Tensor& expected) {
    TF_ASSERT_OK_AND_ASSIGN(auto actual, Run(output));
    ASSERT_TRUE(actual.has_value());
    test::ExpectEqual(*actual, expected);
  }

  void ExpectNull(const Output& output) {
    TF_ASSERT_OK_AND_ASSIGN(auto actual, Run(output));
    ASSERT_FALSE(actual.has_value());
  }

  void ExpectError(const Output& output) { EXPECT_FALSE(Run(output).ok()); }

 protected:
  Scope scope_ = Scope::NewRootScope();
  absl::flat_hash_map<std::pair<const Node*, int>, Tensor> cache_;
  std::optional<EvaluateConstantTensorRunner> runner_ = std::nullopt;
};

template <typename T>
Output Placeholder(const Scope& scope, const PartialTensorShape& shape) {
  return ops::Placeholder(scope, DataTypeToEnum<T>::value,
                          ops::Placeholder::Shape(shape));
}

Output Slice(const Scope& scope, const Output& input, int index) {
  return ops::StridedSlice(
      scope, input, ops::Const(scope, {index}), ops::Const(scope, {index + 1}),
      ops::Const(scope, {1}), ops::StridedSlice::ShrinkAxisMask(1));
}

TEST_F(EvaluateConstantTensorTest, Constant) {
  auto expected = test::AsTensor<float>({1, 2, 3});
  auto op = ops::Const(scope_, expected);
  ExpectTensor(op, expected);
}

TEST_F(EvaluateConstantTensorTest, Shape) {
  auto input = Placeholder<float>(scope_, {2, 3, 5});
  auto shape = ops::Shape(scope_, input);
  ExpectTensor(shape, test::AsTensor<int32_t>({2, 3, 5}));
}

TEST_F(EvaluateConstantTensorTest, ValueOutOfRange) {
  const int64_t dim = std::numeric_limits<int32_t>::max();
  auto input = Placeholder<float>(scope_, {dim});
  auto shape32 = ops::Shape(scope_, input, ops::Shape::OutType(DT_INT32));
  auto shape64 = ops::Shape(scope_, input, ops::Shape::OutType(DT_INT64));
  ExpectError(shape32);
  ExpectTensor(shape64, test::AsTensor<int64_t>({dim}));
}

TEST_F(EvaluateConstantTensorTest, PartialShape) {
  auto input = Placeholder<float>(scope_, {2, -1, 5});
  auto shape = ops::Shape(scope_, input);
  ExpectNull(shape);
}

TEST_F(EvaluateConstantTensorTest, Rank) {
  auto input = Placeholder<float>(scope_, {2, -1, 5});
  auto rank = ops::Rank(scope_, input);
  ExpectTensor(rank, test::AsScalar<int32_t>(3));
}

TEST_F(EvaluateConstantTensorTest, Size) {
  auto input = Placeholder<float>(scope_, {2, 3, 5});
  auto size = ops::Size(scope_, input);
  ExpectTensor(size, test::AsScalar<int32_t>(2 * 3 * 5));
}

TEST_F(EvaluateConstantTensorTest, PartialSize) {
  auto input = Placeholder<float>(scope_, {2, -1, 5});
  auto size = ops::Size(scope_, input);
  ExpectNull(size);
}

TEST_F(EvaluateConstantTensorTest, SliceShape) {
  auto input = Placeholder<float>(scope_, {2, -1, 5});
  auto shape = ops::Shape(scope_, input);
  auto slice0 = Slice(scope_, shape, 0);
  auto slice1 = Slice(scope_, shape, 1);
  auto slice2 = Slice(scope_, shape, 2);
  ExpectTensor(slice0, test::AsScalar<int32_t>(2));
  ExpectNull(slice1);
  ExpectTensor(slice2, test::AsScalar<int32_t>(5));
}

TEST_F(EvaluateConstantTensorTest, UnpackShape) {
  auto input = Placeholder<float>(scope_, {2, -1, 5});
  auto shape = ops::Shape(scope_, input);
  auto unpack = ops::Unstack(scope_, shape, 3, ops::Unstack::Axis(0));
  ExpectTensor(unpack[0], test::AsScalar<int32_t>(2));
  ExpectNull(unpack[1]);
  ExpectTensor(unpack[2], test::AsScalar<int32_t>(5));
}

TEST_F(EvaluateConstantTensorTest, Lookup) {
  auto input = Placeholder<float>(scope_, {2});
  ExpectNull(input);

  auto expected = test::AsTensor<float>({3, 5});
  cache_.emplace(std::make_pair(input.node(), 0), expected);
  ExpectTensor(input, expected);
}

TEST_F(EvaluateConstantTensorTest, ConstantFolding) {
  auto input1 = Placeholder<float>(scope_, {2, -1, 5});
  auto input2 = ops::_Arg(scope_, DT_INT32, 0);
  auto shape = ops::Shape(scope_, input1);
  auto result = ops::Add(scope_, Slice(scope_, shape, 2), input2);

  ExpectNull(result);

  WithRunner().ExpectNull(result);

  cache_.emplace(std::make_pair(input2.node(), 0), test::AsScalar<int32_t>(7));
  WithRunner().ExpectTensor(result, test::AsScalar<int32_t>(5 + 7));
}

TEST_F(EvaluateConstantTensorTest, DoNotEvalPlaceholderWithDefault) {
  auto tensor = test::AsTensor<float>({1, 2, 3});
  auto result1 = ops::Identity(scope_, tensor);
  auto result2 = ops::PlaceholderWithDefault(scope_, tensor, tensor.shape());
  WithRunner().ExpectTensor(result1, tensor);
  WithRunner().ExpectNull(result2);
}

template <bool kEvaluated>
void BM_ConstantFolding(::testing::benchmark::State& state) {
  Scope scope = Scope::NewRootScope();
  auto input1 = Placeholder<float>(scope, {2, -1, 5});
  auto input2 = ops::_Arg(scope, DT_INT32, 0);
  auto input3 = ops::_Arg(scope, DT_INT32, 0);
  auto shape = ops::Shape(scope, input1);
  auto result =
      ops::Mul(scope, ops::Add(scope, Slice(scope, shape, 2), input2), input3);
  TF_CHECK_OK(scope.status());

  const auto& graph = *scope.graph();
  ShapeRefiner refiner(graph.versions(), graph.op_registry());
  for (const auto* node : graph.nodes()) {
    TF_CHECK_OK(refiner.AddNode(node));
  }
  auto tensor2 = test::AsScalar<int32_t>(7);
  auto tensor3 = test::AsScalar<int32_t>(11);
  auto lookup = [&](const Node& node, int index) -> std::optional<Tensor> {
    if (kEvaluated && &node == input2.node()) {
      return tensor2;
    }
    if (&node == input3.node()) {
      return tensor3;
    }
    return std::nullopt;
  };
  GraphRunner graph_runner(Env::Default());
  const EvaluateConstantTensorRunner runner = {
      graph.op_registry(), graph.versions().producer(), &graph_runner};

  for (auto unused : state) {
    auto status_or =
        EvaluateConstantTensor(*result.node(), 0, refiner, lookup, runner);
    TF_CHECK_OK(status_or.status());
    CHECK_EQ(status_or->has_value(), kEvaluated);
  }
}
BENCHMARK_TEMPLATE(BM_ConstantFolding, false);
BENCHMARK_TEMPLATE(BM_ConstantFolding, true);

}  // namespace
}  // namespace tensorflow
