/* Copyright 2022 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value.h"

#include <complex>
#include <cstdint>
#include <optional>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/ADT/ArrayRef.h"
#include "xla/mlir/tools/mlir_interpreter/framework/tensor_or_memref.h"

namespace mlir {
namespace interpreter {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

TEST(InterpreterValueTest, FillUnitTensor) {
  auto t = TensorOrMemref<int64_t>::Empty({});
  t.at({}) = 42;
  InterpreterValue v{t};
  v.Fill([](llvm::ArrayRef<int64_t>) { return InterpreterValue{int64_t{43}}; });
  ASSERT_EQ(t.at({}), 43);
}

TEST(InterpreterValueTest, Fill1DTensor) {
  auto t = TensorOrMemref<int64_t>::Empty({3});
  InterpreterValue v{t};
  v.Fill([](llvm::ArrayRef<int64_t> indices) {
    return InterpreterValue{indices[0]};
  });
  ASSERT_EQ(t.at(0), 0);
  ASSERT_EQ(t.at(1), 1);
  ASSERT_EQ(t.at(2), 2);
}

TEST(InterpreterValueTest, FillTensorOfVector) {
  auto t = TensorOrMemref<int64_t>::Empty({4, 2});
  t.view.num_vector_dims = 1;

  InterpreterValue v{t};
  v.Fill([](llvm::ArrayRef<int64_t> indices) -> InterpreterValue {
    EXPECT_EQ(indices.size(), 1);
    auto r = TensorOrMemref<int64_t>::Empty({2});
    r.view.is_vector = true;
    r.at(0) = indices[0];
    r.at(1) = indices[0] * 10;
    return {r};
  });
  ASSERT_EQ(
      v.ToString(),
      "TensorOrMemref<4xvector<2xi64>>: [[0, 0], [1, 10], [2, 20], [3, 30]]");
}

TEST(InterpreterValueTest, FillZeroSizedTensor) {
  auto t = TensorOrMemref<int64_t>::Empty({0, 1});
  InterpreterValue v{t};
  bool was_called = false;
  v.Fill([&](llvm::ArrayRef<int64_t> indices) {
    was_called = true;
    return InterpreterValue{indices[0]};
  });
  EXPECT_FALSE(was_called);
}

TEST(InterpreterValueTest, TypedAlike) {
  InterpreterValue v{TensorOrMemref<int32_t>::Empty({})};
  auto TypedAlike = v.TypedAlike({1, 2, 3});
  ASSERT_TRUE(
      std::holds_alternative<TensorOrMemref<int32_t>>(TypedAlike.storage));
  ASSERT_THAT(TypedAlike.View().sizes, ElementsAre(1, 2, 3));
}

TEST(InterpreterValueTest, AsUnitTensor) {
  InterpreterValue v{42};
  InterpreterValue wrapped = v.AsUnitTensor();
  ASSERT_THAT(wrapped.View().sizes, IsEmpty());
  ASSERT_EQ(std::get<TensorOrMemref<int32_t>>(wrapped.storage).at({}), 42);
}

TEST(InterpreterValueTest, IsTensor) {
  ASSERT_FALSE(InterpreterValue{42}.IsTensor());
  ASSERT_TRUE(InterpreterValue{TensorOrMemref<int32_t>::Empty({})}.IsTensor());
}

TEST(InterpreterValueTest, AsInt) {
  ASSERT_EQ(InterpreterValue{int64_t{42}}.AsInt(), 42);
  ASSERT_EQ(InterpreterValue{int32_t{42}}.AsInt(), 42);
  ASSERT_EQ(InterpreterValue{int16_t{42}}.AsInt(), 42);
  ASSERT_EQ(InterpreterValue{int8_t{42}}.AsInt(), 42);
  ASSERT_EQ(InterpreterValue{int8_t{-1}}.AsInt(), -1);
}

TEST(InterpreterValueTest, AsUInt) {
  ASSERT_EQ(InterpreterValue{int16_t{-1}}.AsUInt(), 65535);
  ASSERT_EQ(InterpreterValue{int8_t{-1}}.AsUInt(), 255);
}

TEST(InterpreterValueTest, CloneTensor) {
  auto tensor = TensorOrMemref<int64_t>::Empty({3});
  tensor.at(0) = 1;
  tensor.at(1) = 2;
  tensor.at(2) = 3;

  InterpreterValue wrapped{tensor};
  auto clone = wrapped.Clone();
  tensor.at(0) = 4;

  auto& cloned_tensor = std::get<TensorOrMemref<int64_t>>(clone.storage);
  ASSERT_EQ(cloned_tensor.at(0), 1);
  ASSERT_EQ(cloned_tensor.at(1), 2);
  ASSERT_EQ(cloned_tensor.at(2), 3);
}

TEST(InterpreterValueTest, CloneWithLayouts) {
  auto tensor = TensorOrMemref<int64_t>::Empty({3, 5}, {0, 1});
  tensor.at({2, 4}) = 42;

  InterpreterValue wrapped{tensor};
  auto clone = wrapped.Clone();
  ASSERT_EQ(clone.View().strides,
            BufferView::GetStridesForLayout({3, 5}, {1, 0}));
  ASSERT_EQ(clone.ExtractElement({2, 4}).AsInt(), 42);
}

TEST(InterpreterValueTest, CoerceLayoutNoop) {
  auto tensor = TensorOrMemref<int64_t>::Empty({3, 5}, {0, 1});
  tensor.at({2, 4}) = 42;

  InterpreterValue wrapped{tensor};
  auto coerced = wrapped.CoerceLayout({0, 1});
  ASSERT_EQ(tensor.buffer,
            std::get<TensorOrMemref<int64_t>>(coerced.storage).buffer);
}

TEST(InterpreterValueTest, CoerceLayout) {
  auto tensor = TensorOrMemref<int64_t>::Empty({3, 5});
  tensor.at({2, 4}) = 42;

  InterpreterValue wrapped{tensor};
  auto clone = wrapped.CoerceLayout({0, 1});
  ASSERT_EQ(clone.View().strides,
            BufferView::GetStridesForLayout({3, 5}, {0, 1}));
  ASSERT_EQ(clone.ExtractElement({2, 4}).AsInt(), 42);
}

TEST(InterpreterValueTest, CoerceLayoutSquare) {
  auto tensor = TensorOrMemref<float>::Empty({2, 2});
  tensor.at({0, 0}) = 1;
  tensor.at({0, 1}) = 2;
  tensor.at({1, 0}) = 3;
  tensor.at({1, 1}) = 4;

  InterpreterValue wrapped{tensor};
  auto clone = wrapped.CoerceLayout({0, 1});
  auto& cloned_tensor = std::get<TensorOrMemref<float>>(clone.storage);

  EXPECT_EQ(
      *reinterpret_cast<float*>(cloned_tensor.buffer->at(0, sizeof(float))), 1);
  EXPECT_EQ(
      *reinterpret_cast<float*>(cloned_tensor.buffer->at(1, sizeof(float))), 3);
  EXPECT_EQ(
      *reinterpret_cast<float*>(cloned_tensor.buffer->at(2, sizeof(float))), 2);
  EXPECT_EQ(
      *reinterpret_cast<float*>(cloned_tensor.buffer->at(3, sizeof(float))), 4);
}

TEST(InterpreterValueTest, CloneScalar) {
  InterpreterValue value{42};
  auto clone = value.Clone();
  ASSERT_THAT(std::get<int32_t>(clone.storage), 42);
}

TEST(InterpreterValueTest, ToString) {
  InterpreterValue value{TensorOrMemref<int64_t>::Empty({3})};
  ASSERT_EQ(value.ToString(), "TensorOrMemref<3xi64>: [0, 0, 0]");
}

TEST(InterpreterValueTest, ToString2d) {
  InterpreterValue value{TensorOrMemref<int64_t>::Empty({3, 2})};
  ASSERT_EQ(value.ToString(),
            "TensorOrMemref<3x2xi64>: [[0, 0], [0, 0], [0, 0]]");
}

TEST(InterpreterValueTest, ToString0d) {
  InterpreterValue value{TensorOrMemref<int64_t>::Empty({})};
  ASSERT_EQ(value.ToString(), "TensorOrMemref<i64>: 0");
}

TEST(InterpreterValueTest, ToStringComplex) {
  InterpreterValue value{std::complex<float>{}};
  ASSERT_EQ(value.ToString(), "complex<f32>: 0.000000e+00+0.000000e+00i");
}

TEST(CastTest, UnpackTensor) {
  InterpreterValue value{TensorOrMemref<int8_t>::Empty({1, 1})};
  value.InsertElement({0, 0}, {int8_t{1}});
  ASSERT_EQ(InterpreterValueCast<int64_t>(value), 1);
  ASSERT_EQ(InterpreterValueCast<uint8_t>(value), 1);
  ASSERT_EQ(InterpreterValueCast<float>(value), 1.0f);
  ASSERT_EQ(InterpreterValueCast<double>(value), 1.0);

  InterpreterValue non_unit{TensorOrMemref<int8_t>::Empty({2, 2})};
  ASSERT_EQ(InterpreterValueDynCast<int64_t>(non_unit), std::nullopt);
}

TEST(CastTest, IdentityCast) {
  InterpreterValue value{TensorOrMemref<float>::Empty({1, 1})};
  ASSERT_EQ(InterpreterValueCast<InterpreterValue>(value), value);
}

TEST(CastTest, CastToUnsigned) {
  // Note: This is different from `AsUint`, which preserves the size of the
  // original type (i.e. int8_t{-1} results in 255).
  InterpreterValue value{int8_t{-1}};
  ASSERT_EQ(InterpreterValueCast<uint8_t>(value), 255);
  ASSERT_EQ(InterpreterValueCast<uint16_t>(value), 65535);
}

}  // namespace
}  // namespace interpreter
}  // namespace mlir
