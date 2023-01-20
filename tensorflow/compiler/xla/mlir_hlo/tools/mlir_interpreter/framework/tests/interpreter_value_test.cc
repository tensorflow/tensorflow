/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tools/mlir_interpreter/framework/interpreter_value.h"

#include <variant>

#include "llvm/ADT/ArrayRef.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
namespace interpreter {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

TEST(InterpreterValueTest, FillUnitTensor) {
  auto t = TensorOrMemref<int64_t>::empty({});
  t.at({}) = 42;
  InterpreterValue v{t};
  v.fill([](llvm::ArrayRef<int64_t>) { return InterpreterValue{int64_t{43}}; });
  ASSERT_EQ(t.at({}), 43);
}

TEST(InterpreterValueTest, Fill1DTensor) {
  auto t = TensorOrMemref<int64_t>::empty({3});
  InterpreterValue v{t};
  v.fill([](llvm::ArrayRef<int64_t> indices) {
    return InterpreterValue{indices[0]};
  });
  ASSERT_EQ(t.at(0), 0);
  ASSERT_EQ(t.at(1), 1);
  ASSERT_EQ(t.at(2), 2);
}

TEST(InterpreterValueTest, FillTensorOfVector) {
  auto t = TensorOrMemref<int64_t>::empty({4, 2});
  t.view.numVectorDims = 1;

  InterpreterValue v{t};
  v.fill([](llvm::ArrayRef<int64_t> indices) -> InterpreterValue {
    EXPECT_EQ(indices.size(), 1);
    auto r = TensorOrMemref<int64_t>::empty({2});
    r.view.isVector = true;
    r.at(0) = indices[0];
    r.at(1) = indices[0] * 10;
    return {r};
  });
  ASSERT_EQ(
      v.toString(),
      "TensorOrMemref<4xvector<2xi64>>: [[0, 0], [1, 10], [2, 20], [3, 30]]");
}

TEST(InterpreterValueTest, FillZeroSizedTensor) {
  auto t = TensorOrMemref<int64_t>::empty({0, 1});
  InterpreterValue v{t};
  bool wasCalled = false;
  v.fill([&](llvm::ArrayRef<int64_t> indices) {
    wasCalled = true;
    return InterpreterValue{indices[0]};
  });
  EXPECT_FALSE(wasCalled);
}

TEST(InterpreterValueTest, TypedAlike) {
  InterpreterValue v{TensorOrMemref<int32_t>::empty({})};
  auto typedAlike = v.typedAlike({1, 2, 3});
  ASSERT_TRUE(
      std::holds_alternative<TensorOrMemref<int32_t>>(typedAlike.storage));
  ASSERT_THAT(typedAlike.view().sizes, ElementsAre(1, 2, 3));
}

TEST(InterpreterValueTest, AsUnitTensor) {
  InterpreterValue v{42};
  InterpreterValue wrapped = v.asUnitTensor();
  ASSERT_THAT(wrapped.view().sizes, IsEmpty());
  ASSERT_EQ(std::get<TensorOrMemref<int32_t>>(wrapped.storage).at({}), 42);
}

TEST(InterpreterValueTest, IsTensor) {
  ASSERT_FALSE(InterpreterValue{42}.isTensor());
  ASSERT_TRUE(InterpreterValue{TensorOrMemref<int32_t>::empty({})}.isTensor());
}

TEST(InterpreterValueTest, AsInt) {
  ASSERT_EQ(InterpreterValue{int64_t{42}}.asInt(), 42);
  ASSERT_EQ(InterpreterValue{int32_t{42}}.asInt(), 42);
  ASSERT_EQ(InterpreterValue{int16_t{42}}.asInt(), 42);
  ASSERT_EQ(InterpreterValue{int8_t{42}}.asInt(), 42);
  ASSERT_EQ(InterpreterValue{int8_t{-1}}.asInt(), -1);
}

TEST(InterpreterValueTest, AsUInt) {
  ASSERT_EQ(InterpreterValue{int16_t{-1}}.asUInt(), 65535);
  ASSERT_EQ(InterpreterValue{int8_t{-1}}.asUInt(), 255);
}

TEST(InterpreterValueTest, CloneTensor) {
  auto tensor = TensorOrMemref<int64_t>::empty({3});
  tensor.at(0) = 1;
  tensor.at(1) = 2;
  tensor.at(2) = 3;

  InterpreterValue wrapped{tensor};
  auto clone = wrapped.clone();
  tensor.at(0) = 4;

  auto& clonedTensor = std::get<TensorOrMemref<int64_t>>(clone.storage);
  ASSERT_EQ(clonedTensor.at(0), 1);
  ASSERT_EQ(clonedTensor.at(1), 2);
  ASSERT_EQ(clonedTensor.at(2), 3);
}

TEST(InterpreterValueTest, CloneWithLayouts) {
  auto tensor = TensorOrMemref<int64_t>::empty({3, 5}, {0, 1});
  tensor.at({2, 4}) = 42;

  InterpreterValue wrapped{tensor};
  auto clone = wrapped.clone();
  ASSERT_EQ(clone.view().strides,
            BufferView::getStridesForLayout({3, 5}, {1, 0}));
  ASSERT_EQ(clone.extractElement({2, 4}).asInt(), 42);
}

TEST(InterpreterValueTest, CoerceLayoutNoop) {
  auto tensor = TensorOrMemref<int64_t>::empty({3, 5}, {0, 1});
  tensor.at({2, 4}) = 42;

  InterpreterValue wrapped{tensor};
  auto coerced = wrapped.coerceLayout({0, 1});
  ASSERT_EQ(tensor.buffer,
            std::get<TensorOrMemref<int64_t>>(coerced.storage).buffer);
}

TEST(InterpreterValueTest, CoerceLayout) {
  auto tensor = TensorOrMemref<int64_t>::empty({3, 5});
  tensor.at({2, 4}) = 42;

  InterpreterValue wrapped{tensor};
  auto clone = wrapped.coerceLayout({0, 1});
  ASSERT_EQ(clone.view().strides,
            BufferView::getStridesForLayout({3, 5}, {0, 1}));
  ASSERT_EQ(clone.extractElement({2, 4}).asInt(), 42);
}

TEST(InterpreterValueTest, CoerceLayoutSquare) {
  auto tensor = TensorOrMemref<float>::empty({2, 2});
  tensor.at({0, 0}) = 1;
  tensor.at({0, 1}) = 2;
  tensor.at({1, 0}) = 3;
  tensor.at({1, 1}) = 4;

  InterpreterValue wrapped{tensor};
  auto clone = wrapped.coerceLayout({0, 1});
  auto& clonedTensor = std::get<TensorOrMemref<float>>(clone.storage);

  EXPECT_EQ(
      *reinterpret_cast<float*>(clonedTensor.buffer->at(0, sizeof(float))), 1);
  EXPECT_EQ(
      *reinterpret_cast<float*>(clonedTensor.buffer->at(1, sizeof(float))), 3);
  EXPECT_EQ(
      *reinterpret_cast<float*>(clonedTensor.buffer->at(2, sizeof(float))), 2);
  EXPECT_EQ(
      *reinterpret_cast<float*>(clonedTensor.buffer->at(3, sizeof(float))), 4);
}

TEST(InterpreterValueTest, CloneScalar) {
  InterpreterValue value{42};
  auto clone = value.clone();
  ASSERT_THAT(std::get<int32_t>(clone.storage), 42);
}

TEST(InterpreterValueTest, ToString) {
  InterpreterValue value{TensorOrMemref<int64_t>::empty({3})};
  ASSERT_EQ(value.toString(), "TensorOrMemref<3xi64>: [0, 0, 0]");
}

TEST(InterpreterValueTest, ToString2d) {
  InterpreterValue value{TensorOrMemref<int64_t>::empty({3, 2})};
  ASSERT_EQ(value.toString(),
            "TensorOrMemref<3x2xi64>: [[0, 0], [0, 0], [0, 0]]");
}

TEST(InterpreterValueTest, ToString0d) {
  InterpreterValue value{TensorOrMemref<int64_t>::empty({})};
  ASSERT_EQ(value.toString(), "TensorOrMemref<i64>: 0");
}

TEST(InterpreterValueTest, ToStringComplex) {
  InterpreterValue value{std::complex<float>{}};
  ASSERT_EQ(value.toString(), "complex<f32>: 0.000000e+00+0.000000e+00i");
}

TEST(InterpreterValueTest, ToStringDeallocated) {
  InterpreterValue value{TensorOrMemref<int64_t>::empty({})};
  value.buffer()->deallocate();
  ASSERT_EQ(value.toString(), "TensorOrMemref<i64>: <<deallocated>>");
}

TEST(CastTest, UnpackTensor) {
  InterpreterValue value{TensorOrMemref<int8_t>::empty({1, 1})};
  value.insertElement({0, 0}, {int8_t{1}});
  ASSERT_EQ(interpreterValueCast<int64_t>(value), 1);
  ASSERT_EQ(interpreterValueCast<uint8_t>(value), 1);
  ASSERT_EQ(interpreterValueCast<float>(value), 1.0f);
  ASSERT_EQ(interpreterValueCast<double>(value), 1.0);

  InterpreterValue nonUnit{TensorOrMemref<int8_t>::empty({2, 2})};
  ASSERT_EQ(interpreterValueDynCast<int64_t>(nonUnit), std::nullopt);
}

TEST(CastTest, IdentityCast) {
  InterpreterValue value{TensorOrMemref<float>::empty({1, 1})};
  ASSERT_EQ(interpreterValueCast<InterpreterValue>(value), value);
}

TEST(CastTest, CastToUnsigned) {
  // Note: This is different from `AsUint`, which preserves the size of the
  // original type (i.e. int8_t{-1} results in 255).
  InterpreterValue value{int8_t{-1}};
  ASSERT_EQ(interpreterValueCast<uint8_t>(value), 255);
  ASSERT_EQ(interpreterValueCast<uint16_t>(value), 65535);
}

}  // namespace
}  // namespace interpreter
}  // namespace mlir
