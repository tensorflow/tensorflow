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

#include "tensorflow/compiler/xla/mlir/tools/mlir_replay/public/execution_trace_utils.h"

#include <cmath>
#include <complex>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "llvm/ADT/STLExtras.h"
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/mlir_hlo/tools/mlir_interpreter/framework/interpreter_value.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace mlir {
namespace interpreter {
namespace {

class TracedValueRoundTripTest
    : public ::testing::TestWithParam<InterpreterValue> {};

TEST_P(TracedValueRoundTripTest, Run) {
  auto traced_value = ValueToTracedValue(GetParam());
  TF_ASSERT_OK_AND_ASSIGN(auto value, TracedValueToValue(traced_value));
  EXPECT_EQ(GetParam(), value) << GetParam().toString();
}

template <typename T>
InterpreterValue MakeTensor(ArrayRef<int64_t> shape, ArrayRef<T> values) {
  auto result = TensorOrMemref<T>::empty(shape);
  for (auto [indices, value] : llvm::zip(result.view.indices(), values)) {
    result.at(indices) = value;
  }
  return {result};
}

template <typename T>
std::shared_ptr<T> WrapShared(T value) {
  return std::make_shared<T>(std::move(value));
}

INSTANTIATE_TEST_SUITE_P(
    RoundTrip, TracedValueRoundTripTest,
    ::testing::ValuesIn(std::vector<InterpreterValue>{
        {uint8_t{42}},
        {uint16_t{43}},
        {uint32_t{44}},
        {uint64_t{45}},
        {int8_t{-47}},
        {int16_t{-48}},
        {int32_t{-49}},
        {int64_t{-50}},
        {float{42.0}},
        {double{42.0}},
        {std::complex<float>{1.0, 2.0}},
        {std::complex<double>{3.0, 4.0}},
        {true},
        {false},
        {MakeTensor<int16_t>({1, 2}, {42, 43})},
        {MakeTensor<double>({2, 2}, {1.0, -INFINITY, INFINITY, NAN})},
        {MakeTensor<std::complex<double>>({}, {{1.0, 2.0}})},
        {Tuple{SmallVector<std::shared_ptr<InterpreterValue>>{
            WrapShared(InterpreterValue{42}),
            WrapShared(InterpreterValue{43.0}),
        }}}}));

class FromLiteralTest
    : public ::testing::TestWithParam<
          std::pair<std::shared_ptr<xla::Literal>, InterpreterValue>> {};

TEST_P(FromLiteralTest, Run) {
  TF_ASSERT_OK_AND_ASSIGN(auto value, LiteralToValue(*GetParam().first));
  EXPECT_EQ(value, GetParam().second)
      << value.toString() << " vs " << GetParam().second.toString();
}

std::vector<std::pair<std::shared_ptr<xla::Literal>, InterpreterValue>>
MakeInputs() {
  using ::xla::LiteralUtil;
  return {
      {WrapShared(LiteralUtil::CreateR2<uint8_t>({{41, 42}})),
       MakeTensor<uint8_t>({1, 2}, {41, 42})},
      {WrapShared(LiteralUtil::CreateR0<uint16_t>(43)),
       MakeTensor<uint16_t>({}, {43})},
      {WrapShared(LiteralUtil::CreateR0<uint32_t>(44)),
       MakeTensor<uint32_t>({}, {44})},
      {WrapShared(LiteralUtil::CreateR0<uint64_t>(45)),
       MakeTensor<uint64_t>({}, {45})},
      {WrapShared(LiteralUtil::CreateR0<int8_t>(46)),
       MakeTensor<int8_t>({}, {46})},
      {WrapShared(LiteralUtil::CreateR0<int16_t>(47)),
       MakeTensor<int16_t>({}, {47})},
      {WrapShared(LiteralUtil::CreateR0<int32_t>(48)),
       MakeTensor<int32_t>({}, {48})},
      {WrapShared(LiteralUtil::CreateR0<int64_t>(49)),
       MakeTensor<int64_t>({}, {49})},
      {WrapShared(LiteralUtil::CreateR0<float>(50.0)),
       MakeTensor<float>({}, {50.0})},
      {WrapShared(LiteralUtil::CreateR0<double>(51.0)),
       MakeTensor<double>({}, {51.0})},
      {WrapShared(LiteralUtil::CreateR0<std::complex<float>>({52.0, 53.0})),
       MakeTensor<std::complex<float>>({}, {{52.0, 53.0}})},
      {WrapShared(LiteralUtil::CreateR0<std::complex<double>>({54.0, 55.0})),
       MakeTensor<std::complex<double>>({}, {{54.0, 55.0}})},
      {WrapShared(LiteralUtil::CreateR1<bool>({true, false})),
       MakeTensor<bool>({2}, {true, false})},
      {WrapShared(
           LiteralUtil::MakeTupleOwned(LiteralUtil::CreateR0<bool>(true),
                                       LiteralUtil::CreateR0<int8_t>(56))),
       InterpreterValue{Tuple{SmallVector<std::shared_ptr<InterpreterValue>>{
           std::make_shared<InterpreterValue>(MakeTensor<bool>({}, {true})),
           std::make_shared<InterpreterValue>(
               MakeTensor<int8_t>({}, {56}))}}}}};
}

INSTANTIATE_TEST_SUITE_P(Test, FromLiteralTest,
                         ::testing::ValuesIn(MakeInputs()));

}  // namespace
}  // namespace interpreter
}  // namespace mlir
