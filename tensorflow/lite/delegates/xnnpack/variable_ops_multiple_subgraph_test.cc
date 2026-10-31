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

#include <memory>

#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/xnnpack/variable_ops_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace tflite {
namespace xnnpack {

TEST(ReadAssignVariable, TwoSubgraphsReadAssign) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  TfLiteDelegate* delegate = xnnpack_delegate.get();

  VariableOpsTester()
      .NumInputs(0)
      .NumOutputs(2)
      .NumSubgraphs(2)
      .TestTwoSubgraphsReadAssign(delegate);
}

TEST(ReadAssignVariable, TwoSubgraphsReadAssignOneVarHandle) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  TfLiteDelegate* delegate = xnnpack_delegate.get();

  VariableOpsTester()
      .NumInputs(0)
      .NumOutputs(1)
      .NumSubgraphs(2)
      .TestTwoSubgraphsReadAssignOneVarHandle(delegate);
}

TEST(ReadAssignVariable, TwoSubgraphsReadAssignOneVarHandle2) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  TfLiteDelegate* delegate = xnnpack_delegate.get();

  VariableOpsTester()
      .NumInputs(0)
      .NumOutputs(1)
      .NumSubgraphs(2)
      .TestTwoSubgraphsReadAssignOneVarHandle2(delegate);
}

class MalformedVariableOpModelTest
    : public ::testing::TestWithParam<MalformedVariableOpModelCase> {};

TEST_P(MalformedVariableOpModelTest, RejectsMalformedFlatBufferModel) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  VariableOpsTester().TestMalformedVariableOpModelCase(xnnpack_delegate.get(),
                                                       GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    MalformedFlatBufferCases, MalformedVariableOpModelTest,
    ::testing::Values(
        MalformedVariableOpModelCase::kVarHandleZeroOutputs,
        MalformedVariableOpModelCase::kReadVariableZeroInputs,
        MalformedVariableOpModelCase::kReadVariableZeroOutputs,
        MalformedVariableOpModelCase::kAssignVariableZeroInputs,
        MalformedVariableOpModelCase::kAssignVariableOneInput,
        MalformedVariableOpModelCase::kAssignVariableUnexpectedOutput,
        MalformedVariableOpModelCase::kAssignVariableNegativeValueInput,
        MalformedVariableOpModelCase::kAssignVariableOutOfBoundsValueInput,
        MalformedVariableOpModelCase::kReadVariableNegativeValueOutput,
        MalformedVariableOpModelCase::kReadVariableOutOfBoundsValueOutput,
        MalformedVariableOpModelCase::
            kVarHandleAndReadVariableNegativeResourceWire,
        MalformedVariableOpModelCase::
            kVarHandleAndAssignVariableNegativeResourceWire),
    [](const ::testing::TestParamInfo<MalformedVariableOpModelCase>& info) {
      return MalformedVariableOpModelCaseName(info.param);
    });

TEST(ReadAssignVariable, TwoSubgraphsRejectsMismatchedShape) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  VariableOpsTester().TestTwoSubgraphsRejectsMismatchedShape(
      xnnpack_delegate.get());
}

TEST(ReadAssignVariable, TwoSubgraphsRejectsUninitializedRead) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  VariableOpsTester().TestTwoSubgraphsRejectsUninitializedRead(
      xnnpack_delegate.get());
}

TEST(ReadAssignVariable,
     ConflictingLocalResourceVarHandleFallsBackWithoutCorruptingFirstVar) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  VariableOpsTester()
      .TestConflictingLocalResourceVarHandleFallsBackWithoutCorruptingFirstVar(
          xnnpack_delegate.get());
}

}  // namespace xnnpack
}  // namespace tflite
