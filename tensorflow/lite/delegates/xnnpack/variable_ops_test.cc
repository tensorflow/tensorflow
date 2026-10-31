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

TEST(ReadAssignVariable, SimpleAssignOnly) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  VariableOpsTester().NumInputs(1).NumOutputs(0).TestAssignOnly(
      xnnpack_delegate.get());
}

TEST(ReadAssignVariable, SimpleAssignThenRead) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  TfLiteDelegate* delegate = xnnpack_delegate.get();

  VariableOpsTester().NumInputs(1).NumOutputs(1).TestAssignThenRead(delegate);
}

TEST(ReadAssignVariable, AssignTwiceThenRead) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  TfLiteDelegate* delegate = xnnpack_delegate.get();

  VariableOpsTester().NumInputs(2).NumOutputs(1).TestAssignTwiceThenRead(
      delegate);
}

TEST(ReadAssignVariable, SimpleAssignThenReadUsingAnotherVarHandle) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  TfLiteDelegate* delegate = xnnpack_delegate.get();

  VariableOpsTester()
      .NumInputs(1)
      .NumOutputs(1)
      .TestAssignThenReadUsingAnotherVarHandle(delegate);
}

TEST(ReadAssignVariable, TwoVarHandlesAssignThenRead) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  TfLiteDelegate* delegate = xnnpack_delegate.get();

  VariableOpsTester()
      .NumInputs(2)
      .NumOutputs(2)
      .TestTwoVarHandlesAssignThenRead(delegate);
}

TEST(ReadAssignVariable, ValidAssignThenReadSucceeds) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  ASSERT_NE(xnnpack_delegate, nullptr);
  VariableOpsTester().NumInputs(1).NumOutputs(1).TestAssignThenRead(
      xnnpack_delegate.get());
}

TEST(ReadAssignVariable, ResizeAssignOnly) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  VariableOpsTester().NumInputs(1).NumOutputs(0).TestResizeAssignOnly(
      xnnpack_delegate.get());
}

TEST(ReadAssignVariable, ResizeAssignThenRead) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  VariableOpsTester().NumInputs(1).NumOutputs(1).TestResizeAssignThenRead(
      xnnpack_delegate.get());
}

TEST(ReadAssignVariable, RepeatedResizeCyclesHaveNoLeaksOrStaleCapacity) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  VariableOpsTester()
      .NumInputs(1)
      .NumOutputs(1)
      .TestRepeatedResizeCyclesHaveNoLeaksOrStaleCapacity(
          xnnpack_delegate.get());
}

TEST(ReadAssignVariable, RejectsConflictingAliasedVarHandleShapes) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  VariableOpsTester().TestRejectsConflictingAliasedVarHandleShapes(
      xnnpack_delegate.get());
}

TEST(ReadAssignVariable, RejectsConflictingAliasedVarHandleTypes) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  VariableOpsTester().TestRejectsConflictingAliasedVarHandleTypes(
      xnnpack_delegate.get());
}

TEST(ReadAssignVariable, SingleSubgraphRejectsUninitializedRead) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  VariableOpsTester().TestSingleSubgraphRejectsUninitializedRead(
      xnnpack_delegate.get());
}

class VariableOpsGuardRejectTest
    : public ::testing::TestWithParam<VariableOpGuardCase> {};

TEST_P(VariableOpsGuardRejectTest, RejectsInvalidOrOutOfBoundsState) {
  VariableOpsTester().TestGuardRejectCase(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    GuardCases, VariableOpsGuardRejectTest,
    ::testing::Values(
        VariableOpGuardCase::kNullContextInPrepare,
        VariableOpGuardCase::kNullExecutionPlanInPrepare,
        VariableOpGuardCase::kNullParamsInSubgraphCreate,
        VariableOpGuardCase::kNegativeInputTensorsSizeInSubgraphCreate,
        VariableOpGuardCase::kOutOfBoundsOutputTensorIndexInSubgraphCreate,
        VariableOpGuardCase::kNullNodeInputsVarHandle,
        VariableOpGuardCase::kNullNodeInputsAssignVariable,
        VariableOpGuardCase::kNullNodeInputsReadVariable,
        VariableOpGuardCase::kNullNodeOutputsVarHandle,
        VariableOpGuardCase::kNullNodeOutputsAssignVariable,
        VariableOpGuardCase::kNullNodeOutputsReadVariable,
        VariableOpGuardCase::kNegativeInputsSizeVarHandle,
        VariableOpGuardCase::kUnexpectedInputArityVarHandle,
        VariableOpGuardCase::kUnexpectedOutputArityAssignVariable,
        VariableOpGuardCase::kNullVarHandleBuiltinData,
        VariableOpGuardCase::kVarHandleOutputNegativeOne,
        VariableOpGuardCase::kVarHandleOutputNegativeTwo,
        VariableOpGuardCase::kVarHandleOutputOutOfBounds,
        VariableOpGuardCase::kAssignResourceInputNegativeOne,
        VariableOpGuardCase::kAssignResourceInputNegativeTwo,
        VariableOpGuardCase::kAssignResourceInputOutOfBounds,
        VariableOpGuardCase::kAssignValueInputNegativeOne,
        VariableOpGuardCase::kAssignValueInputNegativeTwo,
        VariableOpGuardCase::kAssignValueInputOutOfBounds,
        VariableOpGuardCase::kReadResourceInputNegativeOne,
        VariableOpGuardCase::kReadResourceInputNegativeTwo,
        VariableOpGuardCase::kReadResourceInputOutOfBounds,
        VariableOpGuardCase::kReadValueOutputNegativeOne,
        VariableOpGuardCase::kReadValueOutputNegativeTwo,
        VariableOpGuardCase::kReadValueOutputOutOfBounds,
        VariableOpGuardCase::kWrongResourceTensorType,
        VariableOpGuardCase::kAssignInputNullDims,
        VariableOpGuardCase::kAssignInputScalarRankZero,
        VariableOpGuardCase::kAssignInputNegativeDim,
        VariableOpGuardCase::kAssignInputExcessiveRank,
        VariableOpGuardCase::kAssignInputByteSizeOverflow,
        VariableOpGuardCase::kAssignInputShapeExceedsTensorBytes,
        VariableOpGuardCase::kReadOutputNullDims,
        VariableOpGuardCase::kReadOutputNegativeDim,
        VariableOpGuardCase::kReadOutputExcessiveRank,
        VariableOpGuardCase::kReadOutputByteSizeOverflow,
        VariableOpGuardCase::kReadOutputShapeExceedsTensorBytes,
        VariableOpGuardCase::kConflictingVarHandlesOnSameLocalResourceTensor,
        VariableOpGuardCase::kExternalOutputBytesSmallerThanXnnpackShapeAtInvoke,
        VariableOpGuardCase::kExternalNullDataWithNonZeroXnnpackShapeAtInvoke,
        VariableOpGuardCase::kReshapedExternalInputBytesSmallerThanShapeInPrepare,
        VariableOpGuardCase::kCreateReshapeZeroInputs,
        VariableOpGuardCase::kCreateReshapeOutOfBoundsInput,
        VariableOpGuardCase::kCreateReshapeOutOfBoundsShapeInput,
        VariableOpGuardCase::kCreateMeanZeroInputs,
        VariableOpGuardCase::kCreatePadOutOfBoundsPaddingsInput,
        VariableOpGuardCase::kCreateResizeBilinearZeroInputs,
        VariableOpGuardCase::kCreateSliceLackingInputs,
        VariableOpGuardCase::kCreateStridedSliceLackingInputs,
        VariableOpGuardCase::kCreateSplitMissingInputSlot1,
        VariableOpGuardCase::kCreateSplitOutOfBoundsValueInput,
        VariableOpGuardCase::kCreateSplitOutOfBoundsAxisInput,
        VariableOpGuardCase::kCreateTransposeZeroInputs,
        VariableOpGuardCase::kCreateTransposeOutOfBoundsInput,
        VariableOpGuardCase::kCreateTransposeOutOfBoundsPermInput,
        VariableOpGuardCase::kCreateTransposeConvOutOfBoundsFilterInput,
        VariableOpGuardCase::kCreateTransposeConvOutOfBoundsBiasInput,
        VariableOpGuardCase::kCreateTransposeConvOutOfBoundsOutputShapeInput,
        VariableOpGuardCase::kCreateSliceZeroOutputs,
        VariableOpGuardCase::kCreateStridedSliceZeroOutputs,
        VariableOpGuardCase::kCreateBatchMatMulZeroInputs,
        VariableOpGuardCase::kCreateBatchMatMulZeroOutputs,
        VariableOpGuardCase::kCreateSdpaCustomZeroInputs,
        VariableOpGuardCase::kCreateSdpaCustomZeroOutputs,
        VariableOpGuardCase::kCreateOptionalNegativeInputInDefaultUnaryOp,
        VariableOpGuardCase::kCreateOptionalNegativeOutputInSubgraphCreate,
        VariableOpGuardCase::kCreateLeakyReluNullBuiltinData,
        VariableOpGuardCase::kCreateGeluNullBuiltinData,
        VariableOpGuardCase::kCreateSdpaCustomNegativeInitialDataSize),
    [](const ::testing::TestParamInfo<VariableOpGuardCase>& info) {
      return VariableOpGuardCaseName(info.param);
    });

}  // namespace xnnpack
}  // namespace tflite
