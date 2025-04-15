/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This module provides helper functions for testing the interaction between
// control flow ops and subgraphs.
// For convenience, we mostly only use `kTfLiteInt32` in this module.

#ifndef TENSORFLOW_LITE_KERNELS_SUBGRAPH_TEST_UTIL_H_
#define TENSORFLOW_LITE_KERNELS_SUBGRAPH_TEST_UTIL_H_

#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter_test_util.h"

namespace tflite {
namespace subgraph_test_util {

class SubgraphBuilder {
 public:
  ~SubgraphBuilder();

  // Build a subgraph with ops which support memory sharing.
  // An ADD node consumes a subgraph input so cannot be shared. RESHAPE consumes
  // the output of ADD and may share. The second ADD can't share as it produces
  // a subgraph output.
  void BuildInplaceOpSubgraph(Subgraph* subgraph);

  // Build a subgraph with broadcasting elementwise ops, some of which support
  // sharing and some not.
  void BuildBroadcastingSubgraph(Subgraph* subgraph);

  // Build a subgraph with fictional op OFFSET_ADD which supports sharing of
  // the second input but not the first.
  void BuildOffsetAddSharing(Subgraph* subgraph);

  // Build a subgraph with a dynamic update slice op which operates on
  // a subgraph input tensor. The input buffer cannot be shared with the output.
  void BuildInputDynamicUpdateSliceSubgraph(Subgraph& subgraph);

  // Build a subgraph with a dynamic update slice op which operates on
  // an intermediate tensor. The input buffer can be shared with the output if
  // multiple nodes do not consume the input tensor.
  void BuildInplaceDynamicUpdateSliceSubgraph(Subgraph& subgraph,
                                              bool multiple_consumers);

  // Build a subgraph whose output is not consumed by the parent subgraph.
  void BuildOutputNotConsumedSubgraph(Subgraph& subgraph);

  // Build an if subgraph with float inputs and outputs.
  void BuildFloatIfSubgraph(Subgraph* subgraph, int num_inputs);

  // Build a while subgraph with float inputs and outputs.
  void BuildFloatWhileSubgraph(Subgraph* subgraph, int num_inputs);

  // Build a while body subgraph with delegates to XNNPACK.
  void BuildXNNPACKSubgraph(Subgraph* subgraph);

  // Build a cond subgraph comparing float values.
  void BuildFloatLessCondSubgraph(Subgraph* subgraph, float rhs);

  // Build a body subgraph with a tensor which is both an input and an output.
  void BuildInputIsOutputSubgraph(Subgraph* subgraph);

  // Build a body subgraph with a tensor which is both an input and an output
  // but in different positions.
  void BuildInputIsDifferentOutputSubgraph(Subgraph* subgraph);

  // Build a body subgraph whose output is written by a flex node.
  void BuildFlexOutputSubgraph(Subgraph* subgraph);

  // Build a body subgraph with only a counter.
  void BuildCounterOnlySubgraph(Subgraph* subgraph);

  // Build a subgraph with a single binary op.
  //
  // The op must take in two inputs and have one output.
  void BuildBinaryOpSubgraph(Subgraph* subgraph,
                             TfLiteRegistration* (*Register_OP)(),
                             TfLiteBuiltinOperator builtin_code, void* params,
                             TfLiteType input1_type, TfLiteType input2_type,
                             TfLiteType output_type);

  // Build a subgraph with a single Add op.
  // 2 inputs. 1 output.
  void BuildAddSubgraph(Subgraph* subgraph,
                        TfLiteType operand_type = kTfLiteInt32);

  // Build a subgraph with a single stablehlo Add op.
  // 2 inputs. 1 output.
  void BuildStablehloAddSubgraph(Subgraph* subgraph,
                                 TfLiteType operand_type = kTfLiteInt32);

  // Build a subgraph with a single Maximum op.
  // 2 inputs. 1 output.
  void BuildMaximumSubgraph(Subgraph* subgraph,
                            TfLiteType operand_type = kTfLiteInt32);

  // Build a subgraph with a single stablehlo Maximum op.
  // 2 inputs. 1 output.
  void BuildStablehloMaximumSubgraph(Subgraph* subgraph,
                                     TfLiteType operand_type = kTfLiteInt32);

  // Build a subgraph with a single Minimum op.
  // 2 inputs. 1 output.
  void BuildMinimumSubgraph(Subgraph* subgraph,
                            TfLiteType operand_type = kTfLiteInt32);

  // Build a subgraph with a single stablehlo Minimum op.
  // 2 inputs. 1 output.
  void BuildStablehloMinimumSubgraph(Subgraph* subgraph,
                                     TfLiteType operand_type = kTfLiteInt32);

  // Build a subgraph with a single LogicalOr op.
  // 2 inputs. 1 output.
  void BuildLogicalOrSubgraph(Subgraph* subgraph);

  // Build a subgraph with a single LogicalAnd op.
  // 2 inputs. 1 output.
  void BuildLogicalAndSubgraph(Subgraph* subgraph);

  // Build a subgraph with no ops inside.
  // 2 inputs. 1 output. Routes the second input to the output.
  void BuildOutputIsSecondInputSubgraph(Subgraph* subgraph);

  // Build a subgraph with a single Mul op.
  // 2 inputs. 1 output.
  void BuildMulSubgraph(Subgraph* subgraph,
                        TfLiteType operand_type = kTfLiteInt32);

  // Build a subgraph with a single stablehlo Multiply op.
  // 2 inputs. 1 output.
  void BuildStablehloMulSubgraph(Subgraph* subgraph,
                                 TfLiteType operand_type = kTfLiteInt32);

  // Build a subgraph with a single Pad op.
  // 2 inputs. 1 output.
  void BuildPadSubgraph(Subgraph* subgraph);

  // Build a subgraph with a single If op.
  // 3 inputs:
  //   The 1st input is condition with boolean type.
  //   The 2nd and 3rd inputs are feed input the branch subgraphs.
  // 1 output.
  void BuildIfSubgraph(Subgraph* subgraph);

  // Build a subgraph with a single StableHLO Composite op.
  void BuildCompositeSubgraph(Subgraph* subgraph,
                              const Subgraph* decomposition);

  // Build a subgraph which triggers the reallocation of an inplace output
  // tensor whose corresponding input has not been consumed yet. This tests that
  // the input pointer has be updated.
  void BuildDynamicOpTriggersAllocationOfUnsedInputSubgraph(Subgraph* subgraph);

  // Build a body subgraph which tests all potential inplace write scenarios.
  void BuildAllInplaceScenariosSubgraph(Subgraph* subgraph);
  // Build a subgraph with a single Less op.
  // The subgraph is used as the condition subgraph for testing `While` op.
  // 2 inputs:
  //   The 1st input is a counter with `kTfLiteInt32` type.
  //   The 2nd input is ignored in this subgraph.
  // 1 output with `kTfLiteBool` type.
  //   Equivalent to (input < rhs).
  void BuildLessEqualCondSubgraph(Subgraph* subgraph, int rhs);

  // Build an if subgraph which does not consume an output of ifs body
  // subgraph.
  void BuildOutputNotConsumedIfSubgraph(Subgraph* subgraph);

  // Build a while subgraph which does not consume an output of ifs body
  // subgraph.
  void BuildOutputNotConsumedWhileSubgraph(Subgraph* subgraph);

  // Build a if subgraph with multiple inputs.
  void BuildMultiInputIfSubgraph(Subgraph* subgraph, int num_inputs);

  // Build a while subgraph with multiple inputs.
  void BuildMultiInputWhileSubgraph(Subgraph* subgraph, int num_inputs);

  // Build an if subgraph with multiple inputs and one output which is not
  // consumed.
  void BuildMultiInputIfSubgraphWithUnconsumedOutput(Subgraph* subgraph,
                                                     int num_inputs);

  // Build a while subgraph with multiple inputs and one output which is not
  // consumed.
  void BuildMultiInputWhileSubgraphWithUnconsumedOutput(Subgraph* subgraph,
                                                        int num_inputs);

  // Build a dynamic body subgraph with output tensor aliases.
  void BuildDynamicBodySubgraphWithAliases(Subgraph* subgraph);

  // Build a condition subgraph with a variable number of inputs and outputs.
  void BuildLargeLessEqualCondSubgraph(Subgraph* subgraph, int rhs,
                                       int num_inputs);

  // An accumulate loop body subgraph. Used to produce triangle number
  // sequence. 2 inputs and 2 outputs
  //   Equivalent to (counter, value) -> (counter + 1, counter + 1 + value)
  void BuildAccumulateLoopBodySubgraph(Subgraph* subgraph);

  // An loop body subgraph in which the inputs and outputs may be shared.
  void BuildDeepBodySubgraph(Subgraph* subgraph);

  // A loop body subgraph with arbitrary sized inputs which may be shared.
  void BuildLargeBodySubgraph(Subgraph* subgraph);

  // Build a body subgraph whose output size increases each iteration.
  void BuildDynamicIncreasingSizeSubgraph(Subgraph* subgraph);

  // Build a body subgraph which increasing output size whose input and output
  // tensors could potentially share buffers.
  void BuildLargePadSubgraph(Subgraph* subgraph, std::vector<int> padding);

  // A pad loop body subgraph. When used in a loop it will repeatively enlarge
  // the
  //   tensor.
  // 2 inputs and 2 outputs.
  //   Equivalent to (counter, value) -> (counter + 1, tf.pad(value, padding))
  // Note the padding is created as a constant tensor.
  void BuildPadLoopBodySubgraph(Subgraph* subgraph,
                                const std::vector<int>& padding);

  // Build a subgraph with a single While op.
  // 2 inputs, 2 outputs.
  void BuildWhileSubgraph(Subgraph* subgraph);

  // Build a subgraph that assigns a random value to a variable.
  // No input/output.
  void BuildAssignRandomValueToVariableSubgraph(Subgraph* graph);

  // Build a subgraph with CallOnce op and ReadVariable op.
  // No input and 1 output.
  void BuildCallOnceAndReadVariableSubgraph(Subgraph* graph);

  // Build a subgraph with CallOnce op, ReadVariable op and Add op.
  // No input and 1 output.
  void BuildCallOnceAndReadVariablePlusOneSubgraph(Subgraph* graph);

  // Build a subgraph with a single Less op.
  // The subgraph is used as the condition subgraph for testing `While` op.
  // 3 inputs:
  //   The 1st and 2nd inputs are string tensors, which will be ignored.
  //   The 3rd input is an integner value as a counter in this subgraph.
  // 1 output with `kTfLiteBool` type.
  //   Equivalent to (int_val < rhs).
  void BuildLessEqualCondSubgraphWithDynamicTensor(Subgraph* subgraph, int rhs);

  // Build a subgraph with a single While op, which has 3 inputs and 3 outputs.
  // This subgraph is used for creating/invoking dynamic allocated tensors based
  // on string tensors.
  //   Equivalent to (str1, str2, int_val) ->
  //                 (str1, Fill(str1, int_val + 1), int_val + 1).
  void BuildBodySubgraphWithDynamicTensor(Subgraph* subgraph);

  // Build a subgraph with a single If op, that contains 4 inputs and 3
  // outputs (str1, str2, int_val).
  void BuildIfSubgraphWithDynamicTensor(Subgraph* subgraph);

  // Build a subgraph with a single While op, that contains 3 inputs and 3
  // outputs (str1, str2, int_val).
  void BuildWhileSubgraphWithDynamicTensor(Subgraph* subgraph);

 private:
  template <typename T = int32_t>
  void CreateConstantTensor(Subgraph* subgraph, int tensor_index,
                            const std::vector<int>& shape,
                            const std::vector<T>& data) {
    ASSERT_GT(shape.size(), 0);
    const int num_elements = absl::c_accumulate(shape, 1, std::multiplies<>());
    ASSERT_EQ(data.size(), num_elements);
    const size_t size_in_bytes = sizeof(T) * num_elements;
    // Maybe aligned.
    T* buffer = reinterpret_cast<T*>(malloc(size_in_bytes));
    memcpy(buffer, data.data(), size_in_bytes);
    buffers_.push_back(buffer);
    ASSERT_EQ(subgraph->SetTensorParametersReadOnly(
                  tensor_index, typeToTfLiteType<T>(), "", shape, {},
                  reinterpret_cast<const char*>(buffer), size_in_bytes),
              kTfLiteOk);
  }

  std::vector<void*> buffers_;
};

class ControlFlowOpTest : public InterpreterTest {
 public:
  ControlFlowOpTest() : builder_(new SubgraphBuilder) {}

  ~ControlFlowOpTest() override { builder_.reset(); }

 protected:
  std::unique_ptr<SubgraphBuilder> builder_;
};

// Fill a `TfLiteTensor` with a 32-bits integer vector.
// Preconditions:
// * The tensor must have `kTfLiteInt32` type.
// * The tensor must be allocated.
// * The element count of the tensor must be equal to the length or
//   the vector.
void FillIntTensor(TfLiteTensor* tensor, const std::vector<int32_t>& data);

// Fill a `TfLiteTensor` with a string value.
// Preconditions:
// * The tensor must have `kTfLitString` type.
void FillScalarStringTensor(TfLiteTensor* tensor, const std::string& data);

// Check if the scalar string data of a tensor is as expected.
void CheckScalarStringTensor(const TfLiteTensor* tensor,
                             const std::string& data);

// Check if the shape and string data of a tensor is as expected.
void CheckStringTensor(const TfLiteTensor* tensor,
                       const std::vector<int>& shape,
                       const std::vector<std::string>& data);

// Check if the shape and int32 data of a tensor is as expected.
void CheckIntTensor(const TfLiteTensor* tensor, const std::vector<int>& shape,
                    const std::vector<int32_t>& data);
// Check if the shape and bool data of a tensor is as expected.
void CheckBoolTensor(const TfLiteTensor* tensor, const std::vector<int>& shape,
                     const std::vector<bool>& data);

// Sets the tensor to be readable and writable. Call this on input
// tensors when constructing Subgraphs to test.
void SetupTensor(Subgraph* subgraph, int tensor_index, TfLiteType type);

}  // namespace subgraph_test_util
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SUBGRAPH_TEST_UTIL_H_
