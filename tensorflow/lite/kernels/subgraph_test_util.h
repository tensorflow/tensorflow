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
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"

namespace tflite {
namespace subgraph_test_util {

// TODO(ycling): This file should be renamed as
// `control_flow_test_util` to avoid confusion. I'll do it immediately
// in a separated change.
class SubgraphBuilder {
 public:
  ~SubgraphBuilder();

  // Build a subgraph with a single Add op.
  // 2 inputs. 1 output.
  void BuildAddSubgraph(Subgraph* subgraph);

  // Build a subgraph with a single Mul op.
  // 2 inputs. 1 output.
  void BuildMulSubgraph(Subgraph* subgraph);

  // Build a subgraph with a single Pad op.
  // 2 inputs. 1 output.
  void BuildPadSubgraph(Subgraph* subgraph);

  // Build a subgraph with a single If op.
  // 3 inputs:
  //   The 1st input is condition with boolean type.
  //   The 2nd and 3rd inputs are feed input the branch subgraphs.
  // 1 output.
  void BuildIfSubgraph(Subgraph* subgraph);

  // Build a subgraph with a single Less op.
  // The subgraph is used as the condition subgraph for testing `While` op.
  // 2 inputs:
  //   The 1st input is a counter with `kTfLiteInt32` type.
  //   The 2nd input is ignored in this subgraph.
  // 1 output with `kTfLiteBool` type.
  //   Equivalent to (input < rhs).
  void BuildLessEqualCondSubgraph(Subgraph* subgraph, int rhs);

  // An accumulate loop body subgraph. Used to produce triangle number
  // sequence. 2 inputs and 2 outputs
  //   Equivalent to (counter, value) -> (counter + 1, counter + 1 + value)
  void BuildAccumulateLoopBodySubgraph(Subgraph* subgraph);

  // A pad loop body subgraph. When used in a loop it will repeatively enlarge
  // the
  //   tensor.
  // 2 inputs and 2 outputs.
  //   Equivalent to (counter, value) -> (counter + 1, tf.pad(value, padding))
  // Note the padding is created as a constant tensor.
  void BuildPadLoopBodySubgraph(Subgraph* subgraph,
                                const std::vector<int> padding);

  // Build a subgraph with a single While op.
  // 2 inputs, 2 outputs.
  void BuildWhileSubgraph(Subgraph* subgraph);

 private:
  void CreateConstantInt32Tensor(Subgraph* subgraph, int tensor_index,
                                 const std::vector<int>& shape,
                                 const std::vector<int>& data);
  std::vector<void*> buffers_;
};

class ControlFlowOpTest : public ::testing::Test {
 public:
  ControlFlowOpTest()
      : interpreter_(new Interpreter), builder_(new SubgraphBuilder) {}

  ~ControlFlowOpTest() override {
    interpreter_.reset();
    builder_.reset();
  }

 protected:
  std::unique_ptr<Interpreter> interpreter_;
  std::unique_ptr<SubgraphBuilder> builder_;
};

// Fill a `TfLiteTensor` with a 32-bits integer vector.
// Preconditions:
// * The tensor must have `kTfLiteInt32` type.
// * The tensor must be allocated.
// * The element count of the tensor must be equal to the length or
//   the vector.
void FillIntTensor(TfLiteTensor* tensor, const std::vector<int32_t>& data);

// Check if the shape and int32 data of a tensor is as expected.
void CheckIntTensor(const TfLiteTensor* tensor, const std::vector<int>& shape,
                    const std::vector<int32_t>& data);
// Check if the shape and bool data of a tensor is as expected.
void CheckBoolTensor(const TfLiteTensor* tensor, const std::vector<int>& shape,
                     const std::vector<bool>& data);

}  // namespace subgraph_test_util
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SUBGRAPH_TEST_UTIL_H_
