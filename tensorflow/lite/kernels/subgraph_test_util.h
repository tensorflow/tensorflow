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

#include "tensorflow/lite/core/subgraph.h"

namespace tflite {
namespace subgraph_test_util {

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

// Fill a `TfLiteTensor` with a 32-bits integer vector.
// Preconditions:
// * The tensor must have `kTfLiteInt32` type.
// * The tensor must be allocated.
// * The element count of the tensor must be equal to the length or
//   the vector.
void FillIntTensor(TfLiteTensor* tensor, const std::vector<int32_t>& data);

// Check if the shape and data of a tensor is as expected.
void CheckIntTensor(const TfLiteTensor* tensor, const std::vector<int>& shape,
                    const std::vector<int32_t>& data);

}  // namespace subgraph_test_util
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SUBGRAPH_TEST_UTIL_H_
