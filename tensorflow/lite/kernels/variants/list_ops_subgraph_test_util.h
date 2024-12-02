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
#ifndef TENSORFLOW_LITE_KERNELS_VARIANTS_LIST_OPS_SUBGRAPH_TEST_UTIL_H_
#define TENSORFLOW_LITE_KERNELS_VARIANTS_LIST_OPS_SUBGRAPH_TEST_UTIL_H_

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/variants/list_ops_lib.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

// Helper class for constructing complicated subgraphs for testing.
class ListOpsSubgraphBuilder {
 public:
  // Populates the given Subgraph with ops to add the value of two constants
  // created by `CreateConstantInt32Tensor`.
  void BuildAddConstSubgraph(Subgraph* subgraph);

  // Populates the given Subgraph with a "ListReserve" op whose elements
  // have type `element_type`.
  void BuildReserveSubgraph(Subgraph* subgraph, TensorType element_type);

  // Populates the given Subgraph with "ListStack" op that takes in
  // a "ListReserve". Element types are i32.
  void BuildReserveStackSubgraph(Subgraph* subgraph);

  // Populates the given Subgraph with a "While" op, whose cond and body
  // subgraphs are located at index 1, 2 respectively. The input signatures
  // of both subgraphs 1, 2 are expected to be (kTfLiteInt32, kTfLiteVariant).
  void BuildWhileSubgraph(Subgraph* subgraph);

  // Populates the given Subgraph with a single "Less" op which checks
  // if a given int is less that the constant 3. Also takes a `kTfLiteVariant`
  // tensor in order to be compliant with `BuildWhileSubgraph`.
  void BuildLessThanSubgraph(Subgraph* subgraph);

  // Populates the given Subgraph with a "ListSetItem" op which sets the element
  // at given indice into given tensorlist. Additionally increment and return
  // the given int by 1.
  void BuildSetItemAndIncrementSubgraph(Subgraph* subgraph);

  // Populates the given Subgraph with a "ListReserve" and "ListLength" op.
  void BuildReserveLengthSubgraph(Subgraph* subgraph);

 private:
  // Creates a constant tensor in given Subgraphs at given indice with
  // corresponding data.
  void CreateConstantInt32Tensor(Subgraph* subgraph, int tensor_index,
                                 absl::Span<const int> shape,
                                 absl::Span<const int> data);

  // Custom options usually live in the flatbuffer, so they won't
  // be cleaned up by the `Interpreter`. When we create them in test
  // we need to free when test is done. So we provide a factory function
  // for construction.
  variants::detail::ListReserveOptions* RequestReserveOptions(
      TensorType element_type);
  std::vector<variants::detail::ListReserveOptions> list_reserve_opts_;

  std::vector<std::vector<int32_t>> int_buffers_;
};

class ListOpsSubgraphTest : public ::testing::Test {
 protected:
  Interpreter interpreter_;
  ListOpsSubgraphBuilder builder_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_VARIANTS_LIST_OPS_SUBGRAPH_TEST_UTIL_H_
