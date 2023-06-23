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

#include <memory>
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
  void AddConstSubgraph(Subgraph* subgraph);

  void AddReserveSubgraph(Subgraph* subgraph, TensorType element_type);

 private:
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
