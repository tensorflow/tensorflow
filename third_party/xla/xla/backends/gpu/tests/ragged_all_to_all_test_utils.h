/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_TESTS_RAGGED_ALL_TO_ALL_TEST_UTILS_H_
#define XLA_BACKENDS_GPU_TESTS_RAGGED_ALL_TO_ALL_TEST_UTILS_H_

#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "xla/array.h"

namespace xla::gpu {

class RaggedAllToAllTestUtils {
 public:
  // Computes ragged tensor offsets based on the sizes of the ragged rows.
  // sizes shape: [num_replicas, num_peers, num_updates]
  static Array<int64_t> CalculateOffsetsFromSizes(const Array<int64_t>& sizes,
                                                  int64_t gap = 0);

  // Fill the input and output tensors with random data. An all-to-all is
  // effectively a transpose. We generate a chunk of random data for each update
  // of each pair of replicas and write the chunk starting from the (i, j, k)
  // offset of the input tensor and starting from the (j, i, k) offset of the
  // output tensor.
  static void FillWithRandomData(std::vector<Array<float>>& input_data,
                                 std::vector<Array<float>>& output_data,
                                 const Array<int64_t>& input_offsets,
                                 const Array<int64_t>& output_offsets,
                                 const Array<int64_t>& input_sizes);
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TESTS_RAGGED_ALL_TO_ALL_TEST_UTILS_H_
