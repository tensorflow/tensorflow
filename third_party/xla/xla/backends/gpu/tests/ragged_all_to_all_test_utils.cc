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

#include "xla/backends/gpu/tests/ragged_all_to_all_test_utils.h"

#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "xla/array.h"

namespace xla::gpu::ragged_all_to_all {

Array<int64_t> CalculateOffsetsFromSizes(const Array<int64_t>& sizes,
                                         int64_t gap) {
  int64_t num_replicas = sizes.dim(0);
  int64_t num_peers = sizes.dim(1);
  int64_t num_updates = sizes.dim(2);
  Array<int64_t> offsets(sizes.dimensions());
  for (int i = 0; i < num_replicas; ++i) {
    int64_t cur_offset = 0;
    for (int j = 0; j < num_peers; ++j) {
      for (int k = 0; k < num_updates; ++k) {
        offsets(i, j, k) = cur_offset;
        cur_offset += sizes(i, j, k) + gap;
      }
    }
  }
  return offsets;
}

void FillWithRandomData(std::vector<Array<float>>& input_data,
                        std::vector<Array<float>>& output_data,
                        const Array<int64_t>& input_offsets,
                        const Array<int64_t>& output_offsets,
                        const Array<int64_t>& input_sizes) {
  int64_t num_replicas = input_sizes.dim(0);
  int64_t num_updates_per_replica = input_sizes.dim(2);
  std::vector<int64_t> start_indices(input_data[0].num_dimensions());
  std::vector<int64_t> chunk_sizes{input_data[0].dimensions().begin(),
                                   input_data[0].dimensions().end()};

  for (int i = 0; i < num_replicas; ++i) {
    for (int j = 0; j < num_replicas; ++j) {
      for (int k = 0; k < num_updates_per_replica; ++k) {
        chunk_sizes[0] = input_sizes(i, j, k);

        Array<float> chunk_data(chunk_sizes);
        chunk_data.FillRandomUniform(
            1.0f, 127.0f,
            /*seed=*/(i * num_replicas + j) * num_updates_per_replica + k);

        start_indices[0] = input_offsets(i, j, k);
        input_data[i].UpdateSlice(chunk_data, start_indices);

        start_indices[0] = output_offsets(i, j, k);
        output_data[j].UpdateSlice(chunk_data, start_indices);
      }
    }
  }
}

}  // namespace xla::gpu::ragged_all_to_all
