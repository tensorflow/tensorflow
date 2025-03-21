/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/costs/cost_estimator.h"

namespace tensorflow {
namespace grappler {

Costs CombineCosts(const Costs& left, const Costs& right) {
  CHECK_NE(left.max_memory, kMemoryUnknown);
  CHECK_NE(left.max_per_op_buffers, kMemoryUnknown);
  CHECK_NE(left.max_per_op_streaming, kMemoryUnknown);

  Costs result = left;
  result.execution_time += right.execution_time;
  result.compute_time += right.compute_time;
  result.memory_time += right.memory_time;
  result.network_time += right.network_time;
  result.intermediate_memory_time += right.intermediate_memory_time;
  result.intermediate_memory_read_time += right.intermediate_memory_read_time;
  result.intermediate_memory_write_time += right.intermediate_memory_write_time;
  result.hbm_read_time += right.hbm_read_time;
  result.hbm_write_time += right.hbm_write_time;
  result.hbm_read_time_noderate += right.hbm_read_time_noderate;
  result.hbm_write_time_noderate += right.hbm_write_time_noderate;

  if (right.max_per_op_buffers != kMemoryUnknown) {
    result.max_per_op_buffers =
        std::max(left.max_per_op_buffers, right.max_per_op_buffers);
  }
  if (right.max_per_op_streaming != kMemoryUnknown) {
    result.max_per_op_streaming =
        std::max(left.max_per_op_streaming, right.max_per_op_streaming);
  }

  result.num_ops_total += right.num_ops_total;
  if (right.inaccurate) {
    result.inaccurate = true;
  }
  result.num_ops_with_unknown_shapes += right.num_ops_with_unknown_shapes;
  if (right.max_memory != kMemoryUnknown) {
    result.max_memory += right.max_memory;
  }

  return result;
}

// Multiplies Costs by a scalar.
// Equivalent to applying CombineCosts "multiplier" times.
// Note the field regarding num_ops and max_memory are not multiplied.
Costs MultiplyCosts(const Costs& costs, int multiplier) {
  CHECK_GE(multiplier, 0);
  if (multiplier == 0) {
    return Costs::ZeroCosts();
  }
  if (multiplier == 1) {
    return costs;
  }

  Costs result = costs;
  result.execution_time *= multiplier;
  result.compute_time *= multiplier;
  result.memory_time *= multiplier;
  result.hbm_read_time *= multiplier;
  result.hbm_write_time *= multiplier;
  result.hbm_read_time_noderate *= multiplier;
  result.hbm_write_time_noderate *= multiplier;
  result.network_time *= multiplier;
  result.intermediate_memory_time *= multiplier;
  result.intermediate_memory_read_time *= multiplier;
  result.intermediate_memory_write_time *= multiplier;
  return result;
}

}  // end namespace grappler
}  // end namespace tensorflow
