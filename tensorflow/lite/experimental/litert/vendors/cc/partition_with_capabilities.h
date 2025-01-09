// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_PARTITION_WITH_CAPABILITIES_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_PARTITION_WITH_CAPABILITIES_H_

#include <vector>

#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/conversion.h"

namespace litert {

// Higher-level functions for partitioning by leveraging user-defined
// conversions. This method selects ops for partitioning via a callback that
// checks if an op is supported by the backend.

// Selects ops for partitioning from given subgraph based on given Capability
// check. Returns all ops in the given supbgraph that are supported by the
// backend. Suitable for use in implementing LiteRtCompilerPluginPartition. Any
// allocations of new backend ir types will be done through given external
// allocators.
// NOTE: A missing legalization or any legalization failure will result in
// an op not being supported, rather than a failure of this function.
template <class Ir>
Expected<std::vector<LiteRtOp>> PartitionWithCapabilities(
    const typename Ir::Legalizations& legalizations,
    typename Ir::Capability capability,
    typename Ir::TensorConverterFactory convert_tensor_fact,
    typename Ir::TensorAllocator tensor_allocator,
    typename Ir::OpAllocator op_allocator, const Subgraph& litert_subgraph) {
  std::vector<LiteRtOp> results;

  // Build map for legalization lookup by op code.
  auto map = Ir::MakeLegalizationMap(legalizations);

  // Convert all ops from the given subgraph and check backend support.
  for (const auto& litert_op : litert_subgraph.Ops()) {
    const auto code = litert_op.Code();
    LITERT_LOG(LITERT_INFO, "Checking support for LiteRtOp: %d", code);

    auto it = map.find(code);
    if (it == map.end()) {
      LITERT_LOG(LITERT_WARNING, "No legalization found for LiteRtOp: %d",
                 code);
      continue;
    }

    // Call user-defined conversion.
    auto result = it->second->Legalize(litert_op, convert_tensor_fact,
                                       convert_tensor_fact, tensor_allocator,
                                       op_allocator);
    if (!result) {
      LITERT_LOG(LITERT_WARNING, "Failed to legalize LiteRtOp: %d", code);
      continue;
    }

    if (auto simple_result = GetSimpleConversionResult(*result)) {
      if (capability(*simple_result)) {
        LITERT_LOG(LITERT_INFO, "Selected LiteRtOp: %d", litert_op.Code());
        results.push_back(litert_op.Get());
      }
      continue;
    }

    // Check all ops emitted from a one-to-many conversion are supported.
    if (auto gen_result = GetGeneralConversionResult(*result)) {
      const auto b_ops_start = gen_result->ops.cbegin();
      const auto b_ops_end = gen_result->ops.cend();
      if (std::all_of(b_ops_start, b_ops_end, capability)) {
        LITERT_LOG(LITERT_INFO, "Selected LiteRtOp: %d", litert_op.Code());
        results.push_back(litert_op.Get());
      }
      continue;
    }
  }

  return results;
}

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_PARTITION_WITH_CAPABILITIES_H_
