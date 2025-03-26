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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_CONVERT_GRAPH_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_CONVERT_GRAPH_H_

#include <string>
#include <utility>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/conversion.h"

namespace litert {

// Performs iterative graph conversion with user provided hooks. This function
// traverses the IR in toplogical order, converting ops and tensors with given
// tensor converter and legalizations. Registers converted ops and tensors with
// the backend graph builder after they have been converted. The following are
// true:
// * Each tensor and op will be converted & registered at most once.
// * An ops input and output tensors will be registered before the op is
// converted (and before its registered).
// * The graph builder will be initialized before any registration.
// * The graph builder will be finalized after all registration.
template <class Ir>
LiteRtStatus ConvertGraph(
    const Subgraph& subgraph, std::string graph_name,
    typename Ir::TensorConverterFactory tensor_converter_factory,
    typename Ir::TensorAllocator tensor_alloc,
    typename Ir::OpAllocator op_alloc,
    const typename Ir::Legalizations& legalizations,
    typename Ir::GraphBuilder& builder) {
  // Store mapping between evaluated litert tensors and corresponding backend
  // tensors.
  typename Ir::TensorMap tensor_map;

  // Initialize backend graph builder.
  builder.InitGraph(std::move(graph_name));

  // Convert tensor, add to scope and register in backend graph builder.
  auto handle_tensor = [&tensor_map, &builder](
                           const auto& litert_tensor,
                           auto tensor_converter) -> Ir::TensorResult {
    auto converted = tensor_converter(litert_tensor);
    if (!converted) {
      LITERT_LOG(LITERT_ERROR, "Failed to convert tensor %lu",
                 litert_tensor.Get());
      return converted.Error();
    }

    if (auto status = builder.RegisterTensor(**converted);
        status != kLiteRtStatusOk) {
      LITERT_LOG(LITERT_ERROR, "Failed to register tensor %lu, with status %d",
                 litert_tensor.Get(), status);
      return Error(status);
    }

    tensor_map.insert({litert_tensor.Get(), *converted});
    return *converted;
  };

  // Wrap provided tensor conversion logic for converting subgraph or op input
  // tensors. We want functionality that provides user-defined conversions with
  // tensors to be aware of the tensor map and graph builder registration.
  auto input_tensor_convert_factory = [tensor_converter_factory, &tensor_map,
                                       handle_tensor](auto tensor_alloc) {
    return [tensor_alloc, tensor_converter_factory, &tensor_map,
            handle_tensor](const Tensor& litert_tensor) -> Ir::TensorResult {
      auto tensor_converter = tensor_converter_factory(tensor_alloc);

      // Check if tensor has been converted already.
      auto it = tensor_map.find(litert_tensor.Get());
      const auto in_scope = it != tensor_map.end();
      if (in_scope) {
        LITERT_LOG(LITERT_VERBOSE, "Tensor %lu is in scope",
                   litert_tensor.Get());
        return it->second;
      }

      // If its a subgraph input or constant, we can convert it and add to
      // scope.
      const auto is_cst = litert_tensor.IsConstant();
      const auto is_sg_input = litert_tensor.IsSubgraphInput();
      if (is_sg_input || is_cst) {
        return handle_tensor(litert_tensor, tensor_converter);
      }

      // Tensor must be added to scope before conversion, or not have a parent
      // (e.g. subgraph input or constant) so error at this point.
      LITERT_LOG(LITERT_ERROR, "Tensor %lu not handled", litert_tensor.Get());
      return Error(kLiteRtStatusErrorInvalidArgument);
    };
  };

  // Wrap provided tensor conversion logic for op output tensors. Adds to map
  // and backend graph after conversion.
  auto output_tensor_convert_factory = [tensor_converter_factory,
                                        handle_tensor](auto tensor_alloc) {
    return [tensor_alloc, tensor_converter_factory,
            handle_tensor](const Tensor& litert_tensor) {
      auto tensor_converter = tensor_converter_factory(tensor_alloc);
      return handle_tensor(litert_tensor, tensor_converter);
    };
  };

  // Convert all ops in subgraph in toplogical order.
  auto legalization_map = Ir::MakeLegalizationMap(legalizations);
  for (const auto& op : subgraph.Ops()) {
    auto it = legalization_map.find(op.Code());
    if (it == legalization_map.end()) {
      LITERT_LOG(LITERT_ERROR, "No legalization found for op %d", op.Code());
      return kLiteRtStatusErrorUnsupported;
    }

    auto result = it->second->Legalize(op, input_tensor_convert_factory,
                                       output_tensor_convert_factory,
                                       tensor_alloc, op_alloc);
    if (!result) {
      LITERT_LOG(LITERT_ERROR, "Failed to legalize op %d, with status %d",
                 op.Code(), result.Error().Status());
      return result.Error().Status();
    }

    auto simple_result = GetSimpleConversionResult(*result);
    if (simple_result) {
      if (auto stat = builder.RegisterOp(**simple_result);
          stat != kLiteRtStatusOk) {
        LITERT_LOG(LITERT_ERROR, "Failed to register op %d, with status %d",
                   op.Code(), stat);
        return stat;
      }
    }

    auto general_result = GetGeneralConversionResult(*result);
    if (general_result) {
      for (auto* tensor : general_result->intermediate_tensors) {
        if (auto stat = builder.RegisterTensor(*tensor);
            stat != kLiteRtStatusOk) {
          LITERT_LOG(LITERT_ERROR,
                     "Failed to register tensor %d, with status %d", tensor->id,
                     stat);
          return stat;
        }
      }

      for (auto* op : general_result->ops) {
        if (auto stat = builder.RegisterOp(*op); stat != kLiteRtStatusOk) {
          LITERT_LOG(LITERT_ERROR, "Failed to register op %d, with status %d",
                     op->op_code, stat);
          return stat;
        }
      }
    }
  }

  builder.FinalizeGraph();

  return kLiteRtStatusOk;
}

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_CONVERT_GRAPH_H_
