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

// Utility types for mapping LiteRt IR to arbitrary backend specific
// types. Implementations of these types define mapping for ops and tensors
// that may be used in a stndalone fashion. They also may be composed
// to create lowerings of entire graphs with topology.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_CONVERSION_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_CONVERSION_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/backend_ir.h"

namespace litert {

// Interfaces and types for implementing "conversions" that map LiteRt IR to
// backend IR.
// NOTE: Conversions depend on external memory management for the backend IR
// types. User defined conversions are usually expected to leverage callbacks
// to allocate backend IR types rather than constructing them directly.

// Conversion Result Type
//===---------------------------------------------------------------------------

// Result of a one->many general mapping from LiteRt op to any number of
// backend specific ops. Does not own the memory of the backend ops or tensors.
template <class BackendOp, class BackendTensor>
struct GeneralConversionResult {
  // Ops emitted from translation pattern.
  std::vector<BackendOp*> ops;

  // Any backend tensors used within the results ops. Not relevant when
  // size of backend ops == 1. This does not include input/output tensors of the
  // op being converted.
  std::vector<BackendTensor*> intermediate_tensors;
};

// The result of a one->one specialized mapping from LiteRt op to backend op.
template <class BackendOp>
using SimpleConversionResult = BackendOp*;

// A tag-type for a conversion result that is a non-error non-match.
struct NoMatch {};

// Type union for conversion results.
// TODO(lukeboyer): Update conversion result types to handle the case where
// backend ops add extra inputs.
template <class BackendOp, class BackendTensor>
using ConversionResult =
    std::variant<SimpleConversionResult<BackendOp>,
                 GeneralConversionResult<BackendOp, BackendTensor>, NoMatch>;

// Short hand for holds_alternative.
template <class Result, class BackendOp, class BackendTensor>
bool ConversionIsA(const ConversionResult<BackendOp, BackendTensor>& result) {
  return std::holds_alternative<Result>(result);
}

// Short hand for holds_alternative.
template <class BackendOp, class BackendTensor>
bool ConversionMatched(
    const ConversionResult<BackendOp, BackendTensor>& result) {
  return !std::holds_alternative<NoMatch>(result);
}

// Short hand for holds_alternative.
template <class BackendOp, class BackendTensor>
bool IsSimpleResult(const ConversionResult<BackendOp, BackendTensor>& result) {
  return ConversionIsA<SimpleConversionResult<BackendOp>>(result);
}

// Short hand for holds_alternative.
template <class BackendOp, class BackendTensor>
bool IsGeneralResult(const ConversionResult<BackendOp, BackendTensor>& result) {
  return ConversionIsA<GeneralConversionResult<BackendOp, BackendTensor>>(
      result);
}

// Short hand for std::get. Also checks if match and wraps in expected.
template <class Result, class BackendOp, class BackendTensor>
Expected<Result> GetConversionResult(
    const ConversionResult<BackendOp, BackendTensor>& result) {
  if (ConversionMatched(result)) {
    return Expected<Result>(std::get<Result>(result));
  }
  return Error(kLiteRtStatusLegalizeNoMatch);
}

// Get simple result if there was a match.
template <class BackendOp, class BackendTensor>
Expected<SimpleConversionResult<BackendOp>> GetSimpleConversionResult(
    const ConversionResult<BackendOp, BackendTensor>& result) {
  if (!IsSimpleResult(result)) {
    return Error(kLiteRtStatusErrorInvalidArgument);
  }
  return GetConversionResult<SimpleConversionResult<BackendOp>>(result);
}

// Get general result if there was a match.
template <class BackendOp, class BackendTensor>
Expected<GeneralConversionResult<BackendOp, BackendTensor>>
GetGeneralConversionResult(
    const ConversionResult<BackendOp, BackendTensor>& result) {
  if (!IsGeneralResult(result)) {
    return Error(kLiteRtStatusErrorInvalidArgument);
  }
  return GetConversionResult<GeneralConversionResult<BackendOp, BackendTensor>>(
      result);
}

// Common IR Conversion
//===---------------------------------------------------------------------------

// User defined callback for converting a LiteRt tensor to a backend tensor.
// These are leveraged in various higher-level conversion routines.
// TensorConverters should not stack allocate memory for the backend tensor. In
// most situations, these will be bound to an external allocator.
template <class BackendTensor>
using TensorConverter =
    std::function<Expected<BackendTensor*>(const Tensor& litert_tensor)>;

// User defined callback for creating a TensorConverter. This facilitates
// TensoConverters that are bound to an external allocator.
template <class BackendTensor>
using TensorConverterFactory = std::function<TensorConverter<BackendTensor>(
    TensorAllocator<BackendTensor> alloc)>;

// Mapping from LiteRt tensor to backend tensor, used during iterative graph
// conversions to store current scope.
template <class BackendTensor>
using TensorMap = absl::flat_hash_map<LiteRtTensor, BackendTensor*>;

// User-defined hook that calls backend to determine if an op is supported.
template <class BackendOp>
using Capability = std::function<bool(const BackendOp* op)>;

// Legalization
//===---------------------------------------------------------------------------

// A legalization is a particlar type of user-defined conversion that is
// scheduled for execution on a particular type of LiteRtOp. They may be
// one-to-one or one-to-many conversions.
template <class BackendOp, class BackendTensor>
class Legalization {
 private:
  using Self = Legalization<BackendOp, BackendTensor>;

 public:
  using Result = ConversionResult<BackendOp, BackendTensor>;
  using TensorConverter = TensorConverter<BackendTensor>;
  using TensorConverterFactory = TensorConverterFactory<BackendTensor>;
  using Ptr = std::unique_ptr<Self>;
  using TensorAllocator = TensorAllocator<BackendTensor>;
  using OpAllocator = OpAllocator<BackendOp>;
  using Tensors = std::vector<BackendTensor*>;

  // The type of op to schedule on.
  virtual LiteRtOpCode OpToMatch() const = 0;

  // Invoke this legalization on the given LiteRt op. All new backend IR will be
  // allocated via given allocators. NOTE: In most cases, input and output
  // converters will be the same. They are separated here for compatibility with
  // graph-level conversions routines.
  Expected<Result> Legalize(const Op& litert_op,
                            TensorConverterFactory input_converter,
                            TensorConverterFactory output_converter,
                            TensorAllocator tensor_allocator,
                            OpAllocator op_allocator) const {
    const auto litert_inputs = litert_op.Inputs();
    Tensors inputs(litert_inputs.size());
    auto convert_input = input_converter(tensor_allocator);

    for (size_t i = 0; i < litert_inputs.size(); ++i) {
      const auto& litert_input = litert_inputs[i];
      auto result = convert_input(litert_input);
      if (!result) {
        return result.Error();
      }
      inputs[i] = *result;
    }

    const auto litert_outputs = litert_op.Outputs();
    Tensors outputs(litert_outputs.size());
    auto convert_output = output_converter(tensor_allocator);

    for (size_t i = 0; i < litert_outputs.size(); ++i) {
      const auto& litert_output = litert_outputs[i];
      auto result = convert_output(litert_output);
      if (!result) {
        return result.Error();
      }
      outputs[i] = *result;
    }

    return LegalizeImpl(litert_op, inputs, outputs, tensor_allocator,
                        op_allocator);
  }

  virtual ~Legalization() = default;

 private:
  // The user defined implementation of a legalization. Users must use the
  // given allocators to allocate any new backend IR types (e.g. intermediate
  // ops/tensors in the case of a one-to-many legalization). BackendTensors
  // corresponding to LiteRt inputs and outputs have been pre-converted.
  virtual Expected<Result> LegalizeImpl(const Op& litert_op,
                                        const Tensors& inputs,
                                        const Tensors& outputs,
                                        TensorAllocator tensor_allocator,
                                        OpAllocator op_allocator) const = 0;
};

// Collection of legalizations for a specific backend.
template <class BackendOp, class BackendTensor>
using Legalizations =
    std::vector<typename Legalization<BackendOp, BackendTensor>::Ptr>;

// Map for instance lookup by op code.
template <class BackendOp, class BackendTensor>
using LegalizationMap =
    absl::flat_hash_map<LiteRtOpCode,
                        const Legalization<BackendOp, BackendTensor>*>;

// Construct a LegalizationMap from a collection of legalizations.
// TODO: Consider wrapping the legalization map in a class to avoid
// re-constructing it & better syntax.
template <class BackendOp, class BackendTensor>
LegalizationMap<BackendOp, BackendTensor> MakeLegalizationMap(
    const Legalizations<BackendOp, BackendTensor>& legalizations) {
  LegalizationMap<BackendOp, BackendTensor> map;
  for (const auto& l : legalizations) {
    map.insert({l->OpToMatch(), l.get()});
  }
  return map;
}

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_CONVERSION_H_
