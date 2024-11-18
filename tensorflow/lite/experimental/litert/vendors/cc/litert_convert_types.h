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
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either expruns or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Utility types for mapping LiteRt IR to arbitrary backend specific
// types. Implementations of these types define mapping for ops and tensors
// that may be used in a stndalone fashion. They also may be composed
// to create lowerings of entire graphs with topology.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_LITERT_CONVERT_TYPES_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_LITERT_CONVERT_TYPES_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_detail.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"

namespace litert {

// Types related to inter-dialect graph transformations. NOTE: Anything labeled
// with [USER DEFINED] is pluggable backend specific logic.

//
// Fundamental types for IR mapping.
//

// Defines the mapping between LiteRtTensor types and backend specific
// tensor types.
template <class BackendTensor>
using TensorConverter =
    std::function<Expected<BackendTensor>(const Tensor& litert_tensor)>;

// Result of a one->many general mapping from LiteRt op to any number of
// backend specific ops.
template <class BackendOp, class BackendTensor>
struct GeneralConversionResult {
  // Ops emitted from translation pattern.
  SmallVec<BackendOp> backend_ops;

  // Any backend tensors used within the results ops. Not relevant when
  // size of backend ops == 1.
  SmallVec<BackendTensor> intermediate_tensors;
};

// The result of a one->one specialized mapping from LiteRt op to backend op.
template <class BackendOp>
using SimpleConversionResult = BackendOp;

// A tag-type for a conversion result that is a non-error non-match. Needed
// for compatibility with litert::Expected.
struct NoMatch {};

// Type union for conversion results.
template <class BackendOp, class BackendTensor>
using ConversionResult =
    std::variant<SimpleConversionResult<BackendOp>,
                 GeneralConversionResult<BackendOp, BackendTensor>, NoMatch>;

// Short hand for holds_alternative.
template <class Result, class BackendOp, class BackendTensor>
inline bool LegalizationHoldsAlternative(
    const ConversionResult<BackendOp, BackendTensor>& result) {
  return std::holds_alternative<Result>(result);
}

// Short hand for holds_alternative.
template <class BackendOp, class BackendTensor>
inline bool LegalizationMatched(
    const ConversionResult<BackendOp, BackendTensor>& result) {
  return !std::holds_alternative<NoMatch>(result);
}

// Short hand for holds_alternative.
template <class BackendOp, class BackendTensor>
inline bool LegalizationHasSimpleResult(
    const ConversionResult<BackendOp, BackendTensor>& result) {
  return LegalizationHoldsAlternative<SimpleConversionResult<BackendOp>>(
      result);
}

// Short hand for holds_alternative.
template <class BackendOp, class BackendTensor>
inline bool LegalizationHasGeneralResult(
    const ConversionResult<BackendOp, BackendTensor>& result) {
  return LegalizationHoldsAlternative<
      GeneralConversionResult<BackendOp, BackendTensor>>(result);
}

// Short hand for std::get. Also checks if match and wraps in expected.
template <class Result, class BackendOp, class BackendTensor>
inline Expected<Result> GetConversionResult(
    const ConversionResult<BackendOp, BackendTensor>& result) {
  if (LegalizationMatched(result)) {
    return Expected<Result>(std::get<Result>(result));
  }
  return Error(kLiteRtStatusLegalizeNoMatch);
}

// Get simple result if there was a match.
template <class BackendOp, class BackendTensor>
inline Expected<SimpleConversionResult<BackendOp>> GetSimpleConversionResult(
    const ConversionResult<BackendOp, BackendTensor>& result) {
  return GetConversionResult<SimpleConversionResult<BackendOp>>(result);
}

// Get general result if there was a match.
template <class BackendOp, class BackendTensor>
inline Expected<GeneralConversionResult<BackendOp, BackendTensor>>
GetGeneralConversionResult(
    const ConversionResult<BackendOp, BackendTensor>& result) {
  return GetConversionResult<GeneralConversionResult<BackendOp, BackendTensor>>(
      result);
}

// An op conversion that is scheduled on only LiteRt ops of a particular type.
template <class BackendOp, class BackendTensor>
class LegalizationBase {
  using Self = LegalizationBase<BackendOp, BackendTensor>;

 public:
  using TensorVec = SmallVec<BackendTensor>;
  using Ptr = std::unique_ptr<Self>;
  using PtrVec = std::vector<Ptr>;
  using Result = ConversionResult<BackendOp, BackendTensor>;

  // The op type to schedule this conversion on.
  virtual LiteRtOpCode OpToMatch() const = 0;

  // Run the legalization, transforming the LiteRt op in to backend
  // specific counterpart.
  Expected<Result> Legalize(const Op& litert_op) {
    if (OpToMatch() != litert_op.Code()) {
      LITERT_LOG(LITERT_INFO, "No match on op code.");
      return Expected<Result>(NoMatch{});
    }
    TensorVec inputs;
    for (auto& litert_input : litert_op.Inputs()) {
      auto input = InternalConvertInputTensor(litert_input);
      if (!input) {
        LITERT_LOG(LITERT_ERROR, "Failed internal convert input tensor.");
        return input.Error();
      }
      inputs.emplace_back(std::move(input.Value()));
    }
    TensorVec outputs;
    for (auto& litert_output : litert_op.Outputs()) {
      auto output = InternalConvertOutputTensor(litert_output);
      if (!output) {
        LITERT_LOG(LITERT_ERROR, "Failed internal convert output tensor.");
        return output.Error();
      }
      outputs.emplace_back(std::move(output.Value()));
    }
    return Convert(litert_op, inputs, outputs);
  }

  virtual ~LegalizationBase() = default;

 private:
  template <class Op, class Tensor>
  friend class ScopedFinalizingLegalization;

  // Convert tensors encountered as op inputs. This is an implementation detail
  // not defined by user.
  virtual Expected<BackendTensor> InternalConvertInputTensor(
      const Tensor& litert_tensor) = 0;

  // Convert tensors encountered as op outputs. This is an implementation detail
  // not defined by user.
  virtual Expected<BackendTensor> InternalConvertOutputTensor(
      const Tensor& litert_tensor) = 0;

  // [USER DEFINED]
  // Actual user-defined conversion logic.
  virtual Expected<Result> Convert(const Op& litert_op, TensorVec& inputs,
                                   TensorVec& outputs) = 0;

  // [USER DEFINED]
  // Allow for tensor conversion to be implemented on a per-legalization
  // basis. In most cases this will not be necessary (see derived Legalization
  // class).
  virtual Expected<BackendTensor> ConvertTensor(
      const Tensor& litert_tensor) = 0;

  // Needed for friendship with derived classes (see
  // ScopedFinalizingLegalization).
  static auto GetConvert(Self& inst) {
    return [&](const auto& litert_op, auto& inputs, auto& outputs) -> auto {
      return inst.Convert(litert_op, inputs, outputs);
    };
  }

  // Needed for friendship with derived classes (see
  // ScopedFinalizingLegalization).
  static auto GetConvertTensor(Self& inst) {
    return
        [&](const auto& tensor) -> auto { return inst.ConvertTensor(tensor); };
  }
};

// A legalization with an externally defined TensorConverter. This
// is the common case where most legalizations share the same
// tensor conversion logic.
template <class BackendOp, class BackendTensor>
class Legalization : public LegalizationBase<BackendOp, BackendTensor> {
  using TensorKonverter = TensorConverter<BackendTensor>;

 public:
  using Base = LegalizationBase<BackendOp, BackendTensor>;

  // Construct this legalization with pre-defined tensor conversion logic.
  explicit Legalization(TensorKonverter tensor_converter)
      : tensor_converter_(tensor_converter) {}

 private:
  // Convert tensors encountered as op inputs. For this case this need only be
  // trivial.
  Expected<BackendTensor> InternalConvertInputTensor(
      const Tensor& litert_tensor) override {
    return ConvertTensor(litert_tensor);
  }

  // Convert tensors encountered as op outputs. For this case this need only be
  // trivial.
  Expected<BackendTensor> InternalConvertOutputTensor(
      const Tensor& litert_tensor) override {
    return ConvertTensor(litert_tensor);
  }

  // Run the tensor conversion per the TensorConverter initialized with.
  Expected<BackendTensor> ConvertTensor(const Tensor& litert_tensor) override {
    return tensor_converter_(litert_tensor);
  }

  TensorKonverter tensor_converter_;
};

//
// Pluggable & ready-to-use legalization utility.
//

// Utility for efficiently legalizing from a collection of legalizations.
// A legalizer can have at most one legalization for a given litert op type.
template <class BackendOp, class BackendTensor>
class LegalizerBase {
 public:
  using Legalisation = LegalizationBase<BackendOp, BackendTensor>;
  using Result = ConversionResult<BackendOp, BackendTensor>;

 private:
  using LegalizationRef = std::reference_wrapper<Legalisation>;
  using OptLegalizationRef = std::optional<LegalizationRef>;
  using LegalizationMap =
      absl::flat_hash_map<LiteRtOpCode, typename Legalisation::Ptr>;

 public:
  // Legalize the given op.
  Expected<Result> Legalize(const Op& litert_op) {
    auto legalization = FindLegalization(litert_op.Code());
    if (!legalization) {
      return Error(kLiteRtStatusErrorInvalidArgument);
    }
    return DoLegalize(litert_op, legalization.value());
  }

  // Register the given legalization with this legalizer.
  LiteRtStatus Register(Legalisation::Ptr legalization) {
    if (legalization_map_.contains(legalization->OpToMatch())) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    legalization_map_.insert(
        {legalization->OpToMatch(), std::move(legalization)});
    return kLiteRtStatusOk;
  }

  virtual ~LegalizerBase() = default;

  LegalizerBase() = default;

 private:
  // Internal method that does the actual legalization.
  virtual Expected<Result> DoLegalize(const Op& litert_op,
                                      Legalisation& legalization) = 0;

  // Locate the legalization function associated with given code.
  OptLegalizationRef FindLegalization(LiteRtOpCode code) const {
    if (auto it = legalization_map_.find(code); it != legalization_map_.end()) {
      return *it->second;
    }
    return {};
  }

  LegalizationMap legalization_map_;
};

// A simple legalizer that does not cache any results.
template <class BackendOp, class BackendTensor>
class Legalizer : public LegalizerBase<BackendOp, BackendTensor> {
  using Base = LegalizerBase<BackendOp, BackendTensor>;
  using Self = Legalizer<BackendOp, BackendTensor>;

 public:
  using Ptr = std::unique_ptr<Self>;

 private:
  // Simplest implementation that run the given legalization against given op.
  Expected<typename Base::Result> DoLegalize(
      const Op& litert_op, typename Base::Legalisation& legalization) override {
    return legalization.Legalize(litert_op);
  }
};

// [WIP] A cached legalizer. (Caching could also be done directly in the
// legalization class?)
template <class BackendOp, class BackendTensor>
class CachedLegalizer : public LegalizerBase<BackendOp, BackendTensor> {
  using Base = LegalizerBase<BackendOp, BackendTensor>;

  // [WIP] Cached implementation, returns previously computed BackendOp if it
  // exists.
  // TODO Implement hash code on litert::Op and finish this.
  Expected<typename Base::Result> DoLegalize(
      const Op& litert_op, typename Base::Legalisation& legalization) override {
    return Error(kLiteRtStatusErrorNotFound);
  }
};

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_LITERT_CONVERT_TYPES_H_
