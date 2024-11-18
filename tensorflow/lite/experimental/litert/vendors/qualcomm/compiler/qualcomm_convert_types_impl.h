
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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_COMPILER_QUALCOMM_CONVERT_TYPES_IMPL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_COMPILER_QUALCOMM_CONVERT_TYPES_IMPL_H_

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/litert_convert.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/litert_convert_types.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/common.h"

// Qualcomm specific implementations of convert_types.

namespace litert {
namespace qualcomm {

// Wrapper for qnn ops.
struct QnnOpWrapper {
  // TODO implement Qnn IR wrappers (in different file)
};

// Wrapper for qnn tensors.
struct QnnTensorWrapper {
  // TODO implement Qnn IR wrappers (in different file)
};

// Universal tensor conversion for litert -> qnn. This is defined once and used
// in various higher level routines.
inline Expected<QnnTensorWrapper> ConvertQnnTensor(
    const Tensor& litert_tensor) {
  // TODO map litert tensor to qnn tensor (see IR/qnn_tensor.h).
  return Error(kLiteRtStatusErrorUnsupported);
}

// An example legalization (see legalizations/*.h). Only need to implement
// the `Convert` function below. i/o tensors come pre-converted. Higher-level
// generic utilities will use these to implement the partitioning and graph
// creation.
class MulOpLegalization : public Legalization<QnnOpWrapper, QnnTensorWrapper> {
  using QnnTensorVec = TensorVec;

 public:
  // Get the op to match on.
  LiteRtOpCode OpToMatch() const override { return kLiteRtOpCodeTflMul; }

  MulOpLegalization() : Legalization(ConvertQnnTensor) {}

 private:
  // Convert the given litert to Qualcomm op.
  Expected<Result> Convert(const Op& litert_op, QnnTensorVec& inputs,
                           QnnTensorVec& outputs) override {
    // TODO
    return Error(kLiteRtStatusErrorUnsupported);
  }
};

// A legalizer is a utility for dispatching legalizations. Its essentially
// just a map LiteRtOpCode -> Legalization and abstracts some common
// functionality.. We can simply alias the legalizer to use with our op defs.
using QnnLegalizer = Legalizer<QnnOpWrapper, QnnTensorWrapper>;

// Hook to check whether a given op is supported. This will call the underlying
// Qnn api. This is all that needs to be implemented for partitioning.
inline CompilerCapability<QnnOpWrapper> MakeQnnCompilerCapabilityFunc(
    QnnApi api) {
  return [&](const QnnOpWrapper& qnn_op) -> bool {
    // api.backendValidateOpConfig TODO implement.
    return true;
  };
}

// Hook to finalize a tensor during graph creation. This is leverage during
// graph creation, but not during partitioning/capability checking.
inline TensorFinalizer<QnnTensorWrapper> MakeQnnTensorFinalizer(
    QnnApi qnn_api) {
  return [&](QnnTensorWrapper& backend_tensor) -> LiteRtStatus {
    // qnn_api.tensorCreateGraphTensor TODO implement
    return kLiteRtStatusOk;
  };
}

// Hook to finalize an op during graph creation. This is leverage during
// graph creation, but not during partitioning/capability checking.
inline OpFinalizer<QnnOpWrapper> MakeQnnOpFinalizer(QnnApi qnn_api) {
  return [&](QnnOpWrapper& backend_op) -> LiteRtStatus {
    // qnn_api.graphAddNode ... TODO implement.
    return kLiteRtStatusOk;
  };
}

// Hook to initialize a subgraph graph during graph creation.
inline GraphInitializer MakeQnnGraphInitializer(QnnApi qnn_api) {
  return [&](absl::string_view name) -> LiteRtStatus {
    // qnn_api.graphCreateSubgraph ... TODO implement.
    return kLiteRtStatusOk;
  };
}

// Hook to finalize a subgraph graph during graph creation.
inline GraphFinalizer MakeQnnGraphFinalizer(QnnApi qnn_api) {
  return [&]() -> LiteRtStatus {
    // qnn_api.graphFinalize .. TODO implement.
    return kLiteRtStatusOk;
  };
}

// A graph converter is a utility that implements all of the
// LiteRtCompilerPluginCompile function using the above user defined hooks. We
// only need to specialize the template to use with our qnn logic.
using QnnGraphConverter = GraphConverter<QnnOpWrapper, QnnTensorWrapper>;

}  // namespace qualcomm
}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_COMPILER_QUALCOMM_CONVERT_TYPES_IMPL_H_
