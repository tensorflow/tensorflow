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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_LITERT_CONVERT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_LITERT_CONVERT_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/litert_convert_types.h"

namespace litert {

// Ready-to-use and pluggagle types for mapping LiteRt IR to arbitrary backend
// specific types.

//
// Partitioning
//

// [USER DEFINED]
// Checks whether the backend can support the given op.
template <class BackendOp>
using CompilerCapability = std::function<bool(BackendOp backend_op)>;

// C++ wrapper for CompilerPluginPartition output parameter.
using OpList = std::function<void(LiteRtOp litert_op)>;

// Given a legalizer and compiler capabilities callback, implements the
// LiteRtPluginPartitionModel function.
template <class BackendOp, class BackendTensor>
inline LiteRtStatus PartitionViaCapabilities(
    Legalizer<BackendOp, BackendTensor>& legalizer,
    CompilerCapability<BackendOp> capability, Model& litert_model,
    OpList selected_ops) {
  if (litert_model.NumSubgraphs() != 1) {
    // Only one subgraph support currently implemented.
    return kLiteRtStatusErrorUnsupported;
  }
  auto main_subgraph = litert_model.MainSubgraph();
  if (!main_subgraph) {
    return main_subgraph.Error().Status();
  }

  auto push_if_capable = [&](auto& backend_op, Op& litert_op) -> void {
    auto capable = capability(backend_op);
    if (!capable) {
      return;
    }
    selected_ops(litert_op.Get());
  };

  for (auto& op : main_subgraph->Ops()) {
    auto convert_result = legalizer.Legalize(op);
    if (!convert_result) {
      continue;
    }
    if (auto simple_result = GetSimpleConversionResult(*convert_result);
        simple_result.HasValue()) {
      push_if_capable(*simple_result, op);
    } else if (auto general_result =
                   GetGeneralConversionResult(*convert_result);
               general_result.HasValue()) {
      for (auto& backend_op : general_result->backend_ops) {
        push_if_capable(backend_op, op);
      }
    }
  }

  return kLiteRtStatusOk;
}

//
// Graph Conversion
//

// [USER DEFINED]
// A hook to call after a tensor has been converted (e.g. register with backend
// context).
template <class BackendTensor>
using TensorFinalizer = std::function<LiteRtStatus(BackendTensor& tensor)>;

// [USER DEFINED]
// A hook to call after an op has been converted (e.g. register with backend
// context).
template <class BackendOp>
using OpFinalizer = std::function<LiteRtStatus(BackendOp& op)>;

// [USER DEFINED]
// A hook to call to initialize a backend graph (e.g. register with backend
// context).
using GraphInitializer =
    std::function<LiteRtStatus(absl::string_view graph_name)>;

// [USER DEFINED]
// A hook to call to finalize a backend graph.
using GraphFinalizer = std::function<LiteRtStatus()>;

// Interal legalization wrapper used during graph construction. This invokes
// user-defined finalizer hooks after tensor/op conversion, which normally do
// thing like "register" converter IR with some backend context. This class
// additionally uses a shared scope map which stores the conversion result of
// previously encountered tensors. It is expected that this legalization is
// applied to ops in topological order during graph construction. See
// ConvertSubgraph. Requires BackendTensors be copyable.
template <class BackendOp, class BackendTensor>
class ScopedFinalizingLegalization
    : public LegalizationBase<BackendOp, BackendTensor> {
 public:
  using Tenser = BackendTensor;

 private:
  using Self = ScopedFinalizingLegalization<BackendOp, BackendTensor>;
  using Base = LegalizationBase<BackendOp, BackendTensor>;
  using OpFinaliser = OpFinalizer<BackendOp>;
  using TensorVec = Base::TensorVec;
  using SimpleResult = SimpleConversionResult<BackendOp>;
  using GeneralResult = GeneralConversionResult<BackendOp, BackendTensor>;

 public:
  using Ptr = std::unique_ptr<Self>;

  using TensorFinaliser = TensorFinalizer<BackendTensor>;
  using TensorKonverter = TensorConverter<BackendTensor>;
  using Scope = absl::flat_hash_map<LiteRtTensor, BackendTensor>;
  using SharedScope = std::shared_ptr<Scope>;

  // Create a finalizing legalization from an existing legalization, scope, and
  // various user-defined finalizing hooks.
  static Ptr Create(Base::Ptr wrapped, TensorFinaliser tensor_finalizer,
                    OpFinaliser op_finalizer, SharedScope shared_scope) {
    return std::make_unique<Self>(std::move(wrapped), tensor_finalizer,
                                  op_finalizer, shared_scope);
  }

  // Call wrapped method.
  LiteRtOpCode OpToMatch() const override { return wrapped_->OpToMatch(); }

  // Construct a finalizing legalization from an existing legalization, scope
  // and various user-defined finalizing hooks.
  ScopedFinalizingLegalization(Base::Ptr wrapped,
                               TensorFinaliser tensor_finalizer,
                               OpFinaliser op_finalizer,
                               SharedScope shared_scope)
      : wrapped_(std::move(wrapped)),
        tensor_finalizer_(tensor_finalizer),
        op_finalizer_(op_finalizer),
        shared_scope_(shared_scope) {}

 private:
  // Lookup given tensor in scope.
  Expected<BackendTensor> Lookup(const Tensor& litert_tensor) const {
    LITERT_LOG(LITERT_INFO, "shared scope size: %lu", shared_scope_->size());
    if (auto it = shared_scope_->find(litert_tensor.Get());
        it != shared_scope_->end()) {
      return it->second;
    }
    LITERT_LOG(LITERT_ERROR, "Failed to lookup tensor: %s",
               litert_tensor.Name().data());
    return Error(kLiteRtStatusErrorNotFound);
  }

  // Add given tensor to scope. Tensors should only be evaluated once, so error
  // if tensor is already in scope.
  LiteRtStatus PushToScope(const Tensor& litert_tensor,
                           BackendTensor backend_tensor) {
    return Self::DoPushToScope(*shared_scope_, litert_tensor, backend_tensor);
  }

  // Invoke user-defined tensor finalizer. This should only be called once per
  // tensor.
  LiteRtStatus FinalizeTensor(BackendTensor& backend_tensor) const {
    return tensor_finalizer_(backend_tensor);
  }

  // Invoke user-defined op finalizer. This should only be called once per
  // tensor.
  LiteRtStatus FinalizeOp(BackendOp& backend_op) const {
    return op_finalizer_(backend_op);
  }

  // Instead of converting op input tensor, lookup tensor in scope. Since this
  // legalizations is applied to ops in topological order, it is guaranteed that
  // op input tensors will be already evaluated. Additionally invoke the
  // user-defined finalizer.
  Expected<BackendTensor> InternalConvertInputTensor(
      const Tensor& litert_tensor) override {
    return Lookup(litert_tensor);
  };

  // Convert op output tensor and push it to the scope. Additionally invoke the
  // user-defined finalizer.
  Expected<BackendTensor> InternalConvertOutputTensor(
      const Tensor& litert_tensor) override {
    auto backend_tensor = ConvertTensor(litert_tensor);
    if (!backend_tensor) {
      return backend_tensor;
    }

    LITERT_EXPECT_OK(FinalizeTensor(*backend_tensor));
    shared_scope_->insert({litert_tensor.Get(), *backend_tensor});

    return backend_tensor;
  };

  // Call wrapped method.
  Expected<BackendTensor> ConvertTensor(const Tensor& litert_tensor) override {
    return wrapped_->ConvertTensor(litert_tensor);
  }

  // Call wrapped method.
  Expected<typename Base::Result> WrappedConvert(const Op& litert_op,
                                                 TensorVec& inputs,
                                                 TensorVec& outputs) {
    return wrapped_->GetConvert(*wrapped_)(litert_op, inputs, outputs);
  }

  // Call finalizers on the IR within simple conversion result.
  LiteRtStatus FinalizeSimpleConversionResult(SimpleResult& simple_result) {
    return FinalizeOp(simple_result);
  }

  // Call finalizers on the IR within general conversion result. NOTE
  // intermediate tensors do not need to be pushed to scope since they by
  // definition are only reference by ops within this result.
  LiteRtStatus FinalizeGeneralConversionResult(GeneralResult& general_result) {
    for (auto& backend_op : general_result.backend_ops) {
      LITERT_RETURN_STATUS_IF_NOT_OK(FinalizeOp(backend_op));
    }
    for (auto& intermediate_tensor : general_result.intermediate_tensors) {
      LITERT_RETURN_STATUS_IF_NOT_OK(FinalizeTensor(intermediate_tensor));
    }
    return kLiteRtStatusOk;
  }

  // Call wrapped convert and invoke finalizers on the results as well as the
  // pre-converted input/output tensors. InternalTensorFinalizer is an
  // implementation detail not passed by user.
  Expected<typename Base::Result> Convert(const Op& litert_op,
                                          TensorVec& inputs,
                                          TensorVec& outputs) override {
    auto result = WrappedConvert(litert_op, inputs, outputs);
    if (!result || !LegalizationMatched(*result)) {
      LITERT_LOG(LITERT_WARNING, "Failed to convert op_code: %d",
                 static_cast<int>(litert_op.Code()));
      return result;
    }

    if (auto simple_result = GetSimpleConversionResult(*result);
        simple_result.HasValue()) {
      LITERT_EXPECT_OK(FinalizeSimpleConversionResult(*simple_result));

    } else if (auto general_result = GetGeneralConversionResult(*result);
               general_result.HasValue()) {
      LITERT_EXPECT_OK(FinalizeGeneralConversionResult(*general_result));

    } else {
      return Error(kLiteRtStatusErrorNotFound);
    }

    LITERT_LOG(LITERT_INFO, "Successfully converted op_code: %d",
               static_cast<int>(litert_op.Code()));

    return result;
  }

  Base::Ptr wrapped_;
  // [USER DEFINED]
  TensorFinaliser tensor_finalizer_;
  // [USER DEFINED]
  OpFinaliser op_finalizer_;
  SharedScope shared_scope_;

 public:
  // Helper logic for pushing to tensor map. Used in legalization and
  // externalized for re-use in testing.
  static LiteRtStatus DoPushToScope(Scope& scope, const Tensor& litert_tensor,
                                    BackendTensor backend_tensor) {
    if (scope.contains(litert_tensor.Get())) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    scope.insert({litert_tensor.Get(), backend_tensor});
    return kLiteRtStatusOk;
  }
};

// Ready-to-use and pluggable implementation of iterative graph conversion.
template <class BackendOp, class BackendTensor>
class GraphConverter {
  using TensorKonverter = TensorConverter<BackendTensor>;
  using TensorFinaliser = TensorFinalizer<BackendTensor>;
  using OpFinaliser = OpFinalizer<BackendOp>;
  using Legalisation = LegalizationBase<BackendOp, BackendTensor>;
  using InternalLegalization =
      ScopedFinalizingLegalization<BackendOp, BackendTensor>;
  using Legalizations = typename Legalisation::PtrVec;
  using InternalLegalizations = typename InternalLegalization::PtrVec;
  using Legaliser = Legalizer<BackendOp, BackendTensor>;
  using Scope = typename InternalLegalization::Scope;
  using SharedScope = typename InternalLegalization::SharedScope;

 public:
  // Construct a GraphConverter from given user-defined hooks and legalizations.
  // The result is suitable for running graph conversion one or many times.
  GraphConverter(TensorKonverter tensor_converter,
                 TensorFinaliser tensor_finalizer, OpFinaliser op_finalizer,
                 GraphInitializer graph_initializer,
                 GraphFinalizer graph_finalizer)
      : graph_initializer_(graph_initializer),
        graph_finalizer_(graph_finalizer),
        tensor_converter_(tensor_converter),
        op_finalizer_(op_finalizer),
        tensor_finalizer_(tensor_finalizer),
        shared_scope_(std::make_shared<Scope>()) {}

  // Given a leglizer and various finalizing hooks, implements graph conversion
  // by converting constituent ops/tensors in topological order. This works by
  // storing a mapping to evaluated litert tensors (scope) to their converted
  // counterpart. Input tensors for ops are looked up in scope. Tensors are
  // converted and added to scope whenever when the op that outputs them is
  // converted. Since it evaluates in topological order, all op input tensors
  // are guaranteed to be in scope.
  //
  // User-defined hooks in algorithm:
  // "tensor_converter": Called once for each subgraph input. The interior
  // tensors will be handeled
  //    by the tensor converters defined within given legalizations.
  // "tensor_finalizer": Called once for each tensor after converted.
  // "op_finalizer": Called once for each op after converted.
  // "graph_initializer": Called once at the beggining.
  // "graph_finalizer": Called once at the end.
  LiteRtStatus ConvertGraph(Subgraph litert_subgraph,
                            absl::string_view graph_name) {
    // SETUP

    // Refersh the scope map.
    GetScope().clear();

    // Initialize backend graph with user-defined hook.
    LITERT_RETURN_STATUS_IF_NOT_OK(InitializeGraph(graph_name));

    // CONVERT IR

    // Add subgraph inputs to initial scope.
    for (auto& subgraph_input : litert_subgraph.Inputs()) {
      auto backend_tensor = ConvertTensor(subgraph_input);
      if (!backend_tensor) {
        return backend_tensor.Error().Status();
      }
      GetScope().insert({subgraph_input.Get(), *backend_tensor});
      LITERT_LOG(LITERT_INFO, "Pushed tensor to scope: %s",
                 subgraph_input.Name().data());
    }

    // Legalize each op. The scoped finalizing legalization will handle
    // finalizing and updating scope.
    for (auto& litert_op : litert_subgraph.Ops()) {
      auto result = Legalize(litert_op);
      if (!result) {
        LITERT_LOG(LITERT_ERROR, "Failed to legalize op with OpCode: %d\n",
                   litert_op.Code());
        return result.Error().Status();
      }
      LITERT_LOG(LITERT_INFO, "Legalized op with OpCode: %d\n",
                 litert_op.Code());
    }

    // FINALIZE
    // Note that subgraph outputs will have already been evaluated when their
    // defining op was encountered.

    // Finalize backend graph with user-defined hook.
    LITERT_RETURN_STATUS_IF_NOT_OK(FinalizeGraph());

    return kLiteRtStatusOk;
  }

  // Register the given legalization with the inner legalizer. This
  // will be wrapped with an internal legalization implementation.
  LiteRtStatus Register(Legalisation::Ptr legalization) {
    return legalizer_.Register(
        InternalLegalization::Create(std::move(legalization), tensor_finalizer_,
                                     op_finalizer_, shared_scope_));
  }

 private:
  // Call user-defined tensor converter.
  auto ConvertTensor(const Tensor& litert_tensor) const {
    return tensor_converter_(litert_tensor);
  }

  // Call user-defined graph initializer.
  auto InitializeGraph(absl::string_view graph_name) {
    return graph_initializer_(graph_name);
  }

  // Call user-defined graph finalizer.
  auto FinalizeGraph() { return graph_finalizer_(); }

  // Call inner legalizer to legalize op.
  auto Legalize(const Op& litert_op) { return legalizer_.Legalize(litert_op); }

  // Get the current scope map.
  auto& GetScope() { return *shared_scope_; }

  GraphInitializer graph_initializer_;
  GraphFinalizer graph_finalizer_;
  TensorKonverter tensor_converter_;
  OpFinaliser op_finalizer_;
  TensorFinaliser tensor_finalizer_;
  SharedScope shared_scope_;
  Legaliser legalizer_;
};

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_LITERT_CONVERT_H_
