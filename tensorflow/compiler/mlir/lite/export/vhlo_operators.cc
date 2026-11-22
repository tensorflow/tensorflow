/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/export/vhlo_operators.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "llvm/ADT/APFloat.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/export/types.h"
#include "tensorflow/compiler/mlir/lite/schema/mutable/schema_generated.h"

namespace mlir::TFL::fb_export {
namespace {

using ::flatbuffers::Offset;
using ::tflite::BuiltinOperator;

std::vector<int64_t> GetInt64s(Attribute attr) {
  auto values = cast<vhlo::TensorV1Attr>(attr).getData().getValues<int64_t>();
  return std::vector<int64_t>(values.begin(), values.end());
}

int64_t GetInt(Attribute attr) {
  return cast<vhlo::IntegerV1Attr>(attr).getValue().getSExtValue();
}

bool GetBool(Attribute attr) {
  return cast<vhlo::BooleanV1Attr>(attr).getValue();
}

float GetFloat(Attribute attr) {
  APFloat value = cast<vhlo::FloatV1Attr>(attr).getValue();
  bool loses_info = false;
  value.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven,
                &loses_info);
  return value.convertToFloat();
}

// Returns the builtin code of the element-wise VHLO ops, which have no options.
std::optional<BuiltinOperator> GetElementwiseCode(Operation* op) {
  if (isa<vhlo::AbsOpV1>(op)) return tflite::BuiltinOperator_STABLEHLO_ABS;
  if (isa<vhlo::AddOpV1>(op)) return tflite::BuiltinOperator_STABLEHLO_ADD;
  if (isa<vhlo::AndOpV1>(op)) return tflite::BuiltinOperator_STABLEHLO_AND;
  if (isa<vhlo::CbrtOpV1>(op)) return tflite::BuiltinOperator_STABLEHLO_CBRT;
  if (isa<vhlo::ClampOpV1>(op)) return tflite::BuiltinOperator_STABLEHLO_CLAMP;
  if (isa<vhlo::ConvertOpV1>(op)) {
    return tflite::BuiltinOperator_STABLEHLO_CONVERT;
  }
  if (isa<vhlo::CosineOpV1>(op)) {
    return tflite::BuiltinOperator_STABLEHLO_COSINE;
  }
  if (isa<vhlo::DivOpV1>(op)) return tflite::BuiltinOperator_STABLEHLO_DIVIDE;
  if (isa<vhlo::DynamicUpdateSliceOpV1>(op)) {
    return tflite::BuiltinOperator_STABLEHLO_DYNAMIC_UPDATE_SLICE;
  }
  if (isa<vhlo::ExpOpV1>(op)) {
    return tflite::BuiltinOperator_STABLEHLO_EXPONENTIAL;
  }
  if (isa<vhlo::FloorOpV1>(op)) return tflite::BuiltinOperator_STABLEHLO_FLOOR;
  if (isa<vhlo::LogOpV1>(op)) return tflite::BuiltinOperator_STABLEHLO_LOG;
  if (isa<vhlo::LogisticOpV1>(op)) {
    return tflite::BuiltinOperator_STABLEHLO_LOGISTIC;
  }
  if (isa<vhlo::MaxOpV1>(op)) return tflite::BuiltinOperator_STABLEHLO_MAXIMUM;
  if (isa<vhlo::MinOpV1>(op)) return tflite::BuiltinOperator_STABLEHLO_MINIMUM;
  if (isa<vhlo::MulOpV1>(op)) {
    return tflite::BuiltinOperator_STABLEHLO_MULTIPLY;
  }
  if (isa<vhlo::NegOpV1>(op)) return tflite::BuiltinOperator_STABLEHLO_NEGATE;
  if (isa<vhlo::OrOpV1>(op)) return tflite::BuiltinOperator_STABLEHLO_OR;
  if (isa<vhlo::PowOpV1>(op)) return tflite::BuiltinOperator_STABLEHLO_POWER;
  if (isa<vhlo::RemOpV1>(op)) {
    return tflite::BuiltinOperator_STABLEHLO_REMAINDER;
  }
  if (isa<vhlo::ReshapeOpV1>(op)) {
    return tflite::BuiltinOperator_STABLEHLO_RESHAPE;
  }
  if (isa<vhlo::RsqrtOpV1>(op)) return tflite::BuiltinOperator_STABLEHLO_RSQRT;
  if (isa<vhlo::SelectOpV1>(op)) {
    return tflite::BuiltinOperator_STABLEHLO_SELECT;
  }
  if (isa<vhlo::ShiftLeftOpV1>(op)) {
    return tflite::BuiltinOperator_STABLEHLO_SHIFT_LEFT;
  }
  if (isa<vhlo::SubtractOpV1>(op)) {
    return tflite::BuiltinOperator_STABLEHLO_SUBTRACT;
  }
  if (isa<vhlo::TanhOpV1>(op)) return tflite::BuiltinOperator_STABLEHLO_TANH;
  return std::nullopt;
}

// Adds a composite attribute to a flexbuffer, under `key` if not null (inside
// maps). Returns false for unsupported attributes.
bool AddFlexValue(flexbuffers::Builder& flex, const char* key, Attribute attr) {
  if (auto tensor = dyn_cast<vhlo::TensorV1Attr>(attr)) {
    // Tensors are maps named "_TENSOR_V1_<key>" holding their shape, TFLite
    // type and elements.
    if (key == nullptr) return false;
    DenseElementsAttr data = tensor.getData();
    absl::StatusOr<tflite::TensorType> type =
        GetTensorType(data.getElementType());
    if (!type.ok()) return false;
    const size_t map = flex.StartMap(absl::StrCat("_TENSOR_V1_", key).c_str());
    flex.Vector("TENSOR_SHAPE", [&] {
      for (int64_t dim : data.getType().getShape()) flex.Int(dim);
    });
    flex.Int("TENSOR_TYPE", *type);
    switch (*type) {
      case tflite::TensorType_INT32:
        flex.Vector("TENSOR_DATA", [&] {
          for (int32_t v : data.getValues<int32_t>()) flex.Int(v);
        });
        break;
      case tflite::TensorType_INT64:
        flex.Vector("TENSOR_DATA", [&] {
          for (int64_t v : data.getValues<int64_t>()) flex.Int(v);
        });
        break;
      case tflite::TensorType_FLOAT32:
        flex.Vector("TENSOR_DATA", [&] {
          for (float v : data.getValues<float>()) flex.Float(v);
        });
        break;
      default:
        return false;
    }
    flex.EndMap(map);
    return true;
  }

  if (key != nullptr) flex.Key(key);
  if (auto b = dyn_cast<vhlo::BooleanV1Attr>(attr)) {
    flex.Bool(b.getValue());
  } else if (auto i = dyn_cast<vhlo::IntegerV1Attr>(attr)) {
    flex.Int(i.getValue().getSExtValue());
  } else if (isa<vhlo::FloatV1Attr>(attr)) {
    flex.Float(GetFloat(attr));
  } else if (auto s = dyn_cast<vhlo::StringV1Attr>(attr)) {
    flex.String(s.getValue().str());
  } else if (auto array = dyn_cast<vhlo::ArrayV1Attr>(attr)) {
    const size_t vector = flex.StartVector();
    for (Attribute element : array.getValue()) {
      if (!AddFlexValue(flex, /*key=*/nullptr, element)) return false;
    }
    flex.EndVector(vector, /*typed=*/false, /*fixed=*/false);
  } else {
    return false;
  }
  return true;
}

// Serializes one VHLO op.
class VhloOperatorBuilder {
 public:
  VhloOperatorBuilder(Operation* op, const std::vector<int32_t>& inputs,
                      const std::vector<int32_t>& outputs,
                      ModelContext& context)
      : op_(op),
        inputs_(inputs),
        outputs_(outputs),
        context_(context),
        fbb_(context.builder()) {}

  std::optional<Offset<tflite::Operator>> Build() {
    if (std::optional<BuiltinOperator> code = GetElementwiseCode(op_)) {
      return CreateOperator(*code);
    }
    if (auto op = dyn_cast<vhlo::CompositeOpV1>(op_)) return Build(op);
    if (auto op = dyn_cast<vhlo::CustomCallOpV1>(op_)) return Build(op);
    if (auto op = dyn_cast<vhlo::GatherOpV1>(op_)) return BuildGather(op);
    if (auto op = dyn_cast<vhlo::GatherOpV2>(op_)) return BuildGather(op);
    if (auto op = dyn_cast<vhlo::ScatterOpV1>(op_)) return BuildScatter(op);
    if (auto op = dyn_cast<vhlo::ScatterOpV2>(op_)) return BuildScatter(op);
    if (auto op = dyn_cast<vhlo::PadOpV1>(op_)) return Build(op);
    if (auto op = dyn_cast<vhlo::ReduceWindowOpV1>(op_)) return Build(op);
    if (auto op = dyn_cast<vhlo::RngBitGeneratorOpV1>(op_)) return Build(op);
    op_->emitOpError("is not supported by the TFLite flatbuffer exporter");
    return std::nullopt;
  }

 private:
  Offset<tflite::Operator> CreateOperator(
      BuiltinOperator code,
      tflite::BuiltinOptions2 options_type = tflite::BuiltinOptions2_NONE,
      Offset<void> options = 0) {
    return tflite::CreateOperator(
        fbb_, context_.GetOpcodeIndex(code), fbb_.CreateVector(inputs_),
        fbb_.CreateVector(outputs_), tflite::BuiltinOptions_NONE,
        /*builtin_options=*/0, /*custom_options=*/0,
        tflite::CustomOptionsFormat_FLEXBUFFERS,
        /*mutating_variable_inputs=*/0, /*intermediates=*/0,
        /*large_custom_options_offset=*/0, /*large_custom_options_size=*/0,
        options_type, options);
  }

  Offset<flatbuffers::Vector<int64_t>> Int64Vector(Attribute attr) {
    return fbb_.CreateVector(GetInt64s(attr));
  }

  std::optional<Offset<tflite::Operator>> Build(vhlo::CompositeOpV1 op) {
    const auto decomposition = context_.GetFunctionSubgraph(
        cast<vhlo::StringV1Attr>(op.getDecomposition()).getValue());
    if (!decomposition.has_value()) {
      op.emitOpError("has no decomposition function");
      return std::nullopt;
    }

    flexbuffers::Builder flex;
    const size_t map = flex.StartMap();
    for (auto [key, value] :
         cast<vhlo::DictionaryV1Attr>(op.getCompositeAttributes()).getValue()) {
      const std::string name = cast<vhlo::StringV1Attr>(key).getValue().str();
      if (!AddFlexValue(flex, name.c_str(), value)) {
        op.emitOpError("has unsupported composite attribute: ") << name;
        return std::nullopt;
      }
    }
    flex.EndMap(map);
    flex.Finish();

    const auto options = tflite::CreateStableHLOCompositeOptions(
        fbb_,
        fbb_.CreateString(
            cast<vhlo::StringV1Attr>(op.getName()).getValue().str()),
        *decomposition, fbb_.CreateVector(flex.GetBuffer()),
        tflite::CustomOptionsFormat_FLEXBUFFERS, GetInt(op.getVersion()));
    return CreateOperator(tflite::BuiltinOperator_STABLEHLO_COMPOSITE,
                          tflite::BuiltinOptions2_StableHLOCompositeOptions,
                          options.Union());
  }

  template <typename GatherOp>
  std::optional<Offset<tflite::Operator>> BuildGather(GatherOp op) {
    if constexpr (std::is_same_v<GatherOp, vhlo::GatherOpV2>) {
      if (!GetInt64s(op.getOperandBatchingDims()).empty() ||
          !GetInt64s(op.getStartIndicesBatchingDims()).empty()) {
        op.emitError("stablehlo.gather with batching dims is not supported");
        return std::nullopt;
      }
    }
    const auto options = tflite::CreateStablehloGatherOptions(
        fbb_, Int64Vector(op.getOffsetDims()),
        Int64Vector(op.getCollapsedSliceDims()),
        Int64Vector(op.getStartIndexMap()), GetInt(op.getIndexVectorDim()),
        Int64Vector(op.getSliceSizes()), GetBool(op.getIndicesAreSorted()));
    return CreateOperator(tflite::BuiltinOperator_STABLEHLO_GATHER,
                          tflite::BuiltinOptions2_StablehloGatherOptions,
                          options.Union());
  }

  template <typename ScatterOp>
  std::optional<Offset<tflite::Operator>> BuildScatter(ScatterOp op) {
    if constexpr (std::is_same_v<ScatterOp, vhlo::ScatterOpV2>) {
      if (!GetInt64s(op.getInputBatchingDims()).empty() ||
          !GetInt64s(op.getScatterIndicesBatchingDims()).empty()) {
        op.emitError("stablehlo.scatter with batching dims is not supported");
        return std::nullopt;
      }
    }
    const std::optional<int32_t> body = context_.BuildRegionSubgraph(
        op.getUpdateComputation(), tflite::BuiltinOperator_STABLEHLO_SCATTER);
    if (!body.has_value()) return std::nullopt;
    const auto options = tflite::CreateStablehloScatterOptions(
        fbb_, GetBool(op.getIndicesAreSorted()),
        Int64Vector(op.getUpdateWindowDims()),
        Int64Vector(op.getInsertedWindowDims()),
        Int64Vector(op.getScatterDimsToOperandDims()),
        GetInt(op.getIndexVectorDim()), GetBool(op.getUniqueIndices()), *body);
    return CreateOperator(tflite::BuiltinOperator_STABLEHLO_SCATTER,
                          tflite::BuiltinOptions2_StablehloScatterOptions,
                          options.Union());
  }

  // Custom calls the pipeline can't lower (e.g. from PyTorch PT2E
  // quantization) are kept for downstream consumers.
  std::optional<Offset<tflite::Operator>> Build(vhlo::CustomCallOpV1 op) {
    int32_t api_version = 0;
    switch (
        cast<vhlo::CustomCallApiVersionV1Attr>(op.getApiVersion()).getValue()) {
      case vhlo::CustomCallApiVersionV1::API_VERSION_UNSPECIFIED:
        api_version = 0;
        break;
      case vhlo::CustomCallApiVersionV1::API_VERSION_ORIGINAL:
        api_version = 1;
        break;
      case vhlo::CustomCallApiVersionV1::API_VERSION_STATUS_RETURNING:
        api_version = 2;
        break;
      case vhlo::CustomCallApiVersionV1::API_VERSION_STATUS_RETURNING_UNIFIED:
        api_version = 3;
        break;
      default:
        op.emitOpError("has an unsupported API version");
        return std::nullopt;
    }
    auto backend_config = dyn_cast<vhlo::StringV1Attr>(op.getBackendConfig());
    if (!backend_config) {
      op.emitOpError("must have a string backend config");
      return std::nullopt;
    }

    // Other boolean and string attributes are kept as custom attributes.
    flexbuffers::Builder flex;
    const size_t map = flex.StartMap();
    for (NamedAttribute attr : op->getAttrs()) {
      const std::string name = attr.getName().str();
      if (name == "call_target_name" || name == "backend_config") continue;
      if (auto b = dyn_cast<vhlo::BooleanV1Attr>(attr.getValue())) {
        flex.Bool(name.c_str(), b.getValue());
      } else if (auto s = dyn_cast<vhlo::StringV1Attr>(attr.getValue())) {
        flex.String(name.c_str(), s.getValue().str());
      }
    }
    flex.EndMap(map);
    flex.Finish();

    const auto options = tflite::CreateStablehloCustomCallOptions(
        fbb_,
        fbb_.CreateString(
            cast<vhlo::StringV1Attr>(op.getCallTargetName()).getValue().str()),
        GetBool(op.getHasSideEffect()),
        fbb_.CreateString(backend_config.getValue().str()), api_version,
        /*called_computations=*/0, fbb_.CreateVector(flex.GetBuffer()));
    return CreateOperator(tflite::BuiltinOperator_STABLEHLO_CUSTOM_CALL,
                          tflite::BuiltinOptions2_StablehloCustomCallOptions,
                          options.Union());
  }

  std::optional<Offset<tflite::Operator>> Build(vhlo::PadOpV1 op) {
    const auto options = tflite::CreateStablehloPadOptions(
        fbb_, Int64Vector(op.getEdgePaddingLow()),
        Int64Vector(op.getEdgePaddingHigh()),
        Int64Vector(op.getInteriorPadding()));
    return CreateOperator(tflite::BuiltinOperator_STABLEHLO_PAD,
                          tflite::BuiltinOptions2_StablehloPadOptions,
                          options.Union());
  }

  std::optional<Offset<tflite::Operator>> Build(vhlo::ReduceWindowOpV1 op) {
    const std::optional<int32_t> body = context_.BuildRegionSubgraph(
        op.getBody(), tflite::BuiltinOperator_STABLEHLO_REDUCE_WINDOW);
    if (!body.has_value()) return std::nullopt;
    const auto options = tflite::CreateStablehloReduceWindowOptions(
        fbb_, Int64Vector(op.getWindowDimensions()),
        Int64Vector(op.getWindowStrides()), Int64Vector(op.getBaseDilations()),
        Int64Vector(op.getWindowDilations()), Int64Vector(op.getPadding()),
        *body);
    return CreateOperator(tflite::BuiltinOperator_STABLEHLO_REDUCE_WINDOW,
                          tflite::BuiltinOptions2_StablehloReduceWindowOptions,
                          options.Union());
  }

  std::optional<Offset<tflite::Operator>> Build(vhlo::RngBitGeneratorOpV1 op) {
    tflite::RngAlgorithm algorithm = tflite::RngAlgorithm_DEFAULT;
    switch (cast<vhlo::RngAlgorithmV1Attr>(op.getRngAlgorithm()).getValue()) {
      case vhlo::RngAlgorithmV1::THREE_FRY:
        algorithm = tflite::RngAlgorithm_THREEFRY;
        break;
      case vhlo::RngAlgorithmV1::PHILOX:
        algorithm = tflite::RngAlgorithm_PHILOX;
        break;
      case vhlo::RngAlgorithmV1::DEFAULT:
        break;
    }
    const auto options =
        tflite::CreateStablehloRngBitGeneratorOptions(fbb_, algorithm);
    return CreateOperator(
        tflite::BuiltinOperator_STABLEHLO_RNG_BIT_GENERATOR,
        tflite::BuiltinOptions2_StablehloRngBitGeneratorOptions,
        options.Union());
  }

  Operation* op_;
  const std::vector<int32_t>& inputs_;
  const std::vector<int32_t>& outputs_;
  ModelContext& context_;
  flatbuffers::FlatBufferBuilder& fbb_;
};

}  // namespace

std::optional<Offset<tflite::Operator>> BuildVhloOperator(
    Operation* op, const std::vector<int32_t>& inputs,
    const std::vector<int32_t>& outputs, ModelContext& context) {
  return VhloOperatorBuilder(op, inputs, outputs, context).Build();
}

}  // namespace mlir::TFL::fb_export
