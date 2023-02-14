/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"

#include <stddef.h>
#include <stdlib.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/flatbuffer_operator.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/metrics/error_collector_inst.h"
#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"
#include "tensorflow/compiler/mlir/lite/utils/low_bit_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/stateful_ops_utils.h"
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/lite/delegates/flex/allowlisted_flex_ops.h"
#include "tensorflow/lite/experimental/remat/metadata_util.h"
#include "tensorflow/lite/kernels/internal/kernel_utils.h"
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/versioning/gpu_compatibility.h"
#include "tensorflow/lite/tools/versioning/op_version.h"
#include "tensorflow/lite/tools/versioning/runtime_version.h"
#include "tensorflow/lite/version.h"

using llvm::dyn_cast;
using llvm::formatv;
using llvm::isa;
using llvm::Optional;
using llvm::StringRef;
using llvm::Twine;
using mlir::Dialect;
using mlir::ElementsAttr;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::NoneType;
using mlir::Operation;
using mlir::Region;
using mlir::StringAttr;
using mlir::TensorType;
using mlir::Type;
using mlir::UnknownLoc;
using mlir::Value;
using mlir::WalkResult;
using mlir::func::FuncOp;
using tensorflow::OpOrArgLocNameMapper;
using tensorflow::OpOrArgNameMapper;
using tensorflow::Status;
using tflite::flex::IsAllowlistedFlexOp;
using xla::StatusOr;

template <typename T>
using BufferOffset = flatbuffers::Offset<T>;

template <typename T>
using VectorBufferOffset = flatbuffers::Offset<flatbuffers::Vector<T>>;

using CustomOptionsOffset = VectorBufferOffset<uint8_t>;

namespace error = tensorflow::error;
namespace tfl = mlir::TFL;

ABSL_CONST_INIT const absl::string_view kFlexOpNamePrefix = "Flex";

// Use initial buffer size in flatbuffer builder to be same as the initial size
// used by the TOCO export. (It does not explain rationale for this choice.)
constexpr size_t kInitialBufferSize = 10240;

// Set `isSigned` to false if the `type` is an 8-bit unsigned integer type.
// Since tflite doesn't support unsigned for other types, returns error if
// `isSigned` is set to false for other types.
static StatusOr<tflite::TensorType> GetTFLiteType(Type type,
                                                  bool is_signed = true) {
  if (!is_signed && type.isSignlessInteger(8)) {
    return tflite::TensorType_UINT8;
  }
  if (!is_signed) {
    return Status(error::INVALID_ARGUMENT,
                  "'isSigned' can only be set for 8-bits integer type");
  }

  if (type.isF32()) {
    return tflite::TensorType_FLOAT32;
  } else if (type.isF16()) {
    return tflite::TensorType_FLOAT16;
  } else if (type.isF64()) {
    return tflite::TensorType_FLOAT64;
  } else if (type.isa<mlir::TF::StringType>()) {
    return tflite::TensorType_STRING;
  } else if (type.isa<mlir::TF::Quint8Type>()) {
    return tflite::TensorType_UINT8;
  } else if (auto complex_type = type.dyn_cast<mlir::ComplexType>()) {
    auto ftype = complex_type.getElementType();
    if (ftype.isF32()) {
      return tflite::TensorType_COMPLEX64;
    }
    if (ftype.isF64()) {
      return tflite::TensorType_COMPLEX128;
    }
    return Status(error::INVALID_ARGUMENT, "Unsupported type");
  } else if (auto itype = type.dyn_cast<mlir::IntegerType>()) {
    switch (itype.getWidth()) {
      case 1:
        return tflite::TensorType_BOOL;
      case 4:
        if (itype.isUnsigned()) {
          return Status(error::INVALID_ARGUMENT,
                        "Unsupported 4bit unsigned int type");
        } else {
          return tflite::TensorType_INT4;
        }
      case 8:
        return itype.isUnsigned() ? tflite::TensorType_UINT8
                                  : tflite::TensorType_INT8;
      case 16:
        return itype.isUnsigned() ? tflite::TensorType_UINT16
                                  : tflite::TensorType_INT16;
      case 32:
        return itype.isUnsigned() ? tflite::TensorType_UINT32
                                  : tflite::TensorType_INT32;
      case 64:
        return itype.isUnsigned() ? tflite::TensorType_UINT64
                                  : tflite::TensorType_INT64;
    }
  } else if (auto q_uniform_type =
                 type.dyn_cast<mlir::quant::UniformQuantizedType>()) {
    return GetTFLiteType(q_uniform_type.getStorageType(),
                         q_uniform_type.isSigned());
  } else if (auto q_peraxis_type =
                 type.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
    return GetTFLiteType(q_peraxis_type.getStorageType(),
                         q_peraxis_type.isSigned());
  } else if (auto q_calibrated_type =
                 type.dyn_cast<mlir::quant::CalibratedQuantizedType>()) {
    return GetTFLiteType(q_calibrated_type.getExpressedType());
  } else if (type.isa<mlir::TF::ResourceType>()) {
    return tflite::TensorType_RESOURCE;
  } else if (type.isa<mlir::TF::VariantType>()) {
    return tflite::TensorType_VARIANT;
  }
  // TFLite export fills FLOAT32 for unknown data types. Returning an error
  // for now for safety and this could be revisited when required.
  return Status(error::INVALID_ARGUMENT, "Unsupported type");
}

static bool IsConst(Operation* op) {
  return isa<mlir::func::ConstantOp, mlir::arith::ConstantOp, mlir::TF::ConstOp,
             tfl::ConstOp, tfl::QConstOp, tfl::SparseConstOp,
             tfl::SparseQConstOp, mlir::TFL::NoValueOp>(op);
}

static bool IsTFResourceOp(Operation* op) {
  for (const auto& operand : op->getOperands()) {
    auto elementType = getElementTypeOrSelf(operand.getType());
    if (elementType.isa<mlir::TF::ResourceType>()) {
      return true;
    }
  }
  for (const auto& result : op->getResults()) {
    auto elementType = getElementTypeOrSelf(result.getType());
    if (elementType.isa<mlir::TF::ResourceType>()) {
      return true;
    }
  }
  return false;
}

// Returns whether the current op is not supported by the TF Lite runtime.
static bool IsUnsupportedFlexOp(const std::string& op_name) {
  return op_name == "PartitionedCall" || op_name == "StatefulPartitionedCall";
}

// Create description of operation that could not be converted.
static std::string GetOpDescriptionForDebug(Operation* inst) {
  const int kLargeElementsAttr = 16;
  std::string op_str;
  llvm::raw_string_ostream os(op_str);
  inst->getName().print(os);
  os << "(";
  if (!inst->getOperandTypes().empty()) {
    bool first = true;
    for (Type operand_type : inst->getOperandTypes()) {
      os << (!first ? ", " : "");
      first = false;
      os << operand_type;
    }
  }
  os << ") -> (";
  if (!inst->getResultTypes().empty()) {
    bool first = true;
    for (Type result_type : inst->getResultTypes()) {
      os << (!first ? ", " : "");
      first = false;
      os << result_type;
    }
  }
  os << ")";
  // Print out attributes except for large elementsattributes (which should
  // rarely be the cause why the legalization didn't happen).
  if (!inst->getAttrDictionary().empty()) {
    os << " : {";
    bool first = true;
    for (auto& named_attr : inst->getAttrDictionary()) {
      os << (!first ? ", " : "");
      first = false;
      os << named_attr.getName().getValue() << " = ";
      if (auto element_attr = named_attr.getValue().dyn_cast<ElementsAttr>()) {
        if (element_attr.getNumElements() <= kLargeElementsAttr) {
          element_attr.print(os);
        } else {
          os << "<large>";
        }
      } else {
        named_attr.getValue().print(os);
      }
    }
    os << "}";
  }
  return os.str();
}

// Create a summary with the given information regarding op names and
// descriptions.
static std::string GetOpsSummary(
    const std::map<std::string, std::set<std::string>>& ops,
    const std::string& summary_title) {
  std::string op_str;
  llvm::raw_string_ostream os(op_str);

  std::vector<std::string> keys;
  keys.reserve(ops.size());

  std::vector<std::string> values;
  values.reserve(ops.size());

  for (auto const& op_name_and_details : ops) {
    keys.push_back(op_name_and_details.first);
    for (auto const& op_detail : op_name_and_details.second) {
      values.push_back(op_detail);
    }
  }

  os << summary_title << " ops: " << absl::StrJoin(keys, ", ") << "\n";
  os << "Details:\n\t" << absl::StrJoin(values, "\n\t");

  return os.str();
}

template <typename T>
static bool HasValidTFLiteType(Value value, T& error_handler) {
  // None type is allowed to represent unspecified operands.
  if (value.getType().isa<NoneType>()) return true;

  auto type = value.getType().dyn_cast<TensorType>();
  if (!type) {
    if (auto op = value.getDefiningOp()) {
      error_handler.emitError()
          << '\'' << op << "' should produce value of tensor type instead of "
          << value.getType();
      return false;
    }
    error_handler.emitError("expected tensor type, got ") << value.getType();
    return false;
  }

  Type element_type = type.getElementType();
  auto status = GetTFLiteType(element_type);
  if (!status.ok()) {
    return error_handler.emitError(
               formatv("Failed to convert element type '{0}': {1}",
                       element_type, status.status().error_message())),
           false;
  }
  return true;
}

// Returns true if the module holds all the invariants expected by the
// Translator class.
// TODO(hinsu): Now that translation is done by making a single pass over the
// MLIR module, consider inlining these validation checks at the place where
// these invariants are assumed instead of checking upfront.
static bool IsValidTFLiteMlirModule(ModuleOp module) {
  MLIRContext* context = module.getContext();

  // Verify that module has a function named main.
  FuncOp main_fn = module.lookupSymbol<FuncOp>("main");
  if (!main_fn) {
    int entry_func_count = 0;
    for (auto fn : module.getOps<FuncOp>()) {
      auto attrs = fn->getAttrOfType<mlir::DictionaryAttr>("tf.entry_function");
      if (attrs && !attrs.empty()) {
        ++entry_func_count;
      }
    }

    // Verify that module has a least one enrty function.
    if (entry_func_count == 0) {
      return emitError(UnknownLoc::get(context),
                       "should have a least one entry function"),
             false;
    }
  }

  for (auto fn : module.getOps<FuncOp>()) {
    if (!llvm::hasSingleElement(fn)) {
      return fn.emitError("should have exactly one basic block"), false;
    }
    auto& bb = fn.front();

    for (auto arg : bb.getArguments()) {
      if (!HasValidTFLiteType(arg, fn)) {
        auto elementType = getElementTypeOrSelf(arg.getType());
        if (elementType.isa<mlir::TF::VariantType>()) {
          return fn.emitError(
                     "function argument uses variant type. Currently, the "
                     "variant type is not natively supported in TFLite. Please "
                     "consider not using the variant type: ")
                     << arg.getType(),
                 false;
        }
        return fn.emitError("invalid TFLite type: ") << arg.getType(), false;
      }
    }

    // Verify that all operations except the terminator have exactly one
    // result of type supported by TFLite (or is a ControlType, which
    // will be removed later by ExtractControlEdges.)
    for (auto& inst : bb) {
      if (inst.hasTrait<mlir::OpTrait::IsTerminator>()) break;

      for (auto result : inst.getResults()) {
        if (result.getType().isa<mlir::TFL::ControlType>()) continue;
        if (!HasValidTFLiteType(result, inst)) {
          auto elementType = getElementTypeOrSelf(result.getType());
          if (elementType.isa<mlir::TF::VariantType>()) {
            return inst.emitError(
                       "operand result uses variant type. Currently, the "
                       "variant type is not natively supported in TFLite. "
                       "Please "
                       "consider not using the variant type: ")
                       << result.getType(),
                   false;
          }
          return fn.emitError("invalid TFLite type: ") << result.getType(),
                 false;
        }
      }
    }
  }

  return true;
}

static std::unique_ptr<::tensorflow::NodeDef> GetTensorFlowNodeDef(
    ::mlir::Operation* inst) {
  // We pass empty string for the original node_def name since Flex runtime
  // does not care about this being set correctly on node_def. There is no
  // "easy" (see b/120948529) way yet to get this from MLIR inst.
  auto status_or_node_def = tensorflow::ConvertTFDialectOpToNodeDef(
      inst, /*name=*/"", /*ignore_unregistered_attrs=*/true);
  if (!status_or_node_def.ok()) {
    inst->emitOpError(
        Twine("failed to obtain TensorFlow nodedef with status: " +
              status_or_node_def.status().ToString()));
    return {};
  }
  return std::move(status_or_node_def.value());
}

// Converts a mlir padding StringRef to TfLitePadding.
// Returns std::nullopt if conversion fails.
static std::optional<TfLitePadding> GetTflitePadding(Operation* inst,
                                                     llvm::StringRef padding) {
  const tflite::Padding padding_attr =
      std::move(llvm::StringSwitch<tflite::Padding>(padding)
                    .Case("SAME", tflite::Padding_SAME)
                    .Case("VALID", tflite::Padding_VALID));
  if (padding_attr == tflite::Padding_SAME) {
    return kTfLitePaddingSame;
  }
  if (padding_attr == tflite::Padding_VALID) {
    return kTfLitePaddingValid;
  }

  return inst->emitOpError() << "Invalid padding attribute: " << padding,
         std::nullopt;
}

// Extracts TfLitePoolParams from a TFL custom op.
// Template parameter, TFLOp, should be a TFL custom op containing attributes
// generated from TfLitePoolParams.
// Returns std::nullopt if conversion fails.
template <typename TFLOp>
static std::optional<TfLitePoolParams> GetTflitePoolParams(Operation* inst,
                                                           TFLOp op) {
  TfLitePoolParams pool_params;
  pool_params.stride_height = op.stride_h().getSExtValue();
  pool_params.stride_width = op.stride_w().getSExtValue();
  pool_params.filter_height = op.filter_h().getSExtValue();
  pool_params.filter_width = op.filter_w().getSExtValue();
  const auto padding = GetTflitePadding(inst, op.padding());
  if (padding) {
    pool_params.padding = *padding;
    pool_params.activation = kTfLiteActNone;
    pool_params.computed.padding = TfLitePaddingValues{0, 0, 0, 0};
    return pool_params;
  }

  return std::nullopt;
}

namespace {

using ::mlir::tf_saved_model::kTfSavedModelExportedNamesAttr;
using ::mlir::tf_saved_model::kTfSavedModelIndexPathAttr;

// Helper struct that wraps inputs/outputs of a single SignatureDef.
struct SignatureDefData {
  // Note, we are using maps here to make order deterministic
  // for easily testing only.

  // Inputs defined in the signature def mapped to tensor names.
  std::map<std::string, std::string> inputs;
  // Outputs defined in the signature def mapped to tensor names.
  std::map<std::string, std::string> outputs;
  // Signature key.
  std::string signature_key;
  // Subgraph index.
  uint32_t subgraph_index;
};

// Translates an MLIR module in TFLite dialect to TFLite FlatBuffer.
class Translator {
 public:
  // Translates the given MLIR module into TFLite FlatBuffer format and returns
  // the serialized output. Returns std::nullopt on unsupported, invalid inputs
  // or internal error.
  static std::optional<std::string> Translate(
      ModuleOp module, const toco::TocoFlags& toco_flags,
      const std::unordered_set<std::string>& tags,
      OpOrArgNameMapper* op_or_arg_name_mapper,
      const std::map<std::string, std::string>& metadata);

 private:
  enum class OpType : char { kTfliteBuiltin, kSelectTf, kCustomOp };
  explicit Translator(ModuleOp module, const toco::TocoFlags& toco_flags,
                      const std::unordered_set<std::string>& saved_model_tags,
                      OpOrArgNameMapper* op_or_arg_name_mapper,
                      const std::map<std::string, std::string>& metadata)
      : module_(module),
        name_mapper_(*op_or_arg_name_mapper),
        builder_(kInitialBufferSize),
        saved_model_tags_(saved_model_tags),
        allow_all_select_tf_ops_(toco_flags.allow_all_select_tf_ops()),
        select_user_tf_ops_(toco_flags.select_user_tf_ops().begin(),
                            toco_flags.select_user_tf_ops().end()),
        metadata_(metadata),
        supported_backends_(toco_flags.supported_backends().begin(),
                            toco_flags.supported_backends().end()) {
    // The first buffer must be empty according to the schema definition.
    empty_buffer_ = tflite::CreateBuffer(builder_);
    buffers_.push_back(empty_buffer_);
    if (!toco_flags.force_select_tf_ops()) {
      enabled_op_types_.emplace(OpType::kTfliteBuiltin);
    }
    if (toco_flags.enable_select_tf_ops()) {
      enabled_op_types_.emplace(OpType::kSelectTf);
    }
    if (toco_flags.allow_custom_ops()) {
      enabled_op_types_.emplace(OpType::kCustomOp);
    }
    tf_dialect_ =
        module.getContext()->getOrLoadDialect<mlir::TF::TensorFlowDialect>();
    tfl_dialect_ = module.getContext()
                       ->getOrLoadDialect<mlir::TFL::TensorFlowLiteDialect>();
    // Right now the TF executor dialect is still needed to build NodeDef.
    module.getContext()
        ->getOrLoadDialect<mlir::tf_executor::TensorFlowExecutorDialect>();
  }

  std::optional<std::string> TranslateInternal();

  // Returns TFLite buffer populated with constant value if the operation is
  // TFLite constant operation. Otherwise, returns an empty buffer. Emits error
  // and returns std::nullopt on failure.
  std::optional<BufferOffset<tflite::Buffer>> BuildBuffer(Value value);

  // Build TFLite tensor from the given type. This function is for tfl.lstm
  // intermediates, which should have UniformQuantizedType.
  std::optional<BufferOffset<tflite::Tensor>> BuildTensorFromType(
      mlir::Type type, const std::string& name);

  // Builds TF::VariantType from the given element type. Returns std::nullopt if
  // failure. Returns empty vector if the element type is not TF::VariantType or
  // there is empty TensorType in the TF::VariantType.
  std::optional<std::vector<BufferOffset<tflite::VariantSubType>>>
  BuildTFVariantType(mlir::Type element_type);

  // Builds TFLite tensor from the given value. `buffer_idx` is index of the
  // corresponding buffer. Emits error and returns std::nullopt on failure.
  std::optional<BufferOffset<tflite::Tensor>> BuildTensor(
      Value value, const std::string& name, unsigned buffer_idx,
      const std::optional<BufferOffset<tflite::QuantizationParameters>>&
          quant_parameters);

  // TODO(b/137395003): Legalize tf.IfOp to TFLite dialect, and change the
  // following method to handle TFL::IfOp.
  BufferOffset<tflite::Operator> BuildIfOperator(
      mlir::TF::IfOp op, const std::vector<int32_t>& operands,
      const std::vector<int32_t>& results);

  // Build while operator where cond & body are regions.
  Optional<BufferOffset<tflite::Operator>> BuildWhileOperator(
      mlir::TFL::WhileOp op, const std::vector<int32_t>& operands,
      const std::vector<int32_t>& results);

  // Build call once operator.
  BufferOffset<tflite::Operator> BuildCallOnceOperator(
      mlir::TFL::CallOnceOp op, const std::vector<int32_t>& operands,
      const std::vector<int32_t>& results);

  BufferOffset<tflite::Operator> BuildNumericVerifyOperator(
      mlir::TFL::NumericVerifyOp op, const std::vector<int32_t>& operands,
      const std::vector<int32_t>& results);

  BufferOffset<tflite::Operator> BuildCustomOperator(
      Operation* inst, mlir::TFL::CustomOp op,
      const std::vector<int32_t>& operands,
      const std::vector<int32_t>& results);

  std::optional<CustomOptionsOffset> CreateFlexOpCustomOptions(
      const ::tensorflow::NodeDef& node_def, const mlir::Location& loc);

  std::optional<CustomOptionsOffset> CreateCustomOpCustomOptions(
      const ::tensorflow::NodeDef& node_def, const mlir::Location& loc);

  std::unique_ptr<flexbuffers::Builder> CreateFlexBuilderWithNodeAttrs(
      const ::tensorflow::NodeDef& node_def, const mlir::Location& loc);

  // Returns opcode index for op identified by the op_name, if already
  // available. Otherwise, creates a new OperatorCode using the given `builtin`
  // operator and associates it with `op_name`.
  uint32_t GetOpcodeIndex(const std::string& op_name,
                          tflite::BuiltinOperator builtin);

  // Builds operator for the given operation with specified operand and result
  // tensor indices. Emits an error and returns std::nullopt on failure.
  llvm::Optional<BufferOffset<tflite::Operator>> BuildOperator(
      Operation* inst, std::vector<int32_t> operands,
      const std::vector<int32_t>& results,
      const std::vector<int32_t>& intermediates);

  // Returns the quantization parameters for output value of "quant.stats" op.
  BufferOffset<tflite::QuantizationParameters>
  GetQuantizationForQuantStatsOpOutput(mlir::quantfork::StatisticsOp stats_op);

  // Build a subgraph with a given name out of the region either corresponding
  // to a function's body or while op. Modifies *region by calling
  // ExtractControlEdges.
  std::optional<BufferOffset<tflite::SubGraph>> BuildSubGraph(
      const std::string& name, Region* region, const int index);

  // Modifies *block by unwrapping all ControlNodeOps. The DAG of the control
  // dependencies is returned as a vector of its edges, with node indices into
  // *block.
  std::vector<std::pair<int, int>> ExtractControlEdges(mlir::Block* block);

  // Builds Metadata with the given `name` and buffer `content`.
  BufferOffset<tflite::Metadata> BuildMetadata(StringRef name,
                                               StringRef content);

  // Encodes the `tfl.metadata` dictionary attribute of the module to the
  // metadata section in the final model.
  std::optional<VectorBufferOffset<BufferOffset<tflite::Metadata>>>
  CreateMetadataVector();

  // Builds and returns list of tfl.SignatureDef sections in the model.
  std::optional<VectorBufferOffset<BufferOffset<tflite::SignatureDef>>>
  CreateSignatureDefs(const std::vector<SignatureDefData>& signature_defs);

  // Returns list of offsets for the passed 'items' in TensorMap structure
  // inside the flatbuffer.
  // 'items' is a map from tensor name in signatureDef to tensor name in
  // the subgraph, specified by the 'subgraph_index' argument.
  std::vector<BufferOffset<tflite::TensorMap>> GetList(
      const int subgraph_index,
      const std::map<std::string, std::string>& items);

  // Uses the tf.entry_function attribute (if set) to initialize the op to name
  // mapping.
  void InitializeNamesFromAttribute(FuncOp fn, bool* has_input_attr);

  // Determines if the specified operation op's operand at operand_index
  // is marked as a stateful operand.
  bool IsStatefulOperand(mlir::Operation* op, int operand_index);

  // Returns a unique name for `val`.
  std::string UniqueName(mlir::Value val);

  BufferOffset<tflite::SparsityParameters> BuildSparsityParameters(
      const mlir::TFL::SparsityParameterAttr& s_attr);

  bool EstimateArithmeticCount(int64_t* count);

  // Check compatibility with GPU delegate and returns the compatibility.
  bool CheckGpuDelegateCompatibility(uint8_t* model_buffer_pointer);

  ModuleOp module_;

  tensorflow::OpOrArgNameMapper& name_mapper_;

  flatbuffers::FlatBufferBuilder builder_;
  BufferOffset<tflite::Buffer> empty_buffer_;

  std::vector<BufferOffset<tflite::Buffer>> buffers_;
  // Maps subgraph index and tensor name in the graph to the tensor index.
  absl::flat_hash_map<int, absl::flat_hash_map<std::string, int>>
      tensor_index_map_;

  // Maps op name to index of the corresponding OperatorCode in opcodes_ vector.
  absl::flat_hash_map<std::string, uint32_t> opcode_index_map_;
  std::vector<BufferOffset<tflite::OperatorCode>> opcodes_;

  // Maps function name to index of the corresponding subgraph in the FlatBuffer
  // model.
  absl::flat_hash_map<std::string, int> subgraph_index_map_;
  absl::flat_hash_set<OpType> enabled_op_types_;

  // Points to TensorFlow and TFLite dialects, respectively. nullptr if the
  // dialect is not registered.
  const Dialect* tf_dialect_;
  const Dialect* tfl_dialect_;

  // The failed ops during legalization.
  std::map<std::string, std::set<std::string>> failed_flex_ops_;
  std::map<std::string, std::set<std::string>> failed_custom_ops_;

  // Ops to provide warning messages.
  std::map<std::string, std::set<std::string>> custom_ops_;
  std::map<std::string, std::set<std::string>> flex_ops_;

  // Resource ops to provide warning messages.
  std::map<std::string, std::set<std::string>> resource_ops_;

  // Set of saved model tags, if any.
  const std::unordered_set<std::string> saved_model_tags_;
  // Allows automatic pass through of TF ops as select Tensorflow ops.
  const bool allow_all_select_tf_ops_;
  // User's defined ops allowed with Flex.
  const std::unordered_set<std::string> select_user_tf_ops_;
  // Map of key value pairs of metadata to export.
  const std::map<std::string, std::string> metadata_;
  // User's defined supported backends.
  const std::unordered_set<std::string> supported_backends_;
  // A mapping table to mlir::Operation objects for TFL subgraph and operator
  // index in a flatbuffer.
  std::vector<std::vector<Operation*>> subgraph_op_inst_map_;

  // Will be populated by ExtractControlEdges to contain the control
  // dependencies contained in the ControlNodeOps. Will then be used to populate
  // metadata in the exported flatbuffer file.
  tflite::ModelControlDependencies model_control_dependencies_;
};

bool Translator::EstimateArithmeticCount(int64_t* count) {
  int64_t result = 0;
  bool encounter_undetermined_mac = false;
  module_->walk([&](mlir::TFL::TflArithmeticCountOpInterface op) {
    int64_t mac_count = op.GetArithmeticCount(op);
    if (mac_count < 0) {
      encounter_undetermined_mac = true;
      return;
    }
    result += mac_count;
  });

  *count = result;
  return !encounter_undetermined_mac;
}

std::string Translator::UniqueName(mlir::Value val) {
  return std::string(name_mapper_.GetUniqueName(val));
}

std::optional<BufferOffset<tflite::Buffer>> Translator::BuildBuffer(
    mlir::Value value) {
  auto inst = value.getDefiningOp();
  ElementsAttr attr;
  if (auto cst = dyn_cast<mlir::arith::ConstantOp>(inst)) {
    // arith::ConstantOp have ElementAttr at this point due to validation of the
    // TFLite module.
    attr = cst.getValue().cast<ElementsAttr>();
  } else if (auto cst = dyn_cast<mlir::TF::ConstOp>(inst)) {
    attr = cst.getValue();
  } else if (auto cst = dyn_cast<tfl::ConstOp>(inst)) {
    attr = cst.getValue();
  } else if (auto cst = dyn_cast<tfl::QConstOp>(inst)) {
    attr = cst.getValue();
  } else if (auto cst = dyn_cast<tfl::SparseConstOp>(inst)) {
    attr = cst.getCompressedData();
  } else if (auto cst = dyn_cast<tfl::SparseQConstOp>(inst)) {
    attr = cst.getCompressedData();
  } else {
    return empty_buffer_;
  }

  // TF doesn't currently support 4-bit types (DT_INT4), so we'll run into
  // trouble calling ConvertToTensor(). For now, extract the tensor data from
  // ElementsAttr directly in this and read type from tflite::TensorType instead
  // of tensorflow::DataType.
  auto type = value.getType().cast<TensorType>();
  tflite::TensorType tflite_element_type =
      GetTFLiteType(type.getElementType()).value();
  if (tflite_element_type == tflite::TensorType_INT4) {
    std::vector<uint8_t> data;
    for (mlir::APInt v : attr.getValues<mlir::APInt>()) {
      data.emplace_back(static_cast<uint8_t>(*(v.getRawData())));
    }
    auto packed_buffer = tflite::PackInt4ValuesDensely(data);
    auto buffer_data =
        builder_.CreateVector(packed_buffer.data(), packed_buffer.size());
    return tflite::CreateBuffer(builder_, buffer_data);
  }

  tensorflow::Tensor tensor;
  auto status = tensorflow::ConvertToTensor(attr, &tensor);
  if (!status.ok()) {
    inst->emitError(
        Twine("failed to convert value attribute to tensor with error: " +
              status.ToString()));
    return std::nullopt;
  }

  // TensorFlow and TensorFlow Lite use different string encoding formats.
  // Convert to TensorFlow Lite format is it's a constant string tensor.
  if (tensor.dtype() == tensorflow::DT_STRING) {
    ::tflite::DynamicBuffer dynamic_buffer;
    auto flat = tensor.flat<::tensorflow::tstring>();
    for (int i = 0; i < flat.size(); ++i) {
      const auto& str = flat(i);
      dynamic_buffer.AddString(str.c_str(), str.length());
    }
    char* tensor_buffer;
    int bytes = dynamic_buffer.WriteToBuffer(&tensor_buffer);
    auto buffer_data =
        builder_.CreateVector(reinterpret_cast<uint8_t*>(tensor_buffer), bytes);
    free(tensor_buffer);
    return tflite::CreateBuffer(builder_, buffer_data);
  }

  absl::string_view tensor_data = tensor.tensor_data();
  auto buffer_data = builder_.CreateVector(
      reinterpret_cast<const uint8_t*>(tensor_data.data()), tensor_data.size());
  return tflite::CreateBuffer(builder_, buffer_data);
}

std::optional<std::vector<BufferOffset<tflite::VariantSubType>>>
Translator::BuildTFVariantType(mlir::Type element_type) {
  std::vector<BufferOffset<tflite::VariantSubType>> variant_params;
  auto variant_type = element_type.dyn_cast<mlir::TF::VariantType>();
  if (!variant_type) {
    return variant_params;
  }

  // We only support up to one nested type in tf_type.variant_type.
  if (variant_type.getSubtypes().size() > 1) {
    return std::nullopt;
  }
  if (variant_type.getSubtypes().empty()) {
    return variant_params;
  }
  mlir::TensorType tensor_type = variant_type.getSubtypes().front();
  tflite::TensorType tflite_element_type =
      GetTFLiteType(tensor_type.getElementType()).value();
  std::vector<int32_t> shape;
  if (tensor_type.hasRank()) {
    llvm::ArrayRef<int64_t> shape_ref = tensor_type.getShape();
    shape = std::vector<int32_t>(shape_ref.begin(), shape_ref.end());
  }

  variant_params.push_back(
      tflite::CreateVariantSubType(builder_, builder_.CreateVector(shape),
                                   tflite_element_type, tensor_type.hasRank()));
  return variant_params;
}

std::optional<BufferOffset<tflite::Tensor>> Translator::BuildTensorFromType(
    mlir::Type type, const std::string& name) {
  auto tensor_type = type.cast<TensorType>();

  llvm::ArrayRef<int64_t> shape_ref;
  std::vector<int32_t> shape;

  if (tensor_type.hasRank()) {
    if (tensor_type.hasStaticShape()) {
      shape_ref = tensor_type.getShape();
      shape = std::vector<int32_t>(shape_ref.begin(), shape_ref.end());
    } else {
      return std::nullopt;
    }
  }

  auto element_type = tensor_type.getElementType();
  tflite::TensorType tflite_element_type =
      GetTFLiteType(tensor_type.getElementType()).value();
  std::optional<std::vector<BufferOffset<tflite::VariantSubType>>>
      variant_params = BuildTFVariantType(element_type);
  if (!variant_params.has_value()) {
    return std::nullopt;
  }
  BufferOffset<tflite::QuantizationParameters> q_params = 0;
  if (auto qtype = element_type.dyn_cast<mlir::quant::UniformQuantizedType>()) {
    std::vector<float> scales = {static_cast<float>(qtype.getScale())};
    std::vector<int64_t> zero_points = {qtype.getZeroPoint()};
    q_params = tflite::CreateQuantizationParameters(
        builder_, /*min=*/0, /*max=*/0, builder_.CreateVector<float>(scales),
        builder_.CreateVector<int64_t>(zero_points));
  } else if (auto qtype =
                 element_type
                     .dyn_cast<mlir::quant::CalibratedQuantizedType>()) {
    std::vector<float> mins = {static_cast<float>(qtype.getMin())};
    std::vector<float> maxs = {static_cast<float>(qtype.getMax())};
    q_params = tflite::CreateQuantizationParameters(
        builder_, builder_.CreateVector<float>(mins),
        builder_.CreateVector<float>(maxs));
  }
  return tflite::CreateTensor(
      builder_, builder_.CreateVector(shape), tflite_element_type,
      /*buffer=*/0, builder_.CreateString(name), q_params,
      /*is_variable=*/false, /*sparsity=*/0, /*shape_signature=*/0,
      /*has_rank=*/tensor_type.hasRank(),
      variant_params->empty() ? 0 : builder_.CreateVector(*variant_params));
}

std::optional<BufferOffset<tflite::Tensor>> Translator::BuildTensor(
    Value value, const std::string& name, unsigned buffer_idx,
    const std::optional<BufferOffset<tflite::QuantizationParameters>>&
        quant_parameters) {
  auto type = value.getType().cast<TensorType>();

  // TFLite requires tensor shape only for the inputs and constants.
  // However, we output all known shapes for better round-tripping
  auto check_shape =
      [&](llvm::ArrayRef<int64_t> shape_ref) -> mlir::LogicalResult {
    auto is_out_of_range = [](int64_t dim) {
      return dim > std::numeric_limits<int32_t>::max();
    };

    if (std::any_of(shape_ref.begin(), shape_ref.end(), is_out_of_range))
      return mlir::emitError(
          value.getLoc(),
          "result shape dimensions out of 32 bit int type range");

    return mlir::success();
  };

  std::vector<int32_t> shape;
  std::vector<int32_t> shape_signature;
  auto* inst = value.getDefiningOp();
  if (type.hasStaticShape()) {
    llvm::ArrayRef<int64_t> shape_ref = type.getShape();
    if (mlir::failed(check_shape(shape_ref))) return std::nullopt;

    shape = std::vector<int32_t>(shape_ref.begin(), shape_ref.end());
  } else if (inst && IsConst(inst)) {
    // Const op can have a result of dynamic shaped type (e.g. due to constant
    // folding), but we can still derive the shape of a constant tensor for
    // its attribute type.
    auto tensor_attr = inst->getAttr("value").cast<mlir::TypedAttr>();
    llvm::ArrayRef<int64_t> shape_ref =
        tensor_attr.getType().cast<TensorType>().getShape();
    if (mlir::failed(check_shape(shape_ref))) return std::nullopt;

    shape = std::vector<int32_t>(shape_ref.begin(), shape_ref.end());
  } else if (type.hasRank()) {
    llvm::ArrayRef<int64_t> shape_ref = type.getShape();
    if (mlir::failed(check_shape(shape_ref))) return std::nullopt;

    shape.reserve(shape_ref.size());
    for (auto& dim : shape_ref) {
      // translate dynamic shapes from mlir to tfl values
      shape.push_back(
          dim == mlir::ShapedType::kDynamic ? 1 : static_cast<int>(dim));
      shape_signature.push_back(static_cast<int>(
          dim == mlir::ShapedType::kDynamic ? tensorflow::kTFDynamicSize
                                            : dim));
    }
  }

  BufferOffset<tflite::SparsityParameters> s_params = 0;
  if (auto* inst = value.getDefiningOp()) {
    if (auto cst = dyn_cast<tfl::SparseConstOp>(inst)) {
      s_params = BuildSparsityParameters(cst.getSParam());
    } else if (auto cst = dyn_cast<tfl::SparseQConstOp>(inst)) {
      s_params = BuildSparsityParameters(cst.getSParam());
    }
  }

  Type element_type = type.getElementType();
  tflite::TensorType tflite_element_type =
      GetTFLiteType(type.getElementType()).value();

  std::optional<std::vector<BufferOffset<tflite::VariantSubType>>>
      variant_params = BuildTFVariantType(element_type);
  if (!variant_params.has_value()) {
    return std::nullopt;
  }

  BufferOffset<tflite::QuantizationParameters> q_params;
  if (auto qtype = element_type.dyn_cast<mlir::quant::UniformQuantizedType>()) {
    std::vector<float> scales = {static_cast<float>(qtype.getScale())};
    std::vector<int64_t> zero_points = {qtype.getZeroPoint()};
    q_params = tflite::CreateQuantizationParameters(
        // min and max values are not stored in the quantized type from MLIR, so
        // both are set to 0 in the flatbuffer when they are exported.
        builder_, /*min=*/0, /*max=*/0, builder_.CreateVector<float>(scales),
        builder_.CreateVector<int64_t>(zero_points));
  } else if (auto qtype =
                 element_type
                     .dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
    std::vector<float> scales(qtype.getScales().begin(),
                              qtype.getScales().end());
    std::vector<int64_t> zero_points(qtype.getZeroPoints().begin(),
                                     qtype.getZeroPoints().end());
    q_params = tflite::CreateQuantizationParameters(
        builder_, /*min=*/0, /*max=*/0, builder_.CreateVector<float>(scales),
        builder_.CreateVector<int64_t>(zero_points),
        tflite::QuantizationDetails_NONE, /*details=*/0,
        qtype.getQuantizedDimension());
  } else if (quant_parameters.has_value()) {
    q_params = quant_parameters.value();
  } else {
    q_params = tflite::CreateQuantizationParameters(builder_);
  }
  // Check if the value's uses includes an op and usage at an operand index
  // marked as a stateful. If so, set the tensor's is_variable as true
  // This is v1 ref variable semantics in the TFLite runtime.
  bool is_variable = false;
  for (auto& use : value.getUses()) {
    is_variable = IsStatefulOperand(use.getOwner(), use.getOperandNumber());
    if (is_variable) {
      break;
    }
  }

  bool has_rank = type.hasRank();

  if (shape_signature.empty()) {
    return tflite::CreateTensor(
        builder_, builder_.CreateVector(shape), tflite_element_type,
        (is_variable ? 0 : buffer_idx), builder_.CreateString(name), q_params,
        /*is_variable=*/is_variable, s_params, /*shape_signature=*/0,
        /*has_rank=*/has_rank,
        variant_params->empty() ? 0 : builder_.CreateVector(*variant_params));
  } else {
    return tflite::CreateTensor(
        builder_, builder_.CreateVector(shape), tflite_element_type,
        (is_variable ? 0 : buffer_idx), builder_.CreateString(name), q_params,
        /*is_variable=*/is_variable, s_params,
        /*shape_signature=*/builder_.CreateVector(shape_signature),
        /*has_rank=*/has_rank,
        variant_params->empty() ? 0 : builder_.CreateVector(*variant_params));
  }
}

BufferOffset<tflite::Operator> Translator::BuildIfOperator(
    mlir::TF::IfOp op, const std::vector<int32_t>& operands,
    const std::vector<int32_t>& results) {
  auto opcode_index = GetOpcodeIndex("if", tflite::BuiltinOperator_IF);
  int then_subgraph_index = subgraph_index_map_.at(op.getThenBranch().str());
  int else_subgraph_index = subgraph_index_map_.at(op.getElseBranch().str());
  auto builtin_options = tflite::CreateIfOptions(builder_, then_subgraph_index,
                                                 else_subgraph_index)
                             .Union();
  auto inputs = builder_.CreateVector(operands);
  auto outputs = builder_.CreateVector(results);
  return tflite::CreateOperator(builder_, opcode_index, inputs, outputs,
                                tflite::BuiltinOptions_IfOptions,
                                builtin_options);
}

BufferOffset<tflite::Operator> Translator::BuildCallOnceOperator(
    mlir::TFL::CallOnceOp op, const std::vector<int32_t>& operands,
    const std::vector<int32_t>& results) {
  auto opcode_index =
      GetOpcodeIndex("call_once", tflite::BuiltinOperator_CALL_ONCE);
  int init_subgraph_index =
      subgraph_index_map_.at(op.getSessionInitFunction().str());
  auto builtin_options =
      tflite::CreateCallOnceOptions(builder_, init_subgraph_index).Union();
  auto inputs = builder_.CreateVector(operands);
  auto outputs = builder_.CreateVector(results);
  return tflite::CreateOperator(builder_, opcode_index, inputs, outputs,
                                tflite::BuiltinOptions_CallOnceOptions,
                                builtin_options);
}

llvm::Optional<BufferOffset<tflite::Operator>> Translator::BuildWhileOperator(
    mlir::TFL::WhileOp op, const std::vector<int32_t>& operands,
    const std::vector<int32_t>& results) {
  auto opcode_index = GetOpcodeIndex("while", tflite::BuiltinOperator_WHILE);
  auto get_call_index = [&](mlir::Block& b) -> std::optional<int> {
    if (b.getOperations().size() != 2) return std::nullopt;
    if (auto call_op = dyn_cast<mlir::func::CallOp>(b.front()))
      return subgraph_index_map_.at(call_op.getCallee().str());
    return std::nullopt;
  };
  auto body_subgraph_index = get_call_index(op.getBody().front());
  auto cond_subgraph_index = get_call_index(op.getCond().front());
  if (!body_subgraph_index || !cond_subgraph_index)
    return op.emitOpError("only single call cond/body while export supported"),
           std::nullopt;
  auto builtin_options =
      tflite::CreateWhileOptions(builder_, *cond_subgraph_index,
                                 *body_subgraph_index)
          .Union();
  auto inputs = builder_.CreateVector(operands);
  auto outputs = builder_.CreateVector(results);
  return tflite::CreateOperator(builder_, opcode_index, inputs, outputs,
                                tflite::BuiltinOptions_WhileOptions,
                                builtin_options);
}

BufferOffset<tflite::Operator> Translator::BuildNumericVerifyOperator(
    mlir::TFL::NumericVerifyOp op, const std::vector<int32_t>& operands,
    const std::vector<int32_t>& results) {
  float tolerance = op.getTolerance().convertToFloat();
  bool log_if_failed = op.getLogIfFailed();
  auto fbb = std::make_unique<flexbuffers::Builder>();
  fbb->Map([&]() {
    fbb->Float("tolerance", tolerance);
    fbb->Bool("log_if_failed", log_if_failed);
  });
  fbb->Finish();
  auto f = std::unique_ptr<flexbuffers::Builder>(fbb.release());
  auto custom_option = f->GetBuffer();
  auto opcode_index =
      GetOpcodeIndex("NumericVerify", tflite::BuiltinOperator_CUSTOM);
  return tflite::CreateOperator(
      builder_, opcode_index, builder_.CreateVector(operands),
      builder_.CreateVector(results), tflite::BuiltinOptions_NONE,
      /*builtin_options=*/0, builder_.CreateVector<uint8_t>(custom_option),
      tflite::CustomOptionsFormat_FLEXBUFFERS);
}

BufferOffset<tflite::Operator> Translator::BuildCustomOperator(
    Operation* inst, mlir::TFL::CustomOp op,
    const std::vector<int32_t>& operands, const std::vector<int32_t>& results) {
  const std::string attrs =
      op.getCustomOption().cast<mlir::TFL::ConstBytesAttr>().getValue().str();
  std::vector<uint8_t> custom_option_vector(attrs.size());
  memcpy(custom_option_vector.data(), attrs.data(), attrs.size());
  auto opcode_index =
      GetOpcodeIndex(op.getCustomCode().str(), tflite::BuiltinOperator_CUSTOM);
  return tflite::CreateOperator(
      builder_, opcode_index, builder_.CreateVector(operands),
      builder_.CreateVector(results), tflite::BuiltinOptions_NONE,
      /*builtin_options=*/0,
      builder_.CreateVector<uint8_t>(custom_option_vector),
      tflite::CustomOptionsFormat_FLEXBUFFERS);
}

std::optional<CustomOptionsOffset> Translator::CreateFlexOpCustomOptions(
    const ::tensorflow::NodeDef& node_def, const mlir::Location& loc) {
  std::string node_def_str;
  if (!node_def.SerializeToString(&node_def_str)) {
    return emitError(loc, "failed to serialize tensorflow node_def"),
           std::nullopt;
  }

  auto flex_builder = std::make_unique<flexbuffers::Builder>();
  flex_builder->Vector([&]() {
    flex_builder->String(node_def.op());
    flex_builder->String(node_def_str);
  });
  flex_builder->Finish();
  return builder_.CreateVector(flex_builder->GetBuffer());
}

std::optional<CustomOptionsOffset> Translator::CreateCustomOpCustomOptions(
    const ::tensorflow::NodeDef& node_def, const mlir::Location& loc) {
  auto flex_builder = CreateFlexBuilderWithNodeAttrs(node_def, loc);
  return builder_.CreateVector(flex_builder->GetBuffer());
}

std::unique_ptr<flexbuffers::Builder>
Translator::CreateFlexBuilderWithNodeAttrs(
    const ::tensorflow::NodeDef& node_def, const mlir::Location& loc) {
  auto flex_builder = std::make_unique<flexbuffers::Builder>();
  size_t map_start = flex_builder->StartMap();
  using Item = std::pair<std::string, ::tensorflow::AttrValue>;
  std::vector<Item> attrs(node_def.attr().begin(), node_def.attr().end());
  std::sort(attrs.begin(), attrs.end(),
            [](Item& p1, Item& p2) -> bool { return p1.first < p2.first; });
  for (const Item& pair : attrs) {
    const char* key = pair.first.c_str();
    const ::tensorflow::AttrValue& attr = pair.second;
    switch (attr.value_case()) {
      case ::tensorflow::AttrValue::kS:
        flex_builder->String(key, attr.s());
        break;
      case ::tensorflow::AttrValue::kType: {
        auto status_or_tfl_type = tflite::TfTypeToTflType(attr.type());
        if (status_or_tfl_type.ok()) {
          flex_builder->Int(key, status_or_tfl_type.value());
        } else {
          emitWarning(loc, "ignoring unsupported tensorflow type: ")
              << std::to_string(attr.type());
        }
        break;
      }
      case ::tensorflow::AttrValue::kI:
        flex_builder->Int(key, attr.i());
        break;
      case ::tensorflow::AttrValue::kF:
        flex_builder->Float(key, attr.f());
        break;
      case ::tensorflow::AttrValue::kB:
        flex_builder->Bool(key, attr.b());
        break;
      case tensorflow::AttrValue::kList:
        if (attr.list().s_size() > 0) {
          auto start = flex_builder->StartVector(key);
          for (const std::string& v : attr.list().s()) {
            flex_builder->Add(v);
          }
          flex_builder->EndVector(start, /*typed=*/true, /*fixed=*/false);
        } else if (attr.list().i_size() > 0) {
          auto start = flex_builder->StartVector(key);
          for (const int64_t v : attr.list().i()) {
            flex_builder->Add(v);
          }
          flex_builder->EndVector(start, /*typed=*/true, /*fixed=*/false);
        } else if (attr.list().f_size() > 0) {
          auto start = flex_builder->StartVector(key);
          for (const float v : attr.list().f()) {
            flex_builder->Add(v);
          }
          flex_builder->EndVector(start, /*typed=*/true, /*fixed=*/false);
        } else {
          emitWarning(loc,
                      "ignoring unsupported type in list attribute with key: ")
              << key;
        }
        break;
      default:
        emitWarning(loc, "ignoring unsupported attribute type with key: ")
            << key;
        break;
    }
  }
  flex_builder->EndMap(map_start);
  flex_builder->Finish();
  return flex_builder;
}

uint32_t Translator::GetOpcodeIndex(const std::string& op_name,
                                    tflite::BuiltinOperator builtin) {
  auto it = opcode_index_map_.insert({op_name, 0});

  // If the insert succeeded, the opcode has not been created already. Create a
  // new operator code and update its index value in the map.
  if (it.second) {
    it.first->second = opcodes_.size();
    auto custom_code = builtin == tflite::BuiltinOperator_CUSTOM
                           ? builder_.CreateString(op_name)
                           : BufferOffset<flatbuffers::String>();
    // Use version 0 for builtin op. This is a way to serialize version field to
    // flatbuffer (since 0 is non default) and it will be corrected later.
    int32_t op_version = builtin != tflite::BuiltinOperator_CUSTOM ? 0 : 1;
    opcodes_.push_back(CreateOperatorCode(builder_, /*builtin_code=*/builtin,
                                          custom_code, op_version));
  }
  return it.first->second;
}

llvm::Optional<BufferOffset<tflite::Operator>> Translator::BuildOperator(
    Operation* inst, std::vector<int32_t> operands,
    const std::vector<int32_t>& results,
    const std::vector<int32_t>& intermediates) {
  const auto* dialect = inst->getDialect();
  if (!dialect) {
    inst->emitOpError("dialect is not registered");
    return std::nullopt;
  }

  // If TFLite built in op, create operator as a builtin op.
  if (dialect == tfl_dialect_) {
    // Only if built-in TFLite op emission is enabled, would legalization have
    // converted any TF->TFL.
    if (!enabled_op_types_.contains(OpType::kTfliteBuiltin)) {
      return inst->emitOpError(
                 "is a TFLite builtin op but builtin emission is not enabled"),
             std::nullopt;
    }

    auto builtin_code = GetBuiltinOpCode(inst);
    if (!builtin_code) {
      if (auto verify_op = dyn_cast<mlir::TFL::NumericVerifyOp>(inst)) {
        return BuildNumericVerifyOperator(verify_op, operands, results);
      }
      if (auto custom_op = dyn_cast<mlir::TFL::CustomOp>(inst)) {
        return BuildCustomOperator(inst, custom_op, operands, results);
      }
      if (auto whileOp = dyn_cast<mlir::TFL::WhileOp>(inst)) {
        if (inst->getNumOperands() != inst->getNumResults()) {
          inst->emitOpError(
              "number of operands and results don't match, only canonical "
              "TFL While supported");
          return std::nullopt;
        }
        return BuildWhileOperator(whileOp, operands, results);
      }

      inst->emitOpError("is not a supported TFLite op");
      return std::nullopt;
    }

    if (*builtin_code == tflite::BuiltinOperator_CALL_ONCE) {
      if (auto initOp = dyn_cast<mlir::TFL::CallOnceOp>(inst)) {
        return BuildCallOnceOperator(initOp, operands, results);
      }
    }

    std::string op_name = inst->getName().getStringRef().str();
    uint32_t opcode_index = GetOpcodeIndex(op_name, *builtin_code);

    // If this is TransposeConv we need to do a special case of ignoring the
    // optional tensor, to allow newly created models to run on old runtimes.
    if (*builtin_code == tflite::BuiltinOperator_TRANSPOSE_CONV) {
      if (operands.size() == 4 && operands.at(3) == -1) {
        operands.pop_back();
      }
    }

    auto offset = CreateFlatBufferOperator(inst, opcode_index, operands,
                                           results, intermediates, &builder_);
    if (!offset) {
      inst->emitOpError("is not a supported TFLite op");
    }
    return offset;
  }

  if (dialect == tf_dialect_) {
    if (auto ifOp = dyn_cast<mlir::TF::IfOp>(inst)) {
      return BuildIfOperator(ifOp, operands, results);
    }

    CustomOptionsOffset custom_options;

    // Ops in TF dialect can either be custom ops or flex ops.
    // The reason we go directly from TensorFlow dialect MLIR to tensorflow
    // node instead of going to TF table gen'd ops via generated code is that
    // we do not want to restrict custom and flex op conversion support to
    // only those TF ops that are currently registered in MLIR. The current
    // model is of an open op system.
    //
    //  The following algorithm is followed:
    //   if flex is enabled and the op is allowlisted as flex
    //     we emit op as flex.
    //   if custom is enabled
    //    we emit the op as custom.
    auto node_def = GetTensorFlowNodeDef(inst);
    if (!node_def) {
      return std::nullopt;
    }

    std::string op_name = node_def->op();
    std::string op_desc = GetOpDescriptionForDebug(inst);

    if (IsTFResourceOp(inst)) {
      resource_ops_[op_name].insert(op_desc);
    }

    const bool is_allowed_flex_op =
        !IsUnsupportedFlexOp(node_def->op()) &&
        (IsAllowlistedFlexOp(node_def->op()) ||
         (((select_user_tf_ops_.count(node_def->op()) != 0) ||
           allow_all_select_tf_ops_) &&
          (tensorflow::OpRegistry::Global()->LookUp(node_def->op()) !=
           nullptr)));

    // Flex op case
    // Eventually, the allowlist will go away and we will rely on some TF op
    // trait (e.g. No side effect) to determine if it is a supported "Flex"
    // op or not.
    if (is_allowed_flex_op && enabled_op_types_.contains(OpType::kSelectTf)) {
      // Construct ops as flex op encoding TensorFlow node definition
      // as custom options.
      // Flex ops are named with the kFlexOpNamePrefix prefix to the actual
      // TF op name.
      op_name = std::string(kFlexOpNamePrefix) + node_def->op();
      if (auto options = CreateFlexOpCustomOptions(*node_def, inst->getLoc())) {
        custom_options = *options;
      } else {
        return std::nullopt;
      }

      // Gather flex ops.
      flex_ops_[op_name].insert(op_desc);
    } else if (enabled_op_types_.contains(OpType::kCustomOp)) {
      // Generic case of custom ops - write using flex buffers since that
      // is the only custom options supported by TFLite today.
      op_name = node_def->op();
      if (auto options =
              CreateCustomOpCustomOptions(*node_def, inst->getLoc())) {
        custom_options = *options;
      } else {
        return std::nullopt;
      }

      // Gather custom ops.
      custom_ops_[op_name].insert(op_desc);
    } else {
      // Insert failed op to `flex_ops` or `custom_ops`.
      if (is_allowed_flex_op) {
        failed_flex_ops_[op_name].insert(op_desc);
        tfl::AttachErrorCode(
            inst->emitOpError("is neither a custom op nor a flex op"),
            tflite::metrics::ConverterErrorData::ERROR_NEEDS_FLEX_OPS);
      } else {
        failed_custom_ops_[op_name].insert(op_desc);
        tfl::AttachErrorCode(
            inst->emitOpError("is neither a custom op nor a flex op"),
            tflite::metrics::ConverterErrorData::ERROR_NEEDS_CUSTOM_OPS);
      }
      return std::nullopt;
    }

    uint32_t opcode_index =
        GetOpcodeIndex(op_name, tflite::BuiltinOperator_CUSTOM);
    auto inputs = builder_.CreateVector(operands);
    auto outputs = builder_.CreateVector(results);

    return tflite::CreateOperator(builder_, opcode_index, inputs, outputs,
                                  tflite::BuiltinOptions_NONE,
                                  /*builtin_options=*/0,
                                  /*custom_options=*/custom_options,
                                  tflite::CustomOptionsFormat_FLEXBUFFERS,
                                  /*mutating_variable_inputs=*/0);
  }

  return inst->emitOpError(
             "is not any of a builtin TFLite op, a flex TensorFlow op or a "
             "custom TensorFlow op"),
         std::nullopt;
}

void Translator::InitializeNamesFromAttribute(FuncOp fn, bool* has_input_attr) {
  auto dict_attr = fn->getAttrOfType<mlir::DictionaryAttr>("tf.entry_function");
  if (!dict_attr) return;

  llvm::SmallVector<llvm::StringRef, 2> input_names;
  llvm::SmallVector<llvm::StringRef, 2> output_names;
  if (auto str = dict_attr.get("inputs").dyn_cast_or_null<mlir::StringAttr>()) {
    str.getValue().split(input_names, ',', /*MaxSplit=*/-1,
                         /*KeepEmpty=*/false);
    if (input_names.size() != fn.getNumArguments()) {
      fn.emitWarning() << "invalid entry function specification";
      return;
    }
    for (const auto& it : llvm::enumerate(fn.getArguments())) {
      name_mapper_.InitOpName(it.value(), input_names[it.index()].trim());
    }
    *has_input_attr = true;
  }

  if (auto str =
          dict_attr.get("outputs").dyn_cast_or_null<mlir::StringAttr>()) {
    str.getValue().split(output_names, ',', /*MaxSplit=*/-1,
                         /*KeepEmpty=*/false);
    auto term = fn.back().getTerminator();
    if (output_names.size() != term->getNumOperands()) {
      fn.emitWarning() << "output names (" << output_names.size()
                       << ") != terminator operands (" << term->getNumOperands()
                       << ")";
      return;
    }
    for (const auto& it : llvm::enumerate(term->getOperands())) {
      name_mapper_.InitOpName(it.value(), output_names[it.index()].trim());
    }
  }
}

bool Translator::IsStatefulOperand(mlir::Operation* op, int operand_index) {
  std::vector<int> operand_indices;
  if (!mlir::TFL::IsStatefulOp(op, &operand_indices)) return false;
  return absl::c_find(operand_indices, operand_index) != operand_indices.end();
}

BufferOffset<tflite::QuantizationParameters>
Translator::GetQuantizationForQuantStatsOpOutput(
    mlir::quantfork::StatisticsOp stats_op) {
  auto layer_stats = stats_op.getLayerStats().cast<mlir::DenseFPElementsAttr>();
  std::optional<mlir::ElementsAttr> axis_stats = stats_op.getAxisStats();
  std::optional<uint64_t> axis = stats_op.getAxis();
  std::vector<float> mins, maxs;
  mlir::DenseFPElementsAttr min_max_attr =
      axis_stats.has_value()
          ? axis_stats.value().cast<mlir::DenseFPElementsAttr>()
          : layer_stats;

  for (const auto& index_and_value :
       llvm::enumerate(min_max_attr.getValues<llvm::APFloat>())) {
    const llvm::APFloat value = index_and_value.value();
    if (index_and_value.index() % 2 == 0) {
      mins.push_back(value.convertToFloat());
    } else {
      maxs.push_back(value.convertToFloat());
    }
  }

  return tflite::CreateQuantizationParameters(
      builder_, builder_.CreateVector<float>(mins),
      builder_.CreateVector<float>(maxs), /*scale=*/0, /*zero_point=*/0,
      tflite::QuantizationDetails_NONE, /*details=*/0,
      /*quantized_dimension=*/axis.has_value() ? axis.value() : 0);
}

std::optional<BufferOffset<tflite::SubGraph>> Translator::BuildSubGraph(
    const std::string& name, Region* region, const int index) {
  const auto control_edges = ExtractControlEdges(&region->front());
  bool has_input_attr = false;
  if (auto fn = dyn_cast<FuncOp>(region->getParentOp())) {
    InitializeNamesFromAttribute(fn, &has_input_attr);
  }
  std::vector<BufferOffset<tflite::Tensor>> tensors;
  llvm::DenseMap<Value, int> tensor_index_map;

  // Builds tensor and buffer for argument or operation result. Returns false
  // on failure.
  auto build_tensor_and_buffer = [&](Value value, const int subgraph_index,
                                     const std::string& tensor_name) {
    // NoneType represents optional and may be skipped here.
    if (value.getType().isa<NoneType>()) {
      return true;
    }

    tensor_index_map.insert({value, tensors.size()});
    tensor_index_map_[subgraph_index][tensor_name] = tensors.size();
    std::optional<BufferOffset<tflite::QuantizationParameters>>
        quant_parameters;
    if (value.hasOneUse()) {
      auto stats_op =
          llvm::dyn_cast<mlir::quantfork::StatisticsOp>(*value.user_begin());
      if (stats_op) {
        quant_parameters = GetQuantizationForQuantStatsOpOutput(stats_op);
      }
    }
    auto tensor_or =
        BuildTensor(value, tensor_name, buffers_.size(), quant_parameters);
    if (!tensor_or) return false;
    tensors.push_back(*tensor_or);

    // TODO(ashwinm): Check if for stateful tensors, if it is also needed to
    // make the Buffer empty apart from setting the buffer_idx=0 in the
    // Tensor. This does not seem to affect runtime behavior for RNN/LSTM,
    // but would be good for reducing memory footprint.
    if (value.getDefiningOp()) {
      auto buffer_or = BuildBuffer(value);
      if (!buffer_or) return false;
      buffers_.push_back(*buffer_or);
    } else {
      buffers_.push_back(empty_buffer_);
    }
    return true;
  };

  std::vector<BufferOffset<tflite::Operator>> operators;

  // Maps positions of operations in bb to positions in operators
  llvm::DenseMap<int, int> operation_index_to_operator_index;
  std::vector<Operation*> operators_in_mlir;
  auto& bb = region->front();

  // Main function's arguments are first passed to `input` op so they don't
  // have associated tensor and buffer. Build FlatBuffer tensor and buffer for
  // other functions.
  for (unsigned i = 0, e = bb.getNumArguments(); i < e; ++i) {
    mlir::BlockArgument arg = bb.getArgument(i);
    std::string tensor_name;
    if (has_input_attr)
      tensor_name = std::string(name_mapper_.GetUniqueName(arg));
    if (tensor_name.empty()) tensor_name = absl::StrCat("arg", i);
    if (!build_tensor_and_buffer(arg, index, tensor_name)) return std::nullopt;
  }

  bool failed_once = false;
  for (auto& item : llvm::enumerate(bb)) {
    Operation& inst = item.value();
    const int operation_index = item.index();
    if (inst.hasTrait<mlir::OpTrait::IsTerminator>()) break;
    // For "quant.stats" op, it's used to store the quantization parameters info
    // and its output should be then replaced by its input value.
    if (auto quant_stats_op =
            llvm::dyn_cast<mlir::quantfork::StatisticsOp>(inst)) {
      continue;
    }
    std::vector<int32_t> intermediates;
    // Build intermediate tensors for tfl.lstm and insert these tensors into
    // flatbuffer.
    if (llvm::isa<mlir::TFL::LSTMOp, mlir::TFL::UnidirectionalSequenceLSTMOp>(
            inst)) {
      std::vector<std::string> intermediate_names = {
          "input_to_input_intermediate", "input_to_forget_intermediate",
          "input_to_cell_intermediate", "input_to_output_intermediate",
          "effective_hidden_scale_intermediate"};
      for (const std::string& intermediate : intermediate_names) {
        auto intermediate_attr = inst.getAttr(intermediate);
        if (auto attr = intermediate_attr.dyn_cast_or_null<mlir::TypeAttr>()) {
          Type qtype = attr.getValue();
          auto tensor_or = BuildTensorFromType(
              qtype, name_mapper_.GetUniqueName(intermediate).str());
          if (!tensor_or.has_value()) {
            continue;
          } else {
            intermediates.push_back(tensors.size());
            tensors.push_back(tensor_or.value());
          }
        }
      }
    }

    for (auto val : inst.getResults()) {
      std::string tensor_name = UniqueName(val);
      // For "tfl.numeric_verify" op, the name is used to find out the original
      // activation tensor rather than its own unique name in the visualization
      // or debugging tools.
      auto builtin_code = GetBuiltinOpCode(&inst);
      if (!builtin_code && dyn_cast<mlir::TFL::NumericVerifyOp>(&inst)) {
        // The first operand is the quantized activation, the target of this
        // NumericVerify op.
        auto quantized_op_val = inst.getOperands().front();
        tensor_name = "NumericVerify/" + UniqueName(quantized_op_val) + ":" +
                      std::to_string(tensor_index_map[quantized_op_val]);
      }
      if (!build_tensor_and_buffer(val, index, tensor_name))
        return std::nullopt;
    }

    // Skip constant ops as they don't represent a TFLite operator.
    if (IsConst(&inst)) continue;

    // Fetch operand and result tensor indices.
    std::vector<int32_t> results;
    results.reserve(inst.getNumResults());
    for (auto result : inst.getResults()) {
      results.push_back(tensor_index_map.lookup(result));
    }
    Operation* real_inst = &inst;
    std::vector<int32_t> operands;
    operands.reserve(real_inst->getNumOperands());
    for (auto operand : real_inst->getOperands()) {
      if (operand.getType().isa<NoneType>())
        operands.push_back(kTfLiteOptionalTensor);
      else if (auto stats_op =
                   llvm::dyn_cast_or_null<mlir::quantfork::StatisticsOp>(
                       operand.getDefiningOp()))
        operands.push_back(tensor_index_map.lookup(stats_op.getArg()));
      else
        operands.push_back(tensor_index_map.lookup(operand));
    }

    // CustomTfOp is just a wrapper around a TF op, we export the custom Op
    // not the wrapper, so we fetch the op from the region.
    if (auto custom_op = dyn_cast<mlir::TFL::CustomTfOp>(inst)) {
      // If we have custom op with a region, then use the first op in the
      // region, if it exists, otherwise just use params for custom op.
      if (!custom_op.getBody().empty()) {
        real_inst = &custom_op.getBody().front().front();
      } else {
        module_.emitError(
            "Invalid CustomTfOp: Custom TF Op have empty region.");
      }
    }
    if (auto tfl_operator =
            BuildOperator(real_inst, operands, results, intermediates)) {
      operation_index_to_operator_index.try_emplace(operation_index,
                                                    operators.size());
      operators.push_back(*tfl_operator);
      operators_in_mlir.push_back(real_inst);
    } else {
      failed_once = true;
    }
  }
  if (index + 1 > subgraph_op_inst_map_.size()) {
    subgraph_op_inst_map_.resize(index + 1);
  }
  subgraph_op_inst_map_[index] = operators_in_mlir;
  if (failed_once) return std::nullopt;

  // Get input and output tensor indices for the subgraph.
  std::vector<int32_t> inputs, outputs;
  for (auto arg : bb.getArguments()) {
    inputs.push_back(tensor_index_map[arg]);
  }
  for (auto result : bb.getTerminator()->getOperands()) {
    outputs.push_back(tensor_index_map[result]);
  }
  for (const auto& [from, to] : control_edges) {
    for (int what : {from, to}) {
      if (operation_index_to_operator_index.count(what) == 0) {
        module_.emitError(
            "dangling control edge -- at least one vertex Operation isn't a "
            "flatbuffer Operator.");
      }
    }
    model_control_dependencies_[index].emplace_back(
        operation_index_to_operator_index[from],
        operation_index_to_operator_index[to]);
  }
  return tflite::CreateSubGraph(
      builder_, builder_.CreateVector(tensors), builder_.CreateVector(inputs),
      builder_.CreateVector(outputs), builder_.CreateVector(operators),
      /*name=*/builder_.CreateString(name));
}

BufferOffset<tflite::Metadata> Translator::BuildMetadata(StringRef name,
                                                         StringRef content) {
  auto buffer_index = buffers_.size();
  auto buffer_data = builder_.CreateVector(
      reinterpret_cast<const uint8_t*>(content.data()), content.size());
  buffers_.push_back(tflite::CreateBuffer(builder_, buffer_data));
  return tflite::CreateMetadataDirect(builder_, name.data(), buffer_index);
}

std::optional<VectorBufferOffset<BufferOffset<tflite::Metadata>>>
Translator::CreateMetadataVector() {
  auto dict_attr = module_->getAttrOfType<mlir::DictionaryAttr>("tfl.metadata");
  std::vector<BufferOffset<tflite::Metadata>> metadata;
  if (dict_attr) {
    for (const auto& named_attr : dict_attr) {
      StringRef name = named_attr.getName();
      mlir::Attribute attr = named_attr.getValue();
      if (auto content = attr.dyn_cast<StringAttr>()) {
        metadata.push_back(BuildMetadata(name, content.getValue()));
      } else {
        module_.emitError(
            "all values in tfl.metadata's dictionary key-value pairs should be "
            "string attributes");
        return std::nullopt;
      }
    }
  }
  // Runtime version string is generated after we update the op
  // versions. Here we put a 16-byte dummy string as a placeholder. We choose
  // 16-byte because it's the alignment of buffers in flatbuffer, so it won't
  // cause any waste of space if the actual string is shorter than 16 bytes.
  constexpr std::size_t kByteStringSize = 16;
  metadata.push_back(
      BuildMetadata("min_runtime_version", std::string(kByteStringSize, '\0')));
  for (const auto& kv : metadata_) {
    const std::string& val = kv.second;
    // Only take the first kByteStringSize values.
    const int count = std::min(kByteStringSize, val.length());
    std::string value = std::string(kByteStringSize, '\0')
                            .assign(val.begin(), val.begin() + count);
    metadata.push_back(BuildMetadata(kv.first, value));
  }

  // Populate the model control dependencies metadata entry.
  if (std::any_of(
          model_control_dependencies_.begin(),
          model_control_dependencies_.end(),
          [](const tflite::ControlEdges& edges) { return !edges.empty(); })) {
    metadata.push_back(
        BuildMetadata(tflite::kModelControlDependenciesMetadataKey,
                      tflite::SerializeModelControlDependencies(
                          model_control_dependencies_)));
  }
  return builder_.CreateVector(metadata);
}

// Helper method that returns list of all strings in a StringAttr identified
// by 'attr_key' and values are separated by a comma.
llvm::SmallVector<llvm::StringRef, 2> GetStringsFromAttrWithSeparator(
    mlir::DictionaryAttr attr, const std::string& attr_key) {
  llvm::SmallVector<llvm::StringRef, 2> result;
  if (auto str = attr.get(attr_key).dyn_cast_or_null<mlir::StringAttr>()) {
    str.getValue().split(result, ',', /*MaxSplit=*/-1,
                         /*KeepEmpty=*/false);
  }
  return result;
}

// Helper method that return list of string for all the StringAttr in the
// Attribute identified by 'attr_name'.
std::vector<std::string> GetStringsFromDictionaryAttr(
    const llvm::SmallVector<mlir::DictionaryAttr, 4>& dict_attrs,
    const StringRef attr_name) {
  std::vector<std::string> result;
  for (const auto& arg_attr : dict_attrs) {
    if (!arg_attr) continue;

    auto attrs = arg_attr.getValue();
    for (const auto attr : attrs) {
      if (attr.getName() == attr_name) {
        auto array_attr = attr.getValue().dyn_cast_or_null<mlir::ArrayAttr>();
        if (!array_attr || array_attr.empty()) continue;
        auto string_attr = array_attr[0].dyn_cast_or_null<mlir::StringAttr>();
        if (!string_attr) continue;
        result.push_back(string_attr.getValue().str());
      }
    }
  }
  return result;
}

std::vector<SignatureDefData> BuildSignaturedef(
    FuncOp main_op, const std::string& saved_model_tag,
    const uint32_t subgraph_index, tensorflow::OpOrArgNameMapper& name_mapper) {
  static const char kEntryFunctionAttributes[] = "tf.entry_function";

  // Fetch inputs and outputs from the signature.
  llvm::SmallVector<mlir::DictionaryAttr, 4> arg_attrs, res_attrs;
  main_op.getAllArgAttrs(arg_attrs);
  main_op.getAllResultAttrs(res_attrs);
  std::vector<std::string> sig_def_inputs =
      GetStringsFromDictionaryAttr(arg_attrs, kTfSavedModelIndexPathAttr);
  std::vector<std::string> sig_def_outputs =
      GetStringsFromDictionaryAttr(res_attrs, kTfSavedModelIndexPathAttr);

  // If no defined saved model signature, then return empty list.
  // This can happen when we are converting model not from SavedModel.
  if (sig_def_inputs.empty() && sig_def_outputs.empty()) return {};

  // Fetch function inputs and outputs tensor names.
  auto dict_attr =
      main_op->getAttrOfType<mlir::DictionaryAttr>(kEntryFunctionAttributes);
  if (!dict_attr) return {};

  // Get Input and output tensor names from attribute.
  llvm::SmallVector<llvm::StringRef, 2> input_names =
      GetStringsFromAttrWithSeparator(dict_attr, /*attr_key=*/"inputs");
  llvm::SmallVector<llvm::StringRef, 2> output_names =
      GetStringsFromAttrWithSeparator(dict_attr, /*attr_key=*/"outputs");

  // Verify input size match the number of arguments.
  if (input_names.size() != main_op.getNumArguments()) {
    main_op.emitWarning() << "invalid entry function specification";
    return {};
  }
  // Verify output size match the number of arguments.
  auto term = main_op.back().getTerminator();
  if (output_names.size() != term->getNumOperands()) {
    main_op.emitWarning() << "output names (" << output_names.size()
                          << ") != terminator operands ("
                          << term->getNumOperands() << ")";
    return {};
  }
  // Verify number of tensors for inputs and outputs matches size
  // of the list in the signature def.
  if (input_names.size() != sig_def_inputs.size() ||
      output_names.size() != sig_def_outputs.size()) {
    main_op.emitWarning(
        "Mismatch between signature def inputs/outputs and main function "
        "arguments.");
    return {};
  }
  // Exported method name.
  auto exported_name =
      main_op->getAttrOfType<mlir::ArrayAttr>(kTfSavedModelExportedNamesAttr);
  if (exported_name.empty()) {
    main_op.emitError("Empty exported names for main Function");
    return {};
  }
  // Fill the SignatureDefData container.
  // We create vector of size 1 as TFLite now supports only 1 signatureDef.
  std::vector<SignatureDefData> result(1);
  for (int i = 0; i < input_names.size(); ++i) {
    result[0].inputs[sig_def_inputs[i]] = input_names[i].str();
  }
  for (int i = 0; i < output_names.size(); ++i) {
    // Fetch the name from the actual operand and not rely on names from
    // outputs as deduping can make them invalid after conversion.
    auto& operand = term->getOpOperand(i);
    auto unique_name = std::string(name_mapper.GetUniqueName(operand.get()));
    result[0].outputs[sig_def_outputs[i]] = unique_name;
  }
  if (auto name_attr = exported_name[0].dyn_cast_or_null<StringAttr>())
    result[0].signature_key = name_attr.getValue().str();
  result[0].subgraph_index = subgraph_index;
  return result;
}

std::vector<BufferOffset<tflite::TensorMap>> Translator::GetList(
    const int subgraph_index, const std::map<std::string, std::string>& items) {
  std::vector<BufferOffset<tflite::TensorMap>> result;
  for (const auto& item : items) {
    auto name_buf = builder_.CreateString(item.first);
    tflite::TensorMapBuilder tensor_map_builder(builder_);
    tensor_map_builder.add_name(name_buf);
    tensor_map_builder.add_tensor_index(
        tensor_index_map_[subgraph_index][item.second]);
    result.push_back(tensor_map_builder.Finish());
  }
  return result;
}

std::optional<VectorBufferOffset<BufferOffset<tflite::SignatureDef>>>
Translator::CreateSignatureDefs(
    const std::vector<SignatureDefData>& signature_defs) {
  std::vector<BufferOffset<tflite::SignatureDef>> signature_defs_buffer;
  // When we export each function in the module op, intentionally, we export the
  // entry functions at the beginning of the subgraph list and the
  // subgraph_index is the index in entry functions and at the same, is the
  // index in the subgraph list.
  int subgraph_index = 0;
  for (const auto& signature_def_data : signature_defs) {
    auto inputs = GetList(subgraph_index, signature_def_data.inputs);
    auto outputs = GetList(subgraph_index, signature_def_data.outputs);
    auto inputs_buf = builder_.CreateVector(inputs);
    auto outputs_buf = builder_.CreateVector(outputs);
    auto signature_key_buf =
        builder_.CreateString(signature_def_data.signature_key);
    tflite::SignatureDefBuilder sig_def_builder(builder_);
    sig_def_builder.add_inputs(inputs_buf);
    sig_def_builder.add_outputs(outputs_buf);
    sig_def_builder.add_signature_key(signature_key_buf);
    sig_def_builder.add_subgraph_index(signature_def_data.subgraph_index);
    signature_defs_buffer.push_back(sig_def_builder.Finish());
    ++subgraph_index;
  }

  return builder_.CreateVector(signature_defs_buffer);
}

bool UpdateEntryFunction(ModuleOp module) {
  if (module.lookupSymbol<FuncOp>("main") != nullptr) {
    // We already have an entry function.
    return true;
  }

  int entry_func_count = 0;
  FuncOp entry_func = nullptr;
  for (auto fn : module.getOps<FuncOp>()) {
    auto attrs = fn->getAttrOfType<mlir::DictionaryAttr>("tf.entry_function");
    if (!attrs || attrs.empty()) continue;
    ++entry_func_count;
    entry_func = fn;
  }

  // We should have at least one entry function.
  if (entry_func_count == 0) return false;

  if (entry_func_count == 1) {
    // Update the entry func to main when the entry func is only & one.
    entry_func.setName(StringAttr::get(module.getContext(), "main"));
  }
  return true;
}

std::optional<std::string> Translator::Translate(
    ModuleOp module, const toco::TocoFlags& toco_flags,
    const std::unordered_set<std::string>& tags,
    OpOrArgNameMapper* op_or_arg_name_mapper,
    const std::map<std::string, std::string>& metadata) {
  OpOrArgLocNameMapper default_op_or_arg_name_mapper;
  if (!op_or_arg_name_mapper)
    op_or_arg_name_mapper = &default_op_or_arg_name_mapper;
  if (!UpdateEntryFunction(module)) return std::nullopt;
  if (!IsValidTFLiteMlirModule(module)) return std::nullopt;
  Translator translator(module, toco_flags, tags, op_or_arg_name_mapper,
                        metadata);
  return translator.TranslateInternal();
}

bool Translator::CheckGpuDelegateCompatibility(uint8_t* model_buffer_pointer) {
  bool gpu_compatibile = true;
  auto model = tflite::GetModel(model_buffer_pointer);
  auto subgraphs = model->subgraphs();

  for (int i = 0; i < subgraphs->Length(); ++i) {
    const tflite::SubGraph* subgraph = subgraphs->Get(i);
    for (int j = 0; j < subgraph->operators()->Length(); ++j) {
      const tflite::Operator* op = subgraph->operators()->Get(j);
      const tflite::OperatorCode* op_code =
          model->operator_codes()->Get(op->opcode_index());
      auto status =
          tflite::CheckGpuDelegateCompatibility(op_code, op, subgraph, model);
      if (!status.ok()) {
        gpu_compatibile = false;
        auto inst = subgraph_op_inst_map_[i][j];
        tfl::AttachErrorCode(
            inst->emitOpError()
                << "is not GPU compatible: " << std::string(status.message()),
            tflite::metrics::ConverterErrorData::ERROR_GPU_NOT_COMPATIBLE);
      }
    }
  }
  return gpu_compatibile;
}

std::optional<std::string> Translator::TranslateInternal() {
  // A list of named regions in the module with main function being the first in
  // the list. The main function is required as the first subgraph in the model
  // is entry point for the model.
  std::vector<std::pair<std::string, Region*>> named_regions;
  named_regions.reserve(std::distance(module_.begin(), module_.end()));

  int subgraph_idx = 0;

  // Entry functions for signature defs.
  std::vector<FuncOp> entry_functions;
  std::vector<FuncOp> non_entry_functions;
  FuncOp main_fn = module_.lookupSymbol<FuncOp>("main");
  if (main_fn != nullptr) {
    // Treat the main function as a signature def when the given main function
    // contains on the tf.entry_function attribute.
    auto attrs =
        main_fn->getAttrOfType<mlir::DictionaryAttr>("tf.entry_function");
    if (attrs && !attrs.empty()) {
      entry_functions.push_back(main_fn);
    } else {
      non_entry_functions.push_back(main_fn);
    }
  }

  // Walk over the module collection ops with functions and while ops.
  module_.walk([&](FuncOp fn) {
    if (main_fn == fn) return WalkResult::advance();
    auto attrs = fn->getAttrOfType<mlir::DictionaryAttr>("tf.entry_function");
    if (attrs && !attrs.empty()) {
      entry_functions.push_back(fn);
    } else {
      non_entry_functions.push_back(fn);
    }
    return WalkResult::advance();
  });

  // Assign the subgraph index. Among the given functions, it will put entry
  // functions at the beginning of the list of the subgrahs.
  for (auto fn : entry_functions) {
    subgraph_index_map_[fn.getName().str()] = subgraph_idx++;
    named_regions.emplace_back(fn.getName().str(), &fn.getBody());
  }
  for (auto fn : non_entry_functions) {
    subgraph_index_map_[fn.getName().str()] = subgraph_idx++;
    named_regions.emplace_back(fn.getName().str(), &fn.getBody());
  }

  // Build subgraph for each of the named regions.
  std::vector<BufferOffset<tflite::SubGraph>> subgraphs;
  subgraphs.reserve(named_regions.size());
  model_control_dependencies_.assign(named_regions.size(), {});
  int first_failed_func = -1;

  // When we export each function in the module op, intentionally, we export the
  // entry functions at the beginning of the subgraph list and the
  // subgraph_index is the index in entry functions and at the same, is the
  // index in the subgraph list.
  int subgraph_index = 0;
  for (const auto& it : llvm::enumerate(named_regions)) {
    auto subgraph_or =
        BuildSubGraph(it.value().first, it.value().second, subgraph_index);
    if (!subgraph_or) {
      if (first_failed_func == -1)
        // Record the index of the first region that cannot be converted.
        // Keep looping through all subgraphs in the module to make sure that
        // we collect the list of missing ops from the entire module.
        first_failed_func = it.index();
    } else {
      subgraphs.push_back(*subgraph_or);
      ++subgraph_index;
    }
  }

  if (!resource_ops_.empty()) {
    std::string resource_ops_summary =
        GetOpsSummary(resource_ops_, /*summary_title=*/"Resource");
    LOG(WARNING) << "Graph contains the following resource op(s), that use(s) "
                    "resource type. Currently, the "
                    "resource type is not natively supported in TFLite. Please "
                    "consider not using the resource type if there are issues "
                    "with either TFLite converter or TFLite runtime:\n"
                 << resource_ops_summary;
  }

  if (!flex_ops_.empty()) {
    std::string flex_ops_summary =
        GetOpsSummary(flex_ops_, /*summary_title=*/"Flex");
    LOG(WARNING) << "TFLite interpreter needs to link Flex delegate in order "
                    "to run the model since it contains the following Select TF"
                    "op(s):\n"
                 << flex_ops_summary
                 << "\nSee instructions: "
                    "https://www.tensorflow.org/lite/guide/ops_select";
  }

  if (!custom_ops_.empty()) {
    std::string custom_ops_summary =
        GetOpsSummary(custom_ops_, /*summary_title=*/"Custom");
    LOG(WARNING) << "The following operation(s) need TFLite custom op "
                    "implementation(s):\n"
                 << custom_ops_summary
                 << "\nSee instructions: "
                    "https://www.tensorflow.org/lite/guide/ops_custom";
  }

  if (first_failed_func != -1) {
    std::string failed_flex_ops_summary =
        GetOpsSummary(failed_flex_ops_, /*summary_title=*/"TF Select");
    std::string failed_custom_ops_summary =
        GetOpsSummary(failed_custom_ops_, /*summary_title=*/"Custom");
    std::string err;
    if (!failed_flex_ops_.empty())
      err +=
          "\nSome ops are not supported by the native TFLite runtime, you can "
          "enable TF kernels fallback using TF Select. See instructions: "
          "https://www.tensorflow.org/lite/guide/ops_select \n" +
          failed_flex_ops_summary + "\n";
    if (!failed_custom_ops_.empty())
      err +=
          "\nSome ops in the model are custom ops, "
          "See instructions to implement "
          "custom ops: https://www.tensorflow.org/lite/guide/ops_custom \n" +
          failed_custom_ops_summary + "\n";

    auto& failed_region = named_regions[first_failed_func];
    return failed_region.second->getParentOp()->emitError()
               << "failed while converting: '" << failed_region.first
               << "': " << err,
           std::nullopt;
  }

  // Log MAC count.
  int64_t ops_count;
  if (EstimateArithmeticCount(&ops_count)) {
    const int64_t million = 1e6;
    const int64_t billion = 1e9;
    std::string flops_str;
    std::string mac_str;
    if (ops_count < 10000) {
      flops_str = absl::StrFormat("%ld ", ops_count);
      mac_str = absl::StrFormat("%ld ", ops_count / 2);
    } else if (ops_count < billion) {
      flops_str =
          absl::StrFormat("%.3f M ", static_cast<double>(ops_count) / million);
      mac_str = absl::StrFormat("%.3f M ",
                                static_cast<double>(ops_count / 2) / million);
    } else {
      flops_str =
          absl::StrFormat("%.3f G ", static_cast<double>(ops_count) / billion);
      mac_str = absl::StrFormat("%.3f G ",
                                static_cast<double>(ops_count / 2) / billion);
    }
    LOG(INFO) << "Estimated count of arithmetic ops: " << flops_str
              << " ops, equivalently " << mac_str << " MACs";
  }

  std::string model_description;
  if (auto attr = module_->getAttrOfType<StringAttr>("tfl.description")) {
    model_description = attr.getValue().str();
  } else {
    model_description = "MLIR Converted.";
  }

  // Build the model and finish the model building process.
  auto description = builder_.CreateString(model_description.data());
  VectorBufferOffset<int32_t> metadata_buffer = 0;  // Deprecated
  auto metadata = CreateMetadataVector();
  if (!metadata) return std::nullopt;

  std::vector<SignatureDefData> signature_defs_vec;
  subgraph_index = 0;
  // Build SignatureDefs for the tf.entry_function based func ops.
  for (auto fn : entry_functions) {
    auto signature_defs = BuildSignaturedef(
        fn, saved_model_tags_.empty() ? "" : *saved_model_tags_.begin(),
        subgraph_index, name_mapper_);
    for (const auto& signature_def : signature_defs) {
      signature_defs_vec.push_back(signature_def);
    }
    // When we export each function in the module op, intentionally, we export
    // the entry functions at the beginning of the subgraph list and the
    // subgraph_index is the index in entry functions and at the same, is the
    // index in the subgraph list.
    ++subgraph_index;
  }
  auto signature_defs = CreateSignatureDefs(signature_defs_vec);

  auto model = tflite::CreateModel(builder_, TFLITE_SCHEMA_VERSION,
                                   builder_.CreateVector(opcodes_),
                                   builder_.CreateVector(subgraphs),
                                   description, builder_.CreateVector(buffers_),
                                   metadata_buffer, *metadata, *signature_defs);
  tflite::FinishModelBuffer(builder_, model);
  // There is a limit of 2GB for a flatbuffer.
  if (builder_.GetSize() > 2147483648) {
    LOG(ERROR) << "Model size is bigger than 2gb";
    return std::nullopt;
  }
  tflite::UpdateOpVersion(builder_.GetBufferPointer());
  tflite::UpdateMinimumRuntimeVersionForModel(builder_.GetBufferPointer());
  if (supported_backends_.find("GPU") != supported_backends_.end()) {
    if (!CheckGpuDelegateCompatibility(builder_.GetBufferPointer())) {
      return std::nullopt;
    }
  }

  // Return serialized string for the built FlatBuffer.
  return std::string(reinterpret_cast<const char*>(builder_.GetBufferPointer()),
                     builder_.GetSize());
}

BufferOffset<tflite::SparsityParameters> Translator::BuildSparsityParameters(
    const mlir::TFL::SparsityParameterAttr& s_attr) {
  const int dim_size = s_attr.getDimMetadata().size();
  std::vector<flatbuffers::Offset<tflite::DimensionMetadata>> fb_dim_metadata(
      dim_size);
  for (int i = 0; i < dim_size; i++) {
    const auto dim_metadata =
        s_attr.getDimMetadata()[i].dyn_cast<mlir::TFL::DimensionMetadataAttr>();
    if (dim_metadata.getFormat().getValue() ==
        mlir::TFL::DimensionType::DENSE) {
      fb_dim_metadata[i] = tflite::CreateDimensionMetadata(
          builder_, tflite::DimensionType_DENSE, dim_metadata.getDenseSize());

    } else {
      auto segments = dim_metadata.getSegments();
      std::vector<int> vector_segments(segments.size(), 0);
      for (int j = 0, end = segments.size(); j < end; j++) {
        vector_segments[j] = segments[j];
      }
      tflite::SparseIndexVector segments_type;
      BufferOffset<void> array_segments;
      // The segment array is sorted.
      // TODO(b/147449640): Clean this up with util functions.
      int max_of_segments = vector_segments[segments.size() - 1];
      if (max_of_segments <= UINT8_MAX) {
        segments_type = tflite::SparseIndexVector_Uint8Vector;
        std::vector<uint8_t> uint8_vector(vector_segments.begin(),
                                          vector_segments.end());
        array_segments = tflite::CreateUint8Vector(
                             builder_, builder_.CreateVector(uint8_vector))
                             .Union();
      } else if (max_of_segments <= UINT16_MAX) {
        segments_type = tflite::SparseIndexVector_Uint16Vector;
        std::vector<uint16_t> uint16_vector(vector_segments.begin(),
                                            vector_segments.end());
        array_segments = tflite::CreateUint16Vector(
                             builder_, builder_.CreateVector(uint16_vector))
                             .Union();
      } else {
        segments_type = tflite::SparseIndexVector_Int32Vector;
        array_segments = tflite::CreateInt32Vector(
                             builder_, builder_.CreateVector(vector_segments))
                             .Union();
      }

      auto indices = dim_metadata.getIndices();
      std::vector<int> vector_indices(indices.size(), 0);
      int max_of_indices = 0;
      for (int j = 0, end = indices.size(); j < end; j++) {
        vector_indices[j] = indices[j];
        if (vector_indices[j] > max_of_indices) {
          max_of_indices = vector_indices[j];
        }
      }
      tflite::SparseIndexVector indices_type;
      BufferOffset<void> array_indices;
      if (max_of_indices <= UINT8_MAX) {
        indices_type = tflite::SparseIndexVector_Uint8Vector;
        std::vector<uint8_t> uint8_vector(vector_indices.begin(),
                                          vector_indices.end());
        array_indices = tflite::CreateUint8Vector(
                            builder_, builder_.CreateVector(uint8_vector))
                            .Union();
      } else if (max_of_indices <= UINT16_MAX) {
        indices_type = tflite::SparseIndexVector_Uint16Vector;
        std::vector<uint16_t> uint16_vector(vector_indices.begin(),
                                            vector_indices.end());
        array_indices = tflite::CreateUint16Vector(
                            builder_, builder_.CreateVector(uint16_vector))
                            .Union();
      } else {
        indices_type = tflite::SparseIndexVector_Int32Vector;
        array_indices = tflite::CreateInt32Vector(
                            builder_, builder_.CreateVector(vector_indices))
                            .Union();
      }

      fb_dim_metadata[i] = tflite::CreateDimensionMetadata(
          builder_, tflite::DimensionType_SPARSE_CSR, 0, segments_type,
          array_segments, indices_type, array_indices);
    }
  }

  std::vector<int> traversal_order(dim_size);
  for (int i = 0; i < dim_size; i++) {
    traversal_order[i] = s_attr.getTraversalOrder()[i];
  }
  const int block_map_size = s_attr.getBlockMap().size();
  std::vector<int> block_map(block_map_size);
  for (int i = 0; i < block_map_size; i++) {
    block_map[i] = s_attr.getBlockMap()[i];
  }

  return tflite::CreateSparsityParameters(
      builder_, builder_.CreateVector(traversal_order),
      builder_.CreateVector(block_map), builder_.CreateVector(fb_dim_metadata));
}

std::vector<std::pair<int, int>> Translator::ExtractControlEdges(
    mlir::Block* block) {
  std::vector<std::pair<int, int>> control_edges;

  mlir::IRRewriter rewriter(block->getParentOp()->getContext());

  // Since we're modifying *block, we store integer offsets to block->begin().
  llvm::DenseMap<Operation*, int> control_nodes_at;
  std::vector<Operation*> control_nodes;
  for (const auto& item : llvm::enumerate(*block)) {
    if (llvm::isa<mlir::TFL::ControlNodeOp>(item.value())) {
      control_nodes.push_back(&item.value());
      control_nodes_at.try_emplace(&item.value(), item.index());
    }
  }

  for (auto outer_op : control_nodes) {
    auto control_node_op = dyn_cast<mlir::TFL::ControlNodeOp>(outer_op);
    auto* inner_op = &control_node_op.getBody().front().front();
    auto control_token = control_node_op.getControl();

    // Now go through all uses. Since *block is in executable order, control
    // edges always point to operations we haven't modified yet.
    for (auto& use : control_token.getUses()) {
      auto owner = use.getOwner();
      // Control tokens can only be consumed by other ControlNodeOps,
      assert(llvm::isa<mlir::TFL::ControlNodeOp>(owner));
      assert(control_nodes_at.find(owner) != control_nodes_at.end());
      // Control edge in terms of offsets.
      control_edges.emplace_back(control_nodes_at[outer_op],
                                 control_nodes_at[owner]);
    }
    control_token.dropAllUses();

    // Replace the ControlNodeOp with the wrapped operation.
    rewriter.setInsertionPointAfter(outer_op);
    auto* cloned_inner = rewriter.clone(*inner_op);
    for (auto it :
         llvm::zip(control_node_op.getOutputs(), cloned_inner->getResults())) {
      std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
    }
    rewriter.eraseOp(outer_op);
  }
  return control_edges;
}

}  // namespace

namespace tflite {

bool MlirToFlatBufferTranslateFunction(mlir::ModuleOp module,
                                       const FlatbufferExportOptions& options,
                                       std::string* serialized_flatbuffer) {
  auto maybe_translated = Translator::Translate(
      module, options.toco_flags, options.saved_model_tags,
      options.op_or_arg_name_mapper, options.metadata);
  if (!maybe_translated) return false;
  *serialized_flatbuffer = std::move(*maybe_translated);
  return true;
}

}  // namespace tflite
