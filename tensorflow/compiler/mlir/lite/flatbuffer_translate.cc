/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/flatbuffer_translate.h"

#include <stddef.h>
#include <stdlib.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/flatbuffers.h"  // TF:flatbuffers
#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/QuantOps/QuantTypes.h"  // TF:local_config_mlir
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/IR/Value.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "mlir/Translation.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/flatbuffer_operator.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/stateful_ops_utils.h"
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/delegates/flex/whitelisted_flex_ops.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/versioning/op_version.h"
#include "tensorflow/lite/version.h"

using llvm::dyn_cast;
using llvm::formatv;
using llvm::isa;
using llvm::Optional;
using llvm::StringRef;
using llvm::Twine;
using mlir::Dialect;
using mlir::ElementsAttr;
using mlir::FuncOp;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::NoneType;
using mlir::Operation;
using mlir::StringAttr;
using mlir::TensorType;
using mlir::TranslateFromMLIRRegistration;
using mlir::Type;
using mlir::UnknownLoc;
using mlir::Value;
using tensorflow::OpOrArgLocNameMapper;
using tensorflow::OpOrArgNameMapper;
using tensorflow::Status;
using tflite::flex::IsWhitelistedFlexOp;
using xla::StatusOr;

template <typename T>
using BufferOffset = flatbuffers::Offset<T>;

template <typename T>
using VectorBufferOffset = flatbuffers::Offset<flatbuffers::Vector<T>>;

using CustomOptionsOffset = VectorBufferOffset<uint8_t>;

namespace error = tensorflow::error;
namespace tfl = mlir::TFL;

using llvm::cl::opt;

// These command line flags enable control of the translation implementation.
bool emit_builtin_tflite_ops;
bool emit_custom_ops;
bool emit_select_tf_ops;
bool lower_tensor_list_ops;
bool strip_debug_info;
// NOLINTNEXTLINE
std::string output_arrays_string;

// NOLINTNEXTLINE
static opt<bool, true> emit_builtin_tflite_ops_flag(
    "emit-builtin-tflite-ops",
    llvm::cl::desc(
        "Emit TFLite built in operations in the generated TFLite model"),
    llvm::cl::location(emit_builtin_tflite_ops), llvm::cl::init(true));

// NOLINTNEXTLINE
static opt<bool, true> emit_select_tf_ops_flag(
    "emit-select-tf-ops",
    llvm::cl::desc(
        "Emit Select TF operations (Flex ops) in the generated TFLite model"),
    llvm::cl::location(emit_select_tf_ops), llvm::cl::init(false));

// NOLINTNEXTLINE
static opt<bool, true> emit_custom_ops_flag(
    "emit-custom-ops",
    llvm::cl::desc("Emit Custom operations in the generated TFLite model"),
    llvm::cl::location(emit_custom_ops), llvm::cl::init(false));

// NOLINTNEXTLINE
static opt<bool, true> lower_tensor_list_ops_flag(
    "lower-tensor-list-ops",
    llvm::cl::desc("Lower the TensorList ops within the TFLite dialect"),
    llvm::cl::location(lower_tensor_list_ops), llvm::cl::init(false));

// NOLINTNEXTLINE
static opt<bool, true> strip_debug_info_flag(
    "strip-debug-info", llvm::cl::desc("Strip debug info during export"),
    llvm::cl::location(strip_debug_info), llvm::cl::init(false));

// NOLINTNEXTLINE
static opt<std::string, true> output_arrays_flag(
    "output-arrays", llvm::cl::desc("List of output tensors"),
    llvm::cl::location(output_arrays_string), llvm::cl::init(""));

ABSL_CONST_INIT const absl::string_view kFlexOpNamePrefix = "Flex";

// Use initial buffer size in flatbuffer builder to be same as the initial size
// used by the TOCO export. (It does not explain rationale for this choice.)
constexpr size_t kInitialBufferSize = 10240;

// Set `isSigned` to false if the `type` is an 8-bit unsigned integer type.
// Since tflite doesn't support unsigned for other types, returns error if
// `isSigned` is set to false for other types.
static StatusOr<tflite::TensorType> GetTFLiteType(Type type,
                                                  bool is_signed = true) {
  if (!is_signed && type.isInteger(8)) {
    return tflite::TensorType_UINT8;
  }
  if (!is_signed) {
    return Status(error::INVALID_ARGUMENT,
                  "'isSigned' can only be set for 8-bits integer type");
  }
  switch (type.getKind()) {
    case mlir::StandardTypes::F32:
      return tflite::TensorType_FLOAT32;
    case mlir::StandardTypes::F16:
      return tflite::TensorType_FLOAT16;
    case mlir::TF::TensorFlowTypes::STRING:
      return tflite::TensorType_STRING;
    case mlir::TF::TensorFlowTypes::UINT8:
      return tflite::TensorType_UINT8;
    case mlir::TF::TensorFlowTypes::QUINT8:
      return tflite::TensorType_UINT8;
    case mlir::StandardTypes::Complex: {
      auto ftype = type.cast<mlir::ComplexType>().getElementType();
      if (ftype && ftype.isF32()) {
        return tflite::TensorType_COMPLEX64;
      }
      return Status(error::INVALID_ARGUMENT, "Unsupported type");
    }
    case mlir::StandardTypes::Integer: {
      const auto& itype = type.cast<mlir::IntegerType>();
      switch (itype.getWidth()) {
        case 1:
          return tflite::TensorType_BOOL;
        case 8:
          return tflite::TensorType_INT8;
        case 16:
          return tflite::TensorType_INT16;
        case 32:
          return tflite::TensorType_INT32;
        case 64:
          return tflite::TensorType_INT64;
      }
    }
    case mlir::quant::QuantizationTypes::UniformQuantized: {
      auto qtype = type.cast<mlir::quant::UniformQuantizedType>();
      return GetTFLiteType(qtype.getStorageType(), qtype.isSigned());
    }
    case mlir::quant::QuantizationTypes::UniformQuantizedPerAxis: {
      auto qtype = type.cast<mlir::quant::UniformQuantizedPerAxisType>();
      return GetTFLiteType(qtype.getStorageType(), qtype.isSigned());
    }
    default:
      // TFLite export fills FLOAT32 for unknown data types. Returning an error
      // for now for safety and this could be revisited when required.
      return Status(error::INVALID_ARGUMENT, "Unsupported type");
  }
}

static bool IsConst(Operation* op) {
  return isa<mlir::ConstantOp>(op) || isa<mlir::TF::ConstOp>(op) ||
         isa<tfl::ConstOp>(op) || isa<tfl::QConstOp>(op);
}

template <typename T>
static bool HasValidTFLiteType(Value* value, T& error_handler) {
  // None type is allowed to represent unspecified operands.
  if (value->getType().isa<NoneType>()) return true;

  auto type = value->getType().dyn_cast<TensorType>();
  if (!type) {
    if (auto op = value->getDefiningOp()) {
      error_handler.emitError()
          << '\'' << op << "' should produce value of tensor type instead of "
          << value->getType();
      return false;
    }
    error_handler.emitError("expected tensor type, got ") << value->getType();
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
    return emitError(UnknownLoc::get(context),
                     "should have a function named 'main'"),
           false;
  }

  for (auto fn : module.getOps<FuncOp>()) {
    if (fn.getBlocks().size() != 1) {
      return fn.emitError("should have exactly one basic block"), false;
    }
    auto& bb = fn.getBlocks().front();

    for (auto* arg : bb.getArguments()) {
      if (!HasValidTFLiteType(arg, fn))
        return fn.emitError("invalid TFLite type: ") << arg->getType(), false;
    }

    // Verify that all operations except the terminator have exactly one
    // result of type supported by TFLite.
    for (auto& inst : bb) {
      if (inst.isKnownTerminator()) break;

      for (auto* result : inst.getResults()) {
        if (!HasValidTFLiteType(result, inst))
          return fn.emitError("invalid TFLite type: ") << result->getType(),
                 false;
      }
    }
  }

  return true;
}

static std::unique_ptr<::tensorflow::NodeDef> getTensorFlowNodeDef(
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
  return std::move(status_or_node_def.ValueOrDie());
}

namespace {

// Translates an MLIR module in TFLite dialect to TFLite FlatBuffer.
class Translator {
 public:
  // Translates the given MLIR module into TFLite FlatBuffer format and returns
  // the serialized output. Returns llvm::None on unsupported, invalid inputs or
  // internal error.
  static Optional<std::string> Translate(
      ModuleOp module, bool emit_builtin_tflite_ops, bool emit_select_tf_ops,
      bool emit_custom_ops, OpOrArgNameMapper* op_or_arg_name_mapper);

 private:
  enum class OpType : char { kTfliteBuiltin, kSelectTf, kCustomOp };
  explicit Translator(ModuleOp module, bool emit_builtin_tflite_ops,
                      bool emit_select_tf_ops, bool emit_custom_ops,
                      OpOrArgNameMapper* op_or_arg_name_mapper)
      : module_(module),
        name_mapper_(*op_or_arg_name_mapper),
        builder_(kInitialBufferSize) {
    // The first buffer must be empty according to the schema definition.
    empty_buffer_ = tflite::CreateBuffer(builder_);
    buffers_.push_back(empty_buffer_);
    if (emit_builtin_tflite_ops) {
      enabled_op_types_.emplace(OpType::kTfliteBuiltin);
    }
    if (emit_select_tf_ops) {
      enabled_op_types_.emplace(OpType::kSelectTf);
    }
    if (emit_custom_ops) {
      enabled_op_types_.emplace(OpType::kCustomOp);
    }
    tf_dialect_ = module.getContext()->getRegisteredDialect("tf");
    tfl_dialect_ = module.getContext()->getRegisteredDialect("tfl");
  }

  Optional<std::string> TranslateInternal();

  // Returns TFLite buffer populated with constant value if the operation is
  // TFLite constant operation. Otherwise, returns an empty buffer. Emits error
  // and returns llvm::None on failure.
  Optional<BufferOffset<tflite::Buffer>> BuildBuffer(Operation* inst);

  // Builds TFLite tensor from the given value. `buffer_idx` is index of the
  // corresponding buffer. Emits error and returns llvm::None on failure.
  Optional<BufferOffset<tflite::Tensor>> BuildTensor(Value* value,
                                                     const std::string& name,
                                                     unsigned buffer_idx);

  // TODO(b/137395003): Legalize control flow ops to TFLite dialect, and remove
  // these 2 functions here.
  BufferOffset<tflite::Operator> BuildIfOperator(
      mlir::TF::IfOp op, const std::vector<int32_t>& operands,
      const std::vector<int32_t>& results);
  BufferOffset<tflite::Operator> BuildWhileOperator(
      mlir::TF::WhileOp op, const std::vector<int32_t>& operands,
      const std::vector<int32_t>& results);

  Optional<CustomOptionsOffset> CreateFlexOpCustomOptions(
      const ::tensorflow::NodeDef& node_def, const mlir::Location& loc);

  Optional<CustomOptionsOffset> CreateCustomOpCustomOptions(
      const ::tensorflow::NodeDef& node_def, const mlir::Location& loc);

  std::unique_ptr<flexbuffers::Builder> CreateFlexBuilderWithNodeAttrs(
      const ::tensorflow::NodeDef& node_def, const mlir::Location& loc);

  // Returns opcode index for op identified by the op_name, if already
  // available. Otherwise, creates a new OperatorCode using the given `builtin`
  // operator and associates it with `op_name`.
  uint32_t GetOpcodeIndex(const std::string& op_name,
                          tflite::BuiltinOperator builtin);

  // Builds operator for the given operation with specified operand and result
  // tensor indices. Emits an error and returns llvm::None on failure.
  Optional<BufferOffset<tflite::Operator>> BuildOperator(
      Operation* inst, const std::vector<int32_t>& operands,
      const std::vector<int32_t>& results);

  Optional<BufferOffset<tflite::SubGraph>> BuildSubGraph(FuncOp fn);

  // Builds Metadata with the given `name` and buffer `content`.
  BufferOffset<tflite::Metadata> BuildMetadata(StringRef name,
                                               StringRef content);

  // Encodes the `tfl.metadata` dictionary attribute of the module to the
  // metadata section in the final model.
  Optional<VectorBufferOffset<BufferOffset<tflite::Metadata>>>
  CreateMetadataVector();

  // Uses the tf.entry_function attribute (if set) to initialize the op to name
  // mapping.
  void InitializeNamesFromAttribute(FuncOp fn, bool* has_input_attr);

  // Determines if the specified operation op's operand at operand_index
  // is marked as a stateful operand.
  bool IsStatefulOperand(mlir::Operation* op, int operand_index);

  // Returns a unique name for `op`.
  std::string UniqueName(mlir::Operation* op);

  ModuleOp module_;

  tensorflow::OpOrArgNameMapper& name_mapper_;

  flatbuffers::FlatBufferBuilder builder_;
  BufferOffset<tflite::Buffer> empty_buffer_;

  std::vector<BufferOffset<tflite::Buffer>> buffers_;

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
  std::vector<std::string> failed_flex_ops_;
  std::vector<std::string> failed_custom_ops_;
};

std::string Translator::UniqueName(mlir::Operation* op) {
  return name_mapper_.GetUniqueName(op);
}

Optional<BufferOffset<tflite::Buffer>> Translator::BuildBuffer(
    Operation* inst) {
  ElementsAttr attr;
  if (auto cst = dyn_cast<mlir::ConstantOp>(inst)) {
    // ConstantOp have ElementAttr at this point due to validation of the TFLite
    // module.
    attr = cst.getValue().cast<ElementsAttr>();
  } else if (auto cst = dyn_cast<mlir::TF::ConstOp>(inst)) {
    attr = cst.value();
  } else if (auto cst = dyn_cast<tfl::ConstOp>(inst)) {
    attr = cst.value();
  } else if (auto cst = dyn_cast<tfl::QConstOp>(inst)) {
    attr = cst.value();
  } else {
    return empty_buffer_;
  }
  tensorflow::Tensor tensor;
  auto status = tensorflow::ConvertToTensor(attr, &tensor);
  if (!status.ok()) {
    inst->emitError(
        Twine("failed to convert value attribute to tensor with error: " +
              status.ToString()));
    return llvm::None;
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

Optional<BufferOffset<tflite::Tensor>> Translator::BuildTensor(
    Value* value, const std::string& name, unsigned buffer_idx) {
  auto type = value->getType().cast<TensorType>();

  // TFLite requires tensor shape only for the inputs and constants.
  // However, we output all known shapes for better round-tripping
  auto check_shape =
      [&](llvm::ArrayRef<int64_t> shape_ref) -> mlir::LogicalResult {
    auto is_out_of_range = [](int64_t dim) {
      return dim > std::numeric_limits<int32_t>::max();
    };

    if (std::any_of(shape_ref.begin(), shape_ref.end(), is_out_of_range))
      return mlir::emitError(
          value->getLoc(),
          "result shape dimensions out of 32 bit int type range");

    return mlir::success();
  };

  std::vector<int32_t> shape;
  if (type.hasStaticShape()) {
    llvm::ArrayRef<int64_t> shape_ref = type.getShape();
    if (mlir::failed(check_shape(shape_ref))) return llvm::None;

    shape = std::vector<int32_t>(shape_ref.begin(), shape_ref.end());
  } else if (auto* inst = value->getDefiningOp()) {
    if (IsConst(inst)) {
      // Const op can have a result of dynamic shaped type (e.g. due to constant
      // folding), but we can still derive the shape of a constant tensor for
      // its attribute type.
      mlir::Attribute tensor_attr = inst->getAttr("value");
      llvm::ArrayRef<int64_t> shape_ref =
          tensor_attr.getType().cast<TensorType>().getShape();
      if (mlir::failed(check_shape(shape_ref))) return llvm::None;

      shape = std::vector<int32_t>(shape_ref.begin(), shape_ref.end());
    }
  }
  Type element_type = type.getElementType();
  tflite::TensorType tflite_element_type =
      GetTFLiteType(type.getElementType()).ValueOrDie();

  BufferOffset<tflite::QuantizationParameters> q_params;
  if (auto qtype = element_type.dyn_cast<mlir::quant::UniformQuantizedType>()) {
    q_params = tflite::CreateQuantizationParameters(
        // TODO(fengliuai): min and max values are not stored in the
        // quantized type, so both are set to 0. The model couldn't be imported
        // to TensorFlow because of this.
        builder_, /*min=*/0, /*max=*/0,
        builder_.CreateVector<float>({static_cast<float>(qtype.getScale())}),
        builder_.CreateVector<int64_t>({qtype.getZeroPoint()}));
  } else if (auto qtype =
                 element_type
                     .dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
    std::vector<float> scales(qtype.getScales().begin(),
                              qtype.getScales().end());
    q_params = tflite::CreateQuantizationParameters(
        builder_, /*min=*/0, /*max=*/0, builder_.CreateVector<float>(scales),
        builder_.CreateVector<int64_t>(qtype.getZeroPoints()),
        tflite::QuantizationDetails_NONE, /*details=*/0,
        qtype.getQuantizedDimension());
  } else {
    q_params = tflite::CreateQuantizationParameters(builder_);
  }
  // Check if the value's uses includes an op and usage at an operand index
  // marked as a stateful. If so, set the tensor's is_variable as true
  // This is v1 ref variable semantics in the TFLite runtime.
  bool is_variable = false;
  for (auto& use : value->getUses()) {
    is_variable = IsStatefulOperand(use.getOwner(), use.getOperandNumber());
    if (is_variable) {
      break;
    }
  }
  return tflite::CreateTensor(
      builder_, builder_.CreateVector(shape), tflite_element_type,
      (is_variable ? 0 : buffer_idx), builder_.CreateString(name), q_params,
      /*is_variable=*/is_variable);
}

BufferOffset<tflite::Operator> Translator::BuildIfOperator(
    mlir::TF::IfOp op, const std::vector<int32_t>& operands,
    const std::vector<int32_t>& results) {
  auto opcode_index = GetOpcodeIndex("if", tflite::BuiltinOperator_IF);
  int then_subgraph_index = subgraph_index_map_.at(op.then_branch().str());
  int else_subgraph_index = subgraph_index_map_.at(op.else_branch().str());
  auto builtin_options = tflite::CreateIfOptions(builder_, then_subgraph_index,
                                                 else_subgraph_index)
                             .Union();
  auto inputs = builder_.CreateVector(operands);
  auto outputs = builder_.CreateVector(results);
  return tflite::CreateOperator(builder_, opcode_index, inputs, outputs,
                                tflite::BuiltinOptions_IfOptions,
                                builtin_options);
}

BufferOffset<tflite::Operator> Translator::BuildWhileOperator(
    mlir::TF::WhileOp op, const std::vector<int32_t>& operands,
    const std::vector<int32_t>& results) {
  auto opcode_index = GetOpcodeIndex("while", tflite::BuiltinOperator_WHILE);
  int cond_subgraph_index = subgraph_index_map_.at(op.cond().str());
  int body_subgraph_index = subgraph_index_map_.at(op.body().str());
  auto builtin_options = tflite::CreateWhileOptions(
                             builder_, cond_subgraph_index, body_subgraph_index)
                             .Union();
  auto inputs = builder_.CreateVector(operands);
  auto outputs = builder_.CreateVector(results);
  return tflite::CreateOperator(builder_, opcode_index, inputs, outputs,
                                tflite::BuiltinOptions_WhileOptions,
                                builtin_options);
}

Optional<CustomOptionsOffset> Translator::CreateFlexOpCustomOptions(
    const ::tensorflow::NodeDef& node_def, const mlir::Location& loc) {
  std::string node_def_str;
  if (!node_def.SerializeToString(&node_def_str)) {
    return emitError(loc, "failed to serialize tensorflow node_def"),
           llvm::None;
  }

  auto flex_builder = absl::make_unique<flexbuffers::Builder>();
  flex_builder->Vector([&]() {
    flex_builder->String(node_def.op());
    flex_builder->String(node_def_str);
  });
  flex_builder->Finish();
  return builder_.CreateVector(flex_builder->GetBuffer());
}

Optional<CustomOptionsOffset> Translator::CreateCustomOpCustomOptions(
    const ::tensorflow::NodeDef& node_def, const mlir::Location& loc) {
  std::string node_def_str;
  if (!node_def.SerializeToString(&node_def_str)) {
    return emitError(loc, "failed to serialize tensorflow node_def"),
           llvm::None;
  }
  auto flex_builder = CreateFlexBuilderWithNodeAttrs(node_def, loc);
  return builder_.CreateVector(flex_builder->GetBuffer());
}

std::unique_ptr<flexbuffers::Builder>
Translator::CreateFlexBuilderWithNodeAttrs(
    const ::tensorflow::NodeDef& node_def, const mlir::Location& loc) {
  auto flex_builder = absl::make_unique<flexbuffers::Builder>();
  size_t map_start = flex_builder->StartMap();
  for (const auto& pair : node_def.attr()) {
    const char* key = pair.first.c_str();
    const auto& attr = pair.second;
    switch (attr.value_case()) {
      case ::tensorflow::AttrValue::kS:
        flex_builder->String(key, attr.s());
        break;
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

Optional<BufferOffset<tflite::Operator>> Translator::BuildOperator(
    Operation* inst, const std::vector<int32_t>& operands,
    const std::vector<int32_t>& results) {
  const auto* dialect = inst->getDialect();
  if (!dialect) {
    inst->emitOpError("dialect is not registered");
    return llvm::None;
  }

  // If TFLite built in op, create operator as a builtin op.
  if (dialect == tfl_dialect_) {
    // Only if built-in TFLite op emission is enabled, would legalization have
    // converted any TF->TFL.
    if (!enabled_op_types_.contains(OpType::kTfliteBuiltin)) {
      return inst->emitOpError(
                 "is a TFLite builtin op but builtin emission is not enabled"),
             llvm::None;
    }

    auto builtin_code = GetBuiltinOpCode(inst);
    if (!builtin_code) {
      inst->emitOpError("is not a supported TFLite op");
      return llvm::None;
    }

    std::string op_name = inst->getName().getStringRef().str();
    uint32_t opcode_index = GetOpcodeIndex(op_name, *builtin_code);
    auto offset = CreateFlatBufferOperator(inst, opcode_index, operands,
                                           results, &builder_);
    if (!offset) {
      inst->emitOpError("is not a supported TFLite op");
    }
    return offset;
  }

  if (dialect == tf_dialect_) {
    std::string op_name;
    if (auto ifOp = dyn_cast<mlir::TF::IfOp>(inst)) {
      return BuildIfOperator(ifOp, operands, results);
    } else if (auto whileOp = dyn_cast<mlir::TF::WhileOp>(inst)) {
      return BuildWhileOperator(whileOp, operands, results);
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
    //   if flex is enabled and the op is whitelisted as flex
    //     we emit op as flex.
    //   if custom is enabled
    //    we emit the op as custom.
    auto node_def = getTensorFlowNodeDef(inst);
    if (!node_def) {
      return llvm::None;
    }

    // Flex op case
    // Eventually, the whitelist will go away and we will rely on some TF op
    // trait (e.g. No side effect) to determine if it is a supported "Flex"
    // op or not.
    if (enabled_op_types_.contains(OpType::kSelectTf) &&
        IsWhitelistedFlexOp(node_def->op())) {
      // Construct ops as flex op encoding TensorFlow node definition
      // as custom options.
      // Flex ops are named with the kFlexOpNamePrefix prefix to the actual
      // TF op name.
      op_name = std::string(kFlexOpNamePrefix) + node_def->op();
      if (auto options = CreateFlexOpCustomOptions(*node_def, inst->getLoc())) {
        custom_options = *options;
      } else {
        return llvm::None;
      }
    } else if (enabled_op_types_.contains(OpType::kCustomOp)) {
      // Generic case of custom ops - write using flex buffers since that
      // is the only custom options supported by TFLite today.
      op_name = node_def->op();
      if (auto options =
              CreateCustomOpCustomOptions(*node_def, inst->getLoc())) {
        custom_options = *options;
      } else {
        return llvm::None;
      }
    } else {
      // Insert failed op to `flex_ops` or `custom_ops`.
      if (IsWhitelistedFlexOp(node_def->op())) {
        failed_flex_ops_.push_back(node_def->op());
      } else {
        failed_custom_ops_.push_back(node_def->op());
      }
      return inst->emitOpError("is neither a custom op nor a flex op"),
             llvm::None;
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
         llvm::None;
}

void Translator::InitializeNamesFromAttribute(FuncOp fn, bool* has_input_attr) {
  auto dict_attr = fn.getAttrOfType<mlir::DictionaryAttr>("tf.entry_function");
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
    for (auto it : llvm::enumerate(fn.getArguments())) {
      name_mapper_.InitOpName(it.value(), input_names[it.index()].trim());
    }
    *has_input_attr = true;
  }

  if (auto str =
          dict_attr.get("outputs").dyn_cast_or_null<mlir::StringAttr>()) {
    str.getValue().split(output_names, ',', /*MaxSplit=*/-1,
                         /*KeepEmpty=*/false);
    auto term = fn.getBlocks().back().getTerminator();
    if (output_names.size() != term->getNumOperands()) {
      fn.emitWarning() << "output names (" << output_names.size()
                       << ") != terminator operands (" << term->getNumOperands()
                       << ")";
      return;
    }
    for (const auto& it : llvm::enumerate(term->getOperands())) {
      // TODO(jpienaar): If this isn't due to an op, then we'd need to either
      // ensure the name that will be assigned to the buffer is the same, or
      // insert an op so that we can have a buffer named such. This cannot
      // currently happen due to pseudo_input nodes.
      if (auto op = it.value()->getDefiningOp()) {
        name_mapper_.InitOpName(op, output_names[it.index()].trim());
      } else {
        fn.emitWarning() << "output is not due to an op and '"
                         << output_names[it.index()]
                         << "' may not be a named output";
      }
    }
  }
}

bool Translator::IsStatefulOperand(mlir::Operation* op, int operand_index) {
  std::vector<int> operand_indices;
  // TODO(b/138254427): When the bug is addressed, we'll be able to inspect
  // for the presence of a specific OpTrait using mlir::Operation, without
  // having to cast it to specific ops like below.
  // Until then, when a new RNN/LSTM op is added to TFLite and has stateful
  // tensors as operands, they will need to be added here as well.
  if (!mlir::TFL::IsStatefulOp(op, &operand_indices)) return false;
  return absl::c_find(operand_indices, operand_index) != operand_indices.end();
}

Optional<BufferOffset<tflite::SubGraph>> Translator::BuildSubGraph(FuncOp fn) {
  bool has_input_attr = false;
  InitializeNamesFromAttribute(fn, &has_input_attr);
  std::vector<BufferOffset<tflite::Tensor>> tensors;
  llvm::DenseMap<Value*, int> tensor_index_map;

  // Builds tensor and buffer for argument or operation result. Returns false
  // on failure.
  auto build_tensor_and_buffer = [&](Value* value, const std::string& name) {
    // NoneType represents optional and may be skipped here.
    if (value->getType().isa<NoneType>()) {
      return true;
    }

    tensor_index_map.insert({value, tensors.size()});
    auto tensor_or = BuildTensor(value, name, buffers_.size());
    if (!tensor_or) return false;
    tensors.push_back(*tensor_or);

    // TODO(ashwinm): Check if for stateful tensors, if it is also needed to
    // make the Buffer empty apart from setting the buffer_idx=0 in the Tensor.
    // This does not seem to affect runtime behavior for RNN/LSTM, but would be
    // good for reducing memory footprint.
    if (auto* inst = value->getDefiningOp()) {
      auto buffer_or = BuildBuffer(inst);
      if (!buffer_or) return false;
      buffers_.push_back(*buffer_or);
    } else {
      buffers_.push_back(empty_buffer_);
    }
    return true;
  };

  std::vector<BufferOffset<tflite::Operator>> operators;
  auto& bb = fn.getBlocks().front();

  // Main function's arguments are first passed to `input` op so they don't
  // have associated tensor and buffer. Build FlatBuffer tensor and buffer for
  // other functions.
  for (unsigned i = 0, e = bb.getNumArguments(); i < e; ++i) {
    mlir::BlockArgument* arg = bb.getArgument(i);
    std::string name;
    if (has_input_attr) name = name_mapper_.GetUniqueName(arg);
    if (name.empty()) name = absl::StrCat("arg", i);
    if (!build_tensor_and_buffer(arg, name)) return llvm::None;
  }

  bool failed_once = false;
  for (auto& inst : bb) {
    if (inst.isKnownTerminator()) break;

    std::string name = UniqueName(&inst);
    for (size_t i = 0, e = inst.getNumResults(); i < e; ++i) {
      // Tensors are named by adding result index to name for the particular
      // operation such as name:0, name:1, name:2 etc. Default port is zero so
      // the first result can be specified without the port. This is based on
      // TensorFlow's naming scheme for inputs in the NodeDef proto.
      std::string suffix = i > 0 ? absl::StrCat(":", i) : "";
      if (!build_tensor_and_buffer(inst.getResult(i), name + suffix)) {
        return llvm::None;
      }
    }

    // Skip constant ops as they don't represent a TFLite operator.
    if (IsConst(&inst)) continue;

    // Fetch operand and result tensor indices.
    std::vector<int32_t> operands;
    operands.reserve(inst.getNumOperands());
    for (auto* operand : inst.getOperands()) {
      if (operand->getType().isa<NoneType>())
        operands.push_back(kTfLiteOptionalTensor);
      else
        operands.push_back(tensor_index_map.lookup(operand));
    }
    std::vector<int32_t> results;
    results.reserve(inst.getNumOperands());
    for (auto* result : inst.getResults()) {
      results.push_back(tensor_index_map.lookup(result));
    }

    if (auto tfl_operator = BuildOperator(&inst, operands, results))
      operators.push_back(*tfl_operator);
    else
      failed_once = true;
  }

  if (failed_once) return llvm::None;

  // Get input and output tensor indices for the subgraph.
  std::vector<int32_t> inputs, outputs;
  for (auto* arg : bb.getArguments()) {
    inputs.push_back(tensor_index_map[arg]);
  }
  for (auto* result : bb.getTerminator()->getOperands()) {
    outputs.push_back(tensor_index_map[result]);
  }

  return tflite::CreateSubGraph(
      builder_, builder_.CreateVector(tensors), builder_.CreateVector(inputs),
      builder_.CreateVector(outputs), builder_.CreateVector(operators),
      /*name=*/builder_.CreateString(fn.getName().str()));
}

BufferOffset<tflite::Metadata> Translator::BuildMetadata(StringRef name,
                                                         StringRef content) {
  auto buffer_index = buffers_.size();
  auto buffer_data = builder_.CreateVector(
      reinterpret_cast<const uint8_t*>(content.data()), content.size());
  buffers_.push_back(tflite::CreateBuffer(builder_, buffer_data));
  return tflite::CreateMetadataDirect(builder_, name.data(), buffer_index);
}

Optional<VectorBufferOffset<BufferOffset<tflite::Metadata>>>
Translator::CreateMetadataVector() {
  auto dict_attr = module_.getAttrOfType<mlir::DictionaryAttr>("tfl.metadata");
  if (!dict_attr) return VectorBufferOffset<BufferOffset<tflite::Metadata>>();

  std::vector<BufferOffset<tflite::Metadata>> metadata;
  for (const auto& named_attr : dict_attr) {
    StringRef name = named_attr.first;
    mlir::Attribute attr = named_attr.second;
    if (auto content = attr.dyn_cast<StringAttr>()) {
      metadata.push_back(BuildMetadata(name, content.getValue()));
    } else {
      module_.emitError(
          "all values in tfl.metadata's dictionary key-value pairs should be "
          "string attributes");
      return llvm::None;
    }
  }
  return builder_.CreateVector(metadata);
}

Optional<std::string> Translator::Translate(
    ModuleOp module, bool emit_builtin_tflite_ops, bool emit_select_tf_ops,
    bool emit_custom_ops, OpOrArgNameMapper* op_or_arg_name_mapper) {
  if (!IsValidTFLiteMlirModule(module)) return llvm::None;
  Translator translator(module, emit_builtin_tflite_ops, emit_select_tf_ops,
                        emit_custom_ops, op_or_arg_name_mapper);
  return translator.TranslateInternal();
}

Optional<std::string> Translator::TranslateInternal() {
  // Create a list of functions in the module with main function being the
  // first function in the list. This is required as the first subgraph in the
  // model is entry point for the model.
  std::vector<FuncOp> functions;
  functions.reserve(std::distance(module_.begin(), module_.end()));

  int subgraph_idx = 0;
  FuncOp main_fn = module_.lookupSymbol<FuncOp>("main");
  subgraph_index_map_[main_fn.getName().str()] = subgraph_idx++;
  functions.push_back(main_fn);
  for (auto fn : module_.getOps<FuncOp>()) {
    if (fn == main_fn) continue;

    subgraph_index_map_[fn.getName().str()] = subgraph_idx++;
    functions.push_back(fn);
  }

  // Build subgraph for each of the functions.
  std::vector<BufferOffset<tflite::SubGraph>> subgraphs;
  subgraphs.reserve(functions.size());
  int first_failed_func = -1;
  for (int i = 0; i < functions.size(); ++i) {
    auto subgraph_or = BuildSubGraph(functions[i]);
    if (!subgraph_or) {
      if (first_failed_func == -1)
        // Record the index of the first function that cannot be converted.
        // Keep looping through all subgraphs in the module to make sure that
        // we collect the list of missing ops from the entire module.
        first_failed_func = i;
    } else {
      subgraphs.push_back(*subgraph_or);
    }
  }

  if (first_failed_func != -1) {
    std::string failed_flex_ops_list = absl::StrJoin(failed_flex_ops_, ",");
    std::string failed_custom_ops_list = absl::StrJoin(failed_custom_ops_, ",");
    std::string err;
    if (!failed_flex_ops_list.empty())
      err +=
          "Ops that can be supported by the flex runtime (enabled via setting "
          "the -emit-select-tf-ops flag): " +
          failed_flex_ops_list + ".";
    if (!failed_custom_ops_list.empty())
      err +=
          "Ops that need custom implementation (enabled via setting the "
          "-emit-custom-ops flag): " +
          failed_custom_ops_list;

    return functions[first_failed_func].emitError("failed while converting: '")
               << functions[first_failed_func].getName() << "\'\n"
               << err,
           llvm::None;
  }

  std::string model_description;
  if (auto attr = module_.getAttrOfType<StringAttr>("tfl.description")) {
    model_description = attr.getValue().str();
  } else {
    model_description = "MLIR Converted.";
  }

  // Build the model and finish the model building process.
  auto description = builder_.CreateString(model_description.data());
  VectorBufferOffset<int32_t> metadata_buffer = 0;  // Deprecated
  auto metadata = CreateMetadataVector();
  if (!metadata) return llvm::None;

  auto model = tflite::CreateModel(
      builder_, TFLITE_SCHEMA_VERSION, builder_.CreateVector(opcodes_),
      builder_.CreateVector(subgraphs), description,
      builder_.CreateVector(buffers_), metadata_buffer, *metadata);
  tflite::FinishModelBuffer(builder_, model);
  tflite::UpdateOpVersion(builder_.GetBufferPointer());

  // Return serialized string for the built FlatBuffer.
  return std::string(reinterpret_cast<const char*>(builder_.GetBufferPointer()),
                     builder_.GetSize());
}

}  // namespace

// Translates the given MLIR module in the TFLite dialect to TFLite FlatBuffer
// format. Returns false on success.
//
// TODO(hinsu): Support all valid MLIR modules in TFLite dialect by supporting
// the following:
//
// * Quantization
// * Ops with variable tensors
//
bool tflite::MlirToFlatBufferTranslateFunction(
    ModuleOp module, std::string* serialized_flatbuffer,
    bool emit_builtin_tflite_ops, bool emit_select_tf_ops, bool emit_custom_ops,
    OpOrArgNameMapper* op_or_arg_name_mapper) {
  auto maybe_translated =
      Translator::Translate(module, emit_builtin_tflite_ops, emit_select_tf_ops,
                            emit_custom_ops, op_or_arg_name_mapper);
  if (!maybe_translated) return true;
  *serialized_flatbuffer = std::move(*maybe_translated);
  return false;
}

bool tflite::MlirToFlatBufferTranslateFunction(
    ModuleOp module, std::string* serialized_flatbuffer,
    bool emit_builtin_tflite_ops, bool emit_select_tf_ops,
    bool emit_custom_ops) {
  OpOrArgLocNameMapper op_or_arg_name_mapper;
  return MlirToFlatBufferTranslateFunction(
      module, serialized_flatbuffer, emit_builtin_tflite_ops,
      emit_select_tf_ops, emit_custom_ops, &op_or_arg_name_mapper);
}

static mlir::LogicalResult MlirToFlatBufferFileTranslateFunction(
    ModuleOp module, llvm::raw_ostream& output) {
  std::string serialized_flatbuffer;
  std::unique_ptr<OpOrArgNameMapper> op_or_arg_name_mapper;
  if (strip_debug_info) {
    op_or_arg_name_mapper =
        std::make_unique<tensorflow::OpOrArgStripNameMapper>();
  } else {
    op_or_arg_name_mapper = std::make_unique<OpOrArgLocNameMapper>();
  }
  if (tflite::MlirToFlatBufferTranslateFunction(
          module, &serialized_flatbuffer, emit_builtin_tflite_ops,
          emit_select_tf_ops, emit_custom_ops, op_or_arg_name_mapper.get()))
    return mlir::failure();

  output << serialized_flatbuffer;
  return mlir::success();
}

static TranslateFromMLIRRegistration MLIRToFlatBufferTranslate(
    "mlir-to-tflite-flatbuffer", MlirToFlatBufferFileTranslateFunction);
