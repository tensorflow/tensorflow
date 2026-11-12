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

#include "tensorflow/compiler/mlir/lite/export/export.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/string.h"  // from @flatbuffers
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/core/macros.h"
#include "tensorflow/compiler/mlir/lite/export/constants.h"
#include "tensorflow/compiler/mlir/lite/export/payloads.h"
#include "tensorflow/compiler/mlir/lite/export/tensor_names.h"
#include "tensorflow/compiler/mlir/lite/export/types.h"
#include "tensorflow/compiler/mlir/lite/export/vhlo_operators.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_operator.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/schema/mutable/schema_generated.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_conversion_utils.h"
#include "tensorflow/compiler/mlir/lite/tools/versioning/op_version.h"
#include "tensorflow/compiler/mlir/lite/tools/versioning/runtime_version.h"
#include "tensorflow/compiler/mlir/lite/version.h"

namespace mlir::TFL {
namespace fb_export {
namespace {

using ::flatbuffers::Offset;
using ::mlir::func::FuncOp;

// Function attributes describing signatures, set by the model importer.
constexpr StringLiteral kEntryFunctionAttr = "tf.entry_function";
constexpr StringLiteral kIndexPathAttr = "tf_saved_model.index_path";
constexpr StringLiteral kExportedNamesAttr = "tf_saved_model.exported_names";

// Tensor index of an omitted optional operand.
constexpr int32_t kOptionalTensor = -1;

// Initial value of the offsets and sizes patched once payloads are written. It
// differs from the default value so that the fields are stored, and thereby
// mutable.
constexpr uint64_t kPlaceholder = 1;

// Domains of payload fingerprints.
enum class PayloadKind { kConstant, kBytes, kCustomOptions };

// Returns the `tf.entry_function` attribute of `function`, or null if it isn't
// an entry function.
DictionaryAttr GetEntryFunctionAttr(FuncOp function) {
  auto attr = function->getAttrOfType<DictionaryAttr>(kEntryFunctionAttr);
  return attr && !attr.empty() ? attr : nullptr;
}

// Returns the comma separated names in `attr[key]`.
llvm::SmallVector<StringRef> GetNames(DictionaryAttr attr, StringRef key) {
  llvm::SmallVector<StringRef> names;
  if (auto names_attr = dyn_cast_or_null<StringAttr>(attr.get(key))) {
    names_attr.getValue().split(names, ',', /*MaxSplit=*/-1,
                                /*KeepEmpty=*/false);
  }
  for (StringRef& name : names) name = name.trim();
  return names;
}

// Returns the signature names in the `tf_saved_model.index_path` attributes
// of a function's arguments or results.
std::vector<std::string> GetSignatureNames(ArrayRef<DictionaryAttr> attrs) {
  std::vector<std::string> names;
  for (DictionaryAttr attr : attrs) {
    if (!attr) continue;
    auto path = dyn_cast_or_null<ArrayAttr>(attr.get(kIndexPathAttr));
    if (!path || path.empty()) continue;
    if (auto name = dyn_cast<StringAttr>(path[0])) {
      names.push_back(name.getValue().str());
    }
  }
  return names;
}

bool IsReturned(Value value) {
  return llvm::any_of(value.getUsers(), [](Operation* user) {
    return user->hasTrait<OpTrait::IsTerminator>();
  });
}

class ModelBuilder : public ModelContext {
 public:
  explicit ModelBuilder(ModuleOp module) : module_(module) {}

  absl::Status Export(llvm::raw_pwrite_stream& stream);

  // ModelContext:
  flatbuffers::FlatBufferBuilder& builder() override { return fbb_; }
  uint32_t GetOpcodeIndex(tflite::BuiltinOperator op,
                          StringRef custom_code) override;
  std::optional<int32_t> GetFunctionSubgraph(StringRef name) override;
  std::optional<int32_t> BuildRegionSubgraph(
      Region& region, tflite::BuiltinOperator op) override;

 private:
  // The subgraph being built.
  struct Subgraph {
    int index;
    std::vector<Offset<tflite::Tensor>> tensors;
    llvm::DenseMap<Value, int32_t> tensor_indices;
    std::vector<Offset<tflite::Operator>> operators;
    std::vector<int32_t> inputs;
    std::vector<int32_t> outputs;
  };

  // A function with a signature.
  struct EntryPoint {
    FuncOp function;
    int subgraph;
    std::vector<int32_t> inputs;
    std::vector<int32_t> outputs;
  };

  // Custom options to point to once payloads are written.
  struct CustomOptions {
    int subgraph;
    int op;
    int payload;
  };

  // Returns the defined functions in subgraph order: entry functions first,
  // `main` first among its kind.
  std::vector<FuncOp> GetFunctions();

  // Builds the subgraph at `index` from the single block of `region`, the body
  // of `function` if not null.
  std::optional<Subgraph> BuildSubgraph(StringRef name, Region& region,
                                        int index, FuncOp function);

  // Names the arguments and results of an entry function as specified by its
  // `tf.entry_function` attribute. Returns true if the arguments were named.
  bool NameEntryFunctionTensors(FuncOp function);

  bool AddTensor(Value value, StringRef name, Subgraph& subgraph);
  std::optional<Offset<tflite::QuantizationParameters>> BuildQuantization(
      Type element_type, Location loc, StringRef tensor_name,
      int32_t tensor_index, std::vector<Offset<tflite::Tensor>>& side_tensors);

  bool AddOperator(Operation* op, Subgraph& subgraph);
  std::optional<Offset<tflite::Operator>> BuildTflOperator(
      Operation* op, std::vector<int32_t> inputs,
      const std::vector<int32_t>& outputs, Subgraph& subgraph);
  Offset<tflite::Operator> BuildCustomOperator(
      CustomOp op, const std::vector<int32_t>& inputs,
      const std::vector<int32_t>& outputs, Subgraph& subgraph);

  // Buffers. Each returns the index of the buffer.
  int AddConstantBuffer(const Constant& constant, Type type, bool shareable);
  int AddBytesBuffer(std::string bytes);
  int AddPayloadBuffer(int payload);
  int AddInlineBuffer(StringRef bytes);

  Offset<flatbuffers::Vector<Offset<tflite::Metadata>>> BuildMetadata();
  Offset<flatbuffers::Vector<Offset<tflite::SignatureDef>>>
  BuildSignatureDefs();

  // Writes the flatbuffer followed by the payloads.
  absl::Status Write(llvm::raw_pwrite_stream& stream);

  ModuleOp module_;
  flatbuffers::FlatBufferBuilder fbb_;
  TensorNamer namer_;
  PayloadTable payloads_;

  std::vector<Offset<tflite::Buffer>> buffers_;
  // (buffer index, payload id) of the buffers stored as payloads.
  std::vector<std::pair<int, int>> payload_buffers_;
  // Buffers of shareable constants, by attribute (and encoding) and by
  // fingerprint of their data and type.
  struct SharedBuffer {
    int buffer;
    int payload;
  };
  llvm::DenseMap<std::pair<Attribute, int>, int> buffer_by_attr_;
  absl::flat_hash_map<uint64_t, SharedBuffer> buffer_by_fingerprint_;

  std::vector<Offset<tflite::OperatorCode>> opcodes_;
  std::map<std::pair<tflite::BuiltinOperator, std::string>, uint32_t>
      opcode_indices_;

  std::vector<Offset<tflite::SubGraph>> subgraphs_;
  llvm::StringMap<int> function_subgraphs_;
  std::vector<EntryPoint> entry_points_;
  std::vector<CustomOptions> custom_options_;
};

absl::Status ModelBuilder::Export(llvm::raw_pwrite_stream& stream) {
  // Buffer 0 is the empty buffer of tensors without data.
  buffers_.push_back(tflite::CreateBuffer(fbb_));

  std::vector<FuncOp> functions = GetFunctions();
  const bool has_main = module_.lookupSymbol<FuncOp>("main") != nullptr;
  const int num_entry_functions = llvm::count_if(
      functions, [](FuncOp f) { return GetEntryFunctionAttr(f) != nullptr; });
  if (!has_main && num_entry_functions == 0) {
    module_.emitError("must have a `main` function or entry functions");
    return absl::InvalidArgumentError("No entry function found.");
  }

  subgraphs_.resize(functions.size());
  for (int index = 0; index < functions.size(); ++index) {
    function_subgraphs_[functions[index].getName()] = index;
  }

  bool failed = false;
  for (int index = 0; index < functions.size(); ++index) {
    FuncOp function = functions[index];
    if (!llvm::hasSingleElement(function.getBody())) {
      function.emitError("must have exactly one block");
      failed = true;
      continue;
    }
    const bool is_entry = GetEntryFunctionAttr(function) != nullptr;
    // A lone entry function is the `main` subgraph.
    const StringRef name = !has_main && num_entry_functions == 1 && is_entry
                               ? "main"
                               : function.getName();
    std::optional<Subgraph> subgraph =
        BuildSubgraph(name, function.getBody(), index, function);
    if (!subgraph.has_value()) {
      failed = true;
      continue;
    }
    if (is_entry) {
      entry_points_.push_back({function, index, std::move(subgraph->inputs),
                               std::move(subgraph->outputs)});
    }
  }
  if (failed) {
    return absl::InvalidArgumentError(
        "Failed to export the module to a TFLite flatbuffer; see the emitted "
        "errors.");
  }

  const auto metadata = BuildMetadata();
  const auto signature_defs = BuildSignatureDefs();
  const auto model = tflite::CreateModel(
      fbb_, TFLITE_SCHEMA_VERSION, fbb_.CreateVector(opcodes_),
      fbb_.CreateVector(subgraphs_), fbb_.CreateString("MLIR Converted."),
      fbb_.CreateVector(buffers_), /*metadata_buffer=*/0, metadata,
      signature_defs);
  tflite::FinishModelBuffer(fbb_, model);
  return Write(stream);
}

std::vector<FuncOp> ModelBuilder::GetFunctions() {
  FuncOp main = module_.lookupSymbol<FuncOp>("main");
  std::vector<FuncOp> functions, non_entry_functions;
  auto add = [&](FuncOp function) {
    (GetEntryFunctionAttr(function) ? functions : non_entry_functions)
        .push_back(function);
  };
  if (main) add(main);
  for (FuncOp function : module_.getOps<FuncOp>()) {
    if (function != main && !function.isDeclaration()) add(function);
  }
  functions.insert(functions.end(), non_entry_functions.begin(),
                   non_entry_functions.end());
  return functions;
}

std::optional<ModelBuilder::Subgraph> ModelBuilder::BuildSubgraph(
    StringRef name, Region& region, int index, FuncOp function) {
  Subgraph subgraph;
  subgraph.index = index;
  Block& block = region.front();

  const bool has_input_names = function && NameEntryFunctionTensors(function);
  for (BlockArgument arg : block.getArguments()) {
    std::string tensor_name;
    if (has_input_names) {
      tensor_name = namer_.GetName(arg).str();
    } else if (name == "main") {
      tensor_name = absl::StrCat("arg", arg.getArgNumber());
    } else {
      tensor_name = absl::StrCat(name.str(), "_arg", arg.getArgNumber());
    }
    if (!AddTensor(arg, tensor_name, subgraph)) return std::nullopt;
    subgraph.inputs.push_back(subgraph.tensor_indices.at(arg));
  }

  bool failed = false;
  for (Operation& op : block.without_terminator()) {
    // Keep going after a failure, to report all unsupported ops at once.
    if (!AddOperator(&op, subgraph)) failed = true;
  }
  if (failed) return std::nullopt;

  for (Value result : block.getTerminator()->getOperands()) {
    auto it = subgraph.tensor_indices.find(result);
    if (it == subgraph.tensor_indices.end()) {
      block.getTerminator()->emitOpError(
          "returns a value defined outside of its region");
      return std::nullopt;
    }
    subgraph.outputs.push_back(it->second);
  }

  subgraphs_[index] = tflite::CreateSubGraph(
      fbb_, fbb_.CreateVector(subgraph.tensors),
      fbb_.CreateVector(subgraph.inputs), fbb_.CreateVector(subgraph.outputs),
      fbb_.CreateVector(subgraph.operators), fbb_.CreateString(name.str()));
  return subgraph;
}

bool ModelBuilder::NameEntryFunctionTensors(FuncOp function) {
  DictionaryAttr attr = GetEntryFunctionAttr(function);
  if (!attr) return false;

  bool has_input_names = false;
  if (attr.get("inputs")) {
    const auto names = GetNames(attr, "inputs");
    if (names.size() != function.getNumArguments()) {
      function.emitWarning("invalid entry function specification");
      return false;
    }
    for (auto [arg, name] : llvm::zip(function.getArguments(), names)) {
      namer_.SetName(arg, name);
    }
    has_input_names = true;
  }

  if (attr.get("outputs")) {
    const auto names = GetNames(attr, "outputs");
    Operation* terminator = function.front().getTerminator();
    if (names.size() != terminator->getNumOperands()) {
      function.emitWarning("output names (")
          << names.size() << ") != terminator operands ("
          << terminator->getNumOperands() << ")";
      return has_input_names;
    }
    for (auto [result, name] : llvm::zip(terminator->getOperands(), names)) {
      if (!PrefersHierarchicalName(result, name)) namer_.SetName(result, name);
    }
  }
  return has_input_names;
}

bool ModelBuilder::AddTensor(Value value, StringRef name, Subgraph& subgraph) {
  if (isa<NoneType>(value.getType())) {
    subgraph.tensor_indices[value] = kOptionalTensor;
    return true;
  }
  auto type = dyn_cast<TensorType>(value.getType());
  if (!type) {
    emitError(value.getLoc(), "expected a tensor, got ") << value.getType();
    return false;
  }
  const absl::StatusOr<tflite::TensorType> tflite_type =
      GetTensorType(type.getElementType());
  if (!tflite_type.ok()) {
    emitError(value.getLoc(), tflite_type.status().message());
    return false;
  }

  std::optional<Constant> constant = GetConstant(value.getDefiningOp());
  int buffer = 0;
  if (constant.has_value()) {
    if (constant->encoding != ConstantEncoding::kWidenedToF32) {
      constant->encoding = GetEncoding(*tflite_type);
    }
    if (absl::Status status = CheckConstant(*constant); !status.ok()) {
      emitError(value.getLoc(), status.message());
      return false;
    }
    buffer = AddConstantBuffer(*constant, type, !IsReturned(value));
  }

  // A constant may have a dynamic type (e.g. after folding), but its value
  // has a static shape.
  ArrayRef<int64_t> dims;
  if (constant.has_value() && !type.hasStaticShape()) {
    dims = constant->value.getShapedType().getShape();
  } else if (type.hasRank()) {
    dims = type.getShape();
  }
  std::vector<int32_t> shape, shape_signature;
  for (int64_t dim : dims) {
    if (dim > std::numeric_limits<int32_t>::max()) {
      emitError(value.getLoc(), "dimension out of the int32 range: ") << dim;
      return false;
    }
    const bool is_dynamic = ShapedType::isDynamic(dim);
    shape.push_back(is_dynamic ? 1 : dim);
    shape_signature.push_back(is_dynamic ? -1 : dim);
  }
  const bool has_dynamic_dim = llvm::any_of(
      dims, [](int64_t dim) { return ShapedType::isDynamic(dim); });

  const int32_t index = subgraph.tensors.size();
  std::vector<Offset<tflite::Tensor>> side_tensors;
  const auto quantization = BuildQuantization(
      type.getElementType(), value.getLoc(), name, index, side_tensors);
  if (!quantization.has_value()) return false;

  subgraph.tensor_indices[value] = index;
  subgraph.tensors.push_back(tflite::CreateTensor(
      fbb_, fbb_.CreateVector(shape), *tflite_type, buffer,
      fbb_.CreateString(name.str()), *quantization, /*is_variable=*/false,
      /*sparsity=*/0, has_dynamic_dim ? fbb_.CreateVector(shape_signature) : 0,
      /*has_rank=*/type.hasRank()));
  subgraph.tensors.insert(subgraph.tensors.end(), side_tensors.begin(),
                          side_tensors.end());
  return true;
}

std::optional<Offset<tflite::QuantizationParameters>>
ModelBuilder::BuildQuantization(
    Type element_type, Location loc, StringRef tensor_name,
    int32_t tensor_index, std::vector<Offset<tflite::Tensor>>& side_tensors) {
  if (auto type = dyn_cast<quant::UniformQuantizedType>(element_type)) {
    // min and max aren't known, so they are left as 0.
    return tflite::CreateQuantizationParameters(
        fbb_, /*min=*/0, /*max=*/0,
        fbb_.CreateVector<float>({static_cast<float>(type.getScale())}),
        fbb_.CreateVector<int64_t>({type.getZeroPoint()}));
  }
  if (auto type = dyn_cast<quant::UniformQuantizedPerAxisType>(element_type)) {
    const std::vector<float> scales(type.getScales().begin(),
                                    type.getScales().end());
    return tflite::CreateQuantizationParameters(
        fbb_, /*min=*/0, /*max=*/0, fbb_.CreateVector(scales),
        fbb_.CreateVector(type.getZeroPoints().vec()),
        tflite::QuantizationDetails_NONE, /*details=*/0,
        type.getQuantizedDimension());
  }
  auto blockwise = dyn_cast<BlockwiseQuantType>(element_type);
  if (!blockwise) return tflite::CreateQuantizationParameters(fbb_);

  // Blockwise scales and zero points are tensors of their own, placed right
  // after the quantized tensor.
  auto add_parameter_tensor = [&](const BlockwiseParameter& parameter,
                                  absl::string_view suffix) {
    const int32_t index = tensor_index + 1 + side_tensors.size();
    const int buffer = AddBytesBuffer(parameter.data);
    side_tensors.push_back(tflite::CreateTensor(
        fbb_, fbb_.CreateVector(parameter.shape), parameter.type, buffer,
        fbb_.CreateString(absl::StrCat(tensor_name.str(), suffix)),
        /*quantization=*/0, /*is_variable=*/false, /*sparsity=*/0,
        /*shape_signature=*/0, /*has_rank=*/true));
    return index;
  };

  const absl::StatusOr<BlockwiseParameter> scales =
      SerializeBlockwiseScales(blockwise.getScales());
  if (!scales.ok()) {
    emitError(loc, scales.status().message()) << ": " << blockwise;
    return std::nullopt;
  }
  const int32_t scales_index = add_parameter_tensor(*scales, "_scales");

  // -1 means no zero points, i.e. zero points of 0.
  int32_t zero_points_index = -1;
  if (ElementsAttr zero_points_attr = blockwise.getZeroPoints()) {
    const absl::StatusOr<BlockwiseParameter> zero_points =
        SerializeBlockwiseZeroPoints(zero_points_attr);
    if (!zero_points.ok()) {
      emitError(loc, zero_points.status().message()) << ": " << blockwise;
      return std::nullopt;
    }
    zero_points_index = add_parameter_tensor(*zero_points, "_zero_points");
  }

  const std::vector<int32_t> block_shape(blockwise.getBlockShape().begin(),
                                         blockwise.getBlockShape().end());
  const auto details = tflite::CreateBlockwiseQuantization(
      fbb_, scales_index, zero_points_index, /*block_size=*/0,
      fbb_.CreateVector(block_shape));
  return tflite::CreateQuantizationParameters(
      fbb_, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
      tflite::QuantizationDetails_BlockwiseQuantization, details.Union());
}

bool ModelBuilder::AddOperator(Operation* op, Subgraph& subgraph) {
  // Folded into the constants its users produce.
  if (IsOnlyUsedByFoldedCasts(op)) return true;

  const std::optional<Constant> constant = GetConstant(op);
  const bool is_folded_cast =
      constant.has_value() &&
      constant->encoding == ConstantEncoding::kWidenedToF32;
  for (Value result : op->getResults()) {
    // A folded cast is named after the constant it casts.
    const StringRef name = is_folded_cast
                               ? namer_.GetName(result, op->getOperand(0))
                               : namer_.GetName(result);
    if (!AddTensor(result, name, subgraph)) return false;
  }
  // Constants are tensors, not operators.
  if (constant.has_value() || isa<NoValueOp>(op)) return true;

  std::vector<int32_t> inputs, outputs;
  for (Value operand : op->getOperands()) {
    auto it = subgraph.tensor_indices.find(operand);
    if (it == subgraph.tensor_indices.end()) {
      op->emitOpError("uses a value defined outside of its region");
      return false;
    }
    inputs.push_back(it->second);
  }
  for (Value result : op->getResults()) {
    outputs.push_back(subgraph.tensor_indices.at(result));
  }

  std::optional<Offset<tflite::Operator>> fb_op;
  if (isa<TensorFlowLiteDialect>(op->getDialect())) {
    fb_op = BuildTflOperator(op, std::move(inputs), outputs, subgraph);
  } else if (isa<vhlo::VhloDialect>(op->getDialect())) {
    fb_op = BuildVhloOperator(op, inputs, outputs, *this);
  } else {
    op->emitOpError("is not supported by the TFLite flatbuffer exporter");
  }
  if (!fb_op.has_value()) return false;
  subgraph.operators.push_back(*fb_op);
  return true;
}

std::optional<Offset<tflite::Operator>> ModelBuilder::BuildTflOperator(
    Operation* op, std::vector<int32_t> inputs,
    const std::vector<int32_t>& outputs, Subgraph& subgraph) {
  if (isa<WhileOp, IfOp, CallOnceOp>(op)) {
    op->emitOpError(
        "control flow is not supported by the TFLite flatbuffer exporter");
    return std::nullopt;
  }
  if (auto custom = dyn_cast<CustomOp>(op)) {
    return BuildCustomOperator(custom, inputs, outputs, subgraph);
  }
  const std::optional<tflite::BuiltinOperator> code = GetBuiltinOpCode(op);
  if (!code.has_value()) {
    op->emitOpError("is not a TFLite builtin op");
    return std::nullopt;
  }
  // Drop the omitted bias of TRANSPOSE_CONV, for older runtimes.
  if (*code == tflite::BuiltinOperator_TRANSPOSE_CONV && inputs.size() == 4 &&
      inputs[3] == kOptionalTensor) {
    inputs.pop_back();
  }
  auto fb_op = CreateFlatBufferOperator(op, GetOpcodeIndex(*code, ""), inputs,
                                        outputs, /*intermediates=*/{}, &fbb_);
  if (!fb_op.has_value()) op->emitOpError("is not a supported TFLite op");
  return fb_op;
}

Offset<tflite::Operator> ModelBuilder::BuildCustomOperator(
    CustomOp op, const std::vector<int32_t>& inputs,
    const std::vector<int32_t>& outputs, Subgraph& subgraph) {
  // Custom options can be large (e.g. embedded data), so they are payloads.
  const ConstBytesAttr options = op.getCustomOption();
  const int payload = payloads_.Add(
      llvm::hash_combine(static_cast<int>(PayloadKind::kCustomOptions),
                         options),
      options.getValue().size(), [options](ByteSink sink) {
        const StringRef bytes = options.getValue();
        return sink(absl::string_view(bytes.data(), bytes.size()));
      });
  custom_options_.push_back(
      {subgraph.index, static_cast<int>(subgraph.operators.size()), payload});
  return tflite::CreateOperator(
      fbb_, GetOpcodeIndex(tflite::BuiltinOperator_CUSTOM, op.getCustomCode()),
      fbb_.CreateVector(inputs), fbb_.CreateVector(outputs),
      tflite::BuiltinOptions_NONE, /*builtin_options=*/0,
      /*custom_options=*/0, tflite::CustomOptionsFormat_FLEXBUFFERS,
      /*mutating_variable_inputs=*/0, /*intermediates=*/0,
      /*large_custom_options_offset=*/kPlaceholder,
      /*large_custom_options_size=*/kPlaceholder);
}

uint32_t ModelBuilder::GetOpcodeIndex(tflite::BuiltinOperator op,
                                      StringRef custom_code) {
  auto [it, inserted] = opcode_indices_.try_emplace(
      std::make_pair(op, custom_code.str()), opcodes_.size());
  if (inserted) {
    const auto custom_code_offset = op == tflite::BuiltinOperator_CUSTOM
                                        ? fbb_.CreateString(custom_code.str())
                                        : Offset<flatbuffers::String>();
    // Builtin op versions are set by `UpdateOpVersion` once the model is
    // built; 0 makes the field stored, and thereby mutable.
    const int32_t version = op == tflite::BuiltinOperator_CUSTOM ? 1 : 0;
    opcodes_.push_back(
        tflite::CreateOperatorCode(fbb_, op, custom_code_offset, version));
  }
  return it->second;
}

std::optional<int32_t> ModelBuilder::GetFunctionSubgraph(StringRef name) {
  auto it = function_subgraphs_.find(name);
  if (it == function_subgraphs_.end()) return std::nullopt;
  return it->second;
}

std::optional<int32_t> ModelBuilder::BuildRegionSubgraph(
    Region& region, tflite::BuiltinOperator op) {
  const int32_t index = subgraphs_.size();
  subgraphs_.emplace_back();
  const std::string name =
      absl::StrCat(tflite::EnumNameBuiltinOperator(op), index);
  if (!BuildSubgraph(name, region, index, /*function=*/nullptr)) {
    return std::nullopt;
  }
  return index;
}

int ModelBuilder::AddConstantBuffer(const Constant& constant, Type type,
                                    bool shareable) {
  const std::pair<Attribute, int> attr_key(constant.value,
                                           static_cast<int>(constant.encoding));
  if (shareable) {
    if (auto it = buffer_by_attr_.find(attr_key); it != buffer_by_attr_.end()) {
      return it->second;
    }
  }
  const uint64_t fingerprint = llvm::hash_combine(
      static_cast<int>(PayloadKind::kConstant), GetFingerprint(constant));
  // Distinct attributes can hold the same data, e.g. a weight materialized in
  // each signature. Sharing the buffer (not only its payload) shows consumers
  // like quantizers one weight rather than copies.
  const uint64_t buffer_key = llvm::hash_combine(fingerprint, type);
  if (shareable) {
    if (auto it = buffer_by_fingerprint_.find(buffer_key);
        it != buffer_by_fingerprint_.end()) {
      // The data of `constant` isn't read again, so it can be released with
      // that of the buffer it duplicates.
      payloads_.TieBlob(it->second.payload, GetResourceBlob(constant));
      buffer_by_attr_[attr_key] = it->second.buffer;
      return it->second.buffer;
    }
  }

  const int payload = payloads_.Add(
      fingerprint, GetByteSize(constant),
      [constant](ByteSink sink) { return WriteConstant(constant, sink); });
  payloads_.TieBlob(payload, GetResourceBlob(constant));
  const int buffer = AddPayloadBuffer(payload);
  if (shareable) {
    buffer_by_attr_[attr_key] = buffer;
    buffer_by_fingerprint_[buffer_key] = {buffer, payload};
  }
  return buffer;
}

int ModelBuilder::AddBytesBuffer(std::string bytes) {
  const uint64_t fingerprint = llvm::hash_combine(
      static_cast<int>(PayloadKind::kBytes), StringRef(bytes));
  const uint64_t size = bytes.size();
  const int payload = payloads_.Add(
      fingerprint, size,
      [bytes = std::move(bytes)](ByteSink sink) { return sink(bytes); });
  return AddPayloadBuffer(payload);
}

int ModelBuilder::AddPayloadBuffer(int payload) {
  const int index = buffers_.size();
  buffers_.push_back(tflite::CreateBuffer(fbb_, /*data=*/0,
                                          /*offset=*/kPlaceholder,
                                          /*size=*/kPlaceholder));
  payload_buffers_.emplace_back(index, payload);
  return index;
}

int ModelBuilder::AddInlineBuffer(StringRef bytes) {
  const int index = buffers_.size();
  fbb_.ForceVectorAlignment(bytes.size(), sizeof(uint8_t), kBufferAlignment);
  const auto data = fbb_.CreateVector(
      reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
  buffers_.push_back(tflite::CreateBuffer(fbb_, data));
  return index;
}

Offset<flatbuffers::Vector<Offset<tflite::Metadata>>>
ModelBuilder::BuildMetadata() {
  std::vector<Offset<tflite::Metadata>> metadata;
  auto add = [&](const char* name, StringRef content) {
    const int buffer = AddInlineBuffer(content);
    metadata.push_back(tflite::CreateMetadataDirect(fbb_, name, buffer));
  };
  // Filled by `UpdateMinimumRuntimeVersionForModel`. 16 bytes is the buffer
  // alignment, and more than enough for a version string.
  add(tflite_metadata_min_runtime_version, std::string(16, '\0'));
  // Tells the runtime that buffers are stored after the flatbuffer.
  add(tflite_metadata_buffer_location, "outside flatbuffers");
  return fbb_.CreateVector(metadata);
}

Offset<flatbuffers::Vector<Offset<tflite::SignatureDef>>>
ModelBuilder::BuildSignatureDefs() {
  auto tensor_maps = [&](const std::map<std::string, int32_t>& tensors) {
    std::vector<Offset<tflite::TensorMap>> maps;
    for (const auto& [name, index] : tensors) {
      maps.push_back(tflite::CreateTensorMapDirect(fbb_, name.c_str(), index));
    }
    return fbb_.CreateVector(maps);
  };

  std::vector<Offset<tflite::SignatureDef>> signature_defs;
  for (const EntryPoint& entry : entry_points_) {
    FuncOp function = entry.function;
    llvm::SmallVector<DictionaryAttr> arg_attrs, result_attrs;
    function.getAllArgAttrs(arg_attrs);
    function.getAllResultAttrs(result_attrs);
    const std::vector<std::string> input_names = GetSignatureNames(arg_attrs);
    const std::vector<std::string> output_names =
        GetSignatureNames(result_attrs);
    if (input_names.empty() && output_names.empty()) continue;
    if (input_names.size() != entry.inputs.size() ||
        output_names.size() != entry.outputs.size()) {
      function.emitWarning(
          "signature names don't match the function arguments and results");
      continue;
    }
    auto exported_names =
        function->getAttrOfType<ArrayAttr>(kExportedNamesAttr);
    if (!exported_names || exported_names.empty() ||
        !isa<StringAttr>(exported_names[0])) {
      function.emitWarning("has no exported name for its signature");
      continue;
    }

    // Sorted by name for determinism.
    std::map<std::string, int32_t> inputs, outputs;
    for (auto [name, index] : llvm::zip(input_names, entry.inputs)) {
      inputs[name] = index;
    }
    for (auto [name, index] : llvm::zip(output_names, entry.outputs)) {
      outputs[name] = index;
    }
    const auto inputs_offset = tensor_maps(inputs);
    const auto outputs_offset = tensor_maps(outputs);
    const auto key_offset =
        fbb_.CreateString(cast<StringAttr>(exported_names[0]).getValue().str());
    tflite::SignatureDefBuilder signature_def(fbb_);
    signature_def.add_inputs(inputs_offset);
    signature_def.add_outputs(outputs_offset);
    signature_def.add_signature_key(key_offset);
    signature_def.add_subgraph_index(entry.subgraph);
    signature_defs.push_back(signature_def.Finish());
  }
  return fbb_.CreateVector(signature_defs);
}

absl::Status ModelBuilder::Write(llvm::raw_pwrite_stream& stream) {
  if (fbb_.GetSize() > flatbuffer_size_max) {
    return absl::InvalidArgumentError(
        "The model structure exceeds the 2GB flatbuffer limit.");
  }
  uint8_t* flatbuffer = fbb_.GetBufferPointer();
  const uint64_t flatbuffer_size = fbb_.GetSize();
  // Constant buffers aren't filled yet, so constant inputs look like variable
  // ones to op versioning; no version depends on this as of now.
  tflite::UpdateOpVersion(flatbuffer);
  tflite::UpdateMinimumRuntimeVersionForModel(flatbuffer);

  // Reserve the space of the flatbuffer, which is written once the payload
  // locations are patched in.
  const uint64_t file_start = stream.tell();
  stream.write_zeros(llvm::alignTo(flatbuffer_size, kBufferAlignment));
  if (absl::Status status = payloads_.WriteTo(stream, file_start);
      !status.ok()) {
    return status;
  }

  tflite::Model* model = tflite::GetMutableModel(flatbuffer);
  bool patched = true;
  for (const auto& [buffer_index, payload] : payload_buffers_) {
    const PayloadLocation& location = payloads_.location(payload);
    tflite::Buffer* buffer =
        model->mutable_buffers()->GetMutableObject(buffer_index);
    patched &= buffer->mutate_offset(location.offset);
    patched &= buffer->mutate_size(location.size);
  }
  for (const CustomOptions& options : custom_options_) {
    const PayloadLocation& location = payloads_.location(options.payload);
    tflite::Operator* op = model->mutable_subgraphs()
                               ->GetMutableObject(options.subgraph)
                               ->mutable_operators()
                               ->GetMutableObject(options.op);
    patched &= op->mutate_large_custom_options_offset(location.offset);
    patched &= op->mutate_large_custom_options_size(location.size);
  }
  if (!patched) return absl::InternalError("Failed to patch payload offsets.");

  stream.pwrite(reinterpret_cast<const char*>(flatbuffer), flatbuffer_size,
                file_start);
  return absl::OkStatus();
}

}  // namespace
}  // namespace fb_export

absl::Status ExportFlatbuffer(ModuleOp module,
                              llvm::raw_pwrite_stream& stream) {
  return fb_export::ModelBuilder(module).Export(stream);
}

}  // namespace mlir::TFL
