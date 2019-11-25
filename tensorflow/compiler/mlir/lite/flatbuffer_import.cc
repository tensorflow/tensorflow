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

#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/QuantOps/QuantOps.h"  // TF:local_config_mlir
#include "mlir/Dialect/QuantOps/QuantTypes.h"  // TF:local_config_mlir
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Diagnostics.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/OperationSupport.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/IR/Value.h"  // TF:local_config_mlir
#include "mlir/Support/Functional.h"  // TF:local_config_mlir
#include "mlir/Support/LLVM.h"  // TF:local_config_mlir
#include "mlir/Translation.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/flatbuffer_operator.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_translate.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_translate_flags.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"

using llvm::ArrayRef;
using mlir::Builder;
using mlir::DenseElementsAttr;
using mlir::FuncOp;
using mlir::Location;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OperationState;
using mlir::OwningModuleRef;
using mlir::RankedTensorType;
using mlir::UnrankedTensorType;
using mlir::Value;
using mlir::quant::QuantizedType;
using tflite::TensorT;
using xla::StatusOr;

namespace errors = tensorflow::errors;
namespace tfl = mlir::TFL;

namespace {
bool IsScalar(const TensorT& tensor) {
  // TODO(b/138222071) We can't distinguish scalars and unranked tensors
  // Work out a way to handle this and stub out the code until then
  return tensor.shape.empty() && false;
}

bool IsQuantized(const TensorT& tensor) {
  return (tensor.quantization != nullptr) &&
         !tensor.quantization->zero_point.empty();
}

// Create the MLIR NamedLoc location corresponding to a given tensor
Location TensorLoc(const TensorT& tensor, Builder builder, Location base) {
  if (tensor.name.empty()) {
    return base;
  }
  return mlir::NameLoc::get(builder.getIdentifier(tensor.name), base);
}

// Returns the correct type for a quantized tensor
// We have a special case for constants since they have a higher minimum value.
StatusOr<QuantizedType> GetQuantizedType(const TensorT& tensor, Builder builder,
                                         bool is_constant = false) {
  tflite::QuantizationParametersT& quant_params = *tensor.quantization;
  if (quant_params.details.AsCustomQuantization()) {
    return errors::Unimplemented("Cannot handle experimental quantization");
  }

  bool is_signed = true;
  mlir::IntegerType storage_type;
  if (tensor.type == tflite::TensorType_UINT8) {
    is_signed = false;
    storage_type = builder.getIntegerType(8);
  } else {
    auto raw_elem_type = ConvertElementType(tensor.type, builder);
    if (!raw_elem_type.isa<mlir::IntegerType>()) {
      return errors::InvalidArgument(
          "Quantized tensors must be stored as integers");
    }
    storage_type = raw_elem_type.cast<mlir::IntegerType>();
  }

  // TFlite uses narrow-range [u]int8 for constant buffers of quantized weights.
  // Since we don't know which ones are weights, we represent this optimization
  // as a change in the storage bounds for the type for all constants of this
  // type.
  bool is_weight_buffer = is_constant && (storage_type.getWidth() == 8);

  int64_t storage_min = QuantizedType::getDefaultMinimumForInteger(
                            is_signed, storage_type.getWidth()) +
                        is_weight_buffer;
  int64_t storage_max = QuantizedType::getDefaultMaximumForInteger(
      is_signed, storage_type.getWidth());
  uint32_t flags =
      is_signed ? mlir::quant::QuantizationFlags::FlagValue::Signed : 0;

  // Scale size can't be zero as it is checked before.
  if (quant_params.scale.size() != 1) {
    llvm::SmallVector<double, 4> scales(quant_params.scale.begin(),
                                        quant_params.scale.end());
    return mlir::quant::UniformQuantizedPerAxisType::get(
        flags, storage_type, builder.getF32Type(), scales,
        quant_params.zero_point, quant_params.quantized_dimension, storage_min,
        storage_max);
  }
  return mlir::quant::UniformQuantizedType::get(
      flags, storage_type, builder.getF32Type(), quant_params.scale.at(0),
      quant_params.zero_point.at(0), storage_min, storage_max);
}

// TODO(b/138222071) Remove shapeless_are_scalars once we can reliably
// make that distinction and don't have to rely on context
// (input to main and constants must have static shape)
StatusOr<mlir::TensorType> GetTensorType(const TensorT& tensor, Builder builder,
                                         bool shapeless_are_scalars = false,
                                         bool is_constant = false) {
  mlir::Type elem_type = ConvertElementType(tensor.type, builder);
  // TODO(b/139554398) Store min/max (even for non-quantized tensors) somewhere
  // if it's set
  if (IsQuantized(tensor)) {
    TF_ASSIGN_OR_RETURN(elem_type,
                        GetQuantizedType(tensor, builder, is_constant));
  }

  if (IsScalar(tensor) || (shapeless_are_scalars && tensor.shape.empty())) {
    return RankedTensorType::get({}, elem_type);
  }

  if (!tensor.shape.empty()) {
    llvm::SmallVector<int64_t, 4> shape(tensor.shape.begin(),
                                        tensor.shape.end());
    return RankedTensorType::get(shape, elem_type);
  }

  return UnrankedTensorType::get(elem_type);
}

// Extract the min max information in the tensor and create the quant stats op.
// If the input `tensor` has scale/zero_point, `res` should have quantized
// type, thus none stats op is required and nullptr is retruned.
// If the min max information is invalid, nullptr is returned.
mlir::Operation* ConvertMinMaxToStatsOp(const TensorT& tensor, OpBuilder b,
                                        Value* res) {
  // If the `tensor` has scale/zero_point, it must have been quantized, then the
  // min/max stats is just for comments, so ignore it.
  if (!tensor.quantization || IsQuantized(tensor)) return nullptr;
  // If the result isn't float and unquantizable, the min/max is ignored.
  if (!res->getType()
           .cast<mlir::ShapedType>()
           .getElementType()
           .isa<mlir::FloatType>()) {
    return nullptr;
  }
  auto mins = tensor.quantization->min;
  auto maxs = tensor.quantization->max;
  if (mins.size() != maxs.size() || mins.empty()) return nullptr;

  llvm::SmallVector<llvm::APFloat, 4> min_maxs;
  min_maxs.reserve(mins.size() * 2);
  for (int i = 0; i < mins.size(); ++i) {
    llvm::APFloat min(mins[i]);
    llvm::APFloat max(maxs[i]);
    min_maxs.push_back(min);
    min_maxs.push_back(max);
  }
  // The layer stats contain only the first min/max pairs.
  mlir::ElementsAttr layer_stats = mlir::DenseFPElementsAttr::get(
      mlir::RankedTensorType::get({2}, b.getF32Type()),
      {min_maxs[0], min_maxs[1]});
  mlir::ElementsAttr axis_stats;
  mlir::IntegerAttr axis;
  if (mins.size() > 1) {
    llvm::SmallVector<int64_t, 4> axis_stats_shape{
        static_cast<int64_t>(mins.size()), 2};
    axis_stats = mlir::DenseFPElementsAttr::get(
        mlir::RankedTensorType::get(axis_stats_shape, b.getF32Type()),
        min_maxs);
    // TODO(fengliuai): this quantization dimension isn't correct.
    axis = b.getI64IntegerAttr(tensor.quantization->quantized_dimension);
  }
  return b.create<mlir::quant::StatisticsOp>(b.getUnknownLoc(), res,
                                             layer_stats, axis_stats, axis);
}

StatusOr<std::string> OpNameForOpCode(const tflite::OperatorCodeT opcode) {
  // TODO(krzysd) Support custom ops
  if (opcode.builtin_code == tflite::BuiltinOperator_CUSTOM) {
    return errors::Unimplemented("unsupported custom operation: ",
                                 opcode.custom_code);
  }
  if (opcode.builtin_code == tflite::BuiltinOperator_IF) {
    return std::string("tf.If");
  }
  if (opcode.builtin_code == tflite::BuiltinOperator_WHILE) {
    return std::string("tf.While");
  }

  const char* op_name = tflite::EnumNameBuiltinOperator(opcode.builtin_code);
  std::string lowered_name = llvm::StringRef(op_name).lower();
  return llvm::Twine("tfl.", lowered_name).str();
}

// The buffers in TFLite flatbuffers have their contents stored as a vector of
// bytes that represent little-endian values.
// The read_size parameter is present to allow reading both float16 and float32s
// without a case split.
template <typename T>
std::vector<T> ReadAsLittleEndian(ArrayRef<uint8_t> bytes) {
  std::vector<T> ret;
  size_t read_size = sizeof(T);
  int bytes_len = bytes.size();
  assert(bytes_len % read_size == 0);

  size_t elem_count = bytes_len / read_size;
  ret.reserve(elem_count);

  const char* data_ptr = reinterpret_cast<const char*>(bytes.data());
  for (int i = 0; i < elem_count; i++) {
    ret.push_back(
        llvm::support::endian::readNext<T, llvm::support::little,
                                        llvm::support::unaligned>(data_ptr));
  }
  return ret;
}

tensorflow::TensorProto ConvertTfliteConstTensor(
    const tflite::TensorT& tensor, const std::vector<uint8_t>& buffer) {
  tensorflow::TensorProto ret;
  ret.set_dtype(TflTypeToTfType(tensor.type));

  tensorflow::TensorShapeProto* shape = ret.mutable_tensor_shape();
  shape->set_unknown_rank(false);
  for (auto dim : tensor.shape) {
    shape->add_dim()->set_size(int64_t{dim});
  }
  std::string content;
  content.assign(reinterpret_cast<const char*>(buffer.data()), buffer.size());
  ret.set_tensor_content(content);
  return ret;
}

StatusOr<mlir::ElementsAttr> ConvertFloatBuffer(
    mlir::RankedTensorType shaped_type, mlir::FloatType elem_type,
    const std::vector<uint8_t>& buffer) {
  size_t bytes_len = buffer.size();

  // The bytes of floats are stored little-endian.
  switch (elem_type.getWidth()) {
    case 16: {
      assert(bytes_len % 2 == 0);
      size_t elem_count = bytes_len / 2;
      std::vector<llvm::APFloat> values;
      values.reserve(elem_count);

      const char* data = reinterpret_cast<const char*>(buffer.data());
      auto& semantics = elem_type.getFloatSemantics();

      for (int i = 0; i < elem_count; i++) {
        uint16_t bit_repr =
            llvm::support::endian::readNext<uint16_t, llvm::support::little,
                                            llvm::support::unaligned>(data);
        llvm::APInt int_repr(16, bit_repr);
        values.emplace_back(semantics, int_repr);
      }

      return DenseElementsAttr::get(shaped_type, values);
    }
    case 32: {
      assert(bytes_len % 4 == 0);
      size_t elem_count = bytes_len / 4;
      std::vector<float> values;
      values.reserve(elem_count);

      const char* data = reinterpret_cast<const char*>(buffer.data());

      for (int i = 0; i < elem_count; i++) {
        uint32_t bit_repr =
            llvm::support::endian::readNext<uint32_t, llvm::support::little,
                                            llvm::support::unaligned>(data);
        values.push_back(absl::bit_cast<float>(bit_repr));
      }
      return DenseElementsAttr::get(shaped_type, ArrayRef<float>(values));
    }
  }
  return errors::InvalidArgument("unsupported bit width", elem_type.getWidth());
}

StatusOr<mlir::ElementsAttr> ConvertIntBuffer(
    mlir::RankedTensorType shaped_type, mlir::Type elem_type,
    const std::vector<uint8_t>& buffer) {
  unsigned bit_width;
  mlir::RankedTensorType buffer_type;
  if (auto itype = elem_type.dyn_cast<mlir::IntegerType>()) {
    bit_width = itype.getWidth();
  } else if (auto qtype = elem_type.dyn_cast<QuantizedType>()) {
    bit_width = qtype.getStorageTypeIntegralWidth();
    shaped_type = mlir::RankedTensorType::get(shaped_type.getShape(),
                                              qtype.getStorageType());
  } else {
    return errors::InvalidArgument("unsupported integer constant type");
  }

  switch (bit_width) {
    case 1: {
      // vector<bool> doesn't convert to an ArrayRef
      llvm::SmallVector<bool, 8> values;
      values.reserve(buffer.size());
      for (auto b : buffer) {
        values.emplace_back(b != 0);
      }
      return DenseElementsAttr::get(shaped_type, ArrayRef<bool>(values));
    }
    case 8: {
      return DenseElementsAttr::get(shaped_type, ArrayRef<uint8_t>(buffer));
    }
    case 16: {
      auto values = ReadAsLittleEndian<uint16_t>(buffer);
      return DenseElementsAttr::get(shaped_type, ArrayRef<uint16_t>(values));
    }
    case 32: {
      auto values = ReadAsLittleEndian<uint32_t>(buffer);
      return DenseElementsAttr::get(shaped_type, ArrayRef<uint32_t>(values));
    }
    case 64: {
      auto values = ReadAsLittleEndian<uint64_t>(buffer);
      return DenseElementsAttr::get(shaped_type, ArrayRef<uint64_t>(values));
    }
    default:
      return errors::Unimplemented("Cannot handle bit width ", bit_width);
  }
}

StatusOr<Operation*> BuildConstOp(const tflite::TensorT& tensor,
                                  const std::vector<uint8_t>& buffer,
                                  OpBuilder builder, Location loc) {
  TF_ASSIGN_OR_RETURN(auto type, GetTensorType(tensor, builder,
                                               /*shapeless_are_scalars=*/true,
                                               /*is_constant=*/true));
  auto shaped_type = type.dyn_cast<mlir::RankedTensorType>();
  if (!shaped_type) {
    return errors::Internal("Constant doesn't have a shape");
  }

  auto elem_type = shaped_type.getElementType();

  mlir::ElementsAttr value;
  if (auto float_type = elem_type.dyn_cast<mlir::FloatType>()) {
    TF_ASSIGN_OR_RETURN(value,
                        ConvertFloatBuffer(shaped_type, float_type, buffer));
  } else if (elem_type.isa<mlir::IntegerType>() ||
             elem_type.isa<QuantizedType>()) {
    TF_ASSIGN_OR_RETURN(value,
                        ConvertIntBuffer(shaped_type, elem_type, buffer));
  } else if (elem_type.isa<mlir::ComplexType>() ||
             elem_type.isa<mlir::TF::TensorFlowType>()) {
    auto dialect = elem_type.getContext()->getRegisteredDialect("tf");
    tensorflow::TensorProto repr = ConvertTfliteConstTensor(tensor, buffer);
    std::string mangled = tensorflow::mangling_util::MangleTensor(repr);

    value = mlir::OpaqueElementsAttr::get(dialect, shaped_type, mangled);
  } else {
    return errors::Unimplemented("Constant of unsupported type");
  }

  if (IsQuantized(tensor)) {
    auto op = builder.create<tfl::QConstOp>(
        loc, mlir::TypeAttr::get(shaped_type), value);
    return op.getOperation();
  }
  auto op = builder.create<tfl::ConstOp>(loc, value);
  return op.getOperation();
}

llvm::SmallVector<mlir::NamedAttribute, 4> ConvertSubgraphIdxsToFunctionAttrs(
    tflite::BuiltinOptionsUnion options,
    const std::vector<std::string>& func_names, Builder builder) {
  if (auto* opts = options.AsIfOptions()) {
    uint32_t then_idx = opts->then_subgraph_index;
    auto then_attr = builder.getSymbolRefAttr(func_names.at(then_idx));
    uint32_t else_idx = opts->else_subgraph_index;
    auto else_attr = builder.getSymbolRefAttr(func_names.at(else_idx));

    return {builder.getNamedAttr("then_branch", then_attr),
            builder.getNamedAttr("else_branch", else_attr),
            // TODO(b/139667752): Analyze statelessness correctly
            builder.getNamedAttr("is_stateless", builder.getBoolAttr(false))};
  }
  if (auto* opts = options.AsWhileOptions()) {
    uint32_t cond_idx = opts->cond_subgraph_index;
    auto cond_attr = builder.getSymbolRefAttr(func_names.at(cond_idx));
    uint32_t body_idx = opts->body_subgraph_index;
    auto body_attr = builder.getSymbolRefAttr(func_names.at(body_idx));

    return {builder.getNamedAttr("cond", cond_attr),
            builder.getNamedAttr("body", body_attr),
            // TODO(b/139667752): Analyze statelessness correctly
            builder.getNamedAttr("is_stateless", builder.getBoolAttr(false))};
  }
  return {};
}

// Returns true if this is a basic LSTM op.
bool IsBasicLSTMOp(tflite::BuiltinOptionsUnion op_union) {
  if (const auto* op = op_union.AsLSTMOptions()) {
    return op->kernel_type == tflite::LSTMKernelType_BASIC;
  } else {
    return false;
  }
}

// TODO(krzysd) Handle function calls
StatusOr<Operation*> ConvertOp(
    const tflite::OperatorT& op, const std::vector<Value*>& vals_map,
    Value* optional_arg_marker, const std::vector<std::string>& op_names,
    const std::vector<std::string>& func_names,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tensors, Location loc,
    OpBuilder builder) {
  llvm::SmallVector<Value*, 4> operands;
  llvm::SmallVector<mlir::Type, 2> outputTypes;

  if (op.outputs.empty()) {
    auto err = errors::InvalidArgument("operator with no outputs");
    return emitError(loc, err.ToString()), err;
  }

  const bool is_basic_lstm = IsBasicLSTMOp(op.builtin_options);
  const std::string& op_name =
      is_basic_lstm ? "tfl.basic_lstm" : op_names.at(op.opcode_index);
  OperationState op_state(loc, op_name);

  for (auto input_num : op.inputs) {
    if (input_num == -1) {
      assert(optional_arg_marker != nullptr);
      op_state.addOperands({optional_arg_marker});
    } else {
      op_state.addOperands({vals_map.at(input_num)});
    }
  }

  for (auto output_num : op.outputs) {
    auto& tensor = *tensors.at(output_num);
    auto type_or_err = GetTensorType(tensor, builder);
    if (!type_or_err.ok()) {
      return emitError(loc, type_or_err.status().ToString()),
             type_or_err.status();
    }
    auto type = type_or_err.ConsumeValueOrDie();

    if (op_name == "tfl.quantize") {
      // Special case for quantize: return type must also be in qtype attribute
      op_state.addAttribute("qtype", mlir::TypeAttr::get(type));
    } else if (op_name == "tfl.reshape" && type.hasStaticShape() &&
               op_state.operands.size() == 1) {
      // Special case for reshape: the second op is optional in the old
      // converter and kernel, so we create the second operand, which is
      // required by the new converter, from the result shape.
      auto shape_type =
          RankedTensorType::get({type.getRank()}, builder.getIntegerType(32));
      mlir::SmallVector<mlir::Attribute, 4> shape;
      shape.reserve(type.getRank());
      for (auto s : type.getShape()) {
        shape.push_back(builder.getI32IntegerAttr(static_cast<int32_t>(s)));
      }
      auto output_shape = DenseElementsAttr::get(shape_type, shape);
      auto shape_op = builder.create<tfl::ConstOp>(loc, output_shape);
      op_state.addOperands({shape_op});
    }

    op_state.addTypes({type});
  }

  llvm::SmallVector<mlir::NamedAttribute, 2> attrs;
  mlir::BuiltinOptionsToAttributes(op.builtin_options, builder, attrs);
  op_state.addAttributes(attrs);

  // Handle the conversion from subgraph index to functions for If and While
  auto function_ref_attrs = ConvertSubgraphIdxsToFunctionAttrs(
      op.builtin_options, func_names, builder);
  op_state.addAttributes(function_ref_attrs);

  return builder.createOperation(op_state);
}

// Returns the output tensor indices for the given subgraph. If
// ordered_output_arrays is provided, then return the tensor indices in
// ordered_output_arrays.
StatusOr<llvm::SmallVector<int32_t, 4>> GetOutputTensorIndices(
    const tflite::SubGraphT& subgraph, Location base_loc,
    const std::vector<std::string>& ordered_output_arrays) {
  if (ordered_output_arrays.empty()) {
    return llvm::SmallVector<int32_t, 4>(subgraph.outputs.begin(),
                                         subgraph.outputs.end());
  }

  llvm::SmallVector<int32_t, 4> outputs;
  outputs.resize(ordered_output_arrays.size());
  absl::flat_hash_map<std::string, int> output_order_map;
  for (auto output : llvm::enumerate(ordered_output_arrays)) {
    output_order_map[output.value()] = output.index();
  }

  int tensor_index = 0;
  int found_output_tensors = 0;
  for (const auto& tensor : subgraph.tensors) {
    auto found = output_order_map.find(tensor->name);
    if (found != output_order_map.end()) {
      const int output_index = found->second;
      outputs[output_index] = tensor_index;
      ++found_output_tensors;
    }
    ++tensor_index;
  }

  if (found_output_tensors != ordered_output_arrays.size()) {
    auto err = errors::InvalidArgument(
        "cannot find all nodes in ordered_output_arrays");
    return emitError(base_loc, err.ToString()), err;
  }

  return outputs;
}

// Given a list of tensor indices, returns a string of concatenated tensor names
// wrapped in a NamedAttribute.
template <typename ContainerType>
mlir::NamedAttribute BuildTFEntryFunctionAttribute(
    const tflite::SubGraphT& subgraph, Builder* builder, const std::string name,
    const ContainerType indices) {
  llvm::SmallVector<std::string, 8> tensor_names = mlir::functional::map(
      [&](int i) { return subgraph.tensors.at(i)->name; }, indices);
  return builder->getNamedAttr(
      name, builder->getStringAttr(llvm::join(tensor_names, ",")));
}

// Build a FuncOp from a tflite SubGraph
// The op_names are a mapping from indexes into the TFLite operators array to
// the operator name MLIR expects (tfl.foo_op). The buffers are directly taken
// from the deserialized flatbuffer as we do not have the type information to
// interpret them until this point. The base_loc parameter is the location of
// the flatbuffer as a whole (usually a file). The is_entry_point flag
// controls whether shapeless types are treated as scalars. If
// ordered_output_arrays is not empty, then the imported mlir function will only
// return nodes in ordered_output_arrays in the same order.
StatusOr<FuncOp> ConvertSubgraph(
    const tflite::SubGraphT& subgraph, llvm::StringRef name,
    const std::vector<std::string>& op_names,
    const std::vector<std::string>& func_names,
    const std::vector<std::unique_ptr<tflite::BufferT>>& buffers,
    Location base_loc, Builder builder,
    const std::vector<std::string>& ordered_output_arrays,
    bool is_entry_point) {
  llvm::SmallVector<mlir::Type, 2> ret_types;
  llvm::SmallVector<mlir::Type, 4> input_types;

  auto func_loc = mlir::NameLoc::get(builder.getIdentifier(name), base_loc);

  // Construct function type
  for (auto input : subgraph.inputs) {
    auto& tensor = *subgraph.tensors.at(input);
    // TODO(b/138222071) Graph inputs must have static shape per the exporter,
    // but we cannot differentiate scalars from unranked tensors.
    // Here we reverse the default assumption that shape = [] means unranked.
    // when processing main()
    auto type_or_err = GetTensorType(tensor, builder,
                                     /*shapeless_are_scalars=*/is_entry_point,
                                     /*is_constant=*/false);
    if (!type_or_err.ok()) {
      emitError(func_loc, "error reading argument types")
          << type_or_err.status().ToString();
      return type_or_err.status();
    }
    auto type = type_or_err.ConsumeValueOrDie();
    input_types.push_back(type);
  }

  llvm::SmallVector<bool, 16> is_op_output(subgraph.tensors.size(), false);
  for (auto& op : subgraph.operators) {
    for (auto output : op->outputs) {
      is_op_output[output] = true;
    }
  }

  TF_ASSIGN_OR_RETURN(
      auto func_outputs,
      GetOutputTensorIndices(subgraph, base_loc, ordered_output_arrays));

  for (auto output : func_outputs) {
    bool is_constant = !is_op_output[output];
    auto type_or_err = GetTensorType(*subgraph.tensors.at(output), builder,
                                     /*shapeless_are_scalars=*/is_constant,
                                     /*is_constant=*/is_constant);
    if (!type_or_err.ok()) {
      emitError(func_loc, "error reading return types")
          << type_or_err.status().ToString();
      return type_or_err.status();
    }
    auto type = type_or_err.ConsumeValueOrDie();
    ret_types.push_back(type);
  }
  auto func_type = builder.getFunctionType(input_types, ret_types);

  // Construct function object
  auto func = FuncOp::create(func_loc, name, func_type, /* attrs= */ {});
  func.addEntryBlock();
  auto& body = func.getBody();
  OpBuilder op_builder{body};

  std::vector<Value*> vals_map(subgraph.tensors.size(), nullptr);
  Value* maybe_optional_arg_marker = nullptr;

  // Get or construct MLIR values for each input
  for (int i = 0, e = subgraph.inputs.size(); i < e; i++) {
    auto input_tensor = subgraph.inputs[i];
    const auto& tensor = *subgraph.tensors.at(input_tensor);
    auto loc = TensorLoc(tensor, builder, base_loc);
    if (nullptr != vals_map[input_tensor]) {
      auto err = errors::FailedPrecondition("duplicate input arguments");
      return emitError(loc, err.ToString()), err;
    }
    Value* input_value = func.getArgument(i);

    // If the `tensor` has min/max and doesn't have scale/zero_point
    // information, a stats op is created to use the input_value, then the
    // `tensor` should be mapped to the result of this new stats op.
    if (auto stats_op =
            ConvertMinMaxToStatsOp(tensor, op_builder, input_value)) {
      vals_map[input_tensor] = stats_op->getResult(0);
    } else {
      vals_map[input_tensor] = input_value;
    }
  }

  // Set tf.entry_function attribute
  if (is_entry_point) {
    llvm::SmallVector<mlir::NamedAttribute, 2> attributes;
    if (!subgraph.inputs.empty()) {
      attributes.push_back(BuildTFEntryFunctionAttribute(
          subgraph, &builder, "inputs", subgraph.inputs));
    }
    if (!func_outputs.empty()) {
      attributes.push_back(BuildTFEntryFunctionAttribute(
          subgraph, &builder, "outputs", func_outputs));
    }
    func.setAttr("tf.entry_function", builder.getDictionaryAttr(attributes));
  }

  // Construct MLIR operators from TFLite operators
  for (auto& op : subgraph.operators) {
    for (auto input_num : op->inputs) {
      // The operators in a graph are topologically sorted
      // and so if no previous operation has produced a tensor
      // it must be a constant.
      if (input_num == -1) {
        if (maybe_optional_arg_marker == nullptr) {
          maybe_optional_arg_marker =
              op_builder
                  .create<mlir::ConstantOp>(base_loc, builder.getNoneType(),
                                            builder.getUnitAttr())
                  .getResult();
        }
      } else if (nullptr == vals_map.at(input_num)) {
        auto& const_tensor = *subgraph.tensors[input_num];
        auto const_loc = TensorLoc(const_tensor, builder, base_loc);
        auto op_or_err =
            BuildConstOp(const_tensor, buffers[const_tensor.buffer]->data,
                         op_builder, const_loc);
        if (!op_or_err.ok()) {
          return emitError(const_loc, op_or_err.status().ToString()),
                 op_or_err.status();
        }
        vals_map[input_num] = op_or_err.ValueOrDie()->getResult(0);
      }
    }

    // The NameLoc corresponding to the name of the first output tensor
    auto op_loc =
        op->outputs.empty()
            ? base_loc
            : TensorLoc(*subgraph.tensors[op->outputs[0]], builder, base_loc);
    // If there's an optional argument, maybe_optional_arg_marker has been set
    // to a valid Value*
    TF_ASSIGN_OR_RETURN(
        auto* mlir_op,
        ConvertOp(*op, vals_map, maybe_optional_arg_marker, op_names,
                  func_names, subgraph.tensors, op_loc, op_builder));

    // Add the results to the value maps. There are two cases: 1. the result
    // tensor does not have min/max values, the original op result is used
    // directly; 2. the result tensor has some min/max values, a stats op is
    // created, then the result of the stats op is used.
    for (auto pair : llvm::enumerate(mlir_op->getResults())) {
      int output_tensor_index = op->outputs[pair.index()];
      auto& tensor = *subgraph.tensors[output_tensor_index];
      if (auto stats_op =
              ConvertMinMaxToStatsOp(tensor, op_builder, pair.value())) {
        vals_map[output_tensor_index] = stats_op->getResult(0);
      } else {
        vals_map[output_tensor_index] = pair.value();
      }
    }
  }

  // Construct return values
  llvm::SmallVector<Value*, 4> return_operands;
  for (auto index : func_outputs) {
    if (nullptr == vals_map.at(index)) {
      auto& const_tensor = *subgraph.tensors[index];
      auto const_loc = TensorLoc(const_tensor, builder, base_loc);
      auto op_or_err =
          BuildConstOp(const_tensor, buffers[const_tensor.buffer]->data,
                       op_builder, const_loc);
      if (!op_or_err.ok()) {
        return emitError(const_loc, op_or_err.status().ToString()),
               op_or_err.status();
      }
      vals_map[index] = op_or_err.ValueOrDie()->getResult(0);
    }
    return_operands.push_back(vals_map[index]);
  }

  op_builder.create<mlir::ReturnOp>(base_loc, return_operands);

  return func;
}

// TFLite subgraphs do not necessarily have names, though MLIR functions must
// have them, so we generate a name for subgraphs that are missing one here.
// Note: in TFLite, the first subgraph is the entry point, and in MLIR that
// represents TFLite, this entry point must be called "main"
// TODO(b/131175224,b/132239787) Support multiple entry points
std::string SubgraphName(unsigned index, const tflite::SubGraphT& subgraph) {
  if (subgraph.name.empty()) {
    if (index == 0) {
      return "main";
    } else {
      return llvm::formatv("fn_{0}", index).str();
    }
  } else {
    return subgraph.name;
  }
}
}  // namespace

OwningModuleRef tflite::FlatBufferToMlir(
    absl::string_view buffer, MLIRContext* context, Location base_loc,
    const std::vector<std::string>& ordered_output_arrays) {
  auto model_ptr =
      FlatBufferModel::VerifyAndBuildFromBuffer(buffer.data(), buffer.length());
  if (nullptr == model_ptr) {
    return emitError(base_loc, "couldn't parse flatbuffer"), nullptr;
  }

  std::unique_ptr<ModelT> model(model_ptr->GetModel()->UnPack());

  auto builder = Builder(context);

  std::vector<std::string> operator_names;
  operator_names.reserve(model->operator_codes.size());

  for (auto& opcode : model->operator_codes) {
    auto operator_name_or_error = OpNameForOpCode(*opcode);
    if (!operator_name_or_error.ok()) {
      return emitError(base_loc, operator_name_or_error.status().ToString()),
             nullptr;
    }
    operator_names.push_back(operator_name_or_error.ConsumeValueOrDie());
  }

  std::vector<std::string> func_names;
  for (auto& subgraph : model->subgraphs) {
    func_names.push_back(subgraph->name);
  }

  auto module = mlir::ModuleOp::create(base_loc);
  // We currently don't use this to make decisions, but we could
  // use it in exports or if there are breaking changes
  module.setAttr("tfl.schema_version",
                 builder.getI32IntegerAttr(model->version));
  if (!model->description.empty()) {
    module.setAttr("tfl.description",
                   builder.getStringAttr(model->description));
  }

  if (!ordered_output_arrays.empty() && model->subgraphs.size() > 1) {
    // TODO(b/141485522): support more than one subgraph.
    return emitError(base_loc,
                     "ordered_output_arrays does not support more than one "
                     "subgraph yet"),
           nullptr;
  }

  for (auto e : llvm::enumerate(model->subgraphs)) {
    auto& subgraph = e.value();
    std::string name = SubgraphName(e.index(), *subgraph);
    auto func_or_error = ConvertSubgraph(
        *subgraph, name, operator_names, func_names, model->buffers, base_loc,
        // Only the entry point needs pseudo_input_ops
        // TODO(b/131175224,b/132239787) Support multiple entry points
        builder, ordered_output_arrays,
        /*is_entry_point=*/e.index() == 0);
    if (!func_or_error.ok()) {
      return emitError(base_loc, "could not translate function ")
                 << subgraph->name,
             nullptr;
    }
    module.push_back(func_or_error.ConsumeValueOrDie());
  }
  // TFLite subgraphs do not necessarily have names,

  return OwningModuleRef(module);
}

static OwningModuleRef FlatBufferFileToMlirTrans(llvm::SourceMgr* source_mgr,
                                                 MLIRContext* context) {
  const llvm::MemoryBuffer* input =
      source_mgr->getMemoryBuffer(source_mgr->getMainFileID());
  std::string error;
  auto loc =
      mlir::FileLineColLoc::get(input->getBufferIdentifier(), 0, 0, context);

  // Parses output_arrays_order from command line option.
  absl::flat_hash_set<std::string> output_set;
  std::vector<std::string> output_arrays_order;
  if (!tensorflow::ParseOutputArrayInfo(output_arrays_string, &output_set,
                                        &output_arrays_order)
           .ok()) {
    return emitError(loc, "parsing output array info failed ")
               << output_arrays_string,
           nullptr;
  }

  return tflite::FlatBufferToMlir(
      absl::string_view(input->getBufferStart(), input->getBufferSize()),
      context, loc, output_arrays_order);
}

static mlir::TranslateToMLIRRegistration FlatBufferFileToMlirTransReg(
    "tflite-flatbuffer-to-mlir",
    [](llvm::SourceMgr& source_mgr, MLIRContext* context) {
      return FlatBufferFileToMlirTrans(&source_mgr, context);
    });
