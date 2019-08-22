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
#include "absl/strings/string_view.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Diagnostics.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/OperationSupport.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/IR/Value.h"  // TF:local_config_mlir
#include "mlir/Support/FileUtilities.h"  // TF:local_config_mlir
#include "mlir/Translation.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
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
using mlir::Value;
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

// Create the MLIR NamedLoc location corresponding to a given tensor
Location TensorLoc(const TensorT& tensor, Builder builder, Location base) {
  if (tensor.name.empty()) {
    return base;
  }
  return mlir::NameLoc::get(builder.getIdentifier(tensor.name), base,
                            builder.getContext());
}

mlir::TensorType GetTensorType(const TensorT& tensor, Builder builder) {
  auto elem_type = ConvertElementType(tensor.type, builder);
  if (IsScalar(tensor)) {
    return builder.getTensorType({}, elem_type);
  }

  if (!tensor.shape.empty()) {
    llvm::SmallVector<int64_t, 4> shape(tensor.shape.begin(),
                                        tensor.shape.end());
    return builder.getTensorType(shape, elem_type);
  }

  return builder.getTensorType(elem_type);
}

StatusOr<std::string> OpNameForOpCode(const tflite::OperatorCodeT opcode) {
  // TODO(krzysd) Support "if" and "while" ops
  if (opcode.builtin_code == tflite::BuiltinOperator_CUSTOM) {
    return errors::Unimplemented("unsupported custom operation: ",
                                 opcode.custom_code);
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
    mlir::RankedTensorType shaped_type, mlir::IntegerType elem_type,
    const std::vector<uint8_t>& buffer) {
  switch (elem_type.getWidth()) {
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
      return errors::Unimplemented("Cannot handle bit width",
                                   elem_type.getIntOrFloatBitWidth());
  }
}

StatusOr<tfl::ConstOp> BuildConstOp(const tflite::TensorT& tensor,
                                    const std::vector<uint8_t>& buffer,
                                    OpBuilder builder, Location loc) {
  mlir::TensorType type;
  if (tensor.shape.empty()) {
    // TODO(b/138222071) Scalar constants get typed as unranked tensors,
    // so we have to manually set their shape here
    auto elem_type = ConvertElementType(tensor.type, builder);
    type = builder.getTensorType({}, elem_type);
  } else {
    type = GetTensorType(tensor, builder);
  }

  auto shaped_type = type.dyn_cast<mlir::RankedTensorType>();
  if (!shaped_type) {
    return errors::Internal("Constant doesn't have a shape");
  }

  auto elem_type = shaped_type.getElementType();

  mlir::ElementsAttr value;
  if (auto float_type = elem_type.dyn_cast<mlir::FloatType>()) {
    TF_ASSIGN_OR_RETURN(value,
                        ConvertFloatBuffer(shaped_type, float_type, buffer));
  } else if (auto int_type = elem_type.dyn_cast<mlir::IntegerType>()) {
    TF_ASSIGN_OR_RETURN(value, ConvertIntBuffer(shaped_type, int_type, buffer));
  } else if (elem_type.isa<mlir::TF::TensorFlowType>()) {
    auto& dialect = elem_type.getDialect();
    tensorflow::TensorProto repr = ConvertTfliteConstTensor(tensor, buffer);
    std::string mangled = tensorflow::mangling_util::MangleTensor(repr);

    value = builder.getOpaqueElementsAttr(&dialect, shaped_type, mangled);
  } else {
    return errors::Unimplemented("Constant of unsupported type");
  }
  return builder.create<tfl::ConstOp>(loc, value);
}

// TODO(krzysd) Handle function calls
StatusOr<Operation*> ConvertOp(
    const tflite::OperatorT& op, const std::vector<Value*> vals_map,
    const std::vector<std::string>& op_names,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tensors, Location loc,
    OpBuilder builder) {
  llvm::SmallVector<Value*, 4> operands;
  llvm::SmallVector<mlir::Type, 2> outputTypes;

  if (op.outputs.empty()) {
    auto err = errors::InvalidArgument("operator with no outputs");
    return emitError(loc, err.ToString()), err;
  }

  const std::string& op_name = op_names[op.opcode_index];
  OperationState op_state(loc, op_name);

  for (auto input_num : op.inputs) {
    op_state.addOperands({vals_map[input_num]});
  }

  for (auto output_num : op.outputs) {
    auto& tensor = *tensors[output_num];
    mlir::TensorType type = GetTensorType(tensor, builder);
    // Special case for reshape, which stores its return shape in an option
    // that we need to extract from
    if (auto* opts = op.builtin_options.AsReshapeOptions()) {
      llvm::SmallVector<int64_t, 4> shape(opts->new_shape.begin(),
                                          opts->new_shape.end());
      type = builder.getTensorType(ArrayRef<int64_t>(shape),
                                   type.getElementType());
    }
    op_state.addTypes({type});
  }

  // TODO(krzysd) Handle attributes correctly
  op_state.addAttribute("fused_activation_function",
                        builder.getStringAttr("NONE"));
  return builder.createOperation(op_state);
}

// Build a FuncOp from a tflite SubGraph
// The op_names are a mapping from indexes into the TFLite operators array to
// the operator name MLIR expects (tfl.foo_op). The buffers are directly taken
// from the deserialized flatbuffer as we do not have the type information to
// interpret them until this point. The base_loc parameter is the location of
// the flatbuffer as a whole (usually a file). The add_pseudo_input_ops flag
// controls whether we create the dummy ops for input that the TFLite dialect
// has in the main function (and only the main function).
StatusOr<FuncOp> ConvertSubgraph(
    const tflite::SubGraphT& subgraph, llvm::StringRef name,
    const std::vector<std::string>& op_names,
    const std::vector<std::unique_ptr<tflite::BufferT>>& buffers,
    Location base_loc, Builder builder, bool add_pseudo_input_ops = false) {
  llvm::SmallVector<mlir::Type, 2> ret_types;
  llvm::SmallVector<mlir::Type, 4> input_types;

  // Construct function type
  for (auto input : subgraph.inputs) {
    input_types.push_back(GetTensorType(*subgraph.tensors[input], builder));
  }
  for (auto output : subgraph.outputs) {
    ret_types.push_back(GetTensorType(*subgraph.tensors[output], builder));
  }
  auto func_type = builder.getFunctionType(input_types, ret_types);

  // Construct function object
  auto func_loc = mlir::NameLoc::get(builder.getIdentifier(name), base_loc,
                                     builder.getContext());

  auto func = FuncOp::create(func_loc, name, func_type, /* attrs= */ {});
  func.addEntryBlock();
  auto& body = func.getBody();
  OpBuilder op_builder{body};

  std::vector<Value*> vals_map(subgraph.tensors.size(), nullptr);

  // Get or construct MLIR values for each input
  for (int i = 0, e = subgraph.inputs.size(); i < e; i++) {
    auto input_tensor = subgraph.inputs[i];
    const auto& tensor = *subgraph.tensors[input_tensor];
    auto loc = TensorLoc(tensor, builder, base_loc);
    if (nullptr != vals_map[input_tensor]) {
      auto err = errors::FailedPrecondition("duplicate input arguments");
      return emitError(loc, err.ToString()), err;
    }
    if (add_pseudo_input_ops) {
      auto* input = func.getArgument(i);
      auto op = op_builder.create<tfl::InputOp>(loc, input);
      vals_map[input_tensor] = op.output();
    } else {
      vals_map[input_tensor] = func.getArgument(i);
    }
  }

  // Construct MLIR operators from TFLite operators
  for (auto& op : subgraph.operators) {
    for (auto input_num : op->inputs) {
      // The operators in a graph are topologically sorted
      // and so if no previous operation has produced a tensor
      // it must be a constant.
      if (nullptr == vals_map[input_num]) {
        auto& const_tensor = *subgraph.tensors[input_num];
        auto const_loc = TensorLoc(const_tensor, builder, base_loc);
        auto op_or_err =
            BuildConstOp(const_tensor, buffers[const_tensor.buffer]->data,
                         op_builder, const_loc);
        if (!op_or_err.ok()) {
          return emitError(const_loc, op_or_err.status().ToString()),
                 op_or_err.status();
        }
        vals_map[input_num] = op_or_err.ValueOrDie().output();
      }
    }

    // The NameLoc corresponding to the name of the first output tensor
    auto op_loc =
        op->outputs.empty()
            ? base_loc
            : TensorLoc(*subgraph.tensors[op->outputs[0]], builder, base_loc);
    TF_ASSIGN_OR_RETURN(auto* mlir_op,
                        ConvertOp(*op, vals_map, op_names, subgraph.tensors,
                                  op_loc, op_builder));
    for (auto pair : llvm::enumerate(mlir_op->getResults())) {
      vals_map[op->outputs[pair.index()]] = pair.value();
    }
  }

  // Construct return values
  llvm::SmallVector<Value*, 4> return_operands;
  for (auto index : subgraph.outputs) {
    if (nullptr == vals_map[index]) {
      auto& const_tensor = *subgraph.tensors[index];
      auto const_loc = TensorLoc(const_tensor, builder, base_loc);
      auto op_or_err =
          BuildConstOp(const_tensor, buffers[const_tensor.buffer]->data,
                       op_builder, const_loc);
      if (!op_or_err.ok()) {
        return emitError(const_loc, op_or_err.status().ToString()),
               op_or_err.status();
      }
      vals_map[index] = op_or_err.ValueOrDie().output();
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

OwningModuleRef tflite::FlatBufferToMlir(absl::string_view buffer,
                                         MLIRContext* context,
                                         Location base_loc) {
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

  auto module = mlir::ModuleOp::create(base_loc);
  // We currently don't use this to make decisions, but we could
  // use it in exports or if there are breaking changes
  module.setAttr("tfl.schema_version",
                 builder.getI32IntegerAttr(model->version));
  if (!model->description.empty()) {
    module.setAttr("tfl.description",
                   builder.getStringAttr(model->description));
  }

  for (auto e : llvm::enumerate(model->subgraphs)) {
    auto& subgraph = e.value();
    std::string name = SubgraphName(e.index(), *subgraph);
    auto func_or_error = ConvertSubgraph(
        *subgraph, name, operator_names, model->buffers, base_loc, builder,
        // Only the entry point needs pseudo_input_ops
        // TODO(b/131175224,b/132239787) Support multiple entry points
        /* add_pseudo_input_ops = */ e.index() == 0);
    if (!func_or_error.ok()) {
      return emitError(base_loc, "could not translate function ")
                 << subgraph->name,
             nullptr;
    }
    module.push_back(func_or_error.ConsumeValueOrDie());
  }

  return OwningModuleRef(module);
}

static OwningModuleRef FlatBufferFileToMlirTrans(llvm::StringRef filename,
                                                 MLIRContext* context) {
  std::string error;
  auto loc = mlir::FileLineColLoc::get(filename, 0, 0, context);
  auto buffer = mlir::openInputFile(filename, &error);
  if (nullptr == buffer) {
    return emitError(loc, error), nullptr;
  }

  return tflite::FlatBufferToMlir(
      absl::string_view(buffer->getBufferStart(), buffer->getBufferSize()),
      context, loc);
}

static mlir::TranslateToMLIRRegistration FlatBufferFileToMlirTransReg(
    "tflite-flatbuffer-to-mlir", FlatBufferFileToMlirTrans);
