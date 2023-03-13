/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

// prototype for stablehlo serialization, WIP
// WARNING: converting to stablehlo file is experimental feature, and no runtime
// support is provided

#include "tensorflow/compiler/mlir/lite/stablehlo/serializer/flatbuffer_translator.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/stablehlo/serializer/flatbuffer_operator.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/lite/stablehlo/schema/schema_generated.h"

#define kStablehloOptionalTensor (-1)

using llvm::isa;
using llvm::StringRef;
using llvm::Twine;
using mlir::ElementsAttr;
using mlir::ModuleOp;
using mlir::Operation;
using mlir::StringAttr;
using mlir::TensorType;
using mlir::Value;
using mlir::func::FuncOp;
using tensorflow::OpOrArgLocNameMapper;
using tensorflow::OpOrArgNameMapper;
using xla::StatusOr;

namespace mlir {
namespace odml {

// TODO(b/267689361) this and the following functions should be automatically
// generated similar to operator_converters.inc in tflite
static flatbuffers::Offset<::stablehlo::flatbuf::Operator> CreateAddOperator(
    mlir::stablehlo::AddOp& hlo_op, flatbuffers::FlatBufferBuilder* fbb,
    uint32_t opcode_index, const std::vector<int32_t>& operands,
    const std::vector<int32_t>& results) {
  auto inputs = fbb->CreateVector(operands);
  auto outputs = fbb->CreateVector(results);

  return ::stablehlo::flatbuf::CreateOperator(*fbb, opcode_index, inputs,
                                              outputs);
}

static flatbuffers::Offset<::stablehlo::flatbuf::Operator>
CreateReshapeOperator(mlir::stablehlo::ReshapeOp& hlo_op,
                      flatbuffers::FlatBufferBuilder* fbb,
                      uint32_t opcode_index,
                      const std::vector<int32_t>& operands,
                      const std::vector<int32_t>& results) {
  auto inputs = fbb->CreateVector(operands);
  auto outputs = fbb->CreateVector(results);

  return ::stablehlo::flatbuf::CreateOperator(*fbb, opcode_index, inputs,
                                              outputs);
}

static flatbuffers::Offset<::stablehlo::flatbuf::Operator> CreateDivOperator(
    mlir::stablehlo::DivOp& hlo_op, flatbuffers::FlatBufferBuilder* fbb,
    uint32_t opcode_index, const std::vector<int32_t>& operands,
    const std::vector<int32_t>& results) {
  auto inputs = fbb->CreateVector(operands);
  auto outputs = fbb->CreateVector(results);

  return ::stablehlo::flatbuf::CreateOperator(*fbb, opcode_index, inputs,
                                              outputs);
}

static flatbuffers::Offset<::stablehlo::flatbuf::Operator>
CreateSubtractOperator(mlir::stablehlo::SubtractOp& hlo_op,
                       flatbuffers::FlatBufferBuilder* fbb,
                       uint32_t opcode_index,
                       const std::vector<int32_t>& operands,
                       const std::vector<int32_t>& results) {
  auto inputs = fbb->CreateVector(operands);
  auto outputs = fbb->CreateVector(results);

  return ::stablehlo::flatbuf::CreateOperator(*fbb, opcode_index, inputs,
                                              outputs);
}

static flatbuffers::Offset<::stablehlo::flatbuf::Operator> CreateMulOperator(
    mlir::stablehlo::MulOp hlo_op, flatbuffers::FlatBufferBuilder* fbb,
    uint32_t opcode_index, const std::vector<int32_t>& operands,
    const std::vector<int32_t>& results) {
  auto inputs = fbb->CreateVector(operands);
  auto outputs = fbb->CreateVector(results);

  return ::stablehlo::flatbuf::CreateOperator(*fbb, opcode_index, inputs,
                                              outputs);
}
static flatbuffers::Offset<::stablehlo::flatbuf::Operator> CreateMaxOperator(
    mlir::stablehlo::MaxOp& hlo_op, flatbuffers::FlatBufferBuilder* fbb,
    uint32_t opcode_index, const std::vector<int32_t>& operands,
    const std::vector<int32_t>& results) {
  auto inputs = fbb->CreateVector(operands);
  auto outputs = fbb->CreateVector(results);

  return ::stablehlo::flatbuf::CreateOperator(*fbb, opcode_index, inputs,
                                              outputs);
}

static flatbuffers::Offset<::stablehlo::flatbuf::Operator>
CreateConvertOperator(mlir::stablehlo::ConvertOp& hlo_op,
                      flatbuffers::FlatBufferBuilder* fbb,
                      uint32_t opcode_index,
                      const std::vector<int32_t>& operands,
                      const std::vector<int32_t>& results) {
  auto inputs = fbb->CreateVector(operands);
  auto outputs = fbb->CreateVector(results);

  return ::stablehlo::flatbuf::CreateOperator(*fbb, opcode_index, inputs,
                                              outputs);
}

static flatbuffers::Offset<::stablehlo::flatbuf::Operator> CreateDotOperator(
    mlir::stablehlo::DotOp& hlo_op, flatbuffers::FlatBufferBuilder* fbb,
    uint32_t opcode_index, const std::vector<int32_t>& operands,
    const std::vector<int32_t>& results) {
  auto inputs = fbb->CreateVector(operands);
  auto outputs = fbb->CreateVector(results);

  return ::stablehlo::flatbuf::CreateOperator(*fbb, opcode_index, inputs,
                                              outputs);
}

static flatbuffers::Offset<::stablehlo::flatbuf::Operator> CreateClampOperator(
    mlir::stablehlo::ClampOp& hlo_op, flatbuffers::FlatBufferBuilder* fbb,
    uint32_t opcode_index, const std::vector<int32_t>& operands,
    const std::vector<int32_t>& results) {
  auto inputs = fbb->CreateVector(operands);
  auto outputs = fbb->CreateVector(results);

  return ::stablehlo::flatbuf::CreateOperator(*fbb, opcode_index, inputs,
                                              outputs);
}

static flatbuffers::Offset<::stablehlo::flatbuf::Operator>
CreateLogisticOperator(mlir::stablehlo::LogisticOp& hlo_op,
                       flatbuffers::FlatBufferBuilder* fbb,
                       uint32_t opcode_index,
                       const std::vector<int32_t>& operands,
                       const std::vector<int32_t>& results) {
  auto inputs = fbb->CreateVector(operands);
  auto outputs = fbb->CreateVector(results);

  return ::stablehlo::flatbuf::CreateOperator(*fbb, opcode_index, inputs,
                                              outputs);
}

static flatbuffers::Offset<::stablehlo::flatbuf::Operator>
CreateConcatenateOperator(mlir::stablehlo::ConcatenateOp& hlo_op,
                          flatbuffers::FlatBufferBuilder* fbb,
                          uint32_t opcode_index,
                          const std::vector<int32_t>& operands,
                          const std::vector<int32_t>& results) {
  auto inputs = fbb->CreateVector(operands);
  auto outputs = fbb->CreateVector(results);

  auto options = ::stablehlo::flatbuf::CreateConcatenateOptions(
      *fbb, hlo_op.getDimension());

  return ::stablehlo::flatbuf::CreateOperator(
      *fbb, opcode_index, inputs, outputs,
      ::stablehlo::flatbuf::OperatorOptions_ConcatenateOptions,
      options.Union());
}

static flatbuffers::Offset<::stablehlo::flatbuf::Operator>
CreateConvolutionOperator(mlir::stablehlo::ConvolutionOp& hlo_op,
                          flatbuffers::FlatBufferBuilder* fbb,
                          uint32_t opcode_index,
                          const std::vector<int32_t>& operands,
                          const std::vector<int32_t>& results) {
  auto inputs = fbb->CreateVector(operands);
  auto outputs = fbb->CreateVector(results);

  // converting from mlir struct to std
  std::vector<int64_t> window_strides_vec =
      GetOptionalVector<int64_t>(hlo_op.getWindowStrides(), 0, 0);
  std::vector<int64_t> padding_vec =
      GetOptionalVector<int64_t>(hlo_op.getPadding(), 0, 0);
  std::vector<int64_t> lhs_dilation_vec =
      GetOptionalVector<int64_t>(hlo_op.getLhsDilation(), 0, 0);
  std::vector<int64_t> rhs_dilation_vec =
      GetOptionalVector<int64_t>(hlo_op.getRhsDilation(), 0, 0);
  std::vector<bool> window_reversal_vec =
      GetOptionalVector<bool>(hlo_op.getWindowReversal(), 0, 0);
  const int64_t feature_group_count = hlo_op.getFeatureGroupCount();
  const int64_t batch_group_count = hlo_op.getBatchGroupCount();

  auto conv_dimension_numbers = hlo_op.getDimensionNumbersAttr();

  std::vector<int64_t> input_spatial_dimensions_vec =
      conv_dimension_numbers.getInputSpatialDimensions().vec();
  std::vector<int64_t> kernel_spatial_dimensions_vec =
      conv_dimension_numbers.getKernelSpatialDimensions().vec();
  std::vector<int64_t> output_spatial_dimensions_vec =
      conv_dimension_numbers.getOutputSpatialDimensions().vec();
  const int64_t input_batch_dimension =
      conv_dimension_numbers.getInputBatchDimension();
  const int64_t input_feature_dimension =
      conv_dimension_numbers.getInputFeatureDimension();
  const int64_t kernel_input_feature_dimension =
      conv_dimension_numbers.getKernelInputFeatureDimension();
  const int64_t kernel_output_feature_dimension =
      conv_dimension_numbers.getKernelOutputFeatureDimension();
  const int64_t output_batch_dimension =
      conv_dimension_numbers.getOutputBatchDimension();
  const int64_t output_feature_dimension =
      conv_dimension_numbers.getOutputFeatureDimension();

  // serialize all vectors to flatbuffer
  auto window_strides = fbb->CreateVector(window_strides_vec);
  auto padding = fbb->CreateVector(padding_vec);
  auto lhs_dilation = fbb->CreateVector(lhs_dilation_vec);
  auto rhs_dilation = fbb->CreateVector(rhs_dilation_vec);
  auto input_spatial_dimensions =
      fbb->CreateVector(input_spatial_dimensions_vec);
  auto kernel_spatial_dimensions =
      fbb->CreateVector(kernel_spatial_dimensions_vec);
  auto output_spatial_dimensions =
      fbb->CreateVector(output_spatial_dimensions_vec);
  auto window_reversal = fbb->CreateVector(window_reversal_vec);

  auto options = ::stablehlo::flatbuf::CreateConvolutionOptions(
      *fbb, window_strides, padding, lhs_dilation, rhs_dilation,
      window_reversal, input_batch_dimension, input_feature_dimension,
      input_spatial_dimensions, kernel_input_feature_dimension,
      kernel_output_feature_dimension, kernel_spatial_dimensions,
      output_batch_dimension, output_feature_dimension,
      output_spatial_dimensions, feature_group_count, batch_group_count);

  return ::stablehlo::flatbuf::CreateOperator(
      *fbb, opcode_index, inputs, outputs,
      ::stablehlo::flatbuf::OperatorOptions_ConvolutionOptions,
      options.Union());
}

static flatbuffers::Offset<::stablehlo::flatbuf::Operator>
CreateReduceWindowOperator(mlir::stablehlo::ReduceWindowOp& hlo_op,
                           flatbuffers::FlatBufferBuilder* fbb,
                           uint32_t opcode_index,
                           const std::vector<int32_t>& operands,
                           const std::vector<int32_t>& results,
                           const int subgraph_idx) {
  auto inputs = fbb->CreateVector(operands);
  auto outputs = fbb->CreateVector(results);

  // TODO(zichuanwei@): instead of create these vectors let's just create
  // Flatbuffers vector directly
  std::vector<int64_t> window_dimension_vec(
      GetOptionalVector<int64_t>(hlo_op.getWindowDimensions(), 0, 0));
  std::vector<int64_t> window_strides_vec(
      GetOptionalVector<int64_t>(hlo_op.getWindowStrides(), 0, 0));
  std::vector<int64_t> base_dilations_vec(
      GetOptionalVector<int64_t>(hlo_op.getBaseDilations(), 0, 0));
  std::vector<int64_t> window_dilations_vec(
      GetOptionalVector<int64_t>(hlo_op.getWindowDilations(), 0, 0));
  std::vector<int64_t> padding_vec(
      GetOptionalVector<int64_t>(hlo_op.getPadding(), 0, 0));

  auto window_dimension = fbb->CreateVector(window_dimension_vec);
  auto window_strides = fbb->CreateVector(window_strides_vec);
  auto base_dilations = fbb->CreateVector(base_dilations_vec);
  auto window_dilations = fbb->CreateVector(window_dilations_vec);
  auto padding = fbb->CreateVector(padding_vec);

  auto options = ::stablehlo::flatbuf::CreateReduceWindowOptions(
      *fbb, window_dimension, window_strides, base_dilations, window_dilations,
      padding, subgraph_idx);

  return ::stablehlo::flatbuf::CreateOperator(
      *fbb, opcode_index, inputs, outputs,
      ::stablehlo::flatbuf::OperatorOptions_ReduceWindowOptions,
      options.Union());
}

static flatbuffers::Offset<::stablehlo::flatbuf::Operator>
CreateBroadcastInDimOperator(mlir::stablehlo::BroadcastInDimOp& hlo_op,
                             flatbuffers::FlatBufferBuilder* fbb,
                             uint32_t opcode_index,
                             const std::vector<int32_t>& operands,
                             const std::vector<int32_t>& results) {
  auto inputs = fbb->CreateVector(operands);
  auto outputs = fbb->CreateVector(results);

  std::vector<int64_t> broadcast_dimension_vec =
      GetOptionalVector<int64_t>(hlo_op.getBroadcastDimensions(), 0, 0);

  auto broadcast_dimension = fbb->CreateVector(broadcast_dimension_vec);

  auto options = ::stablehlo::flatbuf::CreateBroadcastInDimOptions(
      *fbb, broadcast_dimension);

  return ::stablehlo::flatbuf::CreateOperator(
      *fbb, opcode_index, inputs, outputs,
      ::stablehlo::flatbuf::OperatorOptions_BroadcastInDimOptions,
      options.Union());
}

static flatbuffers::Offset<::stablehlo::flatbuf::Operator>
CreateResizeBilinearOperator(mlir::stablehlo::CustomCallOp& hlo_op,
                             flatbuffers::FlatBufferBuilder* fbb,
                             uint32_t opcode_index,
                             const std::vector<int32_t>& operands,
                             const std::vector<int32_t>& results) {
  auto inputs = fbb->CreateVector(operands);
  auto outputs = fbb->CreateVector(results);

  auto align_corners =
      hlo_op->getAttr("align_corners").dyn_cast<mlir::BoolAttr>();
  assert(align_corners);
  auto half_pixel_center =
      hlo_op->getAttr("half_pixel_centers").dyn_cast<mlir::BoolAttr>();
  assert(half_pixel_center);

  auto options = ::stablehlo::flatbuf::CreateResizeBilinearOptions(
      *fbb, align_corners.getValue(), half_pixel_center.getValue());

  return ::stablehlo::flatbuf::CreateOperator(
      *fbb, opcode_index, inputs, outputs,
      ::stablehlo::flatbuf::OperatorOptions_ResizeBilinearOptions,
      options.Union());
}

std::optional<flatbuffers::Offset<::stablehlo::flatbuf::Operator>>
CreateFlatBufferOperator(mlir::Operation* op, uint32_t opcode_index,
                         const std::vector<int32_t>& operands,
                         const std::vector<int32_t>& results,
                         flatbuffers::FlatBufferBuilder* fbb,
                         int subgraph_idx = 0) {
  if (auto hlo_op = llvm::dyn_cast<mlir::stablehlo::AddOp>(op))
    return CreateAddOperator(hlo_op, fbb, opcode_index, operands, results);
  if (auto hlo_op = llvm::dyn_cast<mlir::stablehlo::DotOp>(op))
    return CreateDotOperator(hlo_op, fbb, opcode_index, operands, results);
  if (auto hlo_op = llvm::dyn_cast<mlir::stablehlo::LogisticOp>(op))
    return CreateLogisticOperator(hlo_op, fbb, opcode_index, operands, results);
  if (auto hlo_op = llvm::dyn_cast<mlir::stablehlo::DivOp>(op))
    return CreateDivOperator(hlo_op, fbb, opcode_index, operands, results);
  if (auto hlo_op = llvm::dyn_cast<mlir::stablehlo::SubtractOp>(op))
    return CreateSubtractOperator(hlo_op, fbb, opcode_index, operands, results);
  if (auto hlo_op = llvm::dyn_cast<mlir::stablehlo::MulOp>(op))
    return CreateMulOperator(hlo_op, fbb, opcode_index, operands, results);
  if (auto hlo_op = llvm::dyn_cast<mlir::stablehlo::MaxOp>(op))
    return CreateMaxOperator(hlo_op, fbb, opcode_index, operands, results);
  if (auto hlo_op = llvm::dyn_cast<mlir::stablehlo::ReshapeOp>(op))
    return CreateReshapeOperator(hlo_op, fbb, opcode_index, operands, results);
  if (auto hlo_op = llvm::dyn_cast<mlir::stablehlo::ConvolutionOp>(op))
    return CreateConvolutionOperator(hlo_op, fbb, opcode_index, operands,
                                     results);
  if (auto hlo_op = llvm::dyn_cast<mlir::stablehlo::ReduceWindowOp>(op))
    return CreateReduceWindowOperator(hlo_op, fbb, opcode_index, operands,
                                      results, subgraph_idx);
  if (auto hlo_op = llvm::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op))
    return CreateBroadcastInDimOperator(hlo_op, fbb, opcode_index, operands,
                                        results);
  if (auto hlo_op = llvm::dyn_cast<mlir::stablehlo::CustomCallOp>(op))
    return CreateResizeBilinearOperator(hlo_op, fbb, opcode_index, operands,
                                        results);
  if (auto hlo_op = llvm::dyn_cast<mlir::stablehlo::ClampOp>(op))
    return CreateClampOperator(hlo_op, fbb, opcode_index, operands, results);
  if (auto hlo_op = llvm::dyn_cast<mlir::stablehlo::ConcatenateOp>(op))
    return CreateConcatenateOperator(hlo_op, fbb, opcode_index, operands,
                                     results);
  if (auto hlo_op = llvm::dyn_cast<mlir::stablehlo::ConvertOp>(op))
    return CreateConvertOperator(hlo_op, fbb, opcode_index, operands, results);
  return std::nullopt;
}

static StatusOr<::stablehlo::flatbuf::DataType> GetDataType(
    Type type, bool is_signed = true) {
  if (type.isF16()) return ::stablehlo::flatbuf::DataType_FLOAT16;
  if (type.isF32()) return ::stablehlo::flatbuf::DataType_FLOAT32;
  if (type.isF64()) return ::stablehlo::flatbuf::DataType_FLOAT64;
  if (type.isSignlessInteger(8)) return ::stablehlo::flatbuf::DataType_INT8;
  if (type.isSignlessInteger(16)) return ::stablehlo::flatbuf::DataType_INT16;
  if (type.isSignlessInteger(32)) return ::stablehlo::flatbuf::DataType_INT32;
  if (type.isSignlessInteger(64)) return ::stablehlo::flatbuf::DataType_INT64;
  if (type.isUnsignedInteger(8)) return ::stablehlo::flatbuf::DataType_UINT8;
  if (type.isUnsignedInteger(16)) return ::stablehlo::flatbuf::DataType_UINT16;
  if (type.isUnsignedInteger(32)) return ::stablehlo::flatbuf::DataType_UINT32;
  if (type.isUnsignedInteger(64)) return ::stablehlo::flatbuf::DataType_UINT64;
  std::string type_str;
  llvm::raw_string_ostream str_stream(type_str);
  str_stream << type;
  LOG(ERROR) << "unsupported datatype" << type_str;
  return tensorflow::errors::InvalidArgument("unsupported datatype" + type_str);
}

std::optional<::stablehlo::flatbuf::OperatorCode> GetOpCode(
    mlir::Operation* op) {
  if (isa<mlir::stablehlo::AddOp>(op))
    return ::stablehlo::flatbuf::OperatorCode_ADD;
  if (isa<mlir::stablehlo::DotOp>(op))
    return ::stablehlo::flatbuf::OperatorCode_DOT;
  if (isa<mlir::stablehlo::SubtractOp>(op))
    return ::stablehlo::flatbuf::OperatorCode_SUBTRACT;
  if (isa<mlir::stablehlo::DivOp>(op))
    return ::stablehlo::flatbuf::OperatorCode_DIVIDE;
  if (isa<mlir::stablehlo::LogisticOp>(op))
    return ::stablehlo::flatbuf::OperatorCode_LOGISTIC;
  if (isa<mlir::stablehlo::MulOp>(op))
    return ::stablehlo::flatbuf::OperatorCode_MULTIPLY;
  if (isa<mlir::stablehlo::MaxOp>(op))
    return ::stablehlo::flatbuf::OperatorCode_MAXIMUM;
  if (isa<mlir::stablehlo::ReshapeOp>(op))
    return ::stablehlo::flatbuf::OperatorCode_RESHAPE;
  if (isa<mlir::stablehlo::ConvolutionOp>(op))
    return ::stablehlo::flatbuf::OperatorCode_CONVOLUTION;
  if (isa<mlir::stablehlo::BroadcastInDimOp>(op))
    return ::stablehlo::flatbuf::OperatorCode_BROADCAST_IN_DIM;
  if (isa<mlir::stablehlo::ReduceWindowOp>(op))
    return ::stablehlo::flatbuf::OperatorCode_REDUCE_WINDOW;
  if (isa<mlir::stablehlo::ClampOp>(op))
    return ::stablehlo::flatbuf::OperatorCode_CLAMP;
  if (isa<mlir::stablehlo::ConcatenateOp>(op))
    return ::stablehlo::flatbuf::OperatorCode_CONCATENATE;
  if (isa<mlir::stablehlo::ConvertOp>(op))
    return ::stablehlo::flatbuf::OperatorCode_CONVERT;

  // For now we assume the incoming custom op is a resize_bilinear, it is
  // expected any other custom op will cause the program to error out
  if (isa<mlir::stablehlo::CustomCallOp>(op))
    return ::stablehlo::flatbuf::OperatorCode_RESIZE_BILINEAR;

  op->emitError(Twine("unsupported op type " + op->getName().getStringRef()));
  return std::nullopt;
}

static bool IsConst(Operation* op) {
  return isa<mlir::func::ConstantOp, mlir::arith::ConstantOp,
             mlir::stablehlo::ConstantOp>(op);
}

std::optional<std::string> Translator::Translate(
    ModuleOp module, const toco::TocoFlags& toco_flags,
    const std::unordered_set<std::string>& tags,
    OpOrArgNameMapper* op_or_arg_name_mapper,
    const std::map<std::string, std::string>& metadata) {
  OpOrArgLocNameMapper default_op_or_arg_name_mapper;
  if (!op_or_arg_name_mapper)
    op_or_arg_name_mapper = &default_op_or_arg_name_mapper;
  // TODO(b/267689626): sanity checkers not implemented
  Translator translator(module, toco_flags, tags, op_or_arg_name_mapper,
                        metadata);
  return translator.TranslateInternal();
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
        main_fn->getAttrOfType<mlir::DictionaryAttr>(tf_entry_function_);
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

  // collect all reduce window ops, this is only a temporary hack
  // in the future, we should have a function to walk over all ops that have
  // regions contained, the logic in stablehlo is a bit different from tfl
  // dialect in that all subgraphs in tflite a enclosed in func op where
  // stablehlo op maintain their own regions
  std::vector<mlir::stablehlo::ReduceWindowOp> reduce_window;
  module_.walk([&](mlir::stablehlo::ReduceWindowOp op) {
    reduce_window.push_back(op);
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

  // add regions of reduce_window ops into subgraph map. the name will be
  // stablehlo.reduce_window as mlir::region is not assicoate with a name
  for (auto op : reduce_window) {
    reduce_window_subgraph_map_[op] = subgraph_idx++;
    named_regions.emplace_back(op.getOperationName().str(), &op.getBody());
  }

  // Build subgraph for each of the named regions.
  std::vector<BufferOffset<::stablehlo::flatbuf::SubGraph>> subgraphs;
  subgraphs.reserve(named_regions.size());
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
  // TODO(b/267801705) : Add schema version
  auto model = ::stablehlo::flatbuf::CreateModel(
      builder_, 0, builder_.CreateVector(opcodes_),
      builder_.CreateVector(subgraphs), builder_.CreateVector(buffers_));
  ::stablehlo::flatbuf::FinishModelBuffer(builder_, model);
  // There is a limit of 2GB for a flatbuffer.
  if (builder_.GetSize() > 2147483648) {
    LOG(ERROR) << "Model size is bigger than 2gb";
    return std::nullopt;
  }

  // Return serialized string for the built FlatBuffer.
  return std::string(reinterpret_cast<const char*>(builder_.GetBufferPointer()),
                     builder_.GetSize());
}

std::optional<BufferOffset<::stablehlo::flatbuf::Tensor>>
Translator::BuildTensor(Value value, const std::string& name,
                        unsigned buffer_idx) {
  auto type = value.getType().cast<TensorType>();

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

  bool is_variable = !(inst && IsConst(inst));
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

  Type element_type = type.getElementType();
  auto status = GetDataType(element_type);
  if (!status.ok()) return std::nullopt;
  ::stablehlo::flatbuf::DataType data_type = GetDataType(element_type).value();

  return ::stablehlo::flatbuf::CreateTensor(
      builder_, builder_.CreateVector(shape), data_type,
      (is_variable ? 0 : buffer_idx), builder_.CreateString(name));
}

void Translator::InitializeNamesFromAttribute(FuncOp fn, bool* has_input_attr) {
  auto dict_attr = fn->getAttrOfType<mlir::DictionaryAttr>(tf_entry_function_);
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

std::string Translator::UniqueName(mlir::Value val) {
  return std::string(name_mapper_.GetUniqueName(val));
}

std::optional<BufferOffset<::stablehlo::flatbuf::SubGraph>>
Translator::BuildSubGraph(const std::string& name, Region* region, int index) {
  bool has_input_attr = false;
  if (auto fn = dyn_cast<FuncOp>(region->getParentOp())) {
    InitializeNamesFromAttribute(fn, &has_input_attr);
  }
  std::vector<BufferOffset<::stablehlo::flatbuf::Tensor>> tensors;
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
    auto tensor_or = BuildTensor(value, tensor_name, buffers_.size());
    if (!tensor_or) return false;
    tensors.push_back(*tensor_or);

    if (value.getDefiningOp()) {
      auto buffer_or = BuildBuffer(value);
      if (!buffer_or) return false;
      buffers_.push_back(*buffer_or);
    } else {
      // TODO(b/267802872): Tflite will create a buffer entry for every tensor
      // regardless constant or not. in stablehlo serialization, we don't plan
      // to keep this behaviour
      buffers_.push_back(empty_buffer_);
    }
    return true;
  };

  std::vector<BufferOffset<::stablehlo::flatbuf::Operator>> operators;

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

    for (auto val : inst.getResults()) {
      std::string tensor_name = UniqueName(val);
      // For "tfl.numeric_verify" op, the name is used to find out the original
      // activation tensor rather than its own unique name in the visualization
      // or debugging tools.
      // auto builtin_code = GetOpCode(&inst);
      if (!build_tensor_and_buffer(val, index, tensor_name))
        return std::nullopt;
    }

    // Skip constant ops as they don't represent flatbuffer operator.
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
        operands.push_back(kStablehloOptionalTensor);
      else
        operands.push_back(tensor_index_map.lookup(operand));
    }

    if (auto flat_operator = BuildOperator(real_inst, operands, results)) {
      operation_index_to_operator_index.try_emplace(operation_index,
                                                    operators.size());
      operators.push_back(*flat_operator);
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
  return ::stablehlo::flatbuf::CreateSubGraph(
      builder_, builder_.CreateVector(tensors), builder_.CreateVector(inputs),
      builder_.CreateVector(outputs), builder_.CreateVector(operators),
      /*name=*/builder_.CreateString(name));
}

std::optional<BufferOffset<::stablehlo::flatbuf::Buffer>>
Translator::BuildBuffer(mlir::Value value) {
  auto inst = value.getDefiningOp();
  ElementsAttr attr;

  if (auto cst = dyn_cast<mlir::arith::ConstantOp>(inst)) {
    // arith::ConstantOp have ElementAttr at this point due to validation of the
    // TFLite module.
    attr = cst.getValue().cast<ElementsAttr>();
  } else if (auto cst = dyn_cast<mlir::stablehlo::ConstantOp>(inst)) {
    attr = cst.getValue();
  } else {
    return empty_buffer_;
  }

  tensorflow::Tensor tensor;
  auto status = tensorflow::ConvertToTensor(attr, &tensor);
  if (!status.ok()) {
    inst->emitError(
        Twine("failed to convert value attribute to tensor with error: " +
              status.ToString()));
    return std::nullopt;
  }

  absl::string_view tensor_data = tensor.tensor_data();
  auto buffer_data = builder_.CreateVector(
      reinterpret_cast<const uint8_t*>(tensor_data.data()), tensor_data.size());
  return ::stablehlo::flatbuf::CreateBuffer(builder_, buffer_data);
}

uint32_t Translator::GetOpcodeIndex(
    const std::string& op_name, ::stablehlo::flatbuf::OperatorCode op_code) {
  auto it = opcode_index_map_.insert({op_name, 0});

  // If the insert succeeded, the opcode has not been created already. Create a
  // new operator code and update its index value in the map.
  if (it.second) {
    it.first->second = opcodes_.size();
    opcodes_.push_back(op_code);
  }
  return it.first->second;
}

std::optional<BufferOffset<::stablehlo::flatbuf::Operator>>
Translator::BuildOperator(Operation* inst, std::vector<int32_t> operands,
                          const std::vector<int32_t>& results) {
  const auto* dialect = inst->getDialect();
  if (!dialect) {
    inst->emitOpError("dialect is not registered");
    return std::nullopt;
  }

  if (dialect == stablehlo_dialect_) {
    auto op_code = GetOpCode(inst);
    if (op_code == std::nullopt) {
      return inst->emitOpError("op code not found"), std::nullopt;
    }

    auto opcode_index =
        GetOpcodeIndex(inst->getName().getStringRef().str(), op_code.value());
    std::optional<flatbuffers::Offset<::stablehlo::flatbuf::Operator>> offset;
    if (op_code == ::stablehlo::flatbuf::OperatorCode_REDUCE_WINDOW) {
      offset = CreateFlatBufferOperator(
          inst, opcode_index, operands, results, &builder_,
          reduce_window_subgraph_map_
              [llvm::dyn_cast<mlir::stablehlo::ReduceWindowOp>(inst)]);
    } else {
      offset = CreateFlatBufferOperator(inst, opcode_index, operands, results,
                                        &builder_);
    }
    if (!offset) {
      inst->emitOpError("is not a supported stablehlo op");
    }
    return offset;
  }

  return inst->emitOpError("a stableHLO op"), std::nullopt;
}

}  // namespace odml
}  // namespace mlir
