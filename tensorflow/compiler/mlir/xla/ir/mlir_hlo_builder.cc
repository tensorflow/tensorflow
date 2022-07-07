/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.h"

#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/attribute_importer.h"
#include "tensorflow/compiler/mlir/xla/hlo_function_importer.h"
#include "tensorflow/compiler/mlir/xla/hlo_utils.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

static std::string GetMlirOpName(HloOpcode opcode) {
  std::string op_name = HloOpcodeString(opcode);
  absl::c_replace(op_name, '-', '_');
  return mlir::mhlo::MhloDialect::getDialectNamespace().str() + "." + op_name;
}

static std::string ToString(mlir::Type ty) {
  std::string str;
  llvm::raw_string_ostream sstream(str);
  ty.print(sstream);
  sstream.flush();
  return str;
}

// Returns 1D 64-bit dense elements attribute with the given values.
static mlir::DenseIntElementsAttr GetI64ElementsAttr(
    absl::Span<const int64_t> values, mlir::Builder* builder) {
  auto ty = mlir::RankedTensorType::get({static_cast<int64_t>(values.size())},
                                        builder->getIntegerType(64));
  return mlir::DenseIntElementsAttr::get(
      ty, llvm::makeArrayRef(values.data(), values.size()));
}

static mlir::DenseIntElementsAttr ConvertPadding(
    absl::Span<const std::pair<int64_t, int64_t>> padding,
    mlir::Builder* builder) {
  llvm::SmallVector<int64_t, 8> elements;
  elements.reserve(padding.size() * 2);
  for (const auto& vals : padding) {
    elements.push_back(vals.first);
    elements.push_back(vals.second);
  }
  auto ty = mlir::RankedTensorType::get(
      {static_cast<int64_t>(padding.size()), 2}, builder->getIntegerType(64));
  return mlir::DenseIntElementsAttr::get(ty, elements);
}

MlirHloBuilder::~MlirHloBuilder() = default;

StatusOr<XlaOp> MlirHloBuilder::MakeXlaOp(mlir::Value val) {
  mlir::Type ty = val.getType();
  auto shape = std::make_unique<Shape>(TypeToShape(ty));
  if (shape->element_type() == PrimitiveType::PRIMITIVE_TYPE_INVALID) {
    return InvalidArgument("unsupported type: %s", ToString(ty).c_str());
  }

  int64_t handle = reinterpret_cast<int64_t>(val.getAsOpaquePointer());
  handle_to_shape_[handle] = std::move(shape);
  return XlaOp(handle, this);
}

XlaOp MlirHloBuilder::ConstantLiteral(const LiteralSlice& literal) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(mlir::DenseElementsAttr attr,
                        CreateDenseElementsAttrFromLiteral(literal, builder_));
    auto op = builder_.create<mlir::mhlo::ConstantOp>(loc_, attr);
    return MakeXlaOp(op);
  });
}

StatusOr<XlaOp> MlirHloBuilder::ConvGeneralDilatedInternal(
    const Shape& shape, XlaOp lhs, XlaOp rhs, const Window& window,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding,
    absl::Span<const int64_t> lhs_dilation,
    absl::Span<const int64_t> rhs_dilation,
    const ConvolutionDimensionNumbers& dimension_numbers,
    int64_t feature_group_count, int64_t batch_group_count,
    const PrecisionConfig* precision_config) {
  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  mlir::ArrayAttr config_attr;
  if (precision_config)
    config_attr = ConvertPrecisionConfig(precision_config, &builder_);
  auto op = builder_.create<mlir::mhlo::ConvOp>(
      loc_, ty, GetValue(lhs), GetValue(rhs),
      GetI64ElementsAttr(window_strides, &builder_),
      ConvertPadding(padding, &builder_),
      GetI64ElementsAttr(lhs_dilation, &builder_),
      GetI64ElementsAttr(rhs_dilation, &builder_),
      /*window_reversal=*/nullptr,
      ConvertConvDimensionNumbers(dimension_numbers, &builder_),
      builder_.getI64IntegerAttr(feature_group_count),
      builder_.getI64IntegerAttr(batch_group_count), config_attr);
  return MakeXlaOp(op);
}

StatusOr<XlaOp> MlirHloBuilder::FftInternal(
    const Shape& shape, XlaOp operand, FftType fft_type,
    absl::Span<const int64_t> fft_length) {
  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  auto fft_type_attr = mlir::mhlo::symbolizeFftType(FftType_Name(fft_type));
  auto op = builder_.create<mlir::mhlo::FftOp>(
      loc_, ty, GetValue(operand),
      mlir::mhlo::FftTypeAttr::get(builder_.getContext(),
                                   fft_type_attr.getValue()),
      GetI64ElementsAttr(fft_length, &builder_));
  return MakeXlaOp(op);
}

StatusOr<XlaOp> MlirHloBuilder::CustomCallInternal(
    const std::string& call_target_name, absl::Span<const XlaOp> operands,
    const Shape& shape, const std::string& opaque,
    std::optional<absl::Span<const Shape>> operand_shapes_with_layout,
    bool has_side_effect,
    absl::Span<const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
        output_operand_aliasing,
    const Literal* literal, std::optional<Window> window,
    std::optional<ConvolutionDimensionNumbers> dnums,
    CustomCallSchedule schedule, CustomCallApiVersion api_version) {
  TF_RET_CHECK(output_operand_aliasing.empty())
      << "MLIR CustomCallOp does not support output_operand_aliasing yet";
  TF_RET_CHECK(literal == nullptr)
      << "MLIR CustomCallOp does not support literal yet";
  TF_RET_CHECK(!window.has_value())
      << "MLIR CustomCallOp does not support ConvolutionDimensionNumbers yet";
  TF_RET_CHECK(!dnums.has_value())
      << "MLIR CustomCallOp does not support ConvolutionDimensionNumbers yet";
  TF_RET_CHECK(schedule == CustomCallSchedule::SCHEDULE_NONE)
      << "MLIR CustomCallOp does not support custom-call-schedule yet";

  llvm::SmallVector<mlir::NamedAttribute> attributes;
  if (operand_shapes_with_layout.has_value()) {
    TF_ASSIGN_OR_RETURN(mlir::ArrayAttr operand_layouts,
                        ExtractLayoutsFromShapes(
                            operand_shapes_with_layout.value(), &builder_));
    attributes.push_back(
        builder_.getNamedAttr("operand_layouts", operand_layouts));

    mlir::ArrayAttr result_layouts;
    if (shape.IsTuple()) {
      TF_ASSIGN_OR_RETURN(result_layouts,
                          ExtractLayoutsFromTuple(shape, &builder_));
    } else {
      TF_ASSIGN_OR_RETURN(result_layouts,
                          ExtractLayoutsFromShapes({shape}, &builder_));
    }
    attributes.push_back(
        builder_.getNamedAttr("result_layouts", result_layouts));
  }
  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  TF_ASSIGN_OR_RETURN(auto mlir_api_version,
                      ConvertCustomCallApiVersion(api_version));
  attributes.push_back(builder_.getNamedAttr(
      "api_version", mlir::mhlo::CustomCallApiVersionAttr::get(
                         builder_.getContext(), mlir_api_version)));

  attributes.push_back(builder_.getNamedAttr(
      "call_target_name", builder_.getStringAttr(call_target_name)));
  attributes.push_back(builder_.getNamedAttr(
      "has_side_effect", builder_.getBoolAttr(has_side_effect)));
  attributes.push_back(
      builder_.getNamedAttr("backend_config", builder_.getStringAttr(opaque)));

  auto op = builder_.create<mlir::mhlo::CustomCallOp>(
      loc_, ty, GetValues(operands), attributes);
  return MakeXlaOp(op.getResult(0));
}

StatusOr<XlaOp> MlirHloBuilder::ReduceInternal(
    const Shape& shape, absl::Span<const XlaOp> all_operands,
    const XlaComputation& computation,
    absl::Span<const int64_t> dimensions_to_reduce) {
  // Reduce takes two set of variadic operands inputs and init_values.
  // all_operands contains both of these so split operands into two parts.
  int64_t num_args = all_operands.size() / 2;
  auto op = builder_.create<mlir::mhlo::ReduceOp>(
      loc_, GetValues(all_operands.first(num_args)),
      GetValues(all_operands.subspan(num_args)),
      GetI64ElementsAttr(dimensions_to_reduce, &builder_));
  TF_RETURN_IF_ERROR(ImportComputation(computation.proto(), &op.body(),
                                       /*flatten_region_arg_tuple*/ true));
  if (op.getNumResults() == 1) return MakeXlaOp(op.getResult(0));
  auto tuple = builder_.create<mlir::mhlo::TupleOp>(loc_, op.getResults());
  return MakeXlaOp(tuple);
}

StatusOr<XlaOp> MlirHloBuilder::ReduceWindowInternal(
    const Shape& shape, XlaOp operand, XlaOp init_value,
    const XlaComputation& computation, Window window) {
  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  llvm::SmallVector<int64_t, 4> sizes, strides, base_dilations, win_dilations;
  llvm::SmallVector<int64_t, 8> padding;
  for (const auto& dim : window.dimensions()) {
    sizes.push_back(dim.size());
    strides.push_back(dim.stride());
    base_dilations.push_back(dim.base_dilation());
    win_dilations.push_back(dim.window_dilation());
    padding.push_back(dim.padding_low());
    padding.push_back(dim.padding_high());
  }
  auto padding_ty =
      mlir::RankedTensorType::get({static_cast<int64_t>(padding.size()) / 2, 2},
                                  builder_.getIntegerType(64));
  auto op = builder_.create<mlir::mhlo::ReduceWindowOp>(
      loc_, ty, GetValue(operand), GetValue(init_value),
      GetI64ElementsAttr(sizes, &builder_),
      GetI64ElementsAttr(strides, &builder_),
      GetI64ElementsAttr(base_dilations, &builder_),
      GetI64ElementsAttr(win_dilations, &builder_),
      mlir::DenseIntElementsAttr::get(padding_ty, padding));
  TF_RETURN_IF_ERROR(ImportComputation(computation.proto(), &op.body(),
                                       /*flatten_region_arg_tuple*/ true));
  return MakeXlaOp(op.getResult(0));
}

XlaOp MlirHloBuilder::Iota(const Shape& shape, int64_t iota_dimension) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(
        mlir::Type ty,
        ConvertShapeToType<mlir::RankedTensorType>(shape, builder_));
    auto op = builder_.create<mlir::mhlo::IotaOp>(
        loc_, ty,
        builder_.getIntegerAttr(builder_.getI64Type(), iota_dimension));
    return MakeXlaOp(op);
  });
}

StatusOr<XlaOp> MlirHloBuilder::BitcastConvertTypeInternal(const Shape& shape,
                                                           XlaOp operand) {
  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  auto op = builder_.create<mlir::mhlo::BitcastConvertOp>(loc_, ty,
                                                          GetValue(operand));
  return MakeXlaOp(op);
}

StatusOr<XlaOp> MlirHloBuilder::TransposeInternal(
    const Shape& shape, XlaOp operand, absl::Span<const int64_t> permutation) {
  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  auto op = builder_.create<mlir::mhlo::TransposeOp>(
      loc_, ty, GetValue(operand), GetI64ElementsAttr(permutation, &builder_));
  return MakeXlaOp(op);
}

StatusOr<XlaOp> MlirHloBuilder::RevInternal(
    const Shape& shape, XlaOp operand, absl::Span<const int64_t> dimensions) {
  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  auto op = builder_.create<mlir::mhlo::ReverseOp>(
      loc_, ty, GetValue(operand), GetI64ElementsAttr(dimensions, &builder_));
  return MakeXlaOp(op);
}

StatusOr<XlaOp> MlirHloBuilder::SortInternal(const Shape& shape,
                                             absl::Span<const XlaOp> operands,
                                             const XlaComputation& comparator,
                                             int64_t dimension,
                                             bool is_stable) {
  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  llvm::SmallVector<mlir::Type, 4> sort_types = {ty};
  if (auto tuple_ty = ty.dyn_cast<mlir::TupleType>()) {
    sort_types = llvm::to_vector<6>(tuple_ty.getTypes());
  }

  auto op = builder_.create<mlir::mhlo::SortOp>(
      loc_, sort_types, GetValues(operands),
      builder_.getI64IntegerAttr(dimension), builder_.getBoolAttr(is_stable));
  TF_RETURN_IF_ERROR(ImportComputation(comparator.proto(), &op.comparator()));

  if (ty.isa<mlir::TupleType>()) {
    auto tuple = builder_.create<mlir::mhlo::TupleOp>(loc_, op.getResults());
    return MakeXlaOp(tuple);
  }

  return MakeXlaOp(op.getResult(0));
}

StatusOr<XlaOp> MlirHloBuilder::WhileInternal(const Shape& shape,
                                              const XlaComputation& condition,
                                              const XlaComputation& body,
                                              XlaOp init) {
  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));

  llvm::SmallVector<mlir::Value> flattened_operands;
  llvm::SmallVector<mlir::Type> flattened_operand_types;

  HloFunctionImporter::FlattenTupleType(ty, flattened_operand_types);
  HloFunctionImporter::FlattenTupleValue(&builder_, loc_, GetValue(init),
                                         flattened_operands);

  auto op = builder_.create<mlir::mhlo::WhileOp>(loc_, flattened_operand_types,
                                                 flattened_operands);

  TF_RETURN_IF_ERROR(ImportComputation(condition.proto(), &op.cond(),
                                       /*flatten_region_arg_tuple*/ true));
  TF_RETURN_IF_ERROR(ImportComputation(body.proto(), &op.body(),
                                       /*flatten_region_arg_tuple*/ true));

  if (ty.isa<mlir::TupleType>()) {
    llvm::SmallVector<mlir::Value> flattened_results = op->getResults();
    llvm::MutableArrayRef<mlir::Value> flattened_results_ref(flattened_results);
    auto result = HloFunctionImporter::CreateTupleValue(
        &builder_, loc_, flattened_results_ref, ty);
    auto defining_tuple_op = result.getDefiningOp<mlir::mhlo::TupleOp>();
    return MakeXlaOp(defining_tuple_op);
  }

  return MakeXlaOp(op.getResult(0));
}

StatusOr<XlaOp> MlirHloBuilder::ReducePrecisionInternal(
    const Shape& shape, XlaOp operand, const int exponent_bits,
    const int mantissa_bits) {
  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  auto op = builder_.create<mlir::mhlo::ReducePrecisionOp>(
      loc_, ty, GetValue(operand), builder_.getI32IntegerAttr(exponent_bits),
      builder_.getI32IntegerAttr(mantissa_bits));
  return MakeXlaOp(op);
}

StatusOr<XlaOp> MlirHloBuilder::GatherInternal(
    const Shape& shape, XlaOp input, XlaOp start_indices,
    const GatherDimensionNumbers& dimension_numbers,
    absl::Span<const int64_t> slice_sizes, bool indices_are_sorted) {
  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  auto op = builder_.create<mlir::mhlo::GatherOp>(
      loc_, ty, GetValue(input), GetValue(start_indices),
      ConvertGatherDimensionNumbers(dimension_numbers, &builder_),
      GetI64ElementsAttr(slice_sizes, &builder_));
  return MakeXlaOp(op);
}

StatusOr<XlaOp> MlirHloBuilder::ScatterInternal(
    const Shape& shape, absl::Span<const XlaOp> inputs, XlaOp scatter_indices,
    absl::Span<const XlaOp> updates, const XlaComputation& update_computation,
    const ScatterDimensionNumbers& dimension_numbers, bool indices_are_sorted,
    bool unique_indices) {
  // TODO(b/230137437): Allow variadic scatter after adding mhlo support.
  if (inputs.size() != 1) {
    return Unimplemented("Variadic scatter not implemented in mhlo yet.");
  }
  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  auto op = builder_.create<mlir::mhlo::ScatterOp>(
      loc_, ty, GetValue(inputs[0]), GetValue(scatter_indices),
      GetValue(updates[0]),
      ConvertScatterDimensionNumbers(dimension_numbers, &builder_),
      builder_.getBoolAttr(indices_are_sorted),
      builder_.getBoolAttr(unique_indices));

  TF_RETURN_IF_ERROR(
      ImportComputation(update_computation.proto(), &op.update_computation()));
  return MakeXlaOp(op.getResult(0));
}

StatusOr<XlaOp> MlirHloBuilder::SetDimensionSizeInternal(const Shape& shape,
                                                         XlaOp operand,
                                                         XlaOp val,
                                                         int64_t dimension) {
  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  auto op = builder_.create<mlir::mhlo::SetDimensionSizeOp>(
      loc_, ty, GetValue(operand), GetValue(val),
      builder_.getI64IntegerAttr(dimension));
  return MakeXlaOp(op);
}

StatusOr<XlaOp> MlirHloBuilder::RngOpInternal(
    RandomDistribution distribution, absl::Span<const XlaOp> parameters,
    const Shape& shape) {
  mlir::mhlo::RngDistributionAttr attr;
  if (distribution == xla::RandomDistribution::RNG_UNIFORM) {
    attr = mlir::mhlo::RngDistributionAttr::get(
        builder_.getContext(), mlir::mhlo::RngDistribution::UNIFORM);
  } else {
    TF_RET_CHECK(distribution == xla::RandomDistribution::RNG_NORMAL)
        << "Unexpected distribution: " << distribution;
    attr = mlir::mhlo::RngDistributionAttr::get(
        builder_.getContext(), mlir::mhlo::RngDistribution::NORMAL);
  }
  llvm::SmallVector<mlir::NamedAttribute, 1> attributes = {
      builder_.getNamedAttr("rng_distribution", attr)};

  if (shape.is_dynamic())
    return Unimplemented("RngOp with dynamic dims not supported");
  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));

  auto op = builder_.create<mlir::mhlo::RngOp>(
      loc_, ty, GetValue(parameters[0]), GetValue(parameters[1]),
      GetValue(
          ConstantLiteral(LiteralUtil::CreateR1<int64_t>(shape.dimensions()))),
      attr);
  return MakeXlaOp(op.getResult());
}

StatusOr<XlaOp> MlirHloBuilder::RngBitGeneratorInternal(
    const Shape& full_result_shape, RandomAlgorithm algorithm,
    XlaOp initial_state) {
  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         full_result_shape, builder_));

  llvm::SmallVector<mlir::Type> flattened_ret_types;
  HloFunctionImporter::FlattenTupleType(ty, flattened_ret_types);

  auto algorithm_attr = mlir::mhlo::RngAlgorithmAttr::get(
      builder_.getContext(), *mlir::mhlo::symbolizeRngAlgorithm(algorithm));
  auto op = builder_.create<mlir::mhlo::RngBitGeneratorOp>(
      loc_, flattened_ret_types, algorithm_attr, GetValue(initial_state));

  if (ty.isa<mlir::TupleType>()) {
    llvm::SmallVector<mlir::Value> flattened_results = op->getResults();
    llvm::MutableArrayRef<mlir::Value> flattened_results_ref(flattened_results);
    auto result = HloFunctionImporter::CreateTupleValue(
        &builder_, loc_, flattened_results_ref, ty);
    auto defining_tuple_op = result.getDefiningOp<mlir::mhlo::TupleOp>();
    return MakeXlaOp(defining_tuple_op);
  }

  return MakeXlaOp(op.getResult(0));
}

StatusOr<XlaOp> MlirHloBuilder::ReshapeInternal(const Shape& shape,
                                                XlaOp operand,
                                                int64_t inferred_dimension) {
  TF_RETURN_IF_ERROR(first_error());

  if (inferred_dimension != -1)
    return Unimplemented("inferred_dimension not yet supported for Reshape op");
  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  mlir::Value value = GetValue(operand);
  auto op = builder_.create<mlir::mhlo::ReshapeOp>(loc_, ty, value);
  return MakeXlaOp(op.getResult());
}

StatusOr<XlaOp> MlirHloBuilder::DotGeneralInternal(
    const Shape& shape, XlaOp lhs, XlaOp rhs,
    const DotDimensionNumbers& dimension_number,
    const PrecisionConfig* precision_config) {
  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  auto op = builder_.create<mlir::mhlo::DotGeneralOp>(
      loc_, ty, GetValue(lhs), GetValue(rhs),
      ConvertDotDimensionNumbers(dimension_number, &builder_),
      ConvertPrecisionConfig(precision_config, &builder_));
  return MakeXlaOp(op.getResult());
}

StatusOr<XlaOp> MlirHloBuilder::InDimBroadcast(
    const Shape& shape, XlaOp operand,
    absl::Span<const int64_t> broadcast_dimensions) {
  TF_RETURN_IF_ERROR(first_error());
  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  mlir::Value value = GetValue(operand);
  auto op = builder_.create<mlir::mhlo::BroadcastInDimOp>(
      loc_, ty, value, GetI64ElementsAttr(broadcast_dimensions, &builder_));
  return MakeXlaOp(op.getResult());
}

StatusOr<XlaOp> MlirHloBuilder::AddInstruction(
    HloInstructionProto&& instr, HloOpcode opcode,
    absl::Span<const XlaOp> operands) {
  return Unimplemented("MlirHloBuilder does not support op %s",
                       HloOpcodeString(opcode));
}

StatusOr<XlaOp> MlirHloBuilder::Compare(const Shape& shape, XlaOp lhs,
                                        XlaOp rhs,
                                        ComparisonDirection direction,
                                        Comparison::Type type) {
  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  auto op = builder_.create<mlir::mhlo::CompareOp>(
      loc_, ty, GetValue(lhs), GetValue(rhs),
      mlir::mhlo::ComparisonDirectionAttr::get(
          builder_.getContext(), mlir::mhlo::symbolizeComparisonDirection(
                                     ComparisonDirectionToString(direction))
                                     .getValue()),
      mlir::mhlo::ComparisonTypeAttr::get(
          builder_.getContext(),
          mlir::mhlo::symbolizeComparisonType(ComparisonTypeToString(type))
              .getValue()));
  return MakeXlaOp(op.getResult());
}

XlaOp MlirHloBuilder::BinaryOpNoBroadcast(HloOpcode binop, const Shape& shape,
                                          XlaOp lhs, XlaOp rhs) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    return CreateOp(GetMlirOpName(binop), shape, {lhs, rhs});
  });
}

StatusOr<XlaOp> MlirHloBuilder::AddOpWithShape(
    HloOpcode opcode, const Shape& shape, absl::Span<const XlaOp> operands) {
  return CreateOp(GetMlirOpName(opcode), shape,
                  llvm::makeArrayRef<XlaOp>(operands.data(), operands.size()));
}

XlaOp MlirHloBuilder::CreateToken() {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    return MakeXlaOp(builder_.create<mlir::mhlo::CreateTokenOp>(
        loc_, mlir::mhlo::TokenType::get(builder_.getContext())));
  });
}

StatusOr<XlaOp> MlirHloBuilder::TriangularSolveInternal(
    const Shape& shape, XlaOp a, XlaOp b, TriangularSolveOptions options) {
  TF_ASSIGN_OR_RETURN(
      mlir::Type result_ty,
      ConvertShapeToType<mlir::RankedTensorType>(shape, builder_));
  auto op = builder_.create<mlir::mhlo::TriangularSolveOp>(
      loc_, result_ty, GetValue(a), GetValue(b),
      builder_.getBoolAttr(options.left_side()),
      builder_.getBoolAttr(options.lower()),
      builder_.getBoolAttr(options.unit_diagonal()),
      mlir::mhlo::TransposeAttr::get(
          builder_.getContext(),
          ::mlir::mhlo::symbolizeTranspose(
              TriangularSolveOptions::Transpose_Name(options.transpose_a()))
              .getValue()));
  return MakeXlaOp(op);
}

StatusOr<XlaOp> MlirHloBuilder::CholeskyInternal(const Shape& shape, XlaOp a,
                                                 bool lower) {
  TF_ASSIGN_OR_RETURN(
      mlir::Type result_ty,
      ConvertShapeToType<mlir::RankedTensorType>(shape, builder_));
  auto op = builder_.create<mlir::mhlo::CholeskyOp>(
      loc_, result_ty, GetValue(a), builder_.getBoolAttr(lower));
  return MakeXlaOp(op);
}

StatusOr<XlaOp> MlirHloBuilder::InfeedWithTokenInternal(
    const Shape& infeed_instruction_shape, XlaOp token,
    const std::string& config) {
  TF_ASSIGN_OR_RETURN(mlir::Type result_type,
                      ConvertShapeToType<mlir::RankedTensorType>(
                          infeed_instruction_shape, builder_));
  llvm::SmallVector<mlir::Type> flattened_ret_types;
  HloFunctionImporter::FlattenTupleType(result_type, flattened_ret_types);

  mlir::ArrayAttr layout;
  auto op = builder_.create<mlir::mhlo::InfeedOp>(loc_, flattened_ret_types,
                                                  GetValue(token),
                                                  /*infeed_config=*/config,
                                                  /*layout=*/layout);

  llvm::SmallVector<mlir::Value> flattened_results = op->getResults();
  llvm::MutableArrayRef<mlir::Value> flattened_results_ref(flattened_results);
  auto result = HloFunctionImporter::CreateTupleValue(
      &builder_, loc_, flattened_results_ref, result_type);
  auto defining_tuple_op = result.getDefiningOp<mlir::mhlo::TupleOp>();
  return MakeXlaOp(defining_tuple_op);
}

StatusOr<XlaOp> MlirHloBuilder::OutfeedWithTokenInternal(
    XlaOp operand, XlaOp token, const Shape& shape_with_layout,
    const std::string& outfeed_config) {
  auto token_type = mlir::mhlo::TokenType::get(builder_.getContext());
  llvm::SmallVector<mlir::Value> flattened_operands;
  HloFunctionImporter::FlattenTupleValue(&builder_, loc_, GetValue(operand),
                                         flattened_operands);
  return MakeXlaOp(builder_.create<mlir::mhlo::OutfeedOp>(
      loc_, token_type, flattened_operands, GetValue(token), outfeed_config));
}

StatusOr<XlaOp> MlirHloBuilder::ConcatInDimInternal(
    const Shape& shape, absl::Span<const XlaOp> operands, int64_t dimension) {
  TF_ASSIGN_OR_RETURN(
      mlir::Type result_type,
      ConvertShapeToType<mlir::RankedTensorType>(shape, builder_));
  auto mlir_operands = GetValues(operands);
  return MakeXlaOp(builder_.create<mlir::mhlo::ConcatenateOp>(
      loc_, result_type, mlir_operands, builder_.getI64IntegerAttr(dimension)));
}

StatusOr<XlaOp> MlirHloBuilder::GetTupleElementInternal(const Shape& shape,
                                                        XlaOp tuple_data,
                                                        int64_t index) {
  TF_ASSIGN_OR_RETURN(
      mlir::Type result_type,
      ConvertShapeToType<mlir::RankedTensorType>(shape, builder_));
  return MakeXlaOp(builder_.create<mlir::mhlo::GetTupleElementOp>(
      loc_, result_type, GetValue(tuple_data),
      builder_.getI32IntegerAttr(index)));
}

StatusOr<XlaOp> MlirHloBuilder::SliceInternal(
    const Shape& shape, XlaOp operand, absl::Span<const int64_t> start_indices,
    absl::Span<const int64_t> limit_indices,
    absl::Span<const int64_t> strides) {
  return MakeXlaOp(builder_.create<mlir::mhlo::SliceOp>(
      loc_, GetValue(operand), GetI64ElementsAttr(start_indices, &builder_),
      GetI64ElementsAttr(limit_indices, &builder_),
      GetI64ElementsAttr(strides, &builder_)));
}

StatusOr<XlaOp> MlirHloBuilder::DynamicSliceInternal(
    const Shape& shape, XlaOp operand, absl::Span<const XlaOp> start_indices,
    absl::Span<const int64_t> slice_sizes) {
  TF_ASSIGN_OR_RETURN(
      mlir::Type result_ty,
      ConvertShapeToType<mlir::RankedTensorType>(shape, builder_));
  return MakeXlaOp(builder_.create<mlir::mhlo::DynamicSliceOp>(
      loc_, result_ty, GetValue(operand), GetValues(start_indices),
      GetI64ElementsAttr(slice_sizes, &builder_)));
}

StatusOr<XlaOp> MlirHloBuilder::DynamicUpdateSliceInternal(
    const Shape& shape, XlaOp operand, XlaOp update,
    absl::Span<const XlaOp> start_indices) {
  TF_ASSIGN_OR_RETURN(
      mlir::Type result_ty,
      ConvertShapeToType<mlir::RankedTensorType>(shape, builder_));
  return MakeXlaOp(builder_.create<mlir::mhlo::DynamicUpdateSliceOp>(
      loc_, result_ty, GetValue(operand), GetValue(update),
      GetValues(start_indices)));
}

StatusOr<XlaOp> MlirHloBuilder::PadInternal(
    const Shape& shape, XlaOp operand, XlaOp padding_value,
    const PaddingConfig& padding_config) {
  TF_ASSIGN_OR_RETURN(
      mlir::Type result_type,
      ConvertShapeToType<mlir::RankedTensorType>(shape, builder_));
  llvm::SmallVector<int64_t> low, high, internal;
  for (auto& dimension : padding_config.dimensions()) {
    low.push_back(dimension.edge_padding_low());
    high.push_back(dimension.edge_padding_high());
    internal.push_back(dimension.interior_padding());
  }
  return MakeXlaOp(builder_.create<mlir::mhlo::PadOp>(
      loc_, result_type, GetValue(operand), GetValue(padding_value),
      GetI64ElementsAttr(low, &builder_), GetI64ElementsAttr(high, &builder_),
      GetI64ElementsAttr(internal, &builder_)));
}

StatusOr<XlaOp> MlirHloBuilder::TupleInternal(
    const Shape& shape, absl::Span<const XlaOp> elements) {
  mlir::SmallVector<mlir::Value, 4> operands;
  for (auto& element : elements) {
    operands.push_back(GetValue(element));
  }
  return MakeXlaOp(builder_.create<mlir::mhlo::TupleOp>(loc_, operands));
}

StatusOr<XlaOp> MlirHloBuilder::CreateOp(
    const std::string& op_name, const Shape& shape,
    llvm::ArrayRef<XlaOp> operands,
    llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  llvm::SmallVector<mlir::Value, 4> operand_values;
  operand_values.reserve(operands.size());
  for (XlaOp xla_op : operands) {
    operand_values.push_back(GetValue(xla_op));
  }
  TF_ASSIGN_OR_RETURN(mlir::Type ty, ConvertShapeToType<mlir::RankedTensorType>(
                                         shape, builder_));
  mlir::OperationState state(loc_, op_name, operand_values, {ty}, attributes);
  mlir::Operation* op = builder_.create(state);
  return MakeXlaOp(op->getResult(0));
}

Status MlirHloBuilder::ImportComputation(const HloModuleProto& computation,
                                         mlir::Region* region,
                                         bool flatten_region_arg_tuple) {
  TF_ASSIGN_OR_RETURN(auto module_config,
                      xla::HloModule::CreateModuleConfigFromProto(
                          computation, xla::DebugOptions()));
  TF_ASSIGN_OR_RETURN(auto hlo_module, xla::HloModule::CreateFromProto(
                                           computation, module_config));

  return HloFunctionImporter::ImportAsRegion(*hlo_module->entry_computation(),
                                             region, &builder_,
                                             flatten_region_arg_tuple);
}

StatusOr<const Shape*> MlirHloBuilder::GetShapePtr(XlaOp op) const {
  TF_RETURN_IF_ERROR(first_error());
  TF_RETURN_IF_ERROR(CheckOpBuilder(op));
  auto it = handle_to_shape_.find(op.handle());
  if (it == handle_to_shape_.end()) {
    return InvalidArgument("No XlaOp with handle %d", op.handle());
  }
  return it->second.get();
}

}  // namespace xla
