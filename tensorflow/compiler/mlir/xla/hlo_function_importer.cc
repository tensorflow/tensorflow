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

#include "tensorflow/compiler/mlir/xla/hlo_function_importer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/BlockAndValueMapping.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Identifier.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/Region.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

using llvm::APInt;
using llvm::makeArrayRef;
using mlir::DenseElementsAttr;
using mlir::DenseIntElementsAttr;
using mlir::FuncOp;
using mlir::NamedAttribute;
using mlir::Operation;
using mlir::RankedTensorType;
using mlir::ShapedType;
using mlir::Type;
using mlir::Value;

namespace xla {

namespace {
// Note: This sanitization function causes an irreversible many-to-one mapping
// and any solution to mitigate this would cause issues with the reverse
// direction. Longterm solution is to add a function attribute to maintain the
// original HLO naming.
string SanitizeFunctionName(llvm::StringRef name) {
  string output = name;
  llvm::for_each(output, [](char& x) { x = x == '-' ? '_' : x; });
  return output;
}

StatusOr<DenseElementsAttr> CreateDenseAttrFromLiteral(ShapedType type,
                                                       const Literal& literal) {
#define DENSE_ELEMENT_ATTR_BUILDER(xla_type, cpp_type)                 \
  case xla_type: {                                                     \
    auto data_span = literal.data<cpp_type>();                         \
    return DenseElementsAttr::get(                                     \
        type, llvm::makeArrayRef(data_span.data(), data_span.size())); \
  }

  switch (literal.shape().element_type()) {
    DENSE_ELEMENT_ATTR_BUILDER(PrimitiveType::PRED, bool)
    DENSE_ELEMENT_ATTR_BUILDER(PrimitiveType::F32, float)
    DENSE_ELEMENT_ATTR_BUILDER(PrimitiveType::F64, double)
    DENSE_ELEMENT_ATTR_BUILDER(PrimitiveType::S8, int8)
    DENSE_ELEMENT_ATTR_BUILDER(PrimitiveType::S16, int16)
    DENSE_ELEMENT_ATTR_BUILDER(PrimitiveType::S32, int32)
    DENSE_ELEMENT_ATTR_BUILDER(PrimitiveType::S64, int64)
    // TODO(b/130356985): Update once MLIR supports unsigned integers.
    DENSE_ELEMENT_ATTR_BUILDER(PrimitiveType::U8, uint8)
    DENSE_ELEMENT_ATTR_BUILDER(PrimitiveType::U16, uint16)
    DENSE_ELEMENT_ATTR_BUILDER(PrimitiveType::U32, uint32)
    DENSE_ELEMENT_ATTR_BUILDER(PrimitiveType::U64, uint64)
    default:
      return tensorflow::errors::Internal(
          absl::StrCat("Unsupported type: ",
                       PrimitiveType_Name(literal.shape().element_type())));
  }
#undef DENSE_ELEMENT_ATTR_BUILDER
}

// Returns whether the instruction is a default dot operation.
bool DotIsDefault(const HloInstruction* instruction) {
  auto dnums = instruction->dot_dimension_numbers();
  DotDimensionNumbers default_dimension_numbers;
  default_dimension_numbers.add_lhs_contracting_dimensions(
      instruction->operand(0)->shape().dimensions_size() == 1 ? 0 : 1);
  default_dimension_numbers.add_rhs_contracting_dimensions(0);
  return xla::protobuf_util::ProtobufEquals(dnums, default_dimension_numbers);
}
}  // namespace

StatusOr<mlir::FuncOp> HloFunctionImporter::ImportFunction(
    mlir::ModuleOp module, mlir::Builder* builder,
    std::unordered_map<HloComputation*, FuncOp>* function_map,
    HloComputation* computation) {
  HloFunctionImporter importer(module, builder, function_map);
  return importer.ImportFunction(computation);
}

StatusOr<mlir::FuncOp> HloFunctionImporter::ImportFunction(
    HloComputation* computation) {
  auto& imported = (*function_map_)[computation];
  if (imported) return imported;

  llvm::SmallVector<Type, 4> args, rets;
  TF_RETURN_IF_ERROR(
      GetMlirTypes(computation->parameter_instructions(), &args));
  TF_RETURN_IF_ERROR(GetMlirTypes({computation->root_instruction()}, &rets));

  auto func_type = mlir::FunctionType::get(args, rets, context_);

  string computation_name =
      computation->parent()->entry_computation() == computation
          ? "main"
          : SanitizeFunctionName(computation->name());

  // Construct the MLIR function and map arguments.
  llvm::ArrayRef<mlir::NamedAttribute> attrs;
  auto function = mlir::FuncOp::create(mlir::UnknownLoc::get(context_),
                                       computation_name, func_type, attrs);
  module_.push_back(function);

  // Add to the map right away for function calls.
  imported = function;

  mlir::Block* block = function.addEntryBlock();
  TF_RETURN_IF_ERROR(ImportInstructions(computation, block));

  return function;
}

tensorflow::Status HloFunctionImporter::ImportComputation(
    HloComputation* computation, mlir::Region* region) {
  // TODO(hinsu): Store computation name as an attribute for round-trip.
  auto* block = new mlir::Block;
  region->push_back(block);

  llvm::SmallVector<Type, 4> args;
  TF_RETURN_IF_ERROR(
      GetMlirTypes(computation->parameter_instructions(), &args));
  block->addArguments(args);

  return ImportInstructions(computation, block);
}

tensorflow::Status HloFunctionImporter::ImportInstructions(
    HloComputation* computation, mlir::Block* block) {
  // Setup the input parameters.
  const int num_parameters = computation->num_parameters();
  for (int i = 0; i < num_parameters; i++) {
    auto hlo_parameter = computation->parameter_instruction(i);
    instruction_value_map_[hlo_parameter] = block->getArgument(i);
  }

  mlir::OpBuilder builder(block);
  for (auto instruction : computation->MakeInstructionPostOrder()) {
    TF_ASSIGN_OR_RETURN(auto new_operation,
                        ImportInstruction(instruction, &builder));
    if (new_operation) {
      instruction_value_map_[instruction] = new_operation->getResult(0);
    }
  }

  // TODO(suderman): Add location tracking details.
  mlir::Location loc = builder.getUnknownLoc();

  // Setup the return type (HLO only supports a single return value).
  TF_ASSIGN_OR_RETURN(auto result,
                      GetMlirValue(computation->root_instruction()));

  // Create terminator op depending on the parent op of this region.
  if (llvm::isa<FuncOp>(block->getParentOp())) {
    builder.create<mlir::ReturnOp>(loc, result);
  } else {
    builder.create<mlir::xla_hlo::ReturnOp>(loc, result);
  }
  return tensorflow::Status::OK();
}

StatusOr<mlir::Operation*> HloFunctionImporter::ImportInstruction(
    HloInstruction* instruction, mlir::OpBuilder* func_builder) {
  TF_ASSIGN_OR_RETURN(auto operands, GetOperands(instruction));
  TF_ASSIGN_OR_RETURN(auto result_type, ConvertType(instruction->shape()));
  llvm::SmallVector<NamedAttribute, 10> attributes = {builder_->getNamedAttr(
      "name", builder_->getStringAttr(instruction->name()))};
  mlir::Location loc = func_builder->getUnknownLoc();

  switch (instruction->opcode()) {
    case HloOpcode::kParameter: {
      return nullptr;
    }
    case HloOpcode::kConstant: {
      auto attr = CreateDenseAttrFromLiteral(
          result_type.cast<mlir::TensorType>(), instruction->literal());
      if (!attr.ok()) return attr.status();
      mlir::Operation* new_operation =
          func_builder->create<mlir::ConstantOp>(loc, attr.ValueOrDie());
      for (auto attr : attributes) {
        new_operation->setAttr(attr.first, attr.second);
      }
      return new_operation;
    }
    case HloOpcode::kIota: {
      return func_builder
          ->create<mlir::xla_hlo::IotaOp>(
              loc, result_type,
              func_builder->getI64IntegerAttr(
                  static_cast<HloIotaInstruction*>(instruction)
                      ->iota_dimension()))
          .getOperation();
    }
#define MakeAndReturn(mlir_op)                                              \
  {                                                                         \
    mlir::Operation* new_operation =                                        \
        func_builder->create<mlir::xla_hlo::mlir_op>(loc, result_type,      \
                                                     operands, attributes); \
    return new_operation;                                                   \
  }
    case HloOpcode::kBroadcast: {
      // Note that the HLO broadcast is more powerful than the XLA broadcast op.
      // BroadcastInDim offers a superset of the HLO op's functionality.
      if (!instruction->dimensions().empty()) {
        attributes.push_back(builder_->getNamedAttr(
            "broadcast_dimensions",
            ConvertDimensions(instruction->dimensions())));
      }
      MakeAndReturn(BroadcastInDimOp);
    }
    case HloOpcode::kDot: {
      attributes.push_back(ConvertPrecisionConfig(instruction));

      // Consider consolidating DotOps together.
      if (DotIsDefault(instruction)) {
        MakeAndReturn(DotOp);
      }

      attributes.push_back(
          ConvertDotDimensionNumbers(instruction->dot_dimension_numbers()));
      MakeAndReturn(DotGeneralOp);
    }
    case HloOpcode::kCall: {
      TF_ASSIGN_OR_RETURN(FuncOp function,
                          ImportFunction(instruction->to_apply()));
      mlir::Operation* new_operation =
          func_builder->create<mlir::CallOp>(loc, function, operands);
      return new_operation;
    }
    case HloOpcode::kCompare: {
      attributes.push_back(ConvertComparisonDirection(instruction));
      MakeAndReturn(CompareOp);
    }
    case HloOpcode::kGather: {
      auto gather_instruction = static_cast<HloGatherInstruction*>(instruction);
      attributes.push_back(ConvertGatherDimensionNumbers(
          gather_instruction->gather_dimension_numbers()));

      std::vector<int64_t> slice_sizes(
          gather_instruction->gather_slice_sizes().begin(),
          gather_instruction->gather_slice_sizes().end());
      attributes.push_back(
          builder_->getNamedAttr("slice_sizes", Convert(slice_sizes)));
      attributes.push_back(builder_->getNamedAttr(
          "indices_are_sorted",
          builder_->getBoolAttr(gather_instruction->indices_are_sorted())));

      MakeAndReturn(GatherOp);
    }
    case HloOpcode::kDynamicUpdateSlice: {
      return func_builder
          ->create<mlir::xla_hlo::DynamicUpdateSliceOp>(
              loc, result_type, operands[0], operands[1],
              llvm::ArrayRef<Value*>(operands.begin() + 2, operands.end()))
          .getOperation();
    }
    case HloOpcode::kInfeed: {
      attributes.push_back(builder_->getNamedAttr(
          "infeed_config", mlir::StringAttr::get(instruction->infeed_config(),
                                                 builder_->getContext())));
      MakeAndReturn(InfeedOp);
    }
    case HloOpcode::kOutfeed: {
      attributes.push_back(builder_->getNamedAttr(
          "outfeed_config", mlir::StringAttr::get(instruction->outfeed_config(),
                                                  builder_->getContext())));
      MakeAndReturn(OutfeedOp);
    }
    case HloOpcode::kPad: {
      const auto& padding_config = instruction->padding_config();
      llvm::SmallVector<int64_t, 4> edge_padding_low;
      llvm::SmallVector<int64_t, 4> edge_padding_high;
      llvm::SmallVector<int64_t, 4> interior_padding;
      edge_padding_low.reserve(padding_config.dimensions_size());
      edge_padding_high.reserve(padding_config.dimensions_size());
      interior_padding.reserve(padding_config.dimensions_size());

      for (const auto& dimension : padding_config.dimensions()) {
        edge_padding_low.push_back(dimension.edge_padding_low());
        edge_padding_high.push_back(dimension.edge_padding_high());
        interior_padding.push_back(dimension.interior_padding());
      }

      return func_builder
          ->create<mlir::xla_hlo::PadOp>(loc, result_type, operands[0],
                                         operands[1], Convert(edge_padding_low),
                                         Convert(edge_padding_high),
                                         Convert(interior_padding))
          .getOperation();
    }
    case HloOpcode::kSlice: {
      return func_builder
          ->create<mlir::xla_hlo::SliceOp>(
              loc, result_type, operands[0],
              ConvertDimensions(instruction->slice_starts()),
              ConvertDimensions(instruction->slice_limits()),
              ConvertDimensions(instruction->slice_strides()))
          .getOperation();
    }
    case HloOpcode::kConditional: {
      llvm::SmallVector<Type, 4> rets;
      TF_RETURN_IF_ERROR(GetMlirTypes(
          {instruction->true_computation()->root_instruction()}, &rets));

      auto op = func_builder->create<mlir::xla_hlo::ConditionalOp>(
          loc, rets, operands, attributes);
      TF_RETURN_IF_ERROR(ImportComputation(instruction->true_computation(),
                                           &op.true_branch()));
      TF_RETURN_IF_ERROR(ImportComputation(instruction->false_computation(),
                                           &op.false_branch()));
      return op.getOperation();
    }
    case HloOpcode::kConcatenate: {
      // TODO(b/132057942): Support taking an uint64_t instead of an IntegerAttr
      // for concatenate dimension.
      return func_builder
          ->create<mlir::xla_hlo::ConcatenateOp>(
              loc, result_type, operands,
              builder_->getI64IntegerAttr(instruction->concatenate_dimension()))
          .getOperation();
    }
    case HloOpcode::kReduce: {
      // Operands in the first half are reduction inputs and the remaining
      // operands are corresponding initial values.
      size_t num_inputs = operands.size() / 2;
      auto reduce = func_builder->create<mlir::xla_hlo::ReduceOp>(
          loc, result_type, llvm::makeArrayRef(operands).take_front(num_inputs),
          llvm::makeArrayRef(operands).drop_front(num_inputs),
          ConvertDimensions(instruction->dimensions()));
      TF_RETURN_IF_ERROR(
          ImportComputation(instruction->to_apply(), &reduce.body()));
      return reduce.getOperation();
    }
    case HloOpcode::kReverse: {
      return func_builder
          ->create<mlir::xla_hlo::ReverseOp>(
              loc, result_type, operands[0],
              ConvertDimensions(instruction->dimensions()))
          .getOperation();
    }
    case HloOpcode::kRng: {
      auto shape = func_builder->create<mlir::ConstantOp>(
          loc, Convert(result_type.cast<RankedTensorType>().getShape()));
      switch (instruction->random_distribution()) {
        case xla::RNG_UNIFORM:
          return func_builder
              ->create<mlir::xla_hlo::RngUniformOp>(
                  loc, result_type, operands[0], operands[1], shape)
              .getOperation();

        case xla::RNG_NORMAL:
          return func_builder
              ->create<mlir::xla_hlo::RngNormalOp>(
                  loc, result_type, operands[0], operands[1], shape)
              .getOperation();

        default:
          return tensorflow::errors::InvalidArgument(absl::StrCat(
              "Unsupported distribution: ",
              RandomDistributionToString(instruction->random_distribution())));
      }
    }
    case HloOpcode::kWhile: {
      auto op = func_builder->create<mlir::xla_hlo::WhileOp>(
          loc, operands[0]->getType(), operands[0]);
      TF_RETURN_IF_ERROR(
          ImportComputation(instruction->while_condition(), &op.cond()));
      TF_RETURN_IF_ERROR(
          ImportComputation(instruction->while_body(), &op.body()));
      return op.getOperation();
    }
    case HloOpcode::kGetTupleElement: {
      attributes.push_back(builder_->getNamedAttr(
          "index", builder_->getIntegerAttr(builder_->getIntegerType(32),
                                            instruction->tuple_index())));
      MakeAndReturn(GetTupleElementOp);
    };
    case HloOpcode::kGetDimensionSize: {
      attributes.push_back(builder_->getNamedAttr(
          "dimension", builder_->getIntegerAttr(builder_->getIntegerType(32),
                                                instruction->dimension())));
      MakeAndReturn(GetDimensionSizeOp);
    };
    case HloOpcode::kTranspose: {
      attributes.push_back(builder_->getNamedAttr(
          "permutation", ConvertDimensions(instruction->dimensions())));
      MakeAndReturn(TransposeOp);
    }
    case HloOpcode::kConvolution: {
      llvm::SmallVector<int64_t, 4> strides, lhs_dilations, rhs_dilations;
      llvm::SmallVector<int64_t, 8> paddings;
      for (const auto& dim : instruction->window().dimensions()) {
        strides.push_back(dim.stride());
        lhs_dilations.push_back(dim.base_dilation());
        rhs_dilations.push_back(dim.window_dilation());
        paddings.push_back(dim.padding_low());
        paddings.push_back(dim.padding_high());
      }

      attributes.push_back(
          builder_->getNamedAttr("window_strides", Convert(strides)));
      attributes.push_back(ConvertPadding(paddings));
      attributes.push_back(
          builder_->getNamedAttr("lhs_dilations", Convert(lhs_dilations)));
      attributes.push_back(
          builder_->getNamedAttr("rhs_dilations", Convert(rhs_dilations)));
      attributes.push_back(ConvertConvDimensionNumbers(
          instruction->convolution_dimension_numbers()));
      attributes.push_back(builder_->getNamedAttr(
          "feature_group_count",
          builder_->getI64IntegerAttr(instruction->feature_group_count())));
      attributes.push_back(builder_->getNamedAttr(
          "batch_group_count",
          builder_->getI64IntegerAttr(instruction->batch_group_count())));
      attributes.push_back(ConvertPrecisionConfig(instruction));
      MakeAndReturn(ConvOp);
    }

    case HloOpcode::kFft: {
      auto fft_type =
          builder_->getStringAttr(FftType_Name(instruction->fft_type()));

      std::vector<int64_t> fft_length(instruction->fft_length().begin(),
                                      instruction->fft_length().end());

      attributes.push_back(builder_->getNamedAttr("fft_type", fft_type));
      attributes.push_back(
          builder_->getNamedAttr("fft_length", Convert(fft_length)));
      MakeAndReturn(FftOp);
    }
#define NoAttributeCase(hlo_op_code, mlir_op) \
  case HloOpcode::hlo_op_code: {              \
    MakeAndReturn(mlir_op);                   \
  }

      // broadcast dimensions are never added here because they don't exist as
      // part of the HLO instruction. They are only a convenience in the XLA
      // builder API.
      NoAttributeCase(kAdd, AddOp);
      NoAttributeCase(kAfterAll, AfterAllOp);
      NoAttributeCase(kAnd, AndOp);
      NoAttributeCase(kAtan2, Atan2Op);
      NoAttributeCase(kBitcastConvert, BitcastConvertOp);
      NoAttributeCase(kConvert, ConvertOp);
      NoAttributeCase(kClamp, ClampOp);
      NoAttributeCase(kComplex, ComplexOp);
      NoAttributeCase(kCos, CosOp);
      NoAttributeCase(kDivide, DivOp);
      NoAttributeCase(kExp, ExpOp);
      NoAttributeCase(kExpm1, Expm1Op);
      NoAttributeCase(kFloor, FloorOp);
      NoAttributeCase(kImag, ImagOp);
      NoAttributeCase(kLog, LogOp);
      NoAttributeCase(kLog1p, Log1pOp);
      NoAttributeCase(kMaximum, MaxOp);
      NoAttributeCase(kMinimum, MinOp);
      NoAttributeCase(kMultiply, MulOp);
      NoAttributeCase(kNegate, NegOp);
      NoAttributeCase(kNot, NotOp);
      NoAttributeCase(kOr, OrOp);
      NoAttributeCase(kPopulationCount, PopulationCountOp);
      NoAttributeCase(kPower, PowOp);
      NoAttributeCase(kReal, RealOp);
      NoAttributeCase(kRemainder, RemOp);
      NoAttributeCase(kReplicaId, ReplicaIdOp);
      // The dimensions attribute is not present on the HLO Reshape instruction.
      // If dimensions are non-default, the XLA builder implements it as a
      // separate transpose.
      NoAttributeCase(kReshape, ReshapeOp);
      NoAttributeCase(kRsqrt, RsqrtOp);
      NoAttributeCase(kSelect, SelectOp);
      NoAttributeCase(kShiftLeft, ShiftLeftOp);
      NoAttributeCase(kShiftRightArithmetic, ShiftRightArithmeticOp);
      NoAttributeCase(kShiftRightLogical, ShiftRightLogicalOp);
      NoAttributeCase(kSin, SinOp);
      NoAttributeCase(kSubtract, SubOp);
      NoAttributeCase(kTanh, TanhOp);
      NoAttributeCase(kTuple, TupleOp);
      NoAttributeCase(kXor, XorOp);
      // TODO(b/129422361) Copy needs special handling because it is not defined
      // in tensorflow/compiler/xla/client/xla_builder.h.
      // See operation semantics in
      // g3doc/platforms/xla/g3doc/internal/hlo_semantics#copy
      NoAttributeCase(kCopy, CopyOp);
#undef NoAttributeCase
#undef MakeAndReturn
    case HloOpcode::kAddDependency:
      // Arbitrary op code that I suspect we will not implement for quite a
      // while and allows testing handling of unknown ops. Selected because it
      // is not mentioned in xla client anywhere or in the hlo of our sample
      // models.
    default: {
      mlir::OperationState result(loc, "xla_hlo.unknown");
      result.addOperands(operands);
      result.addTypes(result_type);
      for (auto attr : attributes) {
        result.attributes.push_back(attr);
      }

      return func_builder->createOperation(result);
    }
  }
}

StatusOr<llvm::SmallVector<mlir::Value*, 4>> HloFunctionImporter::GetOperands(
    HloInstruction* instruction) {
  llvm::SmallVector<mlir::Value*, 4> operands;
  for (const auto& operand : instruction->operands()) {
    auto input_it = instruction_value_map_.find(operand);
    if (input_it == instruction_value_map_.end()) {
      return tensorflow::errors::Internal(
          absl::StrCat("Could not find input value: ", operand->name(),
                       " for instruction ", instruction->name()));
    }
    operands.push_back(input_it->second);
  }
  return operands;
}

// TODO(suderman): Move to a general library when needed in other places.
StatusOr<mlir::RankedTensorType> HloFunctionImporter::ConvertTensorType(
    const Shape& shape) {
  auto type = shape.element_type();

  llvm::SmallVector<int64_t, 4> array;
  array.reserve(shape.dimensions_size());
  for (auto val : shape.dimensions()) {
    array.push_back(val);
  }

  switch (type) {
    case PrimitiveType::PRED:
      return mlir::RankedTensorType::get(array, builder_->getI1Type());
    case PrimitiveType::F16:
      return mlir::RankedTensorType::get(array, builder_->getF16Type());
    case PrimitiveType::F32:
      return mlir::RankedTensorType::get(array, builder_->getF32Type());
    case PrimitiveType::F64:
      return mlir::RankedTensorType::get(array, builder_->getF64Type());
    case PrimitiveType::S8:
      return mlir::RankedTensorType::get(array, builder_->getIntegerType(8));
    case PrimitiveType::S16:
      return mlir::RankedTensorType::get(array, builder_->getIntegerType(16));
    case PrimitiveType::S32:
      return mlir::RankedTensorType::get(array, builder_->getIntegerType(32));
    case PrimitiveType::S64:
      return mlir::RankedTensorType::get(array, builder_->getIntegerType(64));
    // TODO(b/130356985): Update once MLIR supports unsigned integers.
    case PrimitiveType::U8:
      return mlir::RankedTensorType::get(array, builder_->getIntegerType(8));
    case PrimitiveType::U16:
      return mlir::RankedTensorType::get(array, builder_->getIntegerType(16));
    case PrimitiveType::U32:
      return mlir::RankedTensorType::get(array, builder_->getIntegerType(32));
    case PrimitiveType::U64:
      return mlir::RankedTensorType::get(array, builder_->getIntegerType(64));
    case PrimitiveType::C64:
      return mlir::RankedTensorType::get(
          array, mlir::ComplexType::get(builder_->getF32Type()));
    default:
      return tensorflow::errors::Internal(
          absl::StrCat("Unsupported type: ", PrimitiveType_Name(type)));
  }
}

StatusOr<mlir::Type> HloFunctionImporter::ConvertType(const Shape& shape) {
  if (shape.IsToken()) {
    return mlir::xla_hlo::TokenType::get(builder_->getContext());
  }
  if (shape.IsTuple()) {
    mlir::Type mlir_type;
    llvm::SmallVector<mlir::Type, 4> contents;
    contents.reserve(shape.tuple_shapes_size());
    for (const auto& subtype : shape.tuple_shapes()) {
      TF_ASSIGN_OR_RETURN(auto mlir_subtype, ConvertType(subtype));
      contents.push_back(mlir_subtype);
    }

    return builder_->getTupleType(contents);
  }

  return ConvertTensorType(shape);
}

tensorflow::Status HloFunctionImporter::GetMlirTypes(
    const std::vector<HloInstruction*>& instructions,
    llvm::SmallVectorImpl<mlir::Type>* types) {
  for (auto instruction : instructions) {
    TF_ASSIGN_OR_RETURN(auto ret_type, ConvertType(instruction->shape()));
    types->push_back(ret_type);
  }
  return tensorflow::Status::OK();
}

StatusOr<Value*> HloFunctionImporter::GetMlirValue(
    HloInstruction* instruction) {
  auto lookup = instruction_value_map_.find(instruction);
  if (lookup != instruction_value_map_.end()) {
    return lookup->second;
  }

  return tensorflow::errors::Internal(absl::StrCat(
      "Unable to find value for input: ", instruction->ToString()));
}

mlir::NamedAttribute HloFunctionImporter::ConvertPrecisionConfig(
    HloInstruction* instruction) {
  // TODO(b/129709049) The HLO text format elides this in the all DEFAULT
  // case and the parser sticks it in. Maybe we should too.
  llvm::SmallVector<mlir::Attribute, 4> operand_precision_attrs;

  for (auto prec : instruction->precision_config().operand_precision()) {
    operand_precision_attrs.push_back(
        builder_->getStringAttr(PrecisionConfig_Precision_Name(prec)));
  }
  return builder_->getNamedAttr(
      "precision_config", builder_->getArrayAttr(operand_precision_attrs));
}

mlir::NamedAttribute HloFunctionImporter::ConvertComparisonDirection(
    HloInstruction* instruction) {
  return builder_->getNamedAttr(
      "comparison_direction",
      builder_->getStringAttr(
          ComparisonDirectionToString(instruction->comparison_direction())));
}

mlir::DenseIntElementsAttr HloFunctionImporter::ConvertDimensions(
    llvm::ArrayRef<int64> op_dimensions) {
  llvm::SmallVector<APInt, 8> dimensions;
  dimensions.reserve(op_dimensions.size());
  for (auto value : op_dimensions) dimensions.emplace_back(APInt(64, value));

  return DenseIntElementsAttr::get(
      RankedTensorType::get(dimensions.size(), builder_->getIntegerType(64)),
      dimensions);
}

mlir::DenseIntElementsAttr HloFunctionImporter::Convert(
    llvm::ArrayRef<int64_t> op_dimensions) {
  return DenseIntElementsAttr::get(
      RankedTensorType::get(op_dimensions.size(), builder_->getIntegerType(64)),
      op_dimensions);
}

mlir::NamedAttribute HloFunctionImporter::ConvertPadding(
    llvm::ArrayRef<int64_t> padding) {
  auto ty =
      mlir::RankedTensorType::get({2, static_cast<int64_t>(padding.size()) / 2},
                                  builder_->getIntegerType(64));
  auto attr = DenseIntElementsAttr::get(ty, padding);
  return builder_->getNamedAttr("padding", attr);
}

mlir::NamedAttribute HloFunctionImporter::ConvertDotDimensionNumbers(
    const DotDimensionNumbers& dnums) {
  std::vector<int64_t> rhs_contracting_dimensions(
      dnums.rhs_contracting_dimensions().begin(),
      dnums.rhs_contracting_dimensions().end());
  std::vector<int64_t> lhs_contracting_dimensions(
      dnums.lhs_contracting_dimensions().begin(),
      dnums.lhs_contracting_dimensions().end());
  std::vector<int64_t> rhs_batch_dimensions(
      dnums.rhs_batch_dimensions().begin(), dnums.rhs_batch_dimensions().end());
  std::vector<int64_t> lhs_batch_dimensions(
      dnums.lhs_batch_dimensions().begin(), dnums.lhs_batch_dimensions().end());

  // Push the attributes into our new DictionaryAttr.
  auto lhs_batch_dims_attr = Convert(lhs_batch_dimensions);
  auto rhs_batch_dims_attr = Convert(rhs_batch_dimensions);
  auto lhs_contracting_dims_attr = Convert(lhs_contracting_dimensions);
  auto rhs_contracting_dims_attr = Convert(rhs_contracting_dimensions);

  auto attr = mlir::xla_hlo::DotDimensionNumbers::get(
      lhs_batch_dims_attr, rhs_batch_dims_attr, lhs_contracting_dims_attr,
      rhs_contracting_dims_attr, context_);
  return builder_->getNamedAttr("dot_dimension_numbers", attr);
}

mlir::NamedAttribute HloFunctionImporter::ConvertConvDimensionNumbers(
    const xla::ConvolutionDimensionNumbers& dnums) {
  llvm::SmallVector<int64_t, 4> input_spatial_dims(
      dnums.input_spatial_dimensions().begin(),
      dnums.input_spatial_dimensions().end());
  llvm::SmallVector<int64_t, 4> kernel_spatial_dims(
      dnums.kernel_spatial_dimensions().begin(),
      dnums.kernel_spatial_dimensions().end());
  llvm::SmallVector<int64_t, 4> output_spatial_dims(
      dnums.output_spatial_dimensions().begin(),
      dnums.output_spatial_dimensions().end());
  auto attr = mlir::xla_hlo::ConvDimensionNumbers::get(
      builder_->getI64IntegerAttr(dnums.input_batch_dimension()),
      builder_->getI64IntegerAttr(dnums.input_feature_dimension()),
      Convert(input_spatial_dims),
      builder_->getI64IntegerAttr(dnums.kernel_input_feature_dimension()),
      builder_->getI64IntegerAttr(dnums.kernel_output_feature_dimension()),
      Convert(kernel_spatial_dims),
      builder_->getI64IntegerAttr(dnums.output_batch_dimension()),
      builder_->getI64IntegerAttr(dnums.kernel_output_feature_dimension()),
      Convert(output_spatial_dims), context_);
  return builder_->getNamedAttr("dimension_numbers", attr);
}

mlir::NamedAttribute HloFunctionImporter::ConvertGatherDimensionNumbers(
    const xla::GatherDimensionNumbers& dnums) {
  std::vector<int64_t> offset_dims(dnums.offset_dims().begin(),
                                   dnums.offset_dims().end());
  std::vector<int64_t> collapsed_slice_dims(
      dnums.collapsed_slice_dims().begin(), dnums.collapsed_slice_dims().end());
  std::vector<int64_t> start_index_map(dnums.start_index_map().begin(),
                                       dnums.start_index_map().end());
  auto attr = mlir::xla_hlo::GatherDimensionNumbers::get(
      Convert(offset_dims), Convert(collapsed_slice_dims),
      Convert(start_index_map),
      builder_->getI64IntegerAttr(dnums.index_vector_dim()), context_);
  return builder_->getNamedAttr("dimension_numbers", attr);
}

}  // namespace xla
