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

#include "tensorflow/compiler/mlir/xla/mlir_hlo_to_hlo.h"

#include <memory>

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Matchers.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/TypeUtilities.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

using ::tensorflow::int16;
using ::tensorflow::int32;
using ::tensorflow::int64;
using ::tensorflow::int8;
using ::tensorflow::uint16;
using ::tensorflow::uint32;
using ::tensorflow::uint64;
using ::tensorflow::uint8;

static std::vector<int64> ConvertDenseIntAttr(mlir::DenseIntElementsAttr attr) {
  auto values = attr.getValues<int64>();
  return {values.begin(), values.end()};
}

static std::vector<int64> ConvertDenseIntAttr(
    llvm::Optional<mlir::DenseIntElementsAttr> attr) {
  if (!attr) return {};
  return ConvertDenseIntAttr(*attr);
}

// Converts the broadcast_dimensions attribute into a vector of dimension
// numbers (empty if the attribute is absent).
static std::vector<int64> Convert_broadcast_dimensions(
    llvm::Optional<mlir::DenseIntElementsAttr> broadcast_dimensions) {
  if (!broadcast_dimensions.hasValue()) return {};

  return ConvertDenseIntAttr(*broadcast_dimensions);
}

// Convert a nx2 dense attribute to a list of tuples. This is the way padding
// is defined in hlo.
static std::vector<std::pair<int64, int64>> Convert_padding(
    llvm::Optional<mlir::DenseIntElementsAttr> padding_optional) {
  if (!padding_optional.hasValue()) return {};
  mlir::DenseIntElementsAttr padding = *padding_optional;
  auto it = padding.getValues<int64>().begin();
  std::vector<std::pair<int64, int64>> out(padding.getNumElements() / 2);
  for (auto& item : out) {
    int64 left_pad = *it;
    ++it;
    int64 right_pad = *it;
    ++it;
    item = {left_pad, right_pad};
  }

  return out;
}

static std::vector<xla::ReplicaGroup> Convert_replica_groups(
    mlir::DenseIntElementsAttr groups) {
  int64_t num_groups = groups.getType().getDimSize(0);
  int64_t group_size = groups.getType().getDimSize(1);

  std::vector<xla::ReplicaGroup> result;
  result.reserve(num_groups);
  for (uint64_t i = 0; i < num_groups; ++i) {
    xla::ReplicaGroup group;
    for (uint64_t j = 0; j < group_size; ++j) {
      group.add_replica_ids(groups.getValue<int64_t>({i, j}));
    }
    result.push_back(group);
  }
  return result;
}

#define I64_ELEMENTS_ATTR_TO_VECTOR(attribute)   \
  static std::vector<int64> Convert_##attribute( \
      mlir::DenseIntElementsAttr attribute) {    \
    return ConvertDenseIntAttr(attribute);       \
  }

I64_ELEMENTS_ATTR_TO_VECTOR(broadcast_sizes);
I64_ELEMENTS_ATTR_TO_VECTOR(permutation);
I64_ELEMENTS_ATTR_TO_VECTOR(start_indices);
I64_ELEMENTS_ATTR_TO_VECTOR(limit_indices);
I64_ELEMENTS_ATTR_TO_VECTOR(strides);

#undef I64_ELEMENTS_ATTR_TO_VECTOR

static std::vector<int64> Convert_ArrayRef(llvm::ArrayRef<int64_t> values) {
  return {values.begin(), values.end()};
}

// Converts the precision config array of strings attribute into the
// corresponding XLA proto. All the strings are assumed to be valid names of the
// Precision enum. This should have been checked in the op verify method.
static std::unique_ptr<xla::PrecisionConfig> Convert_precision_config(
    llvm::Optional<mlir::ArrayAttr> optional_precision_config_attr) {
  if (!optional_precision_config_attr.hasValue()) return nullptr;

  auto precision_config = absl::make_unique<xla::PrecisionConfig>();
  for (auto attr : optional_precision_config_attr.getValue()) {
    xla::PrecisionConfig::Precision p;
    auto operand_precision = attr.cast<mlir::StringAttr>().getValue().str();
    // TODO(jpienaar): Update this to ensure this is captured by verify.
    if (xla::PrecisionConfig::Precision_Parse(operand_precision, &p)) {
      precision_config->add_operand_precision(p);
    } else {
      auto* context = attr.getContext();
      mlir::emitError(mlir::UnknownLoc::get(context))
          << "unexpected operand precision " << operand_precision;
      return nullptr;
    }
  }

  return precision_config;
}

static xla::DotDimensionNumbers Convert_dot_dimension_numbers(
    mlir::xla_hlo::DotDimensionNumbers dot_dimension_numbers_attr) {
  xla::DotDimensionNumbers dot_dimension_numbers;

  auto rhs_contracting_dimensions =
      dot_dimension_numbers_attr.rhs_contracting_dimensions()
          .cast<mlir::DenseIntElementsAttr>();
  auto lhs_contracting_dimensions =
      dot_dimension_numbers_attr.lhs_contracting_dimensions()
          .cast<mlir::DenseIntElementsAttr>();
  auto rhs_batch_dimensions =
      dot_dimension_numbers_attr.rhs_batching_dimensions()
          .cast<mlir::DenseIntElementsAttr>();
  auto lhs_batch_dimensions =
      dot_dimension_numbers_attr.lhs_batching_dimensions()
          .cast<mlir::DenseIntElementsAttr>();

  for (auto val : rhs_contracting_dimensions) {
    dot_dimension_numbers.add_rhs_contracting_dimensions(val.getSExtValue());
  }
  for (auto val : lhs_contracting_dimensions) {
    dot_dimension_numbers.add_lhs_contracting_dimensions(val.getSExtValue());
  }

  for (auto val : rhs_batch_dimensions) {
    dot_dimension_numbers.add_rhs_batch_dimensions(val.getSExtValue());
  }

  for (auto val : lhs_batch_dimensions) {
    dot_dimension_numbers.add_lhs_batch_dimensions(val.getSExtValue());
  }

  return dot_dimension_numbers;
}

static xla::ConvolutionDimensionNumbers Convert_convolution_dimension_numbers(
    mlir::xla_hlo::ConvDimensionNumbers input) {
  xla::ConvolutionDimensionNumbers output;

  output.set_input_batch_dimension(
      input.input_batch_dimension().getValue().getSExtValue());
  output.set_input_feature_dimension(
      input.input_feature_dimension().getValue().getSExtValue());

  for (int64 v : input.input_spatial_dimensions().getValues<int64>()) {
    output.add_input_spatial_dimensions(v);
  }

  output.set_kernel_input_feature_dimension(
      input.kernel_input_feature_dimension().getValue().getSExtValue());
  output.set_kernel_output_feature_dimension(
      input.kernel_output_feature_dimension().getValue().getSExtValue());

  for (int64 v : input.kernel_spatial_dimensions().getValues<int64>()) {
    output.add_kernel_spatial_dimensions(v);
  }

  output.set_output_batch_dimension(
      input.output_batch_dimension().getValue().getSExtValue());
  output.set_output_feature_dimension(
      input.output_feature_dimension().getValue().getSExtValue());

  for (int64 v : input.output_spatial_dimensions().getValues<int64>()) {
    output.add_output_spatial_dimensions(v);
  }

  return output;
}

// Converts the comparison_direction string attribute into the XLA enum. The
// string is assumed to correspond to exactly one of the allowed strings
// representing the enum. This should have been checked in the op verify method.
static xla::ComparisonDirection Convert_comparison_direction(
    llvm::StringRef comparison_direction_string) {
  return xla::StringToComparisonDirection(comparison_direction_string.str())
      .ValueOrDie();
}

// Passes through everything except for unique_ptr, on which it calls get().
// This exists to allow the generated code to call XLA functions that take a raw
// pointer. In particular, PrecisionConfig is passed to xla::Dot and xla::Conv
// as a pointer and there is otherwise no way to avoid a memory leak.
template <typename T>
T Unwrap(T t) {
  return t;
}

template <typename T>
T* Unwrap(const std::unique_ptr<T>& t) {
  return t.get();
}

// Convert APInt into an int.
// TODO(hpucha): This should be consolidated into a general place.
static int ConvertAPInt(llvm::APInt i) { return i.getSExtValue(); }

// Convert APFloat to double.
static double ConvertAPFloat(llvm::APFloat value) {
  const auto& semantics = value.getSemantics();
  bool losesInfo = false;
  if (&semantics != &llvm::APFloat::IEEEdouble())
    value.convert(llvm::APFloat::IEEEdouble(),
                  llvm::APFloat::rmNearestTiesToEven, &losesInfo);
  return value.convertToDouble();
}

namespace mlir {
namespace {
class ConvertToHloModule {
 public:
  using ValueLoweringMap = llvm::DenseMap<Value*, xla::XlaOp>;
  using FunctionLoweringMap = llvm::DenseMap<mlir::FuncOp, xla::XlaComputation>;

  // If use_tuple_args is true, then the entry function's arguments are
  // converted to a tuple and passed as a single parameter.
  // Similarly, if return tuple is true, then the entry function's return values
  // are converted to a tuple even when there is only a single return value.
  // Multiple return values are always converted to a tuple and returned as a
  // single value.
  explicit ConvertToHloModule(mlir::ModuleOp module, bool use_tuple_args,
                              bool return_tuple)
      : module_(module),
        module_builder_("main"),
        use_tuple_args_(use_tuple_args),
        return_tuple_(return_tuple) {}

  // Perform the lowering to XLA. This function returns failure if an error was
  // encountered.
  //
  // TODO(hinsu): Check for dynamic shapes and exit instead of crashing.
  LogicalResult Run() {
    for (auto func : module_.getOps<FuncOp>()) {
      if (func.empty()) continue;
      if (failed(RunOnFunction(func))) return failure();
    }
    return success();
  }

  // Lower a specific function to HLO.
  LogicalResult RunOnFunction(mlir::FuncOp f);

  // Lower a `mlir::Region` to a `XlaComputation`
  LogicalResult LowerRegionAsComputation(mlir::Region* region,
                                         xla::XlaComputation* func);

  // Lower a single `Block` to a `XlaComputation`
  LogicalResult LowerBasicBlockAsFunction(Block* block,
                                          xla::XlaBuilder* builder,
                                          bool is_entry_function,
                                          xla::XlaComputation* result);

  ::xla::HloModuleProto ConsumeMainProto() {
    return lowered_computation_[module_.lookupSymbol<mlir::FuncOp>("main")]
        .proto();
  }

  // Lower function call to HLO call instruction
  LogicalResult LowerFunctionCall(
      mlir::CallOp* call_op, xla::XlaBuilder* builder,
      ConvertToHloModule::ValueLoweringMap* value_lowering);

 private:
  LogicalResult Lower(mlir::Operation* inst, bool is_entry_function,
                      xla::XlaBuilder* builder,
                      ConvertToHloModule::ValueLoweringMap* value_lowering,
                      xla::XlaComputation* result);

  // The module being lowered.
  mlir::ModuleOp module_;

  // The top-level XlaBuilder.
  xla::XlaBuilder module_builder_;

  // Map between function and lowered computation.
  FunctionLoweringMap lowered_computation_;

  // Whether the entry function should take a single tuple as input.
  bool use_tuple_args_;

  // Whether to always return a tuple.
  bool return_tuple_;

  // Unique suffix to give to the name of the next lowered region.
  size_t region_id_ = 0;
};

}  // namespace
}  // namespace mlir

namespace {

struct OpLoweringContext {
  llvm::DenseMap<mlir::Value*, xla::XlaOp>* values;
  mlir::ConvertToHloModule* converter;
  xla::XlaBuilder* builder;
};

llvm::SmallVector<xla::XlaOp, 4> GetTuple(mlir::Operation::operand_range values,
                                          OpLoweringContext ctx) {
  llvm::SmallVector<xla::XlaOp, 4> ops;
  for (mlir::Value* value : values) {
    ops.push_back((*ctx.values)[value]);
  }
  return ops;
}

}  // namespace

namespace mlir {
namespace xla_hlo {
namespace {

LogicalResult ExportXlaOp(BroadcastInDimOp op, OpLoweringContext ctx) {
  auto type = op.getType().dyn_cast<RankedTensorType>();
  if (!type) return failure();
  auto& value_map = *ctx.values;
  value_map[op] =
      BroadcastInDim(value_map[op.operand()], Convert_ArrayRef(type.getShape()),
                     Convert_broadcast_dimensions(op.broadcast_dimensions()));
  return success();
}

LogicalResult ExportXlaOp(ConcatenateOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  value_map[op] = xla::ConcatInDim(ctx.builder, GetTuple(op.val(), ctx),
                                   op.dimension().getSExtValue());
  return success();
}

LogicalResult ExportXlaOp(ConditionalOp op, OpLoweringContext ctx) {
  xla::XlaComputation true_branch;
  xla::XlaComputation false_branch;
  auto& value_map = *ctx.values;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.true_branch(),
                                                     &true_branch)) ||
      failed(ctx.converter->LowerRegionAsComputation(&op.false_branch(),
                                                     &false_branch))) {
    return failure();
  }

  value_map[op] =
      xla::Conditional(value_map[op.pred()], value_map[op.true_arg()],
                       true_branch, value_map[op.false_arg()], false_branch);

  return success();
}

LogicalResult ExportXlaOp(ConstOp op, OpLoweringContext ctx) {
  return failure();
}

LogicalResult ExportXlaOp(ConvOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  value_map[op] = xla::ConvGeneralDilated(
      value_map[op.lhs()], value_map[op.rhs()],
      Convert_broadcast_dimensions(op.window_strides()),
      Convert_padding(op.padding()),
      Convert_broadcast_dimensions(op.lhs_dilation()),
      Convert_broadcast_dimensions(op.rhs_dilation()),
      Convert_convolution_dimension_numbers(op.dimension_numbers()),
      op.feature_group_count().getSExtValue(),
      op.batch_group_count().getSExtValue(),
      Convert_precision_config(op.precision_config()).get());
  return success();
}

LogicalResult ExportXlaOp(ConvertOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  value_map[op] = xla::ConvertElementType(
      value_map[op.operand()],
      xla::TypeToPrimitiveType(getElementTypeOrSelf(op.getType())));
  return success();
}

LogicalResult ExportXlaOp(CopyOp op, OpLoweringContext ctx) {
  return failure();
}

LogicalResult ExportXlaOp(DynamicSliceOp op, OpLoweringContext ctx) {
  return failure();
}

LogicalResult ExportXlaOp(DynamicUpdateSliceOp op, OpLoweringContext ctx) {
  return failure();
}

LogicalResult ExportXlaOp(FftOp op, OpLoweringContext ctx) { return failure(); }

LogicalResult ExportXlaOp(GatherOp op, OpLoweringContext ctx) {
  return failure();
}

LogicalResult ExportXlaOp(GetTupleElementOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  value_map[op] = xla::GetTupleElement(value_map[op.getOperand()],
                                       op.index().getSExtValue());
  return success();
}

LogicalResult ExportXlaOp(IotaOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  value_map[op] = xla::Iota(ctx.builder, xla::TypeToShape(op.getType()),
                            op.iota_dimension().getSExtValue());
  return success();
}

LogicalResult ExportXlaOp(PadOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::PaddingConfig padding_config;
  auto edge_padding_low = ConvertDenseIntAttr(op.edge_padding_low());
  auto edge_padding_high = ConvertDenseIntAttr(op.edge_padding_high());
  auto interior_padding = ConvertDenseIntAttr(op.interior_padding());
  for (xla::int64 i = 0; i < edge_padding_low.size(); ++i) {
    auto* dims = padding_config.add_dimensions();
    dims->set_edge_padding_low(edge_padding_low[i]);
    dims->set_edge_padding_high(edge_padding_high[i]);
    dims->set_interior_padding(interior_padding[i]);
  }
  value_map[op] = xla::Pad(value_map[op.getOperand(0)],
                           value_map[op.getOperand(1)], padding_config);
  return success();
}

LogicalResult ExportXlaOp(ReduceOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaComputation body;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.body(), &body))) {
    return failure();
  }
  xla::XlaOp result =
      xla::Reduce(ctx.builder, GetTuple(op.operands(), ctx),
                  GetTuple(op.init_values(), ctx), body,
                  Convert_broadcast_dimensions(op.dimensions()));
  if (op.getNumResults() == 1) {
    value_map[op.getResult(0)] = result;
  } else {
    for (auto item : llvm::enumerate(op.getResults())) {
      value_map[item.value()] = xla::GetTupleElement(result, item.index());
    }
  }
  return success();
}

LogicalResult ExportXlaOp(ReduceWindowOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaComputation body;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.body(), &body))) {
    return failure();
  }
  value_map[op] = xla::ReduceWindowWithGeneralPadding(
      value_map[op.operand()], value_map[op.init_value()], body,
      ConvertDenseIntAttr(op.window_dimensions()),
      ConvertDenseIntAttr(op.window_strides()),
      ConvertDenseIntAttr(op.base_dilations()),
      ConvertDenseIntAttr(op.window_dilations()),
      Convert_padding(op.padding()));
  return success();
}

LogicalResult ExportXlaOp(ReshapeOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  value_map[op] = xla::Reshape(value_map[op.operand()],
                               xla::TypeToShape(op.getType()).dimensions());

  return success();
}

LogicalResult ExportXlaOp(ReturnOp op, OpLoweringContext ctx) {
  // Failure on purpose because `xla_hlo::ReturnOp` will be handled by
  // special purpose logic in `ConvertToHloModule::Lower`.
  return failure();
}

LogicalResult ExportXlaOp(ReverseOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  value_map[op] = xla::Rev(value_map[op.operand()],
                           Convert_broadcast_dimensions(op.dimensions()));
  return success();
}

LogicalResult ExportXlaOp(RngUniformOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  value_map[op] = xla::RngUniform(value_map[op.a()], value_map[op.b()],
                                  xla::TypeToShape(op.getType()));
  return success();
}

LogicalResult ExportXlaOp(SelectAndScatterOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaComputation select;
  xla::XlaComputation scatter;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.select(), &select)) ||
      failed(
          ctx.converter->LowerRegionAsComputation(&op.scatter(), &scatter))) {
    return failure();
  }
  value_map[op] = xla::SelectAndScatterWithGeneralPadding(
      value_map[op.operand()], select,
      ConvertDenseIntAttr(op.window_dimensions()),
      ConvertDenseIntAttr(op.window_strides()), Convert_padding(op.padding()),
      value_map[op.source()], value_map[op.init_value()], scatter);
  return success();
}

LogicalResult ExportXlaOp(SliceOp op, OpLoweringContext ctx) {
  return failure();
}

LogicalResult ExportXlaOp(TupleOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  value_map[op] = xla::Tuple(ctx.builder, GetTuple(op.val(), ctx));
  return success();
}

LogicalResult ExportXlaOp(WhileOp op, OpLoweringContext ctx) {
  xla::XlaComputation condition;
  xla::XlaComputation body;
  auto& value_map = *ctx.values;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.body(), &body)) ||
      failed(ctx.converter->LowerRegionAsComputation(&op.cond(), &condition))) {
    return failure();
  }

  value_map[op] = xla::While(condition, body, value_map[op.getOperand()]);
  return success();
}

}  // namespace
}  // namespace xla_hlo
}  // namespace mlir

#include "tensorflow/compiler/mlir/xla/operator_writers.inc"

namespace mlir {
namespace {

StatusOr<xla::Literal> CreateLiteralFromAttr(Type type, ElementsAttr attr) {
  xla::Shape shape = xla::TypeToShape(type);

#define ELEMENTS_ATTR_TO_LITERAL(xla_type, cpp_type)       \
  case xla_type: {                                         \
    xla::Array<cpp_type> source_data(shape.dimensions());  \
    source_data.SetValues(attr.getValues<cpp_type>());     \
    return xla::LiteralUtil::CreateFromArray(source_data); \
  }

  switch (shape.element_type()) {
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::PRED, bool)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::F32, float)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::F64, double)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::S8, int8)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::S16, int16)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::S32, int32)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::S64, int64)
    // TODO(b/130356985): Update once MLIR supports unsigned integers.
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::U8, uint8)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::U16, uint16)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::U32, uint32)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::U64, uint64)
    default:
      return tensorflow::errors::Internal(absl::StrCat(
          "Unsupported type: ", xla::PrimitiveType_Name(shape.element_type())));
  }
#undef ELEMENTS_ATTR_TO_LITERAL
}

LogicalResult ConvertToHloModule::Lower(
    mlir::Operation* inst, bool is_entry_function, xla::XlaBuilder* builder,
    ConvertToHloModule::ValueLoweringMap* value_lowering,
    xla::XlaComputation* result) {
  if (succeeded(ExportXlaOperator(inst, {value_lowering, this, builder}))) {
    return success();
  }

  auto& value_map = *value_lowering;
  ElementsAttr const_attr;

  if (auto call_op = dyn_cast<mlir::CallOp>(inst)) {
    return LowerFunctionCall(&call_op, builder, &value_map);
  }

  // TODO(jpienaar): This doesn't support layouts yet.
  if (matchPattern(inst, m_Constant(&const_attr))) {
    auto literal_or =
        CreateLiteralFromAttr(*inst->result_type_begin(), const_attr);
    if (!literal_or.ok()) return inst->emitError("unsupported elemental type");
    value_map[inst->getResult(0)] =
        xla::ConstantLiteral(builder, literal_or.ValueOrDie());
    return success();
  }

  if (isa<xla_hlo::ReturnOp>(inst) || isa<mlir::ReturnOp>(inst)) {
    // Construct the return value for the function. If there are multiple
    // values returned, then create a tuple, else return value directly.
    xla::XlaOp return_value;
    unsigned num_return_values = inst->getNumOperands();
    if ((return_tuple_ && is_entry_function) || num_return_values > 1) {
      std::vector<xla::XlaOp> returns(num_return_values);
      for (unsigned i = 0, e = inst->getNumOperands(); i != e; ++i) {
        returns[i] = value_map[inst->getOperand(i)];
      }
      return_value = xla::Tuple(builder, returns);
    } else if (num_return_values == 1) {
      return_value = value_map[inst->getOperand(0)];
    }

    // Build the XlaComputation and check for failures.
    auto computation_or =
        return_value.valid() ? builder->Build(return_value) : builder->Build();
    if (!computation_or.ok()) {
      inst->emitError(llvm::Twine(computation_or.status().error_message()));
      return failure();
    }
    *result = std::move(computation_or.ValueOrDie());
    return success();
  }

  inst->emitError("unable to lower operation of type '" +
                  inst->getName().getStringRef().str() + '\'');
  return failure();
}

LogicalResult ConvertToHloModule::LowerFunctionCall(
    mlir::CallOp* call_op, xla::XlaBuilder* builder,
    ConvertToHloModule::ValueLoweringMap* value_lowering) {
  auto& value_map = *value_lowering;
  mlir::FuncOp callee = module_.lookupSymbol<mlir::FuncOp>(call_op->callee());
  if (failed(RunOnFunction(callee))) return failure();
  std::vector<xla::XlaOp> operands;
  for (auto operand : call_op->getOperands()) {
    operands.push_back(value_map[operand]);
  }
  // Each call to xla::Call would insert a copy of the computation to
  // the HLO. Thus each callsite would have a unique callee in the
  // exported HLO. HLO syntactically does not require all calls to have unique
  // callees, but eventually before lowering call graph is "flattened" to
  // make that true. This is done before lowering because buffer assignment
  // needs this invariant.
  xla::XlaOp call_result =
      xla::Call(builder, lowered_computation_[callee], operands);
  // Use GetTupleElement for multiple outputs
  unsigned num_results = call_op->getNumResults();
  if (num_results > 1) {
    for (unsigned i = 0; i != num_results; ++i) {
      value_map[call_op->getResult(i)] = xla::GetTupleElement(call_result, i);
    }
  } else if (num_results == 1) {
    value_map[call_op->getResult(0)] = call_result;
  }
  return success();
}

LogicalResult ConvertToHloModule::RunOnFunction(mlir::FuncOp f) {
  if (lowered_computation_.count(f)) return success();
  if (f.getBlocks().size() != 1) {
    return f.emitError("only single block Function suppored");
  }

  // Create a sub-builder if this is not the main function.
  std::unique_ptr<xla::XlaBuilder> builder_up;
  bool entry_function = f.getName().str() == "main";
  if (!entry_function)
    builder_up = module_builder_.CreateSubBuilder(f.getName().str());
  auto& builder = entry_function ? module_builder_ : *builder_up;

  xla::XlaComputation computation;
  if (failed(LowerBasicBlockAsFunction(&f.front(), &builder, entry_function,
                                       &computation))) {
    return failure();
  }
  lowered_computation_[f] = std::move(computation);
  return success();
}

LogicalResult ConvertToHloModule::LowerBasicBlockAsFunction(
    Block* block, xla::XlaBuilder* builder, bool is_entry_function,
    xla::XlaComputation* result) {
  auto& bb = *block;
  // Mapping from the Value to lowered XlaOp. The code below lowers in
  // program order and will fail if an operand is unseen. This can be improved.
  ValueLoweringMap lowering;

  // If using tuples as input, then there is only one input parameter that is a
  // tuple.
  if (is_entry_function && use_tuple_args_) {
    std::vector<xla::Shape> arg_shapes;
    arg_shapes.reserve(bb.getNumArguments());
    for (auto& arg : bb.getArguments())
      arg_shapes.push_back(xla::TypeToShape(arg->getType()));
    xla::Shape input_shape = xla::ShapeUtil::MakeTupleShape(arg_shapes);
    auto tuple = xla::Parameter(builder, 0, input_shape, "arg_tuple");
    for (auto& it : llvm::enumerate(bb.getArguments())) {
      lowering[it.value()] = xla::GetTupleElement(tuple, it.index());
    }
  } else {
    for (auto& it : llvm::enumerate(bb.getArguments())) {
      auto* arg = it.value();
      auto num = it.index();
      xla::Shape shape = xla::TypeToShape(arg->getType());
      lowering[arg] =
          xla::Parameter(builder, num, shape, absl::StrCat("Arg_", num));
    }
  }

  for (auto& inst : bb)
    if (failed(Lower(&inst, is_entry_function, builder, &lowering, result)))
      return failure();

  return success();
}

LogicalResult ConvertToHloModule::LowerRegionAsComputation(
    mlir::Region* region, xla::XlaComputation* func) {
  std::unique_ptr<xla::XlaBuilder> builder =
      module_builder_.CreateSubBuilder(absl::StrCat("region_", region_id_++));
  return LowerBasicBlockAsFunction(&region->front(), builder.get(),
                                   /*is_entry_function=*/false, func);
}

}  // namespace

Status ConvertMlirHloToHlo(mlir::ModuleOp module, xla::HloProto* hlo_proto,
                           bool use_tuple_args, bool return_tuple) {
  mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());
  ConvertToHloModule converter(module, use_tuple_args, return_tuple);
  if (failed(converter.Run())) return diag_handler.ConsumeStatus();
  auto hlo_module = converter.ConsumeMainProto();
  hlo_proto->mutable_hlo_module()->Swap(&hlo_module);
  return Status::OK();
}

}  // namespace mlir
