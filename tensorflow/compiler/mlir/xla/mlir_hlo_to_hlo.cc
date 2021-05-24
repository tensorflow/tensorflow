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
#include <string>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/utils/name_utils.h"
#include "tensorflow/compiler/mlir/xla/attribute_exporter.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/quantize.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/stream_executor/lib/statusor.h"

using ::stream_executor::port::StatusOr;
using ::tensorflow::int16;
using ::tensorflow::int32;
using ::tensorflow::int64;
using ::tensorflow::int8;
using ::tensorflow::uint16;
using ::tensorflow::uint32;
using ::tensorflow::uint64;
using ::tensorflow::uint8;

constexpr char kPaddingMapAttr[] = "mhlo.padding_map";
constexpr char kShapeIndicesAttr[] = "shape_indices";
constexpr char kPaddingArgIndicesAttr[] = "padding_arg_indices";
constexpr char kShardingAttr[] = "mhlo.sharding";
constexpr char kFrontendAttributesAttr[] = "mhlo.frontend_attributes";
constexpr char kRepicationAttr[] = "mhlo.is_same_data_across_replicas";

// Array attribute. Same shape as infeed result, but contains a
// minor_to_major array for every tensor.
constexpr char kLayoutAttr[] = "layout";

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

static mlir::LogicalResult GetXlaOp(
    mlir::Value val, const llvm::DenseMap<mlir::Value, xla::XlaOp>& val_map,
    xla::XlaOp* result, mlir::Operation* op) {
  auto iter = val_map.find(val);
  if (iter == val_map.end()) {
    return op->emitOpError(
        "requires all operands to be defined in the parent region for export");
  }
  *result = iter->second;
  return mlir::success();
}

// Convert APInt into an int.
// TODO(hpucha): This should be consolidated into a general place.
static int ConvertAPInt(llvm::APInt i) { return i.getSExtValue(); }

static uint32_t Convertuint32_t(uint32_t i) { return i; }
static uint64_t Convertuint64_t(uint64_t i) { return i; }

// Convert APFloat to double.
static double ConvertAPFloat(llvm::APFloat value) {
  const auto& semantics = value.getSemantics();
  bool losesInfo = false;
  if (&semantics != &llvm::APFloat::IEEEdouble())
    value.convert(llvm::APFloat::IEEEdouble(),
                  llvm::APFloat::rmNearestTiesToEven, &losesInfo);
  return value.convertToDouble();
}

static inline bool Convertbool(bool value) { return value; }

static absl::string_view ConvertStringRef(mlir::StringRef value) {
  return {value.data(), value.size()};
}

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

// Converts StringRef to xla FftType enum
static xla::FftType Convert_fft_type(llvm::StringRef fft_type_str) {
  xla::FftType fft_type_enum;
  // Illegal fft_type string would be caught by the verifier, so 'FftType_Parse'
  // call below should never return false.
  if (!FftType_Parse(std::string(fft_type_str), &fft_type_enum))
    return xla::FftType::FFT;
  return fft_type_enum;
}

static std::vector<std::pair<int64, int64>> Convert_padding(
    llvm::Optional<mlir::DenseIntElementsAttr> padding) {
  return xla::ConvertNx2Attribute(padding).ValueOrDie();
}

static std::vector<std::pair<int64, int64>> Convert_source_target_pairs(
    llvm::Optional<mlir::DenseIntElementsAttr> source_target_pairs) {
  return xla::ConvertNx2Attribute(source_target_pairs).ValueOrDie();
}

static std::vector<xla::ReplicaGroup> Convert_replica_groups(
    mlir::DenseIntElementsAttr groups) {
  return xla::ConvertReplicaGroups(groups).ValueOrDie();
}

// Converts StringRef to xla Transpose enum.
static xla::TriangularSolveOptions::Transpose Convert_transpose_a(
    llvm::StringRef transpose_str) {
  return xla::ConvertTranspose(transpose_str).ValueOrDie();
}

static xla::Layout ExtractLayout(mlir::Operation* op, int rank,
                                 llvm::StringRef attr_name = "minor_to_major") {
  if (auto attr = GetLayoutFromMlirHlo(op, attr_name)) {
    llvm::SmallVector<int64, 4> minor_to_major;
    DCHECK_EQ(rank, attr.size());
    minor_to_major.reserve(attr.size());
    for (const llvm::APInt& i : attr) {
      minor_to_major.push_back(i.getZExtValue());
    }
    return xla::LayoutUtil::MakeLayout(minor_to_major);
  }
  return xla::LayoutUtil::MakeDescendingLayout(rank);
}

#define I64_ELEMENTS_ATTR_TO_VECTOR(attribute)                \
  static std::vector<int64> Convert_##attribute(              \
      llvm::Optional<mlir::DenseIntElementsAttr> attribute) { \
    return ConvertDenseIntAttr(attribute);                    \
  }

I64_ELEMENTS_ATTR_TO_VECTOR(broadcast_sizes);
I64_ELEMENTS_ATTR_TO_VECTOR(permutation);
I64_ELEMENTS_ATTR_TO_VECTOR(start_indices);
I64_ELEMENTS_ATTR_TO_VECTOR(limit_indices);
I64_ELEMENTS_ATTR_TO_VECTOR(strides);
I64_ELEMENTS_ATTR_TO_VECTOR(slice_sizes);
I64_ELEMENTS_ATTR_TO_VECTOR(fft_length);
I64_ELEMENTS_ATTR_TO_VECTOR(dimensions);
I64_ELEMENTS_ATTR_TO_VECTOR(window_strides);
I64_ELEMENTS_ATTR_TO_VECTOR(lhs_dilation);
I64_ELEMENTS_ATTR_TO_VECTOR(rhs_dilation);

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
    mlir::mhlo::DotDimensionNumbers dot_dimension_numbers_attr) {
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

  for (const auto& val : rhs_contracting_dimensions) {
    dot_dimension_numbers.add_rhs_contracting_dimensions(val.getSExtValue());
  }
  for (const auto& val : lhs_contracting_dimensions) {
    dot_dimension_numbers.add_lhs_contracting_dimensions(val.getSExtValue());
  }

  for (const auto& val : rhs_batch_dimensions) {
    dot_dimension_numbers.add_rhs_batch_dimensions(val.getSExtValue());
  }

  for (const auto& val : lhs_batch_dimensions) {
    dot_dimension_numbers.add_lhs_batch_dimensions(val.getSExtValue());
  }

  return dot_dimension_numbers;
}

static xla::ConvolutionDimensionNumbers Convert_dimension_numbers(
    mlir::mhlo::ConvDimensionNumbers input) {
  return xla::ConvertConvDimensionNumbers(input);
}

xla::ChannelHandle Convert_channel_handle(mlir::mhlo::ChannelHandle attr) {
  xla::ChannelHandle channel_handle;
  channel_handle.set_handle(ConvertAPInt(attr.handle().getValue()));
  channel_handle.set_type(static_cast<xla::ChannelHandle::ChannelType>(
      ConvertAPInt(attr.type().getValue())));
  return channel_handle;
}

// Converts the comparison_direction string attribute into the XLA enum. The
// string is assumed to correspond to exactly one of the allowed strings
// representing the enum. This should have been checked in the op verify method.
static xla::ComparisonDirection Convert_comparison_direction(
    llvm::StringRef comparison_direction_string) {
  return xla::StringToComparisonDirection(comparison_direction_string.str())
      .ValueOrDie();
}

static xla::GatherDimensionNumbers Convert_dimension_numbers(
    mlir::mhlo::GatherDimensionNumbers input) {
  xla::GatherDimensionNumbers output;

  auto offset_dims = ConvertDenseIntAttr(input.offset_dims());
  std::copy(offset_dims.begin(), offset_dims.end(),
            tensorflow::protobuf::RepeatedFieldBackInserter(
                output.mutable_offset_dims()));

  auto collapsed_slice_dims = ConvertDenseIntAttr(input.collapsed_slice_dims());
  std::copy(collapsed_slice_dims.begin(), collapsed_slice_dims.end(),
            tensorflow::protobuf::RepeatedFieldBackInserter(
                output.mutable_collapsed_slice_dims()));

  auto start_index_map = ConvertDenseIntAttr(input.start_index_map());
  std::copy(start_index_map.begin(), start_index_map.end(),
            tensorflow::protobuf::RepeatedFieldBackInserter(
                output.mutable_start_index_map()));

  output.set_index_vector_dim(
      ConvertAPInt(input.index_vector_dim().getValue()));
  return output;
}

static xla::ScatterDimensionNumbers Convert_scatter_dimension_numbers(
    mlir::mhlo::ScatterDimensionNumbers input) {
  xla::ScatterDimensionNumbers output;

  auto update_window_dims = ConvertDenseIntAttr(input.update_window_dims());
  std::copy(update_window_dims.begin(), update_window_dims.end(),
            tensorflow::protobuf::RepeatedFieldBackInserter(
                output.mutable_update_window_dims()));

  auto inserted_window_dims = ConvertDenseIntAttr(input.inserted_window_dims());
  std::copy(inserted_window_dims.begin(), inserted_window_dims.end(),
            tensorflow::protobuf::RepeatedFieldBackInserter(
                output.mutable_inserted_window_dims()));

  auto scatter_dims_to_operand_dims =
      ConvertDenseIntAttr(input.scatter_dims_to_operand_dims());
  std::copy(scatter_dims_to_operand_dims.begin(),
            scatter_dims_to_operand_dims.end(),
            tensorflow::protobuf::RepeatedFieldBackInserter(
                output.mutable_scatter_dims_to_operand_dims()));

  output.set_index_vector_dim(
      ConvertAPInt(input.index_vector_dim().getValue()));
  return output;
}

// Extracts sharding from attribute string.
static absl::optional<xla::OpSharding> CreateOpShardingFromStringRef(
    llvm::StringRef sharding) {
  xla::OpSharding sharding_proto;
  if (!sharding_proto.ParseFromString(sharding.str())) return absl::nullopt;
  return sharding_proto;
}

// Returns an OpSharding proto from the "sharding" attribute of the op. If the
// op doesn't have a sharding attribute or the sharding attribute is invalid,
// returns absl::nullopt.
static absl::optional<xla::OpSharding> CreateOpShardingFromAttribute(
    mlir::Operation* op) {
  auto sharding = op->getAttrOfType<mlir::StringAttr>(kShardingAttr);
  if (!sharding) return absl::nullopt;
  return CreateOpShardingFromStringRef(sharding.getValue());
}

// Returns a FrontendAttributes proto from the "frontend_attributes" attribute
// of the op. An empty FrontendAttributes proto is returned if an op does not
// have frontend attributes.
static xla::FrontendAttributes CreateOpFrontendAttributesFromAttribute(
    mlir::Operation* op) {
  xla::FrontendAttributes frontend_attributes;
  auto frontend_attributes_dict =
      op->getAttrOfType<mlir::DictionaryAttr>(kFrontendAttributesAttr);

  if (!frontend_attributes_dict) return frontend_attributes;

  for (const auto& attr : frontend_attributes_dict)
    if (auto value_str_attr = attr.second.dyn_cast<mlir::StringAttr>())
      frontend_attributes.mutable_map()->insert(
          {attr.first.str(), value_str_attr.getValue().str()});

  return frontend_attributes;
}

// Returns a OpMetadata proto based on the location of the op. If the location
// is unknown, an empty proto is returned. `op_name` are populated with the op
// location (converted). FileLineColLoc locations are populated by taking the
// file name and line number, and populating `source_file` and `source_line`
// respectively.
static xla::OpMetadata CreateOpMetadataFromLocation(mlir::Operation* op) {
  xla::OpMetadata metadata;
  if (op->getLoc().isa<mlir::UnknownLoc>()) return metadata;

  std::string name = mlir::GetNameFromLoc(op->getLoc());
  mlir::LegalizeNodeName(name);
  metadata.set_op_name(name);

  if (auto file_line_col_loc = op->getLoc().dyn_cast<mlir::FileLineColLoc>()) {
    metadata.set_source_file(file_line_col_loc.getFilename().str());
    metadata.set_source_line(file_line_col_loc.getLine());
  }

  return metadata;
}

// Checks if all shardings are set.
static bool AllOptionalShardingsAreSet(
    llvm::ArrayRef<absl::optional<xla::OpSharding>> shardings) {
  return llvm::all_of(shardings,
                      [](const absl::optional<xla::OpSharding>& sharding) {
                        return sharding.has_value();
                      });
}

// Extracts argument and result shardings from function.
static void ExtractShardingsFromFunction(
    mlir::FuncOp function,
    llvm::SmallVectorImpl<absl::optional<xla::OpSharding>>* arg_shardings,
    llvm::SmallVectorImpl<absl::optional<xla::OpSharding>>* ret_shardings) {
  arg_shardings->resize(function.getNumArguments(),
                        absl::optional<xla::OpSharding>());
  for (int i = 0, end = function.getNumArguments(); i < end; ++i)
    if (auto sharding =
            function.getArgAttrOfType<mlir::StringAttr>(i, kShardingAttr))
      (*arg_shardings)[i] = CreateOpShardingFromStringRef(sharding.getValue());

  ret_shardings->resize(function.getNumResults(),
                        absl::optional<xla::OpSharding>());
  for (int i = 0, end = function.getNumResults(); i < end; ++i)
    if (auto sharding =
            function.getResultAttrOfType<mlir::StringAttr>(i, kShardingAttr))
      (*ret_shardings)[i] = CreateOpShardingFromStringRef(sharding.getValue());
}

namespace mlir {
namespace {
class ConvertToHloModule {
 public:
  using ValueLoweringMap = llvm::DenseMap<Value, xla::XlaOp>;
  using FunctionLoweringMap = llvm::DenseMap<mlir::FuncOp, xla::XlaComputation>;

  // If use_tuple_args is true, then the entry function's arguments are
  // converted to a tuple and passed as a single parameter.
  // Similarly, if return tuple is true, then the entry function's return values
  // are converted to a tuple even when there is only a single return value.
  // Multiple return values are always converted to a tuple and returned as a
  // single value.
  explicit ConvertToHloModule(
      mlir::ModuleOp module, xla::XlaBuilder& module_builder,
      bool use_tuple_args, bool return_tuple,
      tensorflow::XlaHelpers::ShapeRepresentationFn shape_representation_fn,
      MlirToHloConversionOptions options)
      : module_(module),
        module_builder_(module_builder),
        use_tuple_args_(use_tuple_args),
        return_tuple_(return_tuple),
        shape_representation_fn_(shape_representation_fn),
        options_(options) {
    if (!shape_representation_fn_)
      shape_representation_fn_ = tensorflow::IdentityShapeRepresentationFn();
  }

  // Perform the lowering to XLA. This function returns failure if an error was
  // encountered.
  //
  // TODO(hinsu): Check for dynamic shapes and exit instead of crashing.
  LogicalResult Run() {
    auto main = module_.lookupSymbol<mlir::FuncOp>("main");
    if (!main)
      return module_.emitError(
          "conversion requires module with `main` function");

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
  LogicalResult LowerBasicBlockAsFunction(
      Block* block, xla::XlaBuilder* builder, bool is_entry_function,
      const std::vector<bool>& entry_args_same_across_replicas,
      llvm::ArrayRef<absl::optional<xla::OpSharding>> arg_shardings,
      llvm::ArrayRef<absl::optional<xla::OpSharding>> ret_shardings,
      xla::XlaComputation* result);

  ::xla::HloModuleProto ConsumeMainProto() {
    auto main = module_.lookupSymbol<mlir::FuncOp>("main");
    // This is an invariant check as Run returns failure if there is no main
    // function and so the main proto shouldn't be consumed in that case.
    CHECK(main) << "requires module to have main function";  // Crash Ok.
    return lowered_computation_[main].proto();
  }

  // Lower function call to HLO call instruction
  LogicalResult LowerFunctionCall(
      mlir::CallOp call_op, xla::XlaBuilder* builder,
      ConvertToHloModule::ValueLoweringMap* value_lowering);

  LogicalResult Lower(
      mlir::Operation* inst, bool is_entry_function,
      llvm::ArrayRef<absl::optional<xla::OpSharding>> ret_shardings,
      xla::XlaBuilder* builder,
      ConvertToHloModule::ValueLoweringMap* value_lowering,
      xla::XlaOp* return_value);

  const MlirToHloConversionOptions& GetOptions() const { return options_; }

 private:
  LogicalResult SetEntryTupleShapesAndLeafReplication(
      Block* block, const std::vector<bool>& entry_args_same_across_replicas,
      llvm::SmallVectorImpl<xla::Shape>* arg_shapes,
      std::vector<bool>* leaf_replication);

  LogicalResult SetEntryTupleShardings(
      Block* block, xla::XlaBuilder* builder,
      llvm::ArrayRef<absl::optional<xla::OpSharding>> arg_shardings,
      llvm::SmallVectorImpl<xla::Shape>* arg_shapes);

  // The module being lowered.
  mlir::ModuleOp module_;

  // The top-level XlaBuilder.
  xla::XlaBuilder& module_builder_;

  // Map between function and lowered computation.
  FunctionLoweringMap lowered_computation_;

  // Whether the entry function should take a single tuple as input.
  bool use_tuple_args_;

  // Whether to always return a tuple.
  bool return_tuple_;

  // Shape representation function to determine entry function argument and
  // result shapes.
  tensorflow::XlaHelpers::ShapeRepresentationFn shape_representation_fn_;

  // Unique suffix to give to the name of the next lowered region.
  size_t region_id_ = 0;

  MlirToHloConversionOptions options_;
};

}  // namespace
}  // namespace mlir

namespace {

struct OpLoweringContext {
  llvm::DenseMap<mlir::Value, xla::XlaOp>* values;
  mlir::ConvertToHloModule* converter;
  xla::XlaBuilder* builder;
};

mlir::LogicalResult GetTuple(mlir::Operation* op,
                             mlir::Operation::operand_range values,
                             OpLoweringContext ctx,
                             llvm::SmallVectorImpl<xla::XlaOp>& results) {
  results.reserve(values.size());
  for (mlir::Value value : values) {
    if (failed(GetXlaOp(value, *ctx.values, &results.emplace_back(), op)))
      return mlir::failure();
  }
  return mlir::success();
}

}  // namespace

namespace mlir {
namespace mhlo {
namespace {

LogicalResult ExportXlaOp(AllReduceOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaComputation computation;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.computation(),
                                                     &computation))) {
    return failure();
  }
  auto replica_groups = Convert_replica_groups(op.replica_groups());
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.operand(), value_map, &operand, op))) return failure();

  if (!op.channel_id().hasValue()) {
    value_map[op] = xla::AllReduce(operand, computation, replica_groups,
                                   /*channel_id=*/absl::nullopt);
    return success();
  }
  auto channel_id = Convert_channel_handle(op.channel_id().getValue());
  value_map[op] =
      xla::AllReduce(operand, computation, replica_groups, channel_id);
  return success();
}

LogicalResult ExportXlaOp(BitcastConvertOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.operand(), value_map, &operand, op))) return failure();

  value_map[op] = xla::BitcastConvertType(
      operand, xla::TypeToPrimitiveType(getElementTypeOrSelf(op.getType())));
  return success();
}

LogicalResult ExportXlaOp(BroadcastInDimOp op, OpLoweringContext ctx) {
  auto type = op.getType().dyn_cast<RankedTensorType>();
  if (!type) return failure();
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.operand(), value_map, &operand, op))) return failure();

  value_map[op] =
      BroadcastInDim(operand, Convert_ArrayRef(type.getShape()),
                     Convert_broadcast_dimensions(op.broadcast_dimensions()));
  return success();
}

LogicalResult ExportXlaOp(DotGeneralOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp lhs, rhs;
  if (failed(GetXlaOp(op.lhs(), value_map, &lhs, op))) return mlir::failure();
  if (failed(GetXlaOp(op.rhs(), value_map, &rhs, op))) return mlir::failure();
  xla::PrimitiveType preferred_element_type =
      xla::TypeToPrimitiveType(getElementTypeOrSelf(op.getType()));
  value_map[op] = xla::DotGeneral(
      lhs, rhs, Convert_dot_dimension_numbers(op.dot_dimension_numbers()),
      Unwrap(Convert_precision_config(op.precision_config())),
      preferred_element_type);
  return mlir::success();
}

LogicalResult ExportXlaOp(DynamicBroadcastInDimOp op, OpLoweringContext ctx) {
  // This op has no expression in the legacy export format.
  return failure();
}

LogicalResult ExportXlaOp(DynamicIotaOp op, OpLoweringContext ctx) {
  // This op has no expression in the legacy export format.
  return failure();
}

LogicalResult ExportXlaOp(DynamicReshapeOp op, OpLoweringContext ctx) {
  // This op has no expression in the legacy export format.
  return failure();
}

LogicalResult ExportXlaOp(IfOp op, OpLoweringContext ctx) {
  xla::XlaComputation true_branch;
  xla::XlaComputation false_branch;
  auto& value_map = *ctx.values;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.true_branch(),
                                                     &true_branch)) ||
      failed(ctx.converter->LowerRegionAsComputation(&op.false_branch(),
                                                     &false_branch))) {
    return failure();
  }
  xla::XlaOp pred, true_arg, false_arg;
  if (failed(GetXlaOp(op.pred(), value_map, &pred, op))) return failure();
  if (failed(GetXlaOp(op.true_arg(), value_map, &true_arg, op)))
    return failure();
  if (failed(GetXlaOp(op.false_arg(), value_map, &false_arg, op)))
    return failure();

  value_map[op] =
      xla::Conditional(pred, true_arg, true_branch, false_arg, false_branch);
  return success();
}

LogicalResult ExportXlaOp(CaseOp op, OpLoweringContext ctx) {
  llvm::DenseMap<mlir::Value, xla::XlaOp>& value_map = *ctx.values;
  OperandRange operands = op.branch_operands();
  MutableArrayRef<Region> branches = op.branches();
  llvm::SmallVector<xla::XlaOp, 4> branch_operands(branches.size());
  std::vector<xla::XlaComputation> computations(branches.size());
  std::vector<xla::XlaComputation*> computations_p(branches.size());

  for (unsigned i = 0; i < branches.size(); ++i) {
    xla::XlaOp operand;
    if (failed(GetXlaOp(operands[i], value_map, &operand, op)))
      return failure();
    branch_operands[i] = operand;
    computations_p[i] = &computations[i];
    if (failed(ctx.converter->LowerRegionAsComputation(&branches[i],
                                                       computations_p[i])))
      return failure();
  }
  xla::XlaOp index;
  if (failed(GetXlaOp(op.index(), value_map, &index, op))) return failure();

  xla::XlaOp result = xla::Conditional(index, computations_p, branch_operands);
  if (op.getNumResults() == 1) {
    value_map[op.getResult(0)] = result;
  } else {
    for (auto item : llvm::enumerate(op.getResults())) {
      value_map[item.value()] = xla::GetTupleElement(result, item.index());
    }
  }
  return success();
}

// Specialize CompareOp export to set broadcast_dimensions argument.
mlir::LogicalResult ExportXlaOp(mlir::mhlo::CompareOp op,
                                OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp lhs, rhs;
  if (failed(GetXlaOp(op.lhs(), value_map, &lhs, op))) return mlir::failure();
  if (failed(GetXlaOp(op.rhs(), value_map, &rhs, op))) return mlir::failure();
  auto dir = Convert_comparison_direction(op.comparison_direction());
  auto type_attr = op.compare_typeAttr();

  xla::XlaOp xla_result;
  if (type_attr) {
    auto type =
        xla::StringToComparisonType(type_attr.getValue().str()).ValueOrDie();
    xla_result = xla::Compare(lhs, rhs, /*broadcast_dimensions=*/{}, dir, type);
  } else {
    xla_result = xla::Compare(lhs, rhs, dir);
  }
  value_map[op] = xla_result;
  return mlir::success();
}

LogicalResult ExportXlaOp(ConstOp op, OpLoweringContext ctx) {
  return failure();
}

LogicalResult ExportXlaOp(mlir::mhlo::ConvOp op, OpLoweringContext ctx) {
  // XLA client builder API does not support generating convolution instructions
  // with window reversal.
  if (op.hasWindowReversal()) return failure();
  auto& value_map = *ctx.values;
  xla::XlaOp lhs, rhs;
  if (failed(GetXlaOp(op.lhs(), value_map, &lhs, op))) return mlir::failure();
  if (failed(GetXlaOp(op.rhs(), value_map, &rhs, op))) return mlir::failure();
  xla::PrimitiveType preferred_element_type =
      xla::TypeToPrimitiveType(getElementTypeOrSelf(op.getType()));
  xla::XlaOp xla_result = xla::ConvGeneralDilated(
      lhs, rhs, Convert_window_strides(op.window_strides()),
      Convert_padding(op.padding()), Convert_lhs_dilation(op.lhs_dilation()),
      Convert_rhs_dilation(op.rhs_dilation()),
      Convert_dimension_numbers(op.dimension_numbers()),
      Convertuint64_t(op.feature_group_count()),
      Convertuint64_t(op.batch_group_count()),
      Unwrap(Convert_precision_config(op.precision_config())),
      preferred_element_type);
  value_map[op] = xla_result;
  return mlir::success();
}

LogicalResult ExportXlaOp(ConvertOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.operand(), value_map, &operand, op))) return failure();

  value_map[op] = xla::ConvertElementType(
      operand, xla::TypeToPrimitiveType(getElementTypeOrSelf(op.getType())));
  return success();
}

LogicalResult ExportXlaOp(CustomCallOp op, OpLoweringContext ctx) {
  // XLA client builder API does not support generating custom call instructions
  // with side effect.
  if (op.has_side_effect() || op.getNumResults() != 1) return failure();
  Value result = op.getResult(0);
  llvm::SmallVector<xla::XlaOp> args;
  if (failed(GetTuple(op, op.args(), ctx, args))) return failure();
  auto& value_map = *ctx.values;
  value_map[result] = xla::CustomCall(
      ctx.builder, std::string(op.call_target_name()), args,
      xla::TypeToShape(result.getType()), std::string(op.backend_config()));
  return success();
}

LogicalResult ExportXlaOp(DequantizeOp op, OpLoweringContext ctx) {
  xla::QuantizedRange range(ConvertAPFloat(op.min_range()),
                            ConvertAPFloat(op.max_range()));
  auto& value_map = *ctx.values;
  xla::XlaOp input;
  if (failed(GetXlaOp(op.input(), value_map, &input, op))) return failure();

  auto casted = xla::ConvertElementType(input, xla::U32);
  if (op.is_16bits()) {
    value_map[op] = xla::Dequantize<uint16>(
        casted, range, ConvertStringRef(op.mode()), op.transpose_output());
  } else {
    value_map[op] = xla::Dequantize<uint8>(
        casted, range, ConvertStringRef(op.mode()), op.transpose_output());
  }
  return success();
}

LogicalResult ExportXlaOp(InfeedOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp token;
  if (failed(GetXlaOp(op.token(), value_map, &token, op))) return failure();

  // The shape argument expected by the xla client API is the type of the first
  // element in the result tuple.
  auto result_type = op.getType().cast<mlir::TupleType>().getType(0);
  value_map[op] = xla::InfeedWithToken(token, xla::TypeToShape(result_type),
                                       std::string(op.infeed_config()));
  return success();
}

LogicalResult ExportXlaOp(IotaOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  value_map[op] = xla::Iota(ctx.builder, xla::TypeToShape(op.getType()),
                            op.iota_dimension());
  return success();
}

LogicalResult ExportXlaOp(MapOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaComputation computation;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.computation(),
                                                     &computation))) {
    return failure();
  }
  llvm::SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op, op.operands(), ctx, operands))) return failure();
  value_map[op] = xla::Map(ctx.builder, operands, computation,
                           Convert_dimensions(op.dimensions()));
  return success();
}

LogicalResult ExportXlaOp(OutfeedOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand, token;
  if (failed(GetXlaOp(op.operand(), value_map, &operand, op))) return failure();
  if (failed(GetXlaOp(op.token(), value_map, &token, op))) return failure();

  value_map[op] = xla::OutfeedWithToken(
      operand, token, xla::TypeToShape(op.operand().getType()),
      std::string(op.outfeed_config()));
  return success();
}

LogicalResult ExportXlaOp(PadOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::PaddingConfig padding_config;
  auto edge_padding_low = ConvertDenseIntAttr(op.edge_padding_low());
  auto edge_padding_high = ConvertDenseIntAttr(op.edge_padding_high());
  auto interior_padding = ConvertDenseIntAttr(op.interior_padding());
  for (xla::int64 i = 0, end = edge_padding_low.size(); i < end; ++i) {
    auto* dims = padding_config.add_dimensions();
    dims->set_edge_padding_low(edge_padding_low[i]);
    dims->set_edge_padding_high(edge_padding_high[i]);
    dims->set_interior_padding(interior_padding[i]);
  }
  xla::XlaOp operand, padding_value;
  if (failed(GetXlaOp(op.operand(), value_map, &operand, op))) return failure();
  if (failed(GetXlaOp(op.padding_value(), value_map, &padding_value, op)))
    return failure();

  value_map[op] = xla::Pad(operand, padding_value, padding_config);
  return success();
}

LogicalResult ExportXlaOp(RecvOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result_type = op.getType().cast<mlir::TupleType>().getType(0);
  xla::XlaOp token;
  if (failed(GetXlaOp(op.token(), value_map, &token, op))) return failure();

  if (op.is_host_transfer()) {
    value_map[op] = xla::RecvFromHost(token, xla::TypeToShape(result_type),
                                      Convert_channel_handle(op.channel_id()));
    return success();
  }
  value_map[op] = xla::RecvWithToken(token, xla::TypeToShape(result_type),
                                     Convert_channel_handle(op.channel_id()));
  return success();
}

LogicalResult ExportXlaOp(ReduceOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaComputation body;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.body(), &body))) {
    return failure();
  }
  llvm::SmallVector<xla::XlaOp> operands, init_values;
  if (failed(GetTuple(op, op.inputs(), ctx, operands)) ||
      failed(GetTuple(op, op.init_values(), ctx, init_values))) {
    return failure();
  }
  xla::XlaOp result =
      xla::Reduce(ctx.builder, operands, init_values, body,
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
  llvm::SmallVector<xla::XlaOp> operands, init_values;
  if (failed(GetTuple(op, op.inputs(), ctx, operands)) ||
      failed(GetTuple(op, op.init_values(), ctx, init_values))) {
    return failure();
  }

  xla::XlaOp result = xla::ReduceWindowWithGeneralPadding(
      operands, init_values, body, ConvertDenseIntAttr(op.window_dimensions()),
      ConvertDenseIntAttr(op.window_strides()),
      ConvertDenseIntAttr(op.base_dilations()),
      ConvertDenseIntAttr(op.window_dilations()),
      Convert_padding(op.padding()));

  if (op.getNumResults() == 1) {
    value_map[op.getResult(0)] = result;
  } else {
    for (auto item : llvm::enumerate(op.getResults())) {
      value_map[item.value()] = xla::GetTupleElement(result, item.index());
    }
  }
  return success();
}

LogicalResult ExportXlaOp(ReshapeOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.operand(), value_map, &operand, op))) return failure();

  value_map[op] =
      xla::Reshape(operand, xla::TypeToShape(op.getType()).dimensions());
  return success();
}

LogicalResult ExportXlaOp(ReturnOp op, OpLoweringContext ctx) {
  // Failure on purpose because `mhlo::ReturnOp` will be handled by
  // special purpose logic in `ConvertToHloModule::Lower`.
  return failure();
}

LogicalResult ExportXlaOp(RngBitGeneratorOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  auto xla_arg_1 = value_map[*op.getODSOperands(0).begin()];
  auto xla_result = xla::RngBitGenerator(
      static_cast<xla::RandomAlgorithm>(op.rng_algorithm()), Unwrap(xla_arg_1),
      xla::TypeToShape(result.getType()).tuple_shapes(1));
  value_map[result] = xla_result;
  return mlir::success();
}

LogicalResult ExportXlaOp(RngNormalOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp mu, sigma;
  if (failed(GetXlaOp(op.mu(), value_map, &mu, op))) return failure();
  if (failed(GetXlaOp(op.sigma(), value_map, &sigma, op))) return failure();

  value_map[op] = xla::RngNormal(mu, sigma, xla::TypeToShape(op.getType()));
  return success();
}

LogicalResult ExportXlaOp(RngUniformOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp a, b;
  if (failed(GetXlaOp(op.a(), value_map, &a, op))) return failure();
  if (failed(GetXlaOp(op.b(), value_map, &b, op))) return failure();

  value_map[op] = xla::RngUniform(a, b, xla::TypeToShape(op.getType()));
  return success();
}

LogicalResult ExportXlaOp(ScatterOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaComputation update_computation;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.update_computation(),
                                                     &update_computation))) {
    return failure();
  }
  xla::ScatterDimensionNumbers dimension_numbers =
      Convert_scatter_dimension_numbers(op.scatter_dimension_numbers());
  xla::XlaOp operand, scatter_indices, updates;
  if (failed(GetXlaOp(op.operand(), value_map, &operand, op))) return failure();
  if (failed(GetXlaOp(op.scatter_indices(), value_map, &scatter_indices, op)))
    return failure();
  if (failed(GetXlaOp(op.updates(), value_map, &updates, op))) return failure();

  value_map[op] = xla::Scatter(operand, scatter_indices, updates,
                               update_computation, dimension_numbers,
                               op.indices_are_sorted(), op.unique_indices());
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
  xla::XlaOp operand, source, init_value;
  if (failed(GetXlaOp(op.operand(), value_map, &operand, op))) return failure();
  if (failed(GetXlaOp(op.source(), value_map, &source, op))) return failure();
  if (failed(GetXlaOp(op.init_value(), value_map, &init_value, op)))
    return failure();

  value_map[op] = xla::SelectAndScatterWithGeneralPadding(
      operand, select, ConvertDenseIntAttr(op.window_dimensions()),
      ConvertDenseIntAttr(op.window_strides()), Convert_padding(op.padding()),
      source, init_value, scatter);
  return success();
}

LogicalResult ExportXlaOp(SendOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand, token;
  if (failed(GetXlaOp(op.operand(), value_map, &operand, op))) return failure();
  if (failed(GetXlaOp(op.token(), value_map, &token, op))) return failure();

  if (op.is_host_transfer()) {
    value_map[op] = xla::SendToHost(operand, token,
                                    xla::TypeToShape(op.operand().getType()),
                                    Convert_channel_handle(op.channel_id()));
    return success();
  }
  value_map[op] = xla::SendWithToken(operand, token,
                                     Convert_channel_handle(op.channel_id()));
  return success();
}

LogicalResult ExportXlaOp(SliceOp op, OpLoweringContext ctx) {
  return failure();
}

LogicalResult ExportXlaOp(SortOp op, OpLoweringContext ctx) {
  xla::XlaComputation comparator;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.comparator(),
                                                     &comparator)))
    return failure();

  llvm::SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op, op.operands(), ctx, operands))) return failure();
  auto sorted = xla::Sort(operands, comparator, op.dimension(), op.is_stable());

  auto& value_map = *ctx.values;
  auto shape_or = sorted.builder()->GetShape(sorted);
  if (!shape_or.ok()) {
    return op.emitError(shape_or.status().ToString());
  }

  xla::Shape& shape = shape_or.ValueOrDie();
  if (!shape.IsTuple()) {
    value_map[op.getResult(0)] = sorted;
    return success();
  }

  // MLIR's sort supports multiple returns, untuple all the results of XLA's.
  for (auto it : llvm::enumerate(op.getResults())) {
    value_map[it.value()] = xla::GetTupleElement(sorted, it.index());
  }
  return success();
}

LogicalResult ExportXlaOp(TraceOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.operand(), value_map, &operand, op))) return failure();
  xla::Trace(std::string(op.tag()), operand);
  return success();
}

LogicalResult ExportXlaOp(UnaryEinsumOp op, OpLoweringContext ctx) {
  // Intentional as UnaryEinsumOp is always lowered to the EinsumOp with two
  // operands.
  return failure();
}

LogicalResult ExportXlaOp(WhileOp op, OpLoweringContext ctx) {
  xla::XlaComputation condition;
  xla::XlaComputation body;
  auto& value_map = *ctx.values;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.body(), &body)) ||
      failed(ctx.converter->LowerRegionAsComputation(&op.cond(), &condition))) {
    return failure();
  }

  xla::XlaOp operand;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  value_map[op] = xla::While(condition, body, operand);
  return success();
}

LogicalResult ExportXlaOp(FusionOp op, OpLoweringContext ctx) {
  if (!op.fusion_kind()) {
    op.emitOpError() << "requires fusion kind for HLO translation";
    return failure();
  }

  xla::XlaComputation fused_computation;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.fused_computation(),
                                                     &fused_computation)))
    return failure();

  auto& values = *ctx.values;
  llvm::SmallVector<xla::XlaOp, 4> operands;
  for (auto operand : op.operands()) operands.push_back(values[operand]);

  xla::XlaOp fusion = xla::internal::XlaBuilderFriend::BuildFusion(
      ctx.builder, operands,
      absl::string_view(op.fusion_kind()->data(), op.fusion_kind()->size()),
      fused_computation);
  if (op.getNumResults() == 1) {
    values[op.getResult(0)] = fusion;
  } else {
    for (auto item : llvm::enumerate(op.getResults())) {
      values[item.value()] = xla::GetTupleElement(fusion, item.index());
    }
  }
  return success();
}

LogicalResult ExportXlaOp(BitcastOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.operand(), value_map, &operand, op))) return failure();
  xla::XlaOp bitcast = xla::internal::XlaBuilderFriend::BuildBitcast(
      ctx.builder, operand, xla::TypeToShape(op.getType()));
  value_map[op] = bitcast;
  if (ctx.converter->GetOptions().propagate_bitcast_layouts_to_backend_config) {
    // Encode the source and result layout of the bitcast into the XLA HLO
    // backend config as a protobuf. Note that this is a temporary solution
    // which will go away once XLA:GPU stops falling back to XLA HLO Elemental
    // IR emitters.
    xla::HloInstructionProto* bitcast_proto =
        xla::internal::XlaBuilderFriend::GetInstruction(bitcast);
    xla::HloInstructionProto* operand_proto =
        xla::internal::XlaBuilderFriend::GetInstruction(operand);
    xla::LayoutProto result_layout =
        ExtractLayout(op, bitcast_proto->shape().dimensions_size(),
                      "result_layout")
            .ToProto();
    xla::LayoutProto source_layout =
        ExtractLayout(op, operand_proto->shape().dimensions_size(),
                      "source_layout")
            .ToProto();
    xla::gpu::BitcastBackendConfig bitcast_config;
    *bitcast_config.mutable_source_layout() = source_layout;
    *bitcast_config.mutable_result_layout() = result_layout;
    *bitcast_proto->mutable_backend_config() =
        bitcast_config.SerializeAsString();
  }
  return success();
}

LogicalResult ExportXlaOp(RealDynamicSliceOp op, OpLoweringContext ctx) {
  return failure();
}

LogicalResult ExportXlaOp(DynamicPadOp op, OpLoweringContext ctx) {
  return failure();
}

LogicalResult ExportXlaOp(DynamicGatherOp op, OpLoweringContext ctx) {
  return failure();
}

LogicalResult ExportXlaOp(DynamicConvOp op, OpLoweringContext ctx) {
  return failure();
}

}  // namespace
}  // namespace mhlo
}  // namespace mlir

#include "tensorflow/compiler/mlir/xla/operator_writers.inc"

namespace mlir {
namespace {

StatusOr<xla::Literal> CreateArrayLiteralFromAttr(ElementsAttr attr,
                                                  xla::Layout layout) {
  if (attr.isa<OpaqueElementsAttr>())
    return tensorflow::errors::Unimplemented(
        "Opaque elements attr not supported");

  xla::Shape shape = xla::TypeToShape(attr.getType());

#define ELEMENTS_ATTR_TO_LITERAL(xla_type, cpp_type)                         \
  case xla_type: {                                                           \
    xla::Array<cpp_type> source_data(shape.dimensions());                    \
    source_data.SetValues(attr.getValues<cpp_type>());                       \
    return xla::LiteralUtil::CreateFromArrayWithLayout(source_data, layout); \
  }

  switch (shape.element_type()) {
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::PRED, bool)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::F32, float)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::F64, double)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::S8, int8)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::S16, int16)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::S32, int32)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::S64, int64)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::U8, uint8)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::U16, uint16)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::U32, uint32)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::U64, uint64)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::C64, std::complex<float>)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::C128, std::complex<double>)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::F16, Eigen::half)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::BF16, Eigen::bfloat16)
    default:
      return tensorflow::errors::Internal(absl::StrCat(
          "Unsupported type: ", xla::PrimitiveType_Name(shape.element_type())));
  }
#undef ELEMENTS_ATTR_TO_LITERAL
}

LogicalResult ConvertLayout(mlir::Operation* op, const mlir::ArrayAttr& layout,
                            xla::ShapeProto* shape) {
  // In the case of tuples, ShapeProtos can be nested, and so can the mlir
  // attribute describing the layout. So recurse into the subshapes in both data
  // structures in parallel.
  if (shape->element_type() == xla::TUPLE) {
    auto subshapes = shape->mutable_tuple_shapes();
    if (layout.size() != subshapes->size()) {
      op->emitOpError() << "Expected layout of size " << layout.size()
                        << ", but found " << subshapes->size();
      return failure();
    }
    for (int i = 0; i < subshapes->size(); i++) {
      mlir::Attribute child = layout[i];
      if (child.isa<mlir::UnitAttr>()) {
        // ignore unit attributes, they are used only for tokens.
        continue;
      }
      mlir::ArrayAttr c = child.dyn_cast<mlir::ArrayAttr>();
      if (!c) {
        op->emitOpError() << "Type Error: Expected layout array attribute";
        return failure();
      }
      if (failed(ConvertLayout(op, c, subshapes->Mutable(i)))) {
        return failure();
      }
    }
  } else {
    int rank = shape->dimensions().size();
    if (rank) {
      if (layout.size() != rank) {
        return failure();  // pass error down
      }
      std::vector<int64> array(rank);
      for (int i = 0; i < rank; i++) {
        mlir::IntegerAttr attr = layout[i].dyn_cast<mlir::IntegerAttr>();
        if (!attr) {
          op->emitOpError() << "Type Error: Expected layout integer attribute";
          return failure();
        }
        array[i] = attr.getInt();
      }
      *shape->mutable_layout() = xla::LayoutUtil::MakeLayout(array).ToProto();
    }
  }
  return success();
}

LogicalResult ConvertToHloModule::Lower(
    mlir::Operation* inst, bool is_entry_function,
    llvm::ArrayRef<absl::optional<xla::OpSharding>> ret_shardings,
    xla::XlaBuilder* builder,
    ConvertToHloModule::ValueLoweringMap* value_lowering,
    xla::XlaOp* return_value) {
  // Explicitly fail for ops that are not supported for export.
  if (inst->getDialect() !=
          inst->getContext()->getLoadedDialect<mlir::mhlo::MhloDialect>() &&
      !mlir::isa<mlir::ConstantOp, mlir::CallOp, mlir::tensor::CastOp,
                 mlir::ReturnOp>(inst)) {
    inst->emitOpError("unsupported op for export to XLA");
    return failure();
  }

  *return_value = xla::XlaOp();

  // See MlirToHloConversionOptions for more about layouts.
  auto propagate_layouts = [this](mlir::Operation* inst,
                                  xla::XlaOp xla_op) -> mlir::LogicalResult {
    if (options_.propagate_layouts) {
      auto* shape = xla::internal::XlaBuilderFriend::GetInstruction(xla_op)
                        ->mutable_shape();
      if (shape->tuple_shapes().empty())
        // TODO(kramm): merge this with ConvertLayout.
        *shape->mutable_layout() =
            ExtractLayout(inst, shape->dimensions().size()).ToProto();
    }

    // For infeed ops stemming back to InfeedDequeueTuple, respect the layout
    // attribute, and create the corresponding layout in hlo.
    if (isa<mhlo::InfeedOp>(inst)) {
      mlir::ArrayAttr layout =
          inst->getAttrOfType<mlir::ArrayAttr>(kLayoutAttr);
      if (layout) {
        xla::ShapeProto* shape =
            xla::internal::XlaBuilderFriend::GetInstruction(xla_op)
                ->mutable_shape();

        if (failed(ConvertLayout(inst, layout, shape))) {
          return failure();
        }
      }
    }
    return success();
  };

  if (succeeded(ExportXlaOperator(inst, {value_lowering, this, builder}))) {
    if (inst->getNumResults() == 1) {
      auto iter = value_lowering->find(inst->getResult(0));
      if (iter == value_lowering->end()) {
        inst->emitOpError(
            "inst has a result, but it's not found in value_lowering");
        return failure();
      }
      if (failed(propagate_layouts(inst, iter->second))) {
        return failure();
      }
    }
    return success();
  }

  auto& value_map = *value_lowering;
  ElementsAttr const_attr;

  if (auto call_op = dyn_cast<mlir::CallOp>(inst)) {
    return LowerFunctionCall(call_op, builder, &value_map);
  }

  if (auto op = dyn_cast<mlir::tensor::CastOp>(inst)) {
    Value operand = op.getOperand();
    auto ty = operand.getType().dyn_cast<ShapedType>();
    // If this was a cast from a static shaped tensors, then it is a noop for
    // export to HLO and we can use the operand.
    if (!ty || !ty.hasStaticShape()) {
      inst->emitOpError()
          << "requires static shaped operand for HLO translation";
      return failure();
    }

    xla::XlaOp xla_operand;
    if (failed(GetXlaOp(operand, value_map, &xla_operand, op)))
      return failure();
    value_map[op.getResult()] = xla_operand;
    if (failed(propagate_layouts(inst, xla_operand))) {
      return failure();
    }
    return success();
  }

  if (matchPattern(inst, m_Constant(&const_attr))) {
    xla::Layout layout;
    layout = ExtractLayout(inst, const_attr.getType().getRank());
    auto literal_or = CreateArrayLiteralFromAttr(const_attr, layout);
    if (!literal_or.ok())
      return inst->emitError(literal_or.status().ToString());
    auto constant = xla::ConstantLiteral(builder, literal_or.ValueOrDie());
    value_map[inst->getResult(0)] = constant;

    return success();
  }

  if (isa<mhlo::ReturnOp, mlir::ReturnOp>(inst)) {
    // Construct the return value for the function. If there is a single value
    // returned, then return it directly, else create a tuple and return.
    unsigned num_return_values = inst->getNumOperands();
    if ((return_tuple_ && is_entry_function) || num_return_values != 1) {
      const bool has_ret_shardings =
          !ret_shardings.empty() && AllOptionalShardingsAreSet(ret_shardings);

      std::vector<xla::XlaOp> returns(num_return_values);
      for (OpOperand& ret : inst->getOpOperands()) {
        unsigned index = ret.getOperandNumber();
        xla::XlaOp operand;
        if (failed(GetXlaOp(ret.get(), value_map, &operand, inst)))
          return failure();

        returns[index] = operand;
        if (!is_entry_function || !has_ret_shardings) continue;

        xla::Shape return_shape = xla::TypeToShape(ret.get().getType());
        StatusOr<xla::XlaOp> reshape =
            tensorflow::ReshapeWithCorrectRepresentationAndSharding(
                builder, returns[index], return_shape, shape_representation_fn_,
                ret_shardings[index], /*fast_mem=*/false);
        if (!reshape.ok())
          return inst->emitError() << reshape.status().error_message();

        returns[index] = reshape.ValueOrDie();
      }

      if (has_ret_shardings) {
        xla::OpSharding sharding;
        sharding.set_type(xla::OpSharding::TUPLE);
        for (auto& ret_sharding : ret_shardings)
          *sharding.add_tuple_shardings() = *ret_sharding;

        builder->SetSharding(sharding);
      }

      *return_value = xla::Tuple(builder, returns);
      builder->ClearSharding();
    } else if (num_return_values == 1) {
      xla::XlaOp operand;
      if (failed(GetXlaOp(inst->getOperand(0), value_map, &operand, inst)))
        return failure();

      *return_value = operand;
    }

    return success();
  }

  inst->emitOpError() << "can't be translated to XLA HLO";
  return failure();
}

LogicalResult ConvertToHloModule::LowerFunctionCall(
    mlir::CallOp call_op, xla::XlaBuilder* builder,
    ConvertToHloModule::ValueLoweringMap* value_lowering) {
  auto& value_map = *value_lowering;
  mlir::FuncOp callee = module_.lookupSymbol<mlir::FuncOp>(call_op.callee());
  if (failed(RunOnFunction(callee))) return failure();
  std::vector<xla::XlaOp> operands;
  for (auto operand : call_op.getOperands()) {
    xla::XlaOp xla_operand;
    if (failed(GetXlaOp(operand, value_map, &xla_operand, call_op)))
      return failure();
    operands.push_back(xla_operand);
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
  unsigned num_results = call_op.getNumResults();
  if (num_results > 1) {
    for (unsigned i = 0; i != num_results; ++i) {
      value_map[call_op.getResult(i)] = xla::GetTupleElement(call_result, i);
    }
  } else if (num_results == 1) {
    value_map[call_op.getResult(0)] = call_result;
  }
  return success();
}

LogicalResult ConvertToHloModule::RunOnFunction(mlir::FuncOp f) {
  if (lowered_computation_.count(f)) return success();
  if (!llvm::hasSingleElement(f)) {
    return f.emitError("only single block Function supported");
  }

  // Create a sub-builder if this is not the main function.
  std::unique_ptr<xla::XlaBuilder> builder_up;
  bool entry_function = f.getName() == "main";
  if (!entry_function)
    builder_up = module_builder_.CreateSubBuilder(f.getName().str());
  auto& builder = entry_function ? module_builder_ : *builder_up;

  xla::XlaComputation computation;
  std::vector<bool> entry_args_same_across_replicas;
  llvm::SmallVector<absl::optional<xla::OpSharding>, 4> arg_shardings;
  llvm::SmallVector<absl::optional<xla::OpSharding>, 4> ret_shardings;
  if (entry_function) {
    bool any_arg_replicated = false;
    entry_args_same_across_replicas.reserve(f.getNumArguments());
    for (int64_t i = 0; i < f.getNumArguments(); ++i) {
      auto attr = f.getArgAttrOfType<mlir::UnitAttr>(i, kRepicationAttr);
      entry_args_same_across_replicas.push_back(attr != nullptr);
      any_arg_replicated |= entry_args_same_across_replicas.back();
      // Pass the alias info to the builder so that it will build the alias info
      // into the resulting HloModule.
      auto aliasing_output =
          f.getArgAttrOfType<mlir::IntegerAttr>(i, "tf.aliasing_output");
      if (!aliasing_output) continue;
      if (use_tuple_args_) {
        builder.SetUpAlias(/*output_index=*/{aliasing_output.getInt()},
                           /*param_number=*/0, /*param_index=*/{i});
      } else {
        builder.SetUpAlias(/*output_index=*/{aliasing_output.getInt()},
                           /*param_number=*/i, /*param_index=*/{});
      }
    }
    // Do not populate this field when nothing is replicated, since empty field
    // means no replication. This avoids the need for unrelated tests to handle
    // this field.
    if (!any_arg_replicated) entry_args_same_across_replicas.clear();

    ExtractShardingsFromFunction(f, &arg_shardings, &ret_shardings);
  }
  if (failed(LowerBasicBlockAsFunction(
          &f.front(), &builder, entry_function, entry_args_same_across_replicas,
          arg_shardings, ret_shardings, &computation))) {
    return failure();
  }
  lowered_computation_[f] = std::move(computation);
  return success();
}

LogicalResult ConvertToHloModule::SetEntryTupleShapesAndLeafReplication(
    Block* block, const std::vector<bool>& entry_args_same_across_replicas,
    llvm::SmallVectorImpl<xla::Shape>* arg_shapes,
    std::vector<bool>* leaf_replication) {
  arg_shapes->reserve(block->getNumArguments());
  leaf_replication->reserve(block->getNumArguments());
  for (BlockArgument& arg : block->getArguments()) {
    arg_shapes->push_back(xla::TypeToShape(arg.getType()));
    xla::Shape& arg_shape = arg_shapes->back();
    tensorflow::TensorShape arg_tensor_shape;
    auto status =
        tensorflow::XLAShapeToTensorShape(arg_shape, &arg_tensor_shape);
    if (!status.ok())
      return block->getParentOp()->emitError() << status.error_message();

    tensorflow::DataType dtype;
    status = tensorflow::ConvertToDataType(arg.getType(), &dtype);
    if (!status.ok())
      return block->getParentOp()->emitError() << status.error_message();

    auto arg_shape_status = shape_representation_fn_(arg_tensor_shape, dtype,
                                                     /*use_fast_memory=*/false);
    if (!arg_shape_status.ok())
      return block->getParentOp()->emitError()
             << arg_shape_status.status().error_message();

    arg_shape = std::move(arg_shape_status.ValueOrDie());

    if (entry_args_same_across_replicas.empty()) continue;
    for (int i = 0, e = xla::ShapeUtil::GetLeafCount(arg_shape); i < e; ++i)
      leaf_replication->push_back(
          entry_args_same_across_replicas[arg.getArgNumber()]);
  }

  return success();
}

LogicalResult ConvertToHloModule::SetEntryTupleShardings(
    Block* block, xla::XlaBuilder* builder,
    llvm::ArrayRef<absl::optional<xla::OpSharding>> arg_shardings,
    llvm::SmallVectorImpl<xla::Shape>* arg_shapes) {
  if (!arg_shardings.empty() && AllOptionalShardingsAreSet(arg_shardings)) {
    xla::OpSharding sharding;
    sharding.set_type(xla::OpSharding::TUPLE);
    for (auto arg_sharding : llvm::enumerate(arg_shardings)) {
      auto hlo_sharding = xla::HloSharding::FromProto(*arg_sharding.value());
      if (!hlo_sharding.ok())
        return block->getParentOp()->emitError()
               << hlo_sharding.status().error_message();

      auto status = tensorflow::RewriteLayoutWithShardedShape(
          hlo_sharding.ValueOrDie(), /*use_fast_memory=*/false,
          shape_representation_fn_, &(*arg_shapes)[arg_sharding.index()]);
      if (!status.ok())
        return block->getParentOp()->emitError() << status.error_message();

      *sharding.add_tuple_shardings() = *arg_sharding.value();
    }

    builder->SetSharding(sharding);
  }

  return success();
}

LogicalResult ConvertToHloModule::LowerBasicBlockAsFunction(
    Block* block, xla::XlaBuilder* builder, bool is_entry_function,
    const std::vector<bool>& entry_args_same_across_replicas,
    llvm::ArrayRef<absl::optional<xla::OpSharding>> arg_shardings,
    llvm::ArrayRef<absl::optional<xla::OpSharding>> ret_shardings,
    xla::XlaComputation* result) {
  // Mapping from the Value to lowered XlaOp.
  ValueLoweringMap lowering;

  // If using tuples as input, then there is only one input parameter that is a
  // tuple.
  if (is_entry_function && use_tuple_args_) {
    llvm::SmallVector<xla::Shape, 4> arg_shapes;
    std::vector<bool> leaf_replication;
    if (failed(SetEntryTupleShapesAndLeafReplication(
            block, entry_args_same_across_replicas, &arg_shapes,
            &leaf_replication)))
      return failure();

    if (failed(
            SetEntryTupleShardings(block, builder, arg_shardings, &arg_shapes)))
      return failure();

    xla::Shape input_shape = xla::ShapeUtil::MakeTupleShape(arg_shapes);
    auto tuple =
        xla::Parameter(builder, 0, input_shape, "arg_tuple", leaf_replication);
    builder->ClearSharding();

    bool set_tuple_element_sharding =
        !arg_shardings.empty() && AllOptionalShardingsAreSet(arg_shardings);
    for (BlockArgument& arg : block->getArguments()) {
      if (set_tuple_element_sharding)
        builder->SetSharding(*arg_shardings[arg.getArgNumber()]);
      lowering[arg] = xla::GetTupleElement(tuple, arg.getArgNumber());
    }
    builder->ClearSharding();
  } else {
    for (BlockArgument& arg : block->getArguments()) {
      auto num = arg.getArgNumber();
      xla::Shape shape = xla::TypeToShape(arg.getType());
      if (entry_args_same_across_replicas.empty()) {
        lowering[arg] =
            xla::Parameter(builder, num, shape, absl::StrCat("Arg_", num));
      } else {
        lowering[arg] = xla::Parameter(
            builder, num, shape, absl::StrCat("Arg_", num),
            std::vector<bool>(entry_args_same_across_replicas[num],
                              xla::ShapeUtil::GetLeafCount(shape)));
      }
    }
  }

  xla::XlaOp return_value;
  for (auto& inst : *block)
    if (failed(Lower(&inst, is_entry_function, ret_shardings, builder,
                     &lowering, &return_value)))
      return failure();

  // Build the XlaComputation and check for failures.
  auto computation_or =
      return_value.valid() ? builder->Build(return_value) : builder->Build();
  if (!computation_or.ok()) {
    block->back().emitError(
        llvm::Twine(computation_or.status().error_message()));
    return failure();
  }
  *result = std::move(computation_or.ValueOrDie());
  return success();
}

LogicalResult ConvertToHloModule::LowerRegionAsComputation(
    mlir::Region* region, xla::XlaComputation* func) {
  std::unique_ptr<xla::XlaBuilder> builder =
      module_builder_.CreateSubBuilder(absl::StrCat("region_", region_id_++));
  return LowerBasicBlockAsFunction(
      &region->front(), builder.get(),
      /*is_entry_function=*/false, /*entry_args_same_across_replicas=*/{},
      /*arg_shardings=*/{}, /*ret_shardings=*/{}, func);
}

std::string PaddingMapBadArrayAttrMsg(llvm::StringRef attr_name, int index) {
  return llvm::formatv(
             "requires '{0}' array attribute in '{1}' dict at arg {2}",
             attr_name, kPaddingMapAttr, index)
      .str();
}

std::string PaddingMapMismatchedArraySizeMsg(int arg_index,
                                             int shape_indices_size,
                                             int padding_arg_indices_size) {
  return llvm::formatv(
             "requires '{0}' and '{1}' array attributes in '{2}' dic at arg "
             "{3} to be of the same size, got sizes {4} and {5}",
             kShapeIndicesAttr, kPaddingArgIndicesAttr, kPaddingMapAttr,
             arg_index, shape_indices_size, padding_arg_indices_size)
      .str();
}

std::string PaddingMapBadIntAttrMsg(llvm::StringRef attr_name, int arg_index,
                                    int element_index) {
  return llvm::formatv(
             "requires element {0} in '{1}' array of '{2}' dict at arg {3} "
             "to be an int attribute",
             element_index, attr_name, kPaddingMapAttr, arg_index)
      .str();
}

std::string PaddingMapBadIndexMsg(llvm::StringRef attr_name, int arg_index,
                                  int element_index, int max, int32_t value) {
  return llvm::formatv(
             "requires element {0} in '{1}' array of '{2}' dict at arg {3} "
             "to be in range [0, {4}), got {5}",
             element_index, attr_name, kPaddingMapAttr, arg_index, max, value)
      .str();
}

std::string PaddingMapNegativeShapeIndexMsg(int arg_index, int element_index,
                                            int32_t value) {
  return llvm::formatv(
             "requires element {0} in '{1}' array of '{2}' dict at arg {3} to "
             "be non-negative, got {4}",
             element_index, kShapeIndicesAttr, kPaddingMapAttr, arg_index,
             value)
      .str();
}

std::string PaddingMapUniqueShapeIndexMsg(int arg_index, int element_index,
                                          int32_t value) {
  return llvm::formatv(
             "requires elements in '{0}' array of '{1}' dict at arg {2} to be "
             "unique, got duplicate element {3} at index {4}",
             kShapeIndicesAttr, kPaddingMapAttr, arg_index, value,
             element_index)
      .str();
}

void AddDynamicParameterBindingEntry(xla::DynamicParameterBindingProto* binding,
                                     int arg_index, int32_t shape_index,
                                     int32_t padding_arg_index,
                                     bool use_tuple_args) {
  auto* entry = binding->add_entries();
  entry->set_target_param_dim_num(shape_index);
  if (use_tuple_args) {
    entry->set_target_param_num(0);
    entry->add_target_param_index(arg_index);
    entry->set_dynamic_param_num(0);
    entry->add_dynamic_param_index(padding_arg_index);
  } else {
    entry->set_target_param_num(arg_index);
    entry->set_dynamic_param_num(padding_arg_index);
  }
}

// Validates and populates dynamic parameter bindings from a module's entry
// function `mhlo.padding_map` argument attributes to a `xla::HloModuleProto`
// `DynamicParameterBindingProto`.
LogicalResult AddDynamicParameterBindings(mlir::ModuleOp module,
                                          xla::HloModuleProto* hlo_module_proto,
                                          bool use_tuple_args) {
  auto entry_func = module.lookupSymbol<mlir::FuncOp>("main");
  if (!entry_func) return success();

  auto* dynamic_parameter_binding =
      hlo_module_proto->mutable_dynamic_parameter_binding();
  for (int i = 0, e = entry_func.getNumArguments(); i < e; ++i) {
    auto padding_map_attr = entry_func.getArgAttr(i, kPaddingMapAttr);
    if (!padding_map_attr) continue;
    auto padding_map = padding_map_attr.dyn_cast<DictionaryAttr>();
    if (!padding_map)
      return entry_func.emitError() << "requires '" << kPaddingMapAttr
                                    << "' dict attribute at arg " << i;

    auto shape_indices =
        padding_map.get(kShapeIndicesAttr).dyn_cast_or_null<ArrayAttr>();
    if (!shape_indices)
      return entry_func.emitError(
          PaddingMapBadArrayAttrMsg(kShapeIndicesAttr, i));

    auto padding_arg_indices =
        padding_map.get(kPaddingArgIndicesAttr).dyn_cast_or_null<ArrayAttr>();
    if (!padding_arg_indices)
      return entry_func.emitError(
          PaddingMapBadArrayAttrMsg(kPaddingArgIndicesAttr, i));

    if (shape_indices.size() != padding_arg_indices.size())
      return entry_func.emitError(PaddingMapMismatchedArraySizeMsg(
          i, shape_indices.size(), padding_arg_indices.size()));

    llvm::SmallDenseSet<int32_t, 4> used_shape_indices;
    auto arg_type =
        entry_func.getArgument(i).getType().dyn_cast<RankedTensorType>();
    for (auto shape_and_padding : llvm::enumerate(llvm::zip(
             shape_indices.getValue(), padding_arg_indices.getValue()))) {
      const int element_index = shape_and_padding.index();
      auto shape_index_attr =
          std::get<0>(shape_and_padding.value()).dyn_cast<IntegerAttr>();
      if (!shape_index_attr)
        return entry_func.emitError(
            PaddingMapBadIntAttrMsg(kShapeIndicesAttr, i, element_index));

      auto padding_arg_index_attr =
          std::get<1>(shape_and_padding.value()).dyn_cast<IntegerAttr>();
      if (!padding_arg_index_attr)
        return entry_func.emitError(
            PaddingMapBadIntAttrMsg(kPaddingArgIndicesAttr, i, element_index));

      const int32_t shape_index = shape_index_attr.getInt();
      if (arg_type && (shape_index < 0 || shape_index >= arg_type.getRank()))
        return entry_func.emitError(
            PaddingMapBadIndexMsg(kShapeIndicesAttr, i, element_index,
                                  arg_type.getRank(), shape_index));
      else if (shape_index < 0)
        return entry_func.emitError(
            PaddingMapNegativeShapeIndexMsg(i, element_index, shape_index));

      if (!used_shape_indices.insert(shape_index).second)
        return entry_func.emitError(
            PaddingMapUniqueShapeIndexMsg(i, element_index, shape_index));

      const int32_t padding_arg_index = padding_arg_index_attr.getInt();
      if (padding_arg_index < 0 || padding_arg_index >= e)
        return entry_func.emitError(PaddingMapBadIndexMsg(
            kPaddingArgIndicesAttr, i, element_index, e, padding_arg_index));

      Type padding_arg_type =
          entry_func.getArgument(padding_arg_index).getType();
      if (auto tensor_type = padding_arg_type.dyn_cast<RankedTensorType>())
        if (tensor_type.getRank() != 0)
          return entry_func.emitError()
                 << "requires arg " << padding_arg_index
                 << " to be a scalar for use as a dynamic parameter";

      if (!mlir::getElementTypeOrSelf(padding_arg_type).isSignlessInteger())
        return entry_func.emitError()
               << "requires arg " << padding_arg_index
               << " to be of an int type for use as a dynamic parameter";

      AddDynamicParameterBindingEntry(dynamic_parameter_binding, i, shape_index,
                                      padding_arg_index, use_tuple_args);
    }
  }

  return success();
}

// Runs the PrepareForExport pass on the ModuleOp.
Status PrepareForExport(mlir::ModuleOp module) {
  // Prepare for export to XLA HLO.
  mlir::PassManager pm(module.getContext());
  pm.addNestedPass<mlir::FuncOp>(mhlo::CreatePrepareForExport());
  if (failed(pm.run(module)))
    return tensorflow::errors::Internal("Unable to optimize for XLA export");
  return Status::OK();
}

}  // namespace

Status ConvertRegionToComputation(mlir::Region* region,
                                  xla::XlaComputation* func,
                                  MlirToHloConversionOptions options) {
  mlir::ModuleOp module;
  xla::XlaBuilder module_builder("main");
  ConvertToHloModule converter(module, module_builder, true, true, {}, options);
  if (failed(converter.LowerRegionAsComputation(region, func)))
    return tensorflow::errors::Internal(
        "failed to convert region to computation");
  return Status::OK();
}

Status ConvertMlirHloToHlo(
    mlir::ModuleOp module, xla::HloProto* hlo_proto, bool use_tuple_args,
    bool return_tuple,
    const tensorflow::XlaHelpers::ShapeRepresentationFn shape_representation_fn,
    MlirToHloConversionOptions options) {
  TF_RETURN_IF_ERROR(PrepareForExport(module));
  mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());
  xla::XlaBuilder module_builder("main");
  ConvertToHloModule converter(module, module_builder, use_tuple_args,
                               return_tuple, shape_representation_fn, options);
  if (failed(converter.Run())) return diag_handler.ConsumeStatus();
  auto hlo_module = converter.ConsumeMainProto();
  hlo_proto->mutable_hlo_module()->Swap(&hlo_module);
  if (failed(AddDynamicParameterBindings(
          module, hlo_proto->mutable_hlo_module(), use_tuple_args)))
    return diag_handler.ConsumeStatus();
  return Status::OK();
}

Status BuildHloFromMlirHlo(mlir::Block& block, xla::XlaBuilder& builder,
                           llvm::ArrayRef<xla::XlaOp> xla_params,
                           std::vector<xla::XlaOp>& returns,
                           MlirToHloConversionOptions options) {
  auto module = block.getParentOp()->getParentOfType<mlir::ModuleOp>();
  TF_RETURN_IF_ERROR(PrepareForExport(module));
  ConvertToHloModule converter(module, builder,
                               /*use_tuple_args=*/false, /*return_tuple=*/false,
                               /*shape_representation_fn=*/nullptr, options);

  ConvertToHloModule::ValueLoweringMap lowering;
  if (xla_params.size() != block.getArguments().size())
    return tensorflow::errors::Internal(
        "xla_params size != block arguments size");
  for (BlockArgument& arg : block.getArguments()) {
    auto num = arg.getArgNumber();
    lowering[arg] = xla_params[num];
  }

  mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());
  for (auto& inst : block) {
    if (isa<mhlo::ReturnOp, mlir::ReturnOp>(inst)) {
      returns.resize(inst.getNumOperands());
      for (OpOperand& ret : inst.getOpOperands()) {
        unsigned index = ret.getOperandNumber();
        xla::XlaOp operand;
        if (failed(GetXlaOp(ret.get(), lowering, &operand, &inst)))
          return diag_handler.ConsumeStatus();
        returns[index] = operand;
      }
    } else {
      xla::XlaOp return_value;
      if (failed(converter.Lower(&inst, /*is_entry_function=*/true,
                                 /*ret_shardings=*/{}, &builder, &lowering,
                                 &return_value)))
        return diag_handler.ConsumeStatus();
    }
  }

  return Status::OK();
}

DenseIntElementsAttr GetLayoutFromMlirHlo(mlir::Operation* op,
                                          llvm::StringRef attr_name) {
  CHECK((op->getDialect() ==
             op->getContext()->getLoadedDialect<mlir::mhlo::MhloDialect>() ||
         mlir::isa<mlir::ConstantOp, mlir::memref::TensorLoadOp,
                   mlir::memref::TensorStoreOp>(op)));
  return op->getAttrOfType<mlir::DenseIntElementsAttr>(attr_name);
}

}  // namespace mlir
