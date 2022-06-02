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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
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
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/utils/name_utils.h"
#include "tensorflow/compiler/mlir/xla/attribute_exporter.h"
#include "tensorflow/compiler/mlir/xla/transforms/xla_passes.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/tf2xla/layout_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/quantize.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/stream_executor/lib/statusor.h"

using ::int64_t;
using ::stream_executor::port::StatusOr;
using ::tensorflow::int16;
using ::tensorflow::int32;
using ::tensorflow::int8;
using ::tensorflow::uint16;
using ::tensorflow::uint32;
using ::tensorflow::uint64;
using ::tensorflow::uint8;

constexpr char kShapeIndicesAttr[] = "shape_indices";
constexpr char kPaddingArgIndicesAttr[] = "padding_arg_indices";
constexpr char kShardingAttr[] = "mhlo.sharding";
constexpr char kFrontendAttributesAttr[] = "mhlo.frontend_attributes";
constexpr char kReplicationAttr[] = "mhlo.is_same_data_across_replicas";

// Array attribute. Same shape as infeed result, but contains a
// minor_to_major array for every tensor.
constexpr char kLayoutAttr[] = "layout";
constexpr char kDefaultLayoutAttrName[] = "xla_shape";

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

static std::vector<int64_t> ConvertDenseIntAttr(
    mlir::DenseIntElementsAttr attr) {
  auto values = attr.getValues<int64_t>();
  return {values.begin(), values.end()};
}

static std::vector<int64_t> ConvertDenseIntAttr(
    llvm::Optional<mlir::DenseIntElementsAttr> attr) {
  if (!attr) return {};
  return ConvertDenseIntAttr(*attr);
}

// Converts the broadcast_dimensions attribute into a vector of dimension
// numbers (empty if the attribute is absent).
static std::vector<int64_t> Convert_broadcast_dimensions(
    llvm::Optional<mlir::DenseIntElementsAttr> broadcast_dimensions) {
  if (!broadcast_dimensions.hasValue()) return {};

  return ConvertDenseIntAttr(*broadcast_dimensions);
}

// Converts StringRef to xla FftType enum
static xla::FftType Convert_fft_type(mlir::mhlo::FftType fft_type) {
  xla::FftType fft_type_enum;
  // Illegal fft_type string would be caught by the verifier, so 'FftType_Parse'
  // call below should never return false.
  if (!FftType_Parse(std::string(mlir::mhlo::stringifyFftType(fft_type)),
                     &fft_type_enum))
    return xla::FftType::FFT;
  return fft_type_enum;
}

static std::vector<std::pair<int64_t, int64_t>> Convert_padding(
    llvm::Optional<mlir::DenseIntElementsAttr> padding) {
  return xla::ConvertNx2Attribute(padding).ValueOrDie();
}

static std::vector<std::pair<int64_t, int64_t>> Convert_source_target_pairs(
    llvm::Optional<mlir::DenseIntElementsAttr> source_target_pairs) {
  return xla::ConvertNx2Attribute(source_target_pairs).ValueOrDie();
}

static std::vector<xla::ReplicaGroup> Convert_replica_groups(
    mlir::DenseIntElementsAttr groups) {
  return xla::ConvertReplicaGroups(groups).ValueOrDie();
}

// Converts types and corresponding layouts into xla shapes with layouts.
static std::vector<xla::Shape> ConvertTypesToShapesWithLayout(
    mlir::TypeRange value_types, mlir::ArrayAttr layouts) {
  std::vector<xla::Shape> shapes_with_layout;
  for (auto type_and_layout : llvm::zip(value_types, layouts)) {
    mlir::Type type = std::get<0>(type_and_layout);
    mlir::Attribute layout = std::get<1>(type_and_layout);
    assert(!type.isa<mlir::TupleType>() &&
           "Exporting layout for tuples is not implemented yet");
    shapes_with_layout.emplace_back(xla::TypeToShape(type));
    auto& shape = shapes_with_layout.back();
    shape.mutable_layout()->clear_minor_to_major();
    for (auto l : layout.cast<mlir::DenseIntElementsAttr>()) {
      shape.mutable_layout()->mutable_minor_to_major()->push_back(
          l.getSExtValue());
    }
  }
  return shapes_with_layout;
}

// CustomCallOp result can be of tuple type to pack multiple results into one
// value. If the custom call result is a tuple, then result layouts represent
// the layout of each element of the tuple. Nested tuples are currently not
// supported for export.
static xla::Shape GetCustomCallResultShapeWithLayout(mlir::Type type,
                                                     mlir::ArrayAttr layouts) {
  auto tuple_type = type.dyn_cast<mlir::TupleType>();
  if (!tuple_type) return ConvertTypesToShapesWithLayout({type}, layouts)[0];

  std::vector<xla::Shape> shapes_with_layouts =
      ConvertTypesToShapesWithLayout(tuple_type.getTypes(), layouts);
  return xla::ShapeUtil::MakeTupleShape(shapes_with_layouts);
}

// Converts StringRef to xla Transpose enum.
static xla::TriangularSolveOptions::Transpose Convert_transpose_a(
    mlir::mhlo::Transpose transpose) {
  return xla::ConvertTranspose(mlir::mhlo::stringifyTranspose(transpose))
      .ValueOrDie();
}

static xla::Layout ExtractLayout(
    mlir::Operation* op, int rank,
    llvm::StringRef attr_name = kDefaultLayoutAttrName) {
  if (auto attr = op->getAttrOfType<mlir::DenseIntElementsAttr>(attr_name)) {
    llvm::SmallVector<int64_t, 4> minor_to_major;
    DCHECK_EQ(rank, attr.size());
    minor_to_major.reserve(attr.size());
    for (const llvm::APInt& i : attr) {
      minor_to_major.push_back(i.getZExtValue());
    }
    return xla::LayoutUtil::MakeLayout(minor_to_major);
  }
  return xla::LayoutUtil::MakeDescendingLayout(rank);
}

static xla::Shape ExtractXlaShape(mlir::Operation* op) {
  if (auto attr = op->getAttrOfType<mlir::StringAttr>(kDefaultLayoutAttrName)) {
    return *xla::ParseShape(
        absl::string_view(attr.getValue().data(), attr.getValue().size()));
  } else {
    std::vector<xla::Shape> subshapes;
    for (mlir::Value result : op->getResults()) {
      subshapes.push_back(xla::TypeToShape(result.getType()));
    }
    if (subshapes.size() > 1) {
      return xla::ShapeUtil::MakeTupleShape(subshapes);
    }
    return subshapes[0];
  }
}

#define I64_ELEMENTS_ATTR_TO_VECTOR(attribute)                \
  static std::vector<int64_t> Convert_##attribute(            \
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

static std::vector<int64_t> Convert_ArrayRef(llvm::ArrayRef<int64_t> values) {
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
    auto operand_precision =
        mlir::mhlo::stringifyPrecision(
            attr.cast<mlir::mhlo::PrecisionAttr>().getValue())
            .str();
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
    mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers_attr) {
  xla::DotDimensionNumbers dot_dimension_numbers;

  auto rhs_contracting_dimensions =
      dot_dimension_numbers_attr.getRhsContractingDimensions();
  auto lhs_contracting_dimensions =
      dot_dimension_numbers_attr.getLhsContractingDimensions();
  auto rhs_batch_dimensions =
      dot_dimension_numbers_attr.getRhsBatchingDimensions();
  auto lhs_batch_dimensions =
      dot_dimension_numbers_attr.getLhsBatchingDimensions();

  for (const auto& val : rhs_contracting_dimensions) {
    dot_dimension_numbers.add_rhs_contracting_dimensions(val);
  }
  for (const auto& val : lhs_contracting_dimensions) {
    dot_dimension_numbers.add_lhs_contracting_dimensions(val);
  }

  for (const auto& val : rhs_batch_dimensions) {
    dot_dimension_numbers.add_rhs_batch_dimensions(val);
  }

  for (const auto& val : lhs_batch_dimensions) {
    dot_dimension_numbers.add_lhs_batch_dimensions(val);
  }

  return dot_dimension_numbers;
}

static xla::ConvolutionDimensionNumbers Convert_dimension_numbers(
    mlir::mhlo::ConvDimensionNumbersAttr input) {
  return xla::ConvertConvDimensionNumbers(input);
}

xla::ChannelHandle Convert_channel_handle(mlir::mhlo::ChannelHandle attr) {
  xla::ChannelHandle channel_handle;
  channel_handle.set_handle(ConvertAPInt(attr.handle().getValue()));
  channel_handle.set_type(static_cast<xla::ChannelHandle::ChannelType>(
      ConvertAPInt(attr.type().getValue())));
  return channel_handle;
}

absl::optional<xla::ChannelHandle> Convert_channel_handle(
    llvm::Optional<mlir::mhlo::ChannelHandle> attr) {
  if (!attr.hasValue()) return absl::nullopt;
  return Convert_channel_handle(attr.getValue());
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
    mlir::mhlo::GatherDimensionNumbersAttr input) {
  xla::GatherDimensionNumbers output;

  auto offset_dims = input.getOffsetDims();
  std::copy(offset_dims.begin(), offset_dims.end(),
            tensorflow::protobuf::RepeatedFieldBackInserter(
                output.mutable_offset_dims()));

  auto collapsed_slice_dims = input.getCollapsedSliceDims();
  std::copy(collapsed_slice_dims.begin(), collapsed_slice_dims.end(),
            tensorflow::protobuf::RepeatedFieldBackInserter(
                output.mutable_collapsed_slice_dims()));

  auto start_index_map = input.getStartIndexMap();
  std::copy(start_index_map.begin(), start_index_map.end(),
            tensorflow::protobuf::RepeatedFieldBackInserter(
                output.mutable_start_index_map()));

  output.set_index_vector_dim(input.getIndexVectorDim());
  return output;
}

static xla::ScatterDimensionNumbers Convert_scatter_dimension_numbers(
    mlir::mhlo::ScatterDimensionNumbersAttr input) {
  xla::ScatterDimensionNumbers output;

  auto update_window_dims = input.getUpdateWindowDims();
  std::copy(update_window_dims.begin(), update_window_dims.end(),
            tensorflow::protobuf::RepeatedFieldBackInserter(
                output.mutable_update_window_dims()));

  auto inserted_window_dims = input.getInsertedWindowDims();
  std::copy(inserted_window_dims.begin(), inserted_window_dims.end(),
            tensorflow::protobuf::RepeatedFieldBackInserter(
                output.mutable_inserted_window_dims()));

  auto scatter_dims_to_operand_dims = input.getScatterDimsToOperandDims();
  std::copy(scatter_dims_to_operand_dims.begin(),
            scatter_dims_to_operand_dims.end(),
            tensorflow::protobuf::RepeatedFieldBackInserter(
                output.mutable_scatter_dims_to_operand_dims()));

  output.set_index_vector_dim(input.getIndexVectorDim());
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
    if (auto value_str_attr = attr.getValue().dyn_cast<mlir::StringAttr>())
      frontend_attributes.mutable_map()->insert(
          {attr.getName().str(), value_str_attr.getValue().str()});

  return frontend_attributes;
}

// Returns a OpMetadata proto based on the location of the op. If the location
// is unknown, an empty proto is returned. `op_name` are populated with the op
// location (converted). FileLineColLoc locations are populated by taking the
// file name and line number, and populating `source_file` and `source_line`
// respectively.
static xla::OpMetadata CreateOpMetadataFromLocation(
    mlir::Operation* op, mlir::MlirToHloConversionOptions options) {
  xla::OpMetadata metadata;
  mlir::Location loc = op->getLoc();
  if (loc.isa<mlir::UnknownLoc>()) return metadata;

  std::string name = mlir::GetNameFromLoc(loc);
  if (options.legalize_node_names) {
    mlir::LegalizeNodeName(name);
  }
  metadata.set_op_name(name);
  std::string op_type = mlir::GetOpTypeFromLoc(loc);
  mlir::LegalizeNodeName(op_type);
  metadata.set_op_type(op_type);

  if (auto name_loc = op->getLoc().dyn_cast<mlir::NameLoc>()) {
    loc = name_loc.getChildLoc();
    if (loc.isa<mlir::UnknownLoc>()) return metadata;
  }

  if (auto file_line_col_loc = loc.dyn_cast<mlir::FileLineColLoc>()) {
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
    mlir::func::FuncOp function,
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
  using FunctionLoweringMap =
      llvm::DenseMap<mlir::func::FuncOp, xla::XlaComputation>;

  // If use_tuple_args is true, then the entry function's arguments are
  // converted to a tuple and passed as a single parameter.
  // Similarly, if return tuple is true, then the entry function's return values
  // are converted to a tuple even when there is only a single return value.
  // Multiple return values are always converted to a tuple and returned as a
  // single value.
  explicit ConvertToHloModule(
      mlir::ModuleOp module, xla::XlaBuilder& module_builder,
      bool use_tuple_args, bool return_tuple,
      tensorflow::XlaShapeLayoutHelpers::ShapeDeterminationFns
          shape_determination_fns,
      MlirToHloConversionOptions options)
      : module_(module),
        module_builder_(module_builder),
        use_tuple_args_(use_tuple_args),
        return_tuple_(return_tuple),
        shape_determination_fns_(shape_determination_fns),
        options_(options) {}

  // Perform the lowering to XLA. This function returns failure if an error was
  // encountered.
  //
  // TODO(hinsu): Check for dynamic shapes and exit instead of crashing.
  LogicalResult Run() {
    auto main = module_.lookupSymbol<mlir::func::FuncOp>("main");
    if (!main)
      return module_.emitError(
          "conversion requires module with `main` function");

    for (auto func : module_.getOps<func::FuncOp>()) {
      if (func.empty()) continue;
      if (failed(RunOnFunction(func))) return failure();
    }
    return success();
  }

  // Lower a specific function to HLO.
  LogicalResult RunOnFunction(mlir::func::FuncOp f);

  // Lower a `mlir::Region` to a `XlaComputation`
  LogicalResult LowerRegionAsComputation(
      mlir::Region* region, xla::XlaComputation* func,
      llvm::Optional<llvm::ArrayRef<mlir::Value>> implicit_operands =
          llvm::None,
      bool ensure_single_arg = false);

  // Lower a single `Block` to a `XlaComputation`
  LogicalResult LowerBasicBlockAsFunction(
      Block* block, xla::XlaBuilder* builder, bool is_entry_function,
      bool ensure_single_arg,
      const std::vector<bool>& entry_args_same_across_replicas,
      llvm::ArrayRef<absl::optional<xla::OpSharding>> arg_shardings,
      llvm::ArrayRef<absl::optional<xla::OpSharding>> ret_shardings,
      xla::XlaComputation* result,
      llvm::Optional<llvm::ArrayRef<mlir::Value>> implicit_operands =
          llvm::None);

  ::xla::HloModuleProto ConsumeMainProto() {
    auto main = module_.lookupSymbol<mlir::func::FuncOp>("main");
    // This is an invariant check as Run returns failure if there is no main
    // function and so the main proto shouldn't be consumed in that case.
    CHECK(main) << "requires module to have main function";  // Crash Ok.
    return lowered_computation_[main].proto();
  }

  // Lower function call to HLO call instruction
  LogicalResult LowerFunctionCall(
      mlir::func::CallOp call_op, xla::XlaBuilder* builder,
      ConvertToHloModule::ValueLoweringMap* value_lowering);

  // Look up a symbol with the specified name, returning null if no such name
  // exists.
  func::FuncOp LookUpSymbol(FlatSymbolRefAttr symbol) {
    return module_.lookupSymbol<mlir::func::FuncOp>(symbol);
  }

  // Get Reference to lowered XLA computation for a function.
  xla::XlaComputation& GetLoweredComputation(func::FuncOp func) {
    return lowered_computation_[func];
  }

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

  // Shape determination functions to determine entry function argument and
  // result shapes.
  tensorflow::XlaShapeLayoutHelpers::ShapeDeterminationFns
      shape_determination_fns_;

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

mlir::LogicalResult GetXlaOps(mlir::Operation* op,
                              llvm::ArrayRef<mlir::Value> values,
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

LogicalResult ExportXlaOp(ComputeReshapeShapeOp, OpLoweringContext) {
  // This op has no expression in the legacy export format. It can be expanded
  // to a sequence of operations if needed in the future, but would feed into
  // ops creating unsupported dynamic shapes.
  return failure();
}

LogicalResult ExportXlaOp(CstrReshapableOp, OpLoweringContext) {
  // This op has no expression in the legacy export format.
  return failure();
}

LogicalResult ExportXlaOp(AddDependencyOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp token;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.token(), value_map, &token, op))) return failure();
  if (failed(GetXlaOp(op.operand(), value_map, &operand, op))) return failure();
  auto operand_shape = ctx.builder->GetShape(operand).ConsumeValueOrDie();
  value_map[op] = xla::internal::XlaBuilderFriend::BuildAddDependency(
      ctx.builder, operand, token, operand_shape);
  return success();
}

LogicalResult ExportXlaOp(AllGatherOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.operand(), value_map, &operand, op))) return failure();
  TensorType operand_type = op.operand().getType().cast<TensorType>();
  TensorType result_type = op.getType();
  if (!operand_type.hasStaticShape() || !result_type.hasStaticShape())
    return failure();
  auto all_gather_dim = op.all_gather_dim();
  int64_t shard_count = result_type.getDimSize(all_gather_dim) /
                        operand_type.getDimSize(all_gather_dim);
  value_map[op] = xla::AllGather(operand, all_gather_dim, shard_count,
                                 Convert_replica_groups(op.replica_groups()),
                                 Convert_channel_handle(op.channel_handle()));
  return success();
}

LogicalResult ExportXlaOp(AllReduceOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaComputation computation;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.computation(),
                                                     &computation))) {
    return failure();
  }

  xla::XlaOp operand;
  if (failed(GetXlaOp(op.operand(), value_map, &operand, op))) return failure();

  value_map[op] = xla::AllReduce(operand, computation,
                                 Convert_replica_groups(op.replica_groups()),
                                 Convert_channel_handle(op.channel_handle()));
  return success();
}

LogicalResult ExportXlaOp(ReduceScatterOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.operand(), value_map, &operand, op))) return failure();
  TensorType operand_type = op.operand().getType().cast<TensorType>();
  TensorType result_type = op.getType();
  if (!operand_type.hasStaticShape() || !result_type.hasStaticShape())
    return failure();
  auto scatter_dim = op.scatter_dimension();
  int64_t shard_count = operand_type.getDimSize(scatter_dim) /
                        result_type.getDimSize(scatter_dim);

  xla::XlaComputation computation;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.computation(),
                                                     &computation))) {
    return failure();
  }

  value_map[op] =
      xla::ReduceScatter(operand, computation, scatter_dim, shard_count,
                         Convert_replica_groups(op.replica_groups()),
                         Convert_channel_handle(op.channel_handle()));
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

LogicalResult ExportXlaOp(DotOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp lhs, rhs;
  if (failed(GetXlaOp(op.lhs(), value_map, &lhs, op))) return mlir::failure();
  if (failed(GetXlaOp(op.rhs(), value_map, &rhs, op))) return mlir::failure();
  xla::PrimitiveType preferred_element_type =
      xla::TypeToPrimitiveType(getElementTypeOrSelf(op.getType()));
  value_map[op] = xla::Dot(
      lhs, rhs, Unwrap(Convert_precision_config(op.precision_config())),
      preferred_element_type);
  return mlir::success();
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

  // mhlo.IfOp does not have any operands or blocks-arguments. The computation
  // inside the region-blocks use implicit captures of values defined above.
  // In order to create the xla parameters for functions corresponding to
  // IfOp regions, we need to infer the a region-block's arguments, using all
  // the values used in the region but defined above. Note that in case there
  // are zero implicit capture for a region, we use an empty tuple as the xla
  // parameter.
  //
  // Note that the implicit values used in true and false branch regions might
  // be different and, as a result, the xla parameters for the corresponding
  // regions could have different shapes.
  llvm::SetVector<mlir::Value> implicit_true_operand_set,
      implicit_false_operand_set;
  getUsedValuesDefinedAbove(op.true_branch(), op.true_branch(),
                            implicit_true_operand_set);
  getUsedValuesDefinedAbove(op.false_branch(), op.false_branch(),
                            implicit_false_operand_set);

  llvm::SmallVector<mlir::Value> implicit_true_operands(
      implicit_true_operand_set.begin(), implicit_true_operand_set.end());
  llvm::SmallVector<mlir::Value> implicit_false_operands(
      implicit_false_operand_set.begin(), implicit_false_operand_set.end());

  // Create xla parameters for functions corresponding to ifOp regions using the
  // implicit captures operands. Also export the instructions within those
  // regions.
  if (failed(ctx.converter->LowerRegionAsComputation(
          &op.true_branch(), &true_branch,
          llvm::makeArrayRef(implicit_true_operands),
          /*ensure_single_arg*/ true)) ||
      failed(ctx.converter->LowerRegionAsComputation(
          &op.false_branch(), &false_branch,
          llvm::makeArrayRef(implicit_false_operands),
          /*ensure_single_arg*/ true))) {
    return failure();
  }

  // Create the Xla pred argument.
  xla::XlaOp pred;
  if (failed(GetXlaOp(op.pred(), value_map, &pred, op))) return failure();

  // Create the true branch Xla argument.
  llvm::SmallVector<xla::XlaOp> true_args;
  if (failed(GetXlaOps(op, implicit_true_operands, ctx, true_args)))
    return failure();
  xla::XlaOp true_arg =
      true_args.size() == 1 ? true_args[0] : Tuple(ctx.builder, true_args);

  // Create the false branch Xla argument.
  llvm::SmallVector<xla::XlaOp> false_args;
  if (failed(GetXlaOps(op, implicit_false_operands, ctx, false_args)))
    return failure();
  xla::XlaOp false_arg =
      false_args.size() == 1 ? false_args[0] : Tuple(ctx.builder, false_args);

  // Create XLA Conditional op.
  auto ifop =
      xla::Conditional(pred, true_arg, true_branch, false_arg, false_branch);

  // mhlo.IfOp have multiple returns, untuple all the results of XLA's.
  if (op.getNumResults() == 1) {
    value_map[op.getResult(0)] = ifop;
  } else {
    for (const auto& item : llvm::enumerate(op.getResults())) {
      value_map[item.value()] = xla::GetTupleElement(ifop, item.index());
    }
  }

  return success();
}

LogicalResult ExportXlaOp(CaseOp op, OpLoweringContext ctx) {
  llvm::DenseMap<mlir::Value, xla::XlaOp>& value_map = *ctx.values;
  // OperandRange operands = op.branch_operands();
  MutableArrayRef<Region> branches = op.branches();
  llvm::SmallVector<xla::XlaOp, 4> branch_operands(branches.size());
  std::vector<xla::XlaComputation> computations(branches.size());
  std::vector<xla::XlaComputation*> computations_p(branches.size());

  // mhlo.CaseOp does not have any operands or blocks-arguments. The computation
  // inside the region-blocks use implicit captures of values defined above.
  // In order to create the xla parameters for functions corresponding to
  // CaseOp regions, we need to infer the a region-block's arguments, using all
  // the values used in the region but defined above. Note that in case there
  // are zero implicit captures for a region, we use an empty tuple as the xla
  // parameter.
  //
  // Note that the implicit values used in the regions might
  // be different and, as a result, the xla parameters for the corresponding
  // regions could have different shapes.
  for (unsigned i = 0; i < branches.size(); ++i) {
    llvm::SetVector<mlir::Value> implicit_operand_set;
    getUsedValuesDefinedAbove(branches[i], branches[i], implicit_operand_set);
    llvm::SmallVector<mlir::Value> implicit_operands(
        implicit_operand_set.begin(), implicit_operand_set.end());

    // Create the branches[i]'s Xla argument.
    llvm::SmallVector<xla::XlaOp> args;
    if (failed(GetXlaOps(op, implicit_operands, ctx, args))) return failure();
    branch_operands[i] = args.size() == 1 ? args[0] : Tuple(ctx.builder, args);

    // Create xla parameters for functions corresponding to region branches[i]
    // using the implicit captures operands. Also export the instructions within
    // that region.
    computations_p[i] = &computations[i];
    if (failed(ctx.converter->LowerRegionAsComputation(
            &branches[i], computations_p[i],
            llvm::makeArrayRef(implicit_operands),
            /*ensure_single_arg*/ true)))
      return failure();
  }

  xla::XlaOp index;
  if (failed(GetXlaOp(op.index(), value_map, &index, op))) return failure();

  xla::XlaOp caseop = xla::Conditional(index, computations_p, branch_operands);

  // mhlo.CaseOp have multiple returns, untuple all the results of XLA's.
  if (op.getNumResults() == 1) {
    value_map[op.getResult(0)] = caseop;
  } else {
    for (const auto& item : llvm::enumerate(op.getResults())) {
      value_map[item.value()] = xla::GetTupleElement(caseop, item.index());
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
  auto dir = Convert_comparison_direction(
      mlir::mhlo::stringifyComparisonDirection(op.comparison_direction()));
  auto type_attr = op.compare_typeAttr();

  xla::XlaOp xla_result;
  if (type_attr && type_attr.getValue() != mlir::mhlo::ComparisonType::NOTYPE) {
    auto type = xla::StringToComparisonType(
                    stringifyComparisonType(type_attr.getValue()).str())
                    .ValueOrDie();
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
      xla::ConvertConvDimensionNumbers(op.dimension_numbers()),
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
  if (op.getNumResults() != 1)
    return op.emitOpError() << "with multiple results cannot be exported";

  if (op.called_computations().size() > 1)
    return op.emitOpError()
           << "cannot export with more than one called computations";

  // Custom call can be exported either with called computation or with layout
  // attributes. The XlaBuilder API does not allow both.
  if (!op.called_computations().empty() && op.operand_layouts() &&
      op.result_layouts()) {
    return op.emitOpError() << "cannot export if both called computation and "
                               "layouts are specified";
  }

  Value result = op.getResult(0);
  llvm::SmallVector<xla::XlaOp> args;
  if (failed(GetTuple(op, op.operands(), ctx, args))) return failure();
  auto xla_api_version = xla::ConvertCustomCallApiVersion(op.api_version());
  if (!xla_api_version.ok()) return failure();
  auto& value_map = *ctx.values;

  if (op.called_computations().size() == 1) {
    mlir::func::FuncOp callee = ctx.converter->LookUpSymbol(
        op.called_computations()[0].cast<FlatSymbolRefAttr>());
    if (failed(ctx.converter->RunOnFunction(callee))) return failure();
    xla::XlaComputation& computation =
        ctx.converter->GetLoweredComputation(callee);
    value_map[result] = xla::CustomCallWithComputation(
        ctx.builder, std::string(op.call_target_name()), args, computation,
        xla::TypeToShape(result.getType()), std::string(op.backend_config()),
        op.has_side_effect(),
        /*output_operand_aliasing=*/{},
        /*literal=*/nullptr,
        /*schedule=*/xla::CustomCallSchedule::SCHEDULE_NONE,
        /*api_version=*/*xla_api_version);
    return success();
  }

  if (op.operand_layouts() && op.result_layouts()) {
    auto operand_shapes_with_layout = ConvertTypesToShapesWithLayout(
        op.getOperandTypes(), op.operand_layouts().getValue());
    xla::Shape result_shape_with_layout = GetCustomCallResultShapeWithLayout(
        result.getType(), op.result_layouts().getValue());
    value_map[result] = xla::CustomCallWithLayout(
        ctx.builder, std::string(op.call_target_name()), args,
        result_shape_with_layout, operand_shapes_with_layout,
        std::string(op.backend_config()), op.has_side_effect(),
        /*output_operand_aliasing=*/{},
        /*literal=*/nullptr,
        /*schedule=*/xla::CustomCallSchedule::SCHEDULE_NONE,
        /*api_version=*/*xla_api_version);
    return success();
  }

  value_map[result] = xla::CustomCall(
      ctx.builder, std::string(op.call_target_name()), args,
      xla::TypeToShape(result.getType()), std::string(op.backend_config()),
      op.has_side_effect(), /*output_operand_aliasing=*/{},
      /*literal=*/nullptr,
      /*schedule=*/xla::CustomCallSchedule::SCHEDULE_NONE,
      /*api_version=*/*xla_api_version);
  return success();
}

LogicalResult ExportXlaOp(InfeedOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp token;
  if (failed(GetXlaOp(op.token(), value_map, &token, op))) return failure();

  // mhlo.infeed produces multiple results. The shape argument expected by the
  // xla client API is a tuple type with two element-types:
  // data_type : A tuple containing all the mhlo.infeedOp result types except
  //             the token type.
  // token_type : The last result type of mhlo.infeedOp.
  auto result_types = op.getResultTypes();
  auto num_results = op.getNumResults();

  xla::Shape token_shape = xla::TypeToShape(result_types[num_results - 1]);
  std::vector<xla::Shape> subshapes;
  for (const auto& item : llvm::enumerate(result_types)) {
    if (item.index() == num_results - 1) break;
    subshapes.push_back(xla::TypeToShape(item.value()));
  }

  xla::Shape data_shape = xla::ShapeUtil::MakeTupleShape(subshapes);
  auto xla_result =
      xla::InfeedWithToken(token, data_shape, std::string(op.infeed_config()));
  ctx.builder->ClearSharding();

  if (!subshapes.empty()) {
    auto data_tuple_element = xla::GetTupleElement(xla_result, 0);
    for (const auto& item : llvm::enumerate(op.getResults())) {
      if (item.index() == num_results - 1) break;
      value_map[item.value()] =
          xla::GetTupleElement(data_tuple_element, item.index());
    }
  }

  value_map[op.getResult(num_results - 1)] =
      xla::GetTupleElement(xla_result, 1);

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

  llvm::SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op, op.operands(), ctx, operands))) return failure();

  xla::XlaOp operand = Tuple(ctx.builder, operands);

  std::vector<xla::Shape> subshapes;
  for (auto operand : op.operands())
    subshapes.push_back(xla::TypeToShape(operand.getType()));

  xla::Shape shape_with_layout = xla::ShapeUtil::MakeTupleShape(subshapes);

  xla::XlaOp token;
  if (failed(GetXlaOp(op.token(), value_map, &token, op))) return failure();

  value_map[op] = xla::OutfeedWithToken(operand, token, shape_with_layout,
                                        std::string(op.outfeed_config()));
  return success();
}

LogicalResult ExportXlaOp(PartitionIdOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::Shape shape = xla::TypeToShape(op.getResult().getType());
  value_map[op] =
      xla::internal::XlaBuilderFriend::BuildPartitionId(ctx.builder, shape);
  return success();
}

LogicalResult ExportXlaOp(PadOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::PaddingConfig padding_config;
  auto edge_padding_low = ConvertDenseIntAttr(op.edge_padding_low());
  auto edge_padding_high = ConvertDenseIntAttr(op.edge_padding_high());
  auto interior_padding = ConvertDenseIntAttr(op.interior_padding());
  for (int64_t i = 0, end = edge_padding_low.size(); i < end; ++i) {
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

  xla::XlaOp token;
  if (failed(GetXlaOp(op.token(), value_map, &token, op))) return failure();

  // mhlo.recvOp produces multiple results. The shape argument expected by the
  // xla client API is a tuple type with two element-types:
  // data_type : A tuple containing all the mhlo.RecvOp result types except
  //             the token type.
  // token_type : The last result type of mhlo.recvOp.
  auto result_types = op.getResultTypes();
  auto num_results = op.getNumResults();

  xla::Shape token_shape = xla::TypeToShape(result_types[num_results - 1]);
  std::vector<xla::Shape> subshapes;
  for (const auto& item : llvm::enumerate(result_types)) {
    if (item.index() == num_results - 1) break;
    subshapes.push_back(xla::TypeToShape(item.value()));
  }

  xla::Shape data_shape;
  if (subshapes.size() == 1)
    data_shape = subshapes[0];
  else
    data_shape = xla::ShapeUtil::MakeTupleShape(subshapes);

  xla::XlaOp xla_result;
  if (op.is_host_transfer()) {
    xla_result = xla::RecvFromHost(token, data_shape,
                                   Convert_channel_handle(op.channel_handle()));
  } else {
    xla_result = xla::RecvWithToken(
        token, data_shape, Convert_channel_handle(op.channel_handle()));
  }

  auto data_tuple_element = xla::GetTupleElement(xla_result, 0);
  if (subshapes.size() == 1) {
    value_map[op.getResult(0)] = data_tuple_element;
  } else {
    for (const auto& item : llvm::enumerate(op.getResults())) {
      if (item.index() == num_results - 1) break;
      value_map[item.value()] =
          xla::GetTupleElement(data_tuple_element, item.index());
    }
  }

  value_map[op.getResult(num_results - 1)] =
      xla::GetTupleElement(xla_result, 1);

  return success();
}

LogicalResult ExportXlaOp(ReduceOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaComputation body;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.body(), &body))) {
    return failure();
  }
  llvm::SmallVector<xla::XlaOp> operands, init_values;
  if (failed(GetTuple(op, op.operands(), ctx, operands)) ||
      failed(GetTuple(op, op.init_values(), ctx, init_values))) {
    return failure();
  }
  xla::XlaOp result =
      xla::Reduce(ctx.builder, operands, init_values, body,
                  Convert_broadcast_dimensions(op.dimensions()));
  if (op.getNumResults() == 1) {
    value_map[op.getResult(0)] = result;
  } else {
    for (const auto& item : llvm::enumerate(op.getResults())) {
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
  if (failed(GetTuple(op, op.operands(), ctx, operands)) ||
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
    for (const auto& item : llvm::enumerate(op.getResults())) {
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
  auto results = op.getResults();
  auto xla_arg_1 = value_map[*op.getODSOperands(0).begin()];
  auto xla_result = xla::RngBitGenerator(
      static_cast<xla::RandomAlgorithm>(op.rng_algorithm()), Unwrap(xla_arg_1),
      xla::TypeToShape(results[1].getType()));

  for (const auto& item : llvm::enumerate(results))
    value_map[item.value()] = xla::GetTupleElement(xla_result, item.index());

  return mlir::success();
}

LogicalResult ExportXlaOp(XlaRngGetAndUpdateStateOp op, OpLoweringContext ctx) {
  // This op does not exist in the XLA builder interface.
  (*ctx.values)[op.getResult()] =
      xla::internal::XlaBuilderFriend::BuildRngGetAndUpdateState(
          ctx.builder, static_cast<int64_t>(op.delta()),
          xla::TypeToShape(op.getType()));
  return mlir::success();
}

LogicalResult ExportXlaOp(BatchNormGradOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto results = op.getResults();

  xla::XlaOp operand, scale, mean, variance, grad_output;
  if (failed(GetXlaOp(op.operand(), value_map, &operand, op))) return failure();
  if (failed(GetXlaOp(op.scale(), value_map, &scale, op))) return failure();
  if (failed(GetXlaOp(op.mean(), value_map, &mean, op))) return failure();
  if (failed(GetXlaOp(op.variance(), value_map, &variance, op)))
    return failure();
  if (failed(GetXlaOp(op.grad_output(), value_map, &grad_output, op)))
    return failure();

  auto xla_result =
      xla::BatchNormGrad(operand, scale, mean, variance, grad_output,
                         ConvertAPFloat(op.epsilon()), op.feature_index());

  for (const auto& item : llvm::enumerate(results))
    value_map[item.value()] = xla::GetTupleElement(xla_result, item.index());

  return mlir::success();
}

LogicalResult ExportXlaOp(BatchNormTrainingOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto results = op.getResults();

  xla::XlaOp operand, scale, offset;
  if (failed(GetXlaOp(op.operand(), value_map, &operand, op))) return failure();
  if (failed(GetXlaOp(op.scale(), value_map, &scale, op))) return failure();
  if (failed(GetXlaOp(op.offset(), value_map, &offset, op))) return failure();

  auto xla_result = xla::BatchNormTraining(
      operand, scale, offset, ConvertAPFloat(op.epsilon()), op.feature_index());

  for (const auto& item : llvm::enumerate(results))
    value_map[item.value()] = xla::GetTupleElement(xla_result, item.index());

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

  llvm::SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op, op.operands(), ctx, operands))) return failure();

  xla::XlaOp operand;
  if (operands.size() == 1)
    operand = operands[0];
  else
    operand = Tuple(ctx.builder, operands);

  llvm::SmallVector<xla::XlaOp> many_updates;
  if (failed(GetTuple(op, op.updates(), ctx, many_updates))) return failure();

  xla::XlaOp updates;
  if (many_updates.size() == 1)
    updates = many_updates[0];
  else
    updates = Tuple(ctx.builder, many_updates);
  xla::XlaOp scatter_indices;

  if (failed(GetXlaOp(op.scatter_indices(), value_map, &scatter_indices, op)))
    return failure();

  // Export of varadic scatter currently not supported
  value_map[op.getResult(0)] = xla::Scatter(
      operand, scatter_indices, updates, update_computation, dimension_numbers,
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

  llvm::SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op, op.operands(), ctx, operands))) return failure();

  xla::XlaOp operand;
  if (operands.size() == 1)
    operand = operands[0];
  else
    operand = Tuple(ctx.builder, operands);

  xla::XlaOp token;
  if (failed(GetXlaOp(op.token(), value_map, &token, op))) return failure();

  if (op.is_host_transfer()) {
    value_map[op] = xla::SendToHost(
        operand, token,
        operand.builder()->GetShape(operand).ConsumeValueOrDie(),
        Convert_channel_handle(op.channel_handle()));
    return success();
  }
  value_map[op] = xla::SendWithToken(
      operand, token, Convert_channel_handle(op.channel_handle()));
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
  for (const auto& it : llvm::enumerate(op.getResults())) {
    value_map[it.value()] = xla::GetTupleElement(sorted, it.index());
  }
  return success();
}

LogicalResult ExportXlaOp(TraceOp op, OpLoweringContext ctx) {
  // TODO(atondwal): remove mhlo.trace
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
  if (failed(ctx.converter->LowerRegionAsComputation(
          &op.body(), &body, llvm::None, /*ensure_single_arg*/ true)) ||
      failed(ctx.converter->LowerRegionAsComputation(
          &op.cond(), &condition, llvm::None, /*ensure_single_arg*/ true))) {
    return failure();
  }

  // In case MHLO's whileOp has multiple operands, create xla::Tuple, using
  // those operands, to be used as sole operand of xla::While.
  llvm::SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op, op.getOperands(), ctx, operands))) return failure();

  xla::XlaOp operand = operands[0];
  if (operands.size() > 1) operand = Tuple(ctx.builder, operands);

  auto whileop = xla::While(condition, body, operand);

  auto& value_map = *ctx.values;
  auto shape_or = whileop.builder()->GetShape(whileop);
  if (!shape_or.ok()) {
    return op.emitError(shape_or.status().ToString());
  }

  xla::Shape& shape = shape_or.ValueOrDie();
  if (!shape.IsTuple()) {
    value_map[op.getResult(0)] = whileop;
    return success();
  }

  // mhlo.WhileOp supports multiple returns, untuple all the results of XLA's.
  for (const auto& it : llvm::enumerate(op.getResults())) {
    value_map[it.value()] = xla::GetTupleElement(whileop, it.index());
  }

  return success();
}

LogicalResult ExportXlaOp(OptimizationBarrierOp op, OpLoweringContext ctx) {
  // In case MHLO's OptimizationBarrierOp has multiple operands,
  // create xla::Tuple, using those operands, to be used as
  // sole operand of xla::OptimizationBarrier.
  llvm::SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op, op.getOperands(), ctx, operands))) return failure();
  if (operands.empty()) return success();

  auto& value_map = *ctx.values;
  if (operands.size() == 1) {
    value_map[op.getResult(0)] = xla::OptimizationBarrier(operands[0]);
  } else {
    auto result = xla::OptimizationBarrier(Tuple(ctx.builder, operands));

    for (const auto& it : llvm::enumerate(op.getResults())) {
      value_map[it.value()] = xla::GetTupleElement(result, it.index());
    }
  }

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

  auto fusion_kind_string =
      mlir::mhlo::stringifyFusionKind(op.fusion_kind().getValue());
  xla::XlaOp fusion = xla::internal::XlaBuilderFriend::BuildFusion(
      ctx.builder, operands,
      absl::string_view(fusion_kind_string.data(), fusion_kind_string.size()),
      fused_computation);
  if (op.getNumResults() == 1) {
    values[op.getResult(0)] = fusion;
  } else {
    for (const auto& item : llvm::enumerate(op.getResults())) {
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

LogicalResult ExportXlaOp(PrintOp op, OpLoweringContext ctx) {
  return failure();
}

LogicalResult ExportXlaOp(UniformQuantizeOp op, OpLoweringContext ctx) {
  // Currently, it doesn't have an XLA builder equivalent.
  // TODO(b/230671877): Implement XLA import/export for quantized MHLO ops.
  return failure();
}

LogicalResult ExportXlaOp(UniformDequantizeOp op, OpLoweringContext ctx) {
  // Currently, it doesn't have an XLA builder equivalent.
  // TODO(b/230671877): Implement XLA import/export for quantized MHLO ops.
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
    source_data.SetValues(                                                   \
        attr.cast<DenseElementsAttr>().getValues<cpp_type>());               \
    return xla::LiteralUtil::CreateFromArrayWithLayout(source_data, layout); \
  }

  switch (shape.element_type()) {
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::PRED, bool)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::F32, float)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::F64, double)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::S8, int8)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::S16, int16)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::S32, int32)
    ELEMENTS_ATTR_TO_LITERAL(xla::PrimitiveType::S64, int64_t)
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

    // 'layout' does not take the token attribute into account, so skip the
    // corresponding entry from xla shape proto.
    size_t subshapes_data_size = subshapes->size();
    if (!subshapes->empty() &&
        subshapes->Mutable(subshapes->size() - 1)->element_type() == xla::TOKEN)
      subshapes_data_size = subshapes->size() - 1;

    if (layout.size() != subshapes_data_size) {
      op->emitOpError() << "Expected layout of size " << layout.size()
                        << ", but found " << subshapes->size();
      return failure();
    }
    for (int i = 0; i < subshapes_data_size; i++) {
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
      std::vector<int64_t> array(rank);
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

// Assigns layouts from 'layout' to shape.
// The function accepts any of the following shapes
//   one or more array-shape(s) of infeed data
//   Tuple(Tuple(zero or more array-shape w.r.t data), token_type)
//
// 'layout' of the mhlo.InfedOp 'op' is
//    [zero or more layout for each array-shape w.r.t data]
// 'layout_index' indexes into 'layout' accessing a layout corresponding to a
// shape.
LogicalResult ConvertInfeedtLayout(mlir::Operation* op,
                                   const mlir::ArrayAttr& layout,
                                   xla::ShapeProto* shape,
                                   int64_t layout_index = 0) {
  if (shape->element_type() != xla::TUPLE) {
    // Handles following shape:
    //   single array-shape of infeed data
    mlir::ArrayAttr child_layout =
        layout[layout_index].dyn_cast<mlir::ArrayAttr>();
    if (!child_layout) {
      op->emitOpError() << "Type Error: Expected layout array attribute";
      return failure();
    }

    int rank = shape->dimensions().size();
    if (rank) {
      if (child_layout.size() != rank) {
        return failure();  // pass error down
      }
      std::vector<int64_t> array(rank);
      for (int i = 0; i < rank; i++) {
        mlir::IntegerAttr attr = child_layout[i].dyn_cast<mlir::IntegerAttr>();
        if (!attr) {
          op->emitOpError() << "Type Error: Expected layout integer attribute";
          return failure();
        }
        array[i] = attr.getInt();
      }
      *shape->mutable_layout() = xla::LayoutUtil::MakeLayout(array).ToProto();
    }

    return success();
  }

  auto subshapes = shape->mutable_tuple_shapes();
  auto datashape = subshapes->Mutable(0);

  if (datashape->element_type() == xla::TUPLE) {
    //   Handles following shapes:
    //     (Tuple(zero or more array-shape w.r.t data), token_type)
    auto data_subshapes = datashape->mutable_tuple_shapes();
    if (layout.size() != data_subshapes->size()) {
      op->emitOpError() << "Expected " << data_subshapes->size()
                        << " layout attribute(s) for infeed data, but found "
                        << layout.size();
      return failure();
    }

    for (int i = 0; i < data_subshapes->size(); i++) {
      if (failed(
              ConvertInfeedtLayout(op, layout, data_subshapes->Mutable(i), i)))
        return failure();
    }
  } else {
    //   Handles following shapes:
    //     array-shapes of two or more infeed data
    if (layout.size() != subshapes->size()) {
      op->emitOpError() << "Expected " << subshapes->size()
                        << " layout attribute(s) for infeed data, but found "
                        << layout.size();
      return failure();
    }

    for (int i = 0; i < subshapes->size(); i++) {
      if (failed(ConvertInfeedtLayout(op, layout, subshapes->Mutable(i), i)))
        return failure();
    }
  }

  return success();
}

// MHLO and XLA HLO disagree on the meaning of addition of `pred` / `i1`, so
// there has to be a special case somewhere to account for the difference.  To
// get the expected behavior of an `AddOp` on `i1`, we have to use `xor`.  Since
// the majority of the conversion is generated code, we just sidestep it here
// for this single case, and inline the code to emit an `xor`.
LogicalResult ExportXlaOperatorWrapped(mlir::Operation* inst,
                                       OpLoweringContext ctx) {
  auto op = dyn_cast<mlir::mhlo::AddOp>(inst);
  if (op && op.getResult()
                .getType()
                .cast<mlir::TensorType>()
                .getElementType()
                .isSignlessInteger(1)) {
    auto& value_map = *ctx.values;
    auto result = op.getResult();
    xla::XlaOp xla_arg_0;
    if (failed(GetXlaOp(op.lhs(), value_map, &xla_arg_0, op)))
      return mlir::failure();
    xla::XlaOp xla_arg_1;
    if (failed(GetXlaOp(op.rhs(), value_map, &xla_arg_1, op)))
      return mlir::failure();
    auto xla_result = xla::Xor(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
    value_map[result] = xla_result;
    return mlir::success();
  }

  return ExportXlaOperator(inst, ctx);
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
      !mlir::isa<mlir::func::ConstantOp, mlir::arith::ConstantOp,
                 mlir::func::CallOp, mlir::tensor::CastOp,
                 mlir::func::ReturnOp>(inst)) {
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
      // TODO(kramm): merge this with ConvertLayout.
      *shape = ExtractXlaShape(inst).ToProto();
    }

    return success();
  };

  if (succeeded(
          ExportXlaOperatorWrapped(inst, {value_lowering, this, builder}))) {
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
    // For infeed ops stemming back to InfeedDequeueTuple, respect the
    // layout attribute, and create the corresponding layout in hlo.
    if (isa<mhlo::InfeedOp>(inst)) {
      mlir::ArrayAttr layout =
          inst->getAttrOfType<mlir::ArrayAttr>(kLayoutAttr);

      if (layout) {
        // We propagate layout to the following three ops:
        // L1: For each data-result of mhlo.InfeedOp, we find the exported
        // xla::kGetTupleElement and propagate the layout.
        //
        // L2: For the token-result of mhlo.InfeedOp (result at last index),
        // we extract the xla::kInfeed op using the corresponding
        // xla::kGetTupleElement and propagate the layout to it.
        //
        // L3: In case there are non-zero data-results, there exists an
        // additional xla::kGetTupleElement accessing a tuple of the
        // data-results. We need to propagate the layout to that
        // xla::kGetTupleElement as well.
        auto num_results = inst->getNumResults();
        bool propagate_layout_to_data_tuple = true;
        for (unsigned i = 0; i < num_results; i++) {
          auto iter = value_lowering->find(inst->getResult(i));
          if (iter == value_lowering->end()) {
            inst->emitOpError() << "inst's result value at index " << i
                                << " has no match in value_lowering";
            return failure();
          }
          auto xla_gte_op = iter->second;
          xla::HloInstructionProto* get_tuple_element_proto =
              xla::internal::XlaBuilderFriend::GetInstruction(xla_gte_op);

          assert(xla::StringToHloOpcode(get_tuple_element_proto->opcode())
                         .ValueOrDie() == xla::HloOpcode::kGetTupleElement &&
                 "The token-result of mhlo.InfeedOp should be mapped to a "
                 "xla::HloOpcode::kGetTupleElement");

          if (i == num_results - 1) {
            // L2
            xla::HloInstructionProto* xla_infeed_op_proto =
                xla::internal::XlaBuilderFriend::GetInstructionByHandle(
                    xla_gte_op.builder(),
                    get_tuple_element_proto->operand_ids(0));

            assert(xla::StringToHloOpcode(xla_infeed_op_proto->opcode())
                           .ValueOrDie() == xla::HloOpcode::kInfeed &&
                   "Expected xla::HloOpcode::kInfeed op");

            auto* shape = xla_infeed_op_proto->mutable_shape();
            if (failed(ConvertInfeedtLayout(inst, layout, shape)))
              return failure();

          } else {
            // L1
            auto* shape = get_tuple_element_proto->mutable_shape();
            if (failed(ConvertInfeedtLayout(inst, layout, shape, i)))
              return failure();

            // L3
            if (propagate_layout_to_data_tuple) {
              xla::HloInstructionProto* data_tuple_proto =
                  xla::internal::XlaBuilderFriend::GetInstructionByHandle(
                      xla_gte_op.builder(),
                      get_tuple_element_proto->operand_ids(0));
              auto* data_tuple_shape = data_tuple_proto->mutable_shape();

              assert(xla::StringToHloOpcode(data_tuple_proto->opcode())
                             .ValueOrDie() ==
                         xla::HloOpcode::kGetTupleElement &&
                     "Expected a xla:tupleOp for all the data results.");
              if (failed(ConvertInfeedtLayout(inst, layout, data_tuple_shape)))
                return failure();
            }
            propagate_layout_to_data_tuple = false;
          }
        }
      }
    }
    return success();
  }

  auto& value_map = *value_lowering;
  ElementsAttr const_attr;

  if (auto call_op = dyn_cast<mlir::func::CallOp>(inst)) {
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
    if (!inst->getResult(0).getType().isa<ShapedType>()) {
      return inst->emitError(
          "expected shaped type during constant mhlo -> hlo translation");
    }

    auto literal_or =
        CreateArrayLiteralFromAttr(const_attr, ExtractXlaShape(inst).layout());
    if (!literal_or.ok())
      return inst->emitError(literal_or.status().ToString());
    auto constant = xla::ConstantLiteral(builder, literal_or.ValueOrDie());
    value_map[inst->getResult(0)] = constant;

    return success();
  }

  if (isa<mhlo::ReturnOp, mlir::func::ReturnOp>(inst)) {
    // Construct the return value for the function. If there is a single value
    // returned, then return it directly, else create a tuple and return.
    unsigned num_return_values = inst->getNumOperands();
    const bool has_ret_shardings =
        !ret_shardings.empty() && AllOptionalShardingsAreSet(ret_shardings);
    if ((return_tuple_ && is_entry_function) || num_return_values != 1) {
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
                builder, returns[index], return_shape, shape_determination_fns_,
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

      if (has_ret_shardings) {
        auto tuple = Tuple(builder, {operand});
        builder->SetSharding(*ret_shardings[0]);
        *return_value = GetTupleElement(tuple, 0);
        builder->ClearSharding();
      } else {
        *return_value = operand;
      }
    }

    return success();
  }

  inst->emitOpError() << "can't be translated to XLA HLO";
  return failure();
}

LogicalResult ConvertToHloModule::LowerFunctionCall(
    mlir::func::CallOp call_op, xla::XlaBuilder* builder,
    ConvertToHloModule::ValueLoweringMap* value_lowering) {
  auto& value_map = *value_lowering;
  mlir::func::FuncOp callee =
      module_.lookupSymbol<mlir::func::FuncOp>(call_op.getCallee());
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

LogicalResult ConvertToHloModule::RunOnFunction(mlir::func::FuncOp f) {
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
      auto attr = f.getArgAttrOfType<mlir::UnitAttr>(i, kReplicationAttr);
      entry_args_same_across_replicas.push_back(attr != nullptr);
      any_arg_replicated |= entry_args_same_across_replicas.back();
      // Pass the alias info to the builder so that it will build the alias info
      // into the resulting HloModule.
      auto aliasing_output =
          f.getArgAttrOfType<mlir::IntegerAttr>(i, "tf.aliasing_output");
      if (!aliasing_output) continue;
      xla::ShapeIndex output_index;
      if ((return_tuple_ && entry_function) || f.getNumResults() != 1) {
        output_index = {aliasing_output.getInt()};
      } else {
        if (aliasing_output.getInt() != 0) {
          return f.emitError(
              "Aliasing output must be 0 if only one output exists");
        }
        output_index = {};
      }
      if (use_tuple_args_) {
        builder.SetUpAlias(output_index, /*param_number=*/0,
                           /*param_index=*/{i});
      } else {
        builder.SetUpAlias(output_index, /*param_number=*/i,
                           /*param_index=*/{});
      }
    }
    // Do not populate this field when nothing is replicated, since empty field
    // means no replication. This avoids the need for unrelated tests to handle
    // this field.
    if (!any_arg_replicated) entry_args_same_across_replicas.clear();

    ExtractShardingsFromFunction(f, &arg_shardings, &ret_shardings);
  }
  if (failed(LowerBasicBlockAsFunction(&f.front(), &builder, entry_function,
                                       false, entry_args_same_across_replicas,
                                       arg_shardings, ret_shardings,
                                       &computation))) {
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

    tensorflow::DataType arg_dtype;
    status = tensorflow::ConvertToDataType(arg.getType(), &arg_dtype);
    if (!status.ok())
      return block->getParentOp()->emitError() << status.error_message();

    CHECK(shape_determination_fns_.layout_preference_fn &&  // Crash OK
          shape_determination_fns_.shape_representation_fn);
    auto layout_preference = shape_determination_fns_.layout_preference_fn(
        arg_tensor_shape, arg_dtype, absl::nullopt);
    auto arg_shape_status = shape_determination_fns_.shape_representation_fn(
        arg_tensor_shape, arg_dtype, /*use_fast_memory=*/false,
        layout_preference);
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
    for (const auto& arg_sharding : llvm::enumerate(arg_shardings)) {
      auto hlo_sharding = xla::HloSharding::FromProto(*arg_sharding.value());
      if (!hlo_sharding.ok())
        return block->getParentOp()->emitError()
               << hlo_sharding.status().error_message();

      auto status = tensorflow::RewriteLayoutWithShardedShape(
          hlo_sharding.ValueOrDie(), /*use_fast_memory=*/false,
          shape_determination_fns_, &(*arg_shapes)[arg_sharding.index()]);
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
    bool ensure_single_arg,
    const std::vector<bool>& entry_args_same_across_replicas,
    llvm::ArrayRef<absl::optional<xla::OpSharding>> arg_shardings,
    llvm::ArrayRef<absl::optional<xla::OpSharding>> ret_shardings,
    xla::XlaComputation* result,
    llvm::Optional<llvm::ArrayRef<mlir::Value>> implicit_operands) {
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
    if (ensure_single_arg) {
      // Applicable for mhlo.IfOp or mhlo.CaseOp or mhlo.WhileOp.
      llvm::SmallVector<xla::Shape> arg_shapes;

      auto args_size = block->getNumArguments();
      if (implicit_operands) args_size = implicit_operands->size();

      arg_shapes.reserve(args_size);
      if (implicit_operands) {
        for (auto implicit_operand : *implicit_operands)
          arg_shapes.push_back(xla::TypeToShape(implicit_operand.getType()));
      } else {
        for (BlockArgument& arg : block->getArguments())
          arg_shapes.push_back(xla::TypeToShape(arg.getType()));
      }

      if (args_size > 1) {
        auto tuple = xla::Parameter(builder, 0,
                                    xla::ShapeUtil::MakeTupleShape(arg_shapes),
                                    "arg_tuple");

        if (implicit_operands) {
          int arg_index = 0;
          for (auto implicit_operand : *implicit_operands)
            lowering[implicit_operand] =
                xla::GetTupleElement(tuple, arg_index++);
        } else {
          for (BlockArgument& arg : block->getArguments())
            lowering[arg] = xla::GetTupleElement(tuple, arg.getArgNumber());
        }
      } else if (args_size == 1) {
        if (implicit_operands) {
          lowering[(*implicit_operands)[0]] =
              xla::Parameter(builder, 0, arg_shapes[0], "Arg_");
        } else {
          lowering[block->getArgument(0)] =
              xla::Parameter(builder, 0, arg_shapes[0], "Arg_");
        }
      } else {
        // Applicable only for IfOp or CaseOp. No implicit operands implies no
        // xla parameters. In this case, we create an empty tuple as the
        // block-parameter.
        xla::Parameter(builder, 0, xla::ShapeUtil::MakeTupleShape(arg_shapes),
                       "arg_empty_tuple");
      }
    } else {
      for (BlockArgument& arg : block->getArguments()) {
        auto num = arg.getArgNumber();
        xla::Shape shape = xla::TypeToShape(arg.getType());
        if (!arg_shardings.empty() && arg_shardings[num]) {
          builder->SetSharding(*arg_shardings[num]);
        }
        if (entry_args_same_across_replicas.empty()) {
          lowering[arg] =
              xla::Parameter(builder, num, shape, absl::StrCat("Arg_", num));
        } else {
          lowering[arg] = xla::Parameter(
              builder, num, shape, absl::StrCat("Arg_", num),
              std::vector<bool>(entry_args_same_across_replicas[num],
                                xla::ShapeUtil::GetLeafCount(shape)));
        }
        builder->ClearSharding();
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
    mlir::Region* region, xla::XlaComputation* func,
    llvm::Optional<llvm::ArrayRef<mlir::Value>> implicit_operands,
    bool ensure_single_arg) {
  std::unique_ptr<xla::XlaBuilder> builder =
      module_builder_.CreateSubBuilder(absl::StrCat("region_", region_id_++));
  return LowerBasicBlockAsFunction(&region->front(), builder.get(),
                                   /*is_entry_function=*/false,
                                   /*ensure_single_arg*/ ensure_single_arg,
                                   /*entry_args_same_across_replicas=*/{},
                                   /*arg_shardings=*/{}, /*ret_shardings=*/{},
                                   func, implicit_operands);
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

// Runs the PrepareForExport pass on the ModuleOp.
Status PrepareForExport(mlir::ModuleOp module) {
  // Prepare for export to XLA HLO.
  mlir::PassManager pm(module.getContext());
  pm.addNestedPass<mlir::func::FuncOp>(mhlo::CreatePrepareForExport());
  if (failed(pm.run(module)))
    return tensorflow::errors::Internal("Unable to optimize for XLA export");
  return ::tensorflow::OkStatus();
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
  return ::tensorflow::OkStatus();
}

Status ConvertMlirHloToHlo(
    mlir::ModuleOp module, xla::HloProto* hlo_proto, bool use_tuple_args,
    bool return_tuple,
    const tensorflow::XlaShapeLayoutHelpers::ShapeDeterminationFns
        shape_determination_fns,
    MlirToHloConversionOptions options) {
  TF_RETURN_IF_ERROR(PrepareForExport(module));
  mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());
  xla::XlaBuilder module_builder("main");
  ConvertToHloModule converter(module, module_builder, use_tuple_args,
                               return_tuple, shape_determination_fns, options);
  if (failed(converter.Run())) return diag_handler.ConsumeStatus();
  auto hlo_module = converter.ConsumeMainProto();
  StringRef module_name = module.getName() ? *module.getName() : "main";
  hlo_module.set_name(module_name.str());
  hlo_proto->mutable_hlo_module()->Swap(&hlo_module);
  return ::tensorflow::OkStatus();
}

Status BuildHloFromMlirHlo(mlir::Block& block, xla::XlaBuilder& builder,
                           llvm::ArrayRef<xla::XlaOp> xla_params,
                           std::vector<xla::XlaOp>& returns,
                           MlirToHloConversionOptions options) {
  auto module = block.getParentOp()->getParentOfType<mlir::ModuleOp>();
  TF_RETURN_IF_ERROR(PrepareForExport(module));
  ConvertToHloModule converter(module, builder,
                               /*use_tuple_args=*/false, /*return_tuple=*/false,
                               /*shape_determination_fns=*/{}, options);

  ConvertToHloModule::ValueLoweringMap lowering;
  // xla_params should only include non-constant parameters the block arguments
  // correspond to.
  if (xla_params.size() != block.getArguments().size())
    return tensorflow::errors::Internal("xla_params size (", xla_params.size(),
                                        ") != block arguments size (",
                                        block.getArguments().size(), ")");
  for (BlockArgument& arg : block.getArguments()) {
    auto num = arg.getArgNumber();
    lowering[arg] = xla_params[num];
  }

  mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());
  for (auto& inst : block) {
    if (isa<mhlo::ReturnOp, mlir::func::ReturnOp>(inst)) {
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

  return ::tensorflow::OkStatus();
}

}  // namespace mlir
