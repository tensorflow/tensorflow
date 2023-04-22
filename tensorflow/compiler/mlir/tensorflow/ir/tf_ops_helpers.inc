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

// This is a simple include file used to simplify the splitting of the
// tf_ops.cc file. The helpers in here should be refactored and moved to
// tf_verifiers or tf_ops.
// TODO(jpienaar): Remove this file post refactoring.

//===----------------------------------------------------------------------===//
// TF op helper functions
//===----------------------------------------------------------------------===//

// Returns the RankedTensorType for the given operand. TensorFlow constant ops
// may have non-static shape because the shape is not propagated during constant
// folding. If the defining op for the given operand is a constant op, this
// routine uses the constant op's attribute to get the actual shape.
static RankedTensorType GetRankedTensorTypeForOperand(Value operand) {
  DenseElementsAttr attr;
  if (matchPattern(operand, m_Constant(&attr))) {
    return attr.getType().dyn_cast<RankedTensorType>();
  }
  return operand.getType().dyn_cast<RankedTensorType>();
}

// Returns true if the given `value` is of ranked float tensor type with the
// given `rank`.
static inline bool IsOfRankedFloatTensorType(RankedTensorType type, int rank) {
  return type && type.getRank() == rank &&
         type.getElementType().isa<FloatType>();
}

// Returns true if the given `value` has the specified rank or has unranked
// type.
static inline bool IsOfRankOrUnranked(Value value, int64_t rank) {
  RankedTensorType type = GetRankedTensorTypeForOperand(value);
  return !type || type.getRank() == rank;
}

// Returns true if the given `value` has at least the specified rank or has
// unranked type.
static inline bool HasRankAtLeast(Value value, int64_t rank) {
  RankedTensorType type = GetRankedTensorTypeForOperand(value);
  return !type || type.getRank() >= rank;
}

// Returns true if the given `value` has at most the specified rank or has
// unranked type.
static inline bool HasRankAtMost(Value value, int64_t rank) {
  RankedTensorType type = GetRankedTensorTypeForOperand(value);
  return !type || type.getRank() <= rank;
}

static bool IsUnknownDimOrRank(int64_t dim_or_rank) {
  return dim_or_rank == -1;
}

// Returns the tf.Equal/tf.NotEqual result type given `x` and `y` and inputs. If
// `incompatible_shape_error` is true, reports error if `x` and `y` has
// incompatible shapes. Otherwise, returns a tensor type with unknown rank.
static Type DeduceEqualCmpOpType(Builder *builder, Location loc, Value x,
                                 Value y, BoolAttr incompatible_shape_error) {
  auto result_type =
      OpTrait::util::getBroadcastedType(x.getType(), y.getType());
  if (!result_type) {
    if (incompatible_shape_error.getValue()) {
      mlir::emitError(loc, "non-broadcastable operands");
    } else {
      return UnrankedTensorType::get(builder->getI1Type());
    }
  }

  auto ranked_type = result_type.dyn_cast<RankedTensorType>();
  if (!ranked_type) return UnrankedTensorType::get(builder->getI1Type());

  return RankedTensorType::get(ranked_type.getShape(), builder->getI1Type());
}

// Returns dimension index for the given TensorFlow axis that supports negative
// indexing.
static int64_t GetDimForAxis(int64_t axis, int64_t rank) {
  return axis >= 0 ? axis : axis + rank;
}

// Infers output type for reduction ops such as SumOp, MaxOp etc.
// TODO(b/e667204a): Move this logic to shape inference once it supports custom
// inference functions.
static Type InferReductionOpType(Value input, Value reduction_indices,
                                 BoolAttr keep_dims, Builder *builder) {
  Type input_ty = input.getType();
  Type element_ty = getElementTypeOrSelf(input_ty);

  // Output type is unranked if input type is not ranked.
  auto ranked_ty = input_ty.dyn_cast<RankedTensorType>();
  if (!ranked_ty) return UnrankedTensorType::get(element_ty);
  int64_t rank = ranked_ty.getRank();

  DenseIntElementsAttr indices;
  if (!matchPattern(reduction_indices, m_Constant(&indices))) {
    // Output type is unranked if reduction indices are not constant and reduced
    // dimensions are not kept.
    if (!keep_dims.getValue()) return UnrankedTensorType::get(element_ty);

    // Otherwise, output type has same rank as the input.
    return RankedTensorType::get(SmallVector<int64_t, 4>(rank, -1), element_ty);
  }

  int64_t num_reduce_dim = 0;
  llvm::SmallVector<bool, 4> is_reduce_dim(rank, false);
  for (const APInt &index : indices.getValues<APInt>()) {
    int64_t dim = GetDimForAxis(index.getSExtValue(), rank);
    // Invalid input.
    if (dim < 0 || dim >= rank) return UnrankedTensorType::get(element_ty);

    if (!is_reduce_dim[dim]) {
      is_reduce_dim[dim] = true;
      num_reduce_dim++;
    }
  }

  ArrayRef<int64_t> shape = ranked_ty.getShape();
  SmallVector<int64_t, 4> out_shape;
  out_shape.reserve(rank - (keep_dims.getValue() ? 0 : num_reduce_dim));
  for (int64_t i = 0; i < rank; ++i) {
    if (!is_reduce_dim[i])
      out_shape.push_back(shape[i]);
    else if (keep_dims.getValue())
      out_shape.push_back(1);
  }
  return RankedTensorType::get(out_shape, element_ty);
}

// Returns the equivalent Value skipping through identity nodes.
Value LookThroughIdentity(Value result) {
  while (isa_and_nonnull<IdentityOp, IdentityNOp>(result.getDefiningOp())) {
    auto op_result = result.cast<OpResult>();
    result = op_result.getOwner()->getOperand(op_result.getResultNumber());
  }
  return result;
}

// Verifies that the given types are cast compatible. If not, emits appropriate
// error for the given op. If mask_one_dim is set to true, then the types are
// allowed to have one mismatching dimension. Masking one of the dimensions is
// useful for ops like Concat that requires all ranked inputs to have the same
// rank and match dimension sizes for all but one of the dimensions.
static LogicalResult VerifyTypesCompatibility(
    Operation::operand_type_range types, bool mask_one_dim, Operation *op) {
  constexpr int64_t kUninitialized = -1;
  int64_t common_rank = kUninitialized;
  llvm::SmallVector<int64_t, 4> common_dims;
  int64_t dim_to_mask = kUninitialized;

  // Initialize common_rank with rank of the first ranked type and verify that
  // following ranked types have the same rank.
  // Similarly, initialize each of the dimensions with the first type that has
  // the dimension size available and verify that all following types have the
  // same size for the dimension. However, if mask_one_dim is true, note down
  // the dimension index on the first mismatch and ignore dimension at that
  // index in following types.
  for (Type ty : types) {
    RankedTensorType ranked_ty = ty.dyn_cast<RankedTensorType>();
    if (!ranked_ty) continue;

    int64_t rank = ranked_ty.getRank();
    if (common_rank == kUninitialized) {
      common_rank = rank;
      common_dims.resize(common_rank, kUninitialized);
    } else if (common_rank != rank) {
      return op->emitError()
             << "operand type " << ranked_ty
             << " is not compatible with preceding operands; expected rank: "
             << common_rank;
    }

    for (int64_t i = 0, e = common_rank; i != e; i++) {
      if (i == dim_to_mask) continue;

      int64_t dim = ranked_ty.getDimSize(i);
      if (dim == kUninitialized) continue;

      int64_t &common_dim = common_dims[i];
      if (common_dim == kUninitialized) {
        common_dim = dim;
      } else if (common_dim != dim) {
        // If mask_one_dim is true, do not emit an error if this is the only
        // dimension with mismatches. Note down the dimension to mask it from
        // the following types.
        if (mask_one_dim && dim_to_mask == kUninitialized) {
          dim_to_mask = i;
          continue;
        }

        return op->emitError() << "operand type " << ranked_ty
                               << " is not compatible with preceding operands; "
                                  "expected dimension at index "
                               << i << ": " << common_dim;
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Helper functions detect device capabilities from RuntimeDevices.
//===----------------------------------------------------------------------===//

namespace {
using DeviceNameUtils = ::tensorflow::DeviceNameUtils;
using ParsedName = ::tensorflow::DeviceNameUtils::ParsedName;

bool IsGpuDevice(const DeviceNameUtils::ParsedName &device) {
  return device.type == ::tensorflow::DEVICE_GPU;
}

}  // namespace

// Returns true if at least one GPU device is available at runtime.
bool CanUseGpuDevice(const RuntimeDevices &devices) {
  return llvm::any_of(devices.device_names(), IsGpuDevice);
}

// Returns true if all of the GPUs available at runtime support TensorCores
// (NVIDIA compute capability >= 7.0).
bool CanUseTensorCores(const RuntimeDevices &devices) {
  auto has_tensor_cores = [&](const DeviceNameUtils::ParsedName &device) {
    auto md = devices.GetGpuDeviceMetadata(device);
    return md ? md->cc_major().getInt() >= 7 : false;
  };
  return llvm::all_of(
      llvm::make_filter_range(devices.device_names(), IsGpuDevice),
      has_tensor_cores);
}

// Returns true if operation does not have explicit device placement that would
// prevent it from running on GPU device.
bool CanUseGpuDevice(Operation *op) {
  auto device_attr = op->getAttrOfType<StringAttr>("device");
  if (!device_attr || device_attr.getValue().empty()) return true;

  DeviceNameUtils::ParsedName device;
  if (!DeviceNameUtils::ParseFullName(device_attr.getValue().str(), &device))
    return false;

  // We can't use GPU if operation explicitly placed on non-GPU device.
  return !device.has_type || device.type == ::tensorflow::DEVICE_GPU;
}

//===----------------------------------------------------------------------===//
// TF op helper functions to work with layout transformation.
//===----------------------------------------------------------------------===//

SmallVector<int64_t, 4> ReversePermutation(ArrayRef<int64_t> permutation) {
  SmallVector<int64_t, 4> reverse(permutation.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    reverse[permutation[i]] = i;
  }
  return reverse;
}

SmallVector<int64_t, 4> GetDataFormatPermutation(StringRef from, StringRef to) {
  if (from == "NHWC" && to == "NCHW") {
    return {0, 3, 1, 2};
  } else if (from == "NCHW" && to == "NHWC") {
    return {0, 2, 3, 1};
  } else {
    return {};
  }
}

// Shuffle elements in the `attr` according to the permutation. Optional
// `inner_size` allows to shuffle array attributes created from rank 2 tensors
// on outer dimension only.
ArrayAttr ShuffleArrayAttr(ArrayAttr attr, ArrayRef<int64_t> permutation,
                           int inner_size = 1) {
  if (attr.empty()) return attr;

  assert(attr.size() % inner_size == 0);
  assert(attr.size() / inner_size == permutation.size());

  SmallVector<Attribute, 8> values{attr.begin(), attr.end()};
  SmallVector<Attribute, 8> shuffled(values.size());

  for (size_t i = 0; i < permutation.size(); ++i) {
    for (size_t j = 0; j < inner_size; ++j) {
      shuffled[i * inner_size + j] = values[permutation[i] * inner_size + j];
    }
  }

  return ArrayAttr::get(attr.getContext(), shuffled);
}

// Shuffle ranked tensor dimensions according to the permutation.
Type ShuffleRankedTensorType(Type type, ArrayRef<int64_t> permutation) {
  if (auto ranked_type = type.dyn_cast<RankedTensorType>()) {
    ArrayRef<int64_t> shape = ranked_type.getShape();
    assert(permutation.size() == shape.size());

    SmallVector<int64_t, 4> new_shape(permutation.size());
    for (size_t i = 0; i < permutation.size(); ++i)
      new_shape[i] = shape[permutation[i]];

    return RankedTensorType::get(new_shape, ranked_type.getElementType());
  }

  return type;
}

static bool AreCancellablePermutations(DenseIntElementsAttr perm0,
                                       DenseIntElementsAttr perm1) {
  if (perm0.getNumElements() == 0 || perm1.getNumElements() == 0) return false;
  if (perm0.getNumElements() != perm1.getNumElements()) return false;

  SmallVector<int64_t, 8> perm0_values;
  for (const auto &value : perm0.getIntValues())
    perm0_values.push_back(value.getSExtValue());

  SmallVector<int64_t, 8> perm1_values;
  for (const auto &value : perm1.getIntValues())
    perm1_values.push_back(value.getSExtValue());

  for (int i = 0; i < perm0_values.size(); ++i) {
    if (perm0_values[perm1_values[i]] != i) return false;
  }

  return true;
}

// Default implementation of `LayoutSensitiveInterface::UpdateDataFormat` for
// layout sensitive operations that do not have any additional layout dependent
// attributes besides `data_format` string.
template <typename Op>
LogicalResult UpdateDataFormat(StringRef data_format, Op *op) {
  auto perm = GetDataFormatPermutation(op->data_format(), data_format);
  if (perm.empty()) return failure();

  // Update data format attribute.
  (*op)->setAttr("data_format", StringAttr::get(op->getContext(), data_format));

  // Update types for all layout sensitive results.
  auto layout_sensitive = cast<LayoutSensitiveInterface>(op->getOperation());
  for (unsigned idx : layout_sensitive.GetLayoutDependentResults()) {
    OpResult result = op->getOperation()->getResult(idx);
    result.setType(ShuffleRankedTensorType(result.getType(), perm));
  }

  return success();
}

// Default implementation for folding operand transpose into the operation.
// See `FoldOperandsTransposeInterface::FoldOperandsPermutation`.
template <typename Op>
LogicalResult FoldOperandsPermutation(
    ArrayRef<int64_t> permutation, Op *op,
    ArrayRef<std::pair<StringRef, ArrayAttr>> shuffle_attrs = {}) {
  MLIRContext *context = (*op)->template getParentOfType<ModuleOp>().getContext();

  // We only support NHWC <-> NCHW permutations.
  static constexpr std::array<int64_t, 4> kNchwToNhwc = {0, 2, 3, 1};
  static constexpr std::array<int64_t, 4> kNhwcToNchw = {0, 3, 1, 2};

  // Operation data format after folding `permutation`.
  StringRef target_data_format = [&]() -> StringRef {
    if (op->data_format() == "NHWC" && permutation.equals(kNchwToNhwc)) {
      return "NCHW";  // cancel NCHW->NHWC operand permutation
    } else if (op->data_format() == "NCHW" && permutation.equals(kNhwcToNchw)) {
      return "NHWC";  // cancel NHWC->NCHW operand permutation
    } else {
      return "";
    }
  }();
  if (target_data_format.empty()) return failure();

  // To fold operand `permutation` into the `op` we need shuffle all layout
  // dependent attributes and types with a reverse permutation, and change
  // operation data format to `target_data_format`.
  //
  // Example:
  //   %1 = SomeOp(...)   {data_format = NHWC}
  //   %2 = Transpose(%1) {permutation = NHWC->NCHW}
  //   %3 = Op(%2)        {data_format = NCHW}
  //
  // To bypass %2 we have to change data format to shuffle data format from NCHW
  // to NHWC, which is the reverse of operand permutation (function argument).
  auto reverse_permutation =
      GetDataFormatPermutation(op->data_format(), target_data_format);
  if (reverse_permutation.empty()) return failure();

  (*op)->setAttr("data_format", StringAttr::get(context, target_data_format));

  for (auto pair : shuffle_attrs) {
    StringRef attr_name = pair.first;
    ArrayAttr attr_value = pair.second;
    (*op)->setAttr(attr_name, ShuffleArrayAttr(attr_value, reverse_permutation));
  }

  auto fold = cast<FoldOperandsTransposeInterface>(op->getOperation());
  for (unsigned idx : fold.GetLayoutDependentResults()) {
    OpResult result = op->getOperation()->getResult(idx);
    result.setType(
        ShuffleRankedTensorType(result.getType(), reverse_permutation));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Rewrite Pattern for removing trivial Arithmetic op.
//===----------------------------------------------------------------------===//

namespace {
// Fold Arithmetic Op if one of the operands is a constant known to be an
// Identity (e.g. X+0, X*1, etc...). For commutative operations fold if
// known identity value is either lhs or rhs.
template <
    typename OpT,
    typename std::enable_if<llvm::is_one_of<
        OpT, AddV2Op, SubOp, MulOp, DivOp, RealDivOp>::value>::type * = nullptr>
OpFoldResult IdentityArithmeticOpFolder(OpT arithmetic_op,
                                        ArrayRef<Attribute> operands) {
  auto lhs_type = arithmetic_op.x().getType().template cast<ShapedType>();
  auto rhs_type = arithmetic_op.y().getType().template cast<ShapedType>();
  auto result_type =
      arithmetic_op.getResult().getType().template cast<ShapedType>();

  // We can fold arithmetic operation only of we can prove that we will not
  // accidentally hide a broadcasting error.
  auto is_valid_broadcasting = [](ShapedType operand_ty, ShapedType identity_ty,
                                  ShapedType result_ty) -> bool {
    // Scalar identity is broadcastable to any operand shape, we only need to
    // check that operand has the same shape as a result.
    bool scalar_identity = identity_ty.hasRank() && identity_ty.getRank() == 0;
    if (scalar_identity) return operand_ty == result_ty;

    // If identity is not a scalar, we must verify that all shapes are equal
    // and statically known.
    //
    // TODO(ezhulenev): Fold if identity shape is statically know to be
    // broadcastable to the operand shape.
    return operand_ty == result_ty && identity_ty == result_ty &&
           result_ty.hasStaticShape();
  };

  // Check that we have a constant operand on one side (candidate for identity).
  const bool is_commutative =
      (std::is_same<OpT, AddV2Op>::value || std::is_same<OpT, MulOp>::value);
  auto lhs_attr = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  auto rhs_attr = operands[1].dyn_cast_or_null<DenseElementsAttr>();
  if (!rhs_attr && !(is_commutative && lhs_attr)) return {};

  // Mul and Div ops have identity value one while AddV2 and SubOp have identity
  // value zero.
  const int identity =
      (std::is_same<OpT, MulOp>::value || std::is_same<OpT, DivOp>::value ||
       std::is_same<OpT, RealDivOp>::value)
          ? 1
          : 0;

  Type element_ty = lhs_type.getElementType();
  Attribute identity_attr;
  if (auto ty = element_ty.template dyn_cast<FloatType>()) {
    identity_attr = FloatAttr::get(ty, static_cast<double>(identity));
  } else if (auto ty = element_ty.template dyn_cast<IntegerType>()) {
    identity_attr = IntegerAttr::get(ty, static_cast<int64_t>(identity));
  } else {
    return {};
  }

  // Fold: Op(Operand, Identity) -> Operand.
  if (rhs_attr && is_valid_broadcasting(lhs_type, rhs_type, result_type)) {
    if (rhs_attr.isSplat() && rhs_attr.getSplatValue() == identity_attr)
      return arithmetic_op.x();
  }

  // Fold: Op(Identity, Operand) -> Operand for commutative operations.
  if (lhs_attr && is_commutative &&
      is_valid_broadcasting(rhs_type, lhs_type, result_type)) {
    if (lhs_attr.isSplat() && lhs_attr.getSplatValue() == identity_attr)
      return arithmetic_op.y();
  }

  return {};
}
}  // namespace

// Verifies an reduction op's `input` and reduction `dims`.
static LogicalResult VerifyReductionInputAndDims(Value input, Value dims,
                                                 Location loc) {
  auto dims_type = dims.getType().dyn_cast<RankedTensorType>();
  if (!dims_type) return success();
  if (dims_type.getRank() > 1)
    return emitError(loc, "dimensions can only be 0D or 1D tensor");

  auto input_type = input.getType().dyn_cast<RankedTensorType>();
  if (!input_type) return success();
  int64_t rank = input_type.getRank();

  DenseIntElementsAttr dims_attr;
  if (!matchPattern(dims, m_Constant(&dims_attr))) return success();
  for (const auto &dim_pair : llvm::enumerate(dims_attr)) {
    int64_t cur_dim = dim_pair.value().getSExtValue();
    if (cur_dim < -rank || cur_dim >= rank)
      return emitError(loc)
             << dim_pair.index() << "-th dimension should be in the range of [-"
             << rank << ", " << rank << ")";
  }

  return success();
}

// A type range with description (in singular form) attached to it.
using TypeRangeWithDesc = std::pair<TypeRange, StringRef>;

LogicalResult VerifyTypeRangesAreCompatible(Operation *op,
                                            TypeRangeWithDesc range0,
                                            TypeRangeWithDesc range1) {
  if (range0.first.size() != range1.first.size()) {
    return op->emitOpError()
           << range0.second << "s (size = " << range0.first.size() << ")"
           << " should have the same number of values as " << range1.second
           << "s (size = " << range1.first.size() << ")";
  }

  for (auto it : llvm::enumerate(llvm::zip(range0.first, range1.first))) {
    int index = it.index();
    Type type0 = std::get<0>(it.value());
    Type type1 = std::get<1>(it.value());
    if (!AreCastCompatible({type0, type1}))
      return op->emitOpError(llvm::formatv(
          "{0} type {1} is incompatible with {2} type {3} at index {4}",
          range0.second, type0, range1.second, type1, index));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Function control flow canonicalization.
//===----------------------------------------------------------------------===//

// Eliminate attributes that are not needed, but can get attached to Ops
// during import.
template <typename Op>
struct DropAttributes : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  // Drop the "output_shapes" attribute.
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    bool found = !!op->removeAttr("output_shapes");
    return success(found);
  }
};

//===----------------------------------------------------------------------===//
// TF op helper functions for handling resource handles and ids.
//===----------------------------------------------------------------------===//

// Returns device of op if present. If op has no device set, an empty string ref
// is returned instead.
llvm::StringRef GetDeviceOrEmpty(Operation *op) {
  if (auto device_attr = op->getAttrOfType<StringAttr>("device"))
    return device_attr.getValue();
  return llvm::StringRef();
}

// Returns resource handle value and id for resource op based on attributes. If
// a resource handle is anonymous, a new id is always returned.
ResourceHandleValueAndId GetResourceHandleValueAndIdBase(
    llvm::StringRef container, llvm::StringRef shared_name,
    llvm::StringRef device, Value resource,
    llvm::SmallDenseMap<ResourceHandle, int64_t> &resource_handle_id_map,
    int64_t &next_id) {
  // Always create a new ID for anonymous handle.
  if (IsResourceHandleAnonymous(shared_name)) return {resource, next_id++};

  ResourceHandle handle(container, shared_name, device, /*op=*/nullptr);
  auto emplace_res = resource_handle_id_map.try_emplace(handle, next_id);
  // New ID created, increment next_id.
  if (emplace_res.second) ++next_id;
  return {resource, emplace_res.first->second};
}

// Helper function to create TF op while copying all underscore attributes from
// another TF op.
// TODO(jpienaar): This is a workaround until behavior is established.
template <typename OpTy, typename... Args>
OpTy CreateTfOp(RewriterBase& b, Operation *op, Args &&... args) {
  auto ret = b.create<OpTy>(op->getLoc(), std::forward<Args>(args)...);
  CopyDeviceAndUnderscoredAttributes(op, ret.getOperation());
  return ret;
}

// Helper function to replace TF op with another op while copying all underscore
// attributes from the TF op.
// TODO(jpienaar): This is a workaround until behavior is established.
template <typename OpTy, typename... Args>
OpTy ReplaceTfOpWithNewOp(RewriterBase& b, Operation *op, Args &&... args) {
  auto ret = CreateTfOp<OpTy>(b, op, std::forward<Args>(args)...);
  b.replaceOp(op, ret.getOperation()->getResults());
  return ret;
}