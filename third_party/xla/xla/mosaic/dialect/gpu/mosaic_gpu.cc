/* Copyright 2024 The JAX Authors.

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

#include "xla/mosaic/dialect/gpu/mosaic_gpu.h"

#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"  // IWYU pragma: keep
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/tsl/platform/statusor.h"

// Generated definitions.
#include "xla/mosaic/dialect/gpu/mosaic_gpu_dialect.cc.inc"
#include "xla/mosaic/dialect/gpu/mosaic_gpu_enums.cc.inc"
#define GET_ATTRDEF_CLASSES
#include "xla/mosaic/dialect/gpu/mosaic_gpu_attrdefs.cc.inc"
#define GET_TYPEDEF_CLASSES
#include "xla/mosaic/dialect/gpu/mosaic_gpu_types.cc.inc"
#define GET_OP_CLASSES
#include "xla/mosaic/dialect/gpu/mosaic_gpu_ops.cc.inc"

namespace mosaic_gpu {
namespace {

using ::mlir::FloatType;
using ::mlir::ImplicitLocOpBuilder;
using ::mlir::IntegerType;
using ::mlir::MLIRContext;
using ::mlir::Type;
using ::mlir::TypeRange;
using ::mlir::Value;
using ::mlir::ValueRange;

using Index = ::mlir::TypedValue<::mlir::IndexType>;
using Integer = ::mlir::TypedValue<::mlir::IntegerType>;

Integer ToI64(ImplicitLocOpBuilder& b, Index index) {
  return llvm::cast<Integer>(
      b.create<mlir::arith::IndexCastOp>(b.getI64Type(), index).getResult());
}

template <typename T>
Value Constant(ImplicitLocOpBuilder& b, T scalar, IntegerType type) {
  return b.create<mlir::arith::ConstantOp>(
      type, mlir::IntegerAttr::get(type, scalar));
}

template <typename T>
Value Constant(ImplicitLocOpBuilder& b, T scalar, FloatType type) {
  return b.create<mlir::arith::ConstantOp>(type,
                                           mlir::FloatAttr::get(type, scalar));
}

// Given a range of values of the same type, produces a LLVM array that contains
// all of them in order. Returns a pointer to the start of the newly created
// array.
absl::StatusOr<Pointer> ToLLVMArray(ImplicitLocOpBuilder& b,
                                    ValueRange values) {
  if (values.empty()) {
    return absl::InvalidArgumentError("Can not pack an empty array of values.");
  }

  Type element_type = values.front().getType();

  MLIRContext* ctx = b.getContext();
  mlir::LLVM::LLVMPointerType pointer_type =
      mlir::LLVM::LLVMPointerType::get(ctx);
  Pointer array_pointer = b.create<mlir::LLVM::AllocaOp>(
      pointer_type, element_type, Constant(b, values.size(), b.getI64Type()));

  for (auto [i, value] : llvm::enumerate(values)) {
    if (value.getType() != element_type) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Expected all values to have the same type, but got ",
          MlirToString(value.getType()), " and ", MlirToString(element_type)));
    }

    auto element_pointer = llvm::cast<Pointer>(
        b.create<mlir::LLVM::GEPOp>(
             pointer_type, element_type, array_pointer,
             mlir::ArrayRef<mlir::LLVM::GEPArg>(mlir::LLVM::GEPArg(i)))
            .getResult());
    b.create<mlir::LLVM::StoreOp>(value, element_pointer);
  }

  return array_pointer;
}

// Extracts a pointer to the start of the parameter memref.
Pointer FromMemref(ImplicitLocOpBuilder& b, Memref memref) {
  Index aligned_pointer_as_index =
      b.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(memref);

  mlir::LLVM::LLVMPointerType pointer_type =
      mlir::LLVM::LLVMPointerType::get(b.getContext());

  Value alloc_pointer = b.create<mlir::LLVM::IntToPtrOp>(
      pointer_type, ToI64(b, aligned_pointer_as_index));

  Type tensor_element_type = memref.getType().getElementType();

  return mlir::cast<Pointer>(
      b.create<mlir::LLVM::GEPOp>(
           pointer_type, tensor_element_type, alloc_pointer,
           mlir::ArrayRef<mlir::LLVM::GEPArg>(
               mlir::LLVM::GEPArg(ToI64(b, aligned_pointer_as_index))))
          .getResult());
}

}  // anonymous namespace

// TODO(bchetioui): add swizzling.
absl::Status InitTmaDescriptor(mlir::OpBuilder& builder,
                               Pointer host_pointer_to_descriptor,
                               Memref gmem_ref,
                               mlir::ArrayRef<int64_t> slice_shape) {
  ImplicitLocOpBuilder b(
      mlir::NameLoc::get(builder.getStringAttr("InitTmaDescriptor")), builder);

  mlir::memref::ExtractStridedMetadataOp extract_strided_metadata_op =
      b.create<mlir::memref::ExtractStridedMetadataOp>(gmem_ref);

  Type tensor_element_type = gmem_ref.getType().getElementType();

  Pointer tensor_base_pointer = FromMemref(b, gmem_ref);

  int64_t tensor_rank = gmem_ref.getType().getRank();
  ValueRange sizes = extract_strided_metadata_op.getSizes();
  ValueRange strides = extract_strided_metadata_op.getStrides();

  if (tensor_rank != slice_shape.size()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Slice shape should have the same rank as the target tensor "
        "but got ",
        slice_shape.size(), " != ", tensor_rank));
  }

  std::vector<Value> sizes_as_i64;
  std::vector<Value> strides_as_i64;
  std::vector<Value> slice_as_i64;
  sizes_as_i64.reserve(tensor_rank);
  strides_as_i64.reserve(tensor_rank);
  slice_as_i64.reserve(tensor_rank);
  for (auto [size, stride, slice_dim] :
       llvm::zip(sizes, strides, slice_shape)) {
    sizes_as_i64.push_back(ToI64(b, llvm::cast<Index>(size)));
    strides_as_i64.push_back(ToI64(b, llvm::cast<Index>(stride)));
    slice_as_i64.push_back(Constant(b, slice_dim, b.getI64Type()));
  }

  TF_ASSIGN_OR_RETURN(Pointer sizes_array, ToLLVMArray(b, sizes_as_i64));
  TF_ASSIGN_OR_RETURN(Pointer strides_array, ToLLVMArray(b, strides_as_i64));
  TF_ASSIGN_OR_RETURN(Pointer slice_array, ToLLVMArray(b, slice_as_i64));

  IntegerType i64 = b.getI64Type();

  int64_t elem_bitwidth = tensor_element_type.getIntOrFloatBitWidth();

  if (elem_bitwidth < 8) {
    return absl::UnimplementedError("Sub-byte types are not yet supported.");
  }

  // TODO(bchetioui): connect this to runtime.
  b.create<mlir::func::CallOp>(
      kRuntimeTmaDescriptorInitializerName, TypeRange{},
      ValueRange{/*tma_desc=*/host_pointer_to_descriptor,
                 /*base_addr=*/tensor_base_pointer,
                 /*elem_bytewidth=*/Constant(b, elem_bitwidth / 8, i64),
                 /*rank=*/Constant(b, tensor_rank, i64),
                 /*sizes=*/sizes_array,
                 /*strides=*/strides_array,
                 // TODO(bchetioui): implement swizzling.
                 /*swizzle_bytes=*/Constant(b, 0, i64),
                 /*window_shape=*/slice_array});

  return absl::OkStatus();
}

void DeclareRuntimeFunctions(mlir::OpBuilder& builder) {
  MLIRContext* ctx = builder.getContext();
  mlir::LLVM::LLVMPointerType ptr = mlir::LLVM::LLVMPointerType::get(ctx);
  IntegerType i64 = builder.getI64Type();

  builder
      .create<mlir::func::FuncOp>(
          builder.getUnknownLoc(), kRuntimeTmaDescriptorInitializerName,
          builder.getFunctionType(
              TypeRange{ptr, ptr, i64, i64, ptr, ptr, i64, ptr}, TypeRange{}))
      .setVisibility(mlir::func::FuncOp::Visibility::Private);
}

bool IsContiguous(mlir::MemRefType type) {
  return type.getLayout().isIdentity() ||
         (type.hasStaticShape() && type.getNumElements() > 0 &&
          mlir::memref::isStaticShapeAndContiguousRowMajor(type));
}

namespace {
llvm::LogicalResult VerifyCommonLoadStoreOp(
    mlir::Location loc, mlir::MemRefType gmem_type, absl::string_view gmem_name,
    mlir::MemRefType smem_type, absl::string_view smem_name,
    mlir::ArrayRef<int64_t> slice_lengths, int num_indices) {
  auto error = [loc](auto... params) {
    return emitError(loc, llvm::formatv(params...));
  };

  if (!IsContiguous(smem_type)) {
    return error("The `{0}` memref must be contiguous.", smem_name);
  }
  if (gmem_type.getElementType() != smem_type.getElementType()) {
    return error(
        "The `source` and `destination` memrefs must have the same element "
        "type.");
  }
  if (absl::c_any_of(slice_lengths, [](int64_t s) { return s < -1; })) {
    return error(
        "The `slice_lengths` attribute must not contain values less than -1.");
  }
  if (gmem_type.getRank() !=
      smem_type.getRank() + absl::c_count(slice_lengths, -1)) {
    return error(
        "The rank of the `{0}` must be equal to the rank of the "
        "`{1}` plus the number of collapsed dimensions as indicated "
        "by -1 values in `slice_lengths`.",
        gmem_name, smem_name);
  }
  if (num_indices != gmem_type.getRank()) {
    return error("The size of `indices` must be equal to the rank of `{0}`.",
                 gmem_name);
  }
  if (slice_lengths.size() != gmem_type.getRank()) {
    return error(
        "The size of `slice_lengths` must be equal to the rank of `{0}`.",
        gmem_name);
  }
  return llvm::success();
}
}  // namespace

llvm::LogicalResult AsyncLoadOp::verify() {
  auto r = VerifyCommonLoadStoreOp(getLoc(), getSource().getType(), "source",
                                   getDestination().getType(), "destination",
                                   getSliceLengths(), getIndices().size());
  if (failed(r)) {
    return r;
  }

  for (int i = 0; i < getCollective().size(); ++i) {
    for (int k = i + 1; k < getCollective().size(); ++k)
      if (getCollective()[i] == getCollective()[k]) {
        return emitError(
            "The `collective` attribute must not contain duplicate "
            "dimensions.");
      }
  }

  return llvm::success();
}

llvm::LogicalResult AsyncStoreOp::verify() {
  return VerifyCommonLoadStoreOp(getLoc(), getDestination().getType(),
                                 "destination", getSource().getType(), "source",
                                 getSliceLengths(), getIndices().size());
}

namespace {
// This is the size of the M dimension in all wgmma instructions. It is fixed,
// unlike the K and N dimensions.
constexpr int kWgmmaSizeM = 64;
}  // namespace

llvm::LogicalResult WGMMAOp::verify() {
  auto error = [this](auto... params) {
    return emitOpError(llvm::formatv(params...));
  };

  auto a_shaped_type = mlir::cast<mlir::ShapedType>(getA().getType());
  mlir::Type element_type = a_shaped_type.getElementType();
  if (element_type != getB().getType().getElementType()) {
    return error("The `a` and `b` inputs must have the same element type.");
  }

  auto a_shape = a_shaped_type.getShape();
  if (a_shape.size() != 2) {
    return error("The `a` input must have rank 2.");
  }

  auto b_shape = getB().getType().getShape();
  if (b_shape.size() != 2) {
    return error("The `b` input must have rank 2.");
  }

  auto accShape = getAccumulator().getType().getShape();
  if (accShape.size() != 2) {
    return error("The accumulator must have rank 2.");
  }

  if (accShape[0] % kWgmmaSizeM) {
    return error(
        "The accumulator's first dimension must be a multiple of {0}, but got "
        "{1}.",
        kWgmmaSizeM, accShape[0]);
  }

  int M = accShape[0];  // groups_m * 64
  if (M != a_shape[0] && M != a_shape[1]) {
    return error(
        "The accumulator's first dimension {0} must be equal to one "
        "of the dimensions of `a` - ({1}, {2}).",
        M, a_shape[0], a_shape[1]);
  }
  int K = (a_shape[0] == M ? a_shape[1] : a_shape[0]);  // groups_k * k
  if (K != b_shape[0] && K != b_shape[1]) {
    return error(
        "`a`'s contracting dimension {0} must be equal to one "
        "of the dimensions of `b` - ({1}, {2}).",
        K, b_shape[0], b_shape[1]);
  }
  int N = (b_shape[0] == K ? b_shape[1] : b_shape[0]);  // groups_n * k
  if (N != accShape[1]) {
    return error(
        "`b`'s non-contracting dimension {0} must be equal to the "
        "accumulator's second dimension {1}.",
        N, accShape[1]);
  }

  return llvm::success();
}

llvm::LogicalResult CustomPrimitiveOp::verify() {
  int num_vector_operands = 0;
  int num_smem_ref_operands = 0;
  mlir::Attribute smem = mlir::gpu::AddressSpaceAttr::get(
      getContext(), mlir::gpu::AddressSpace::Workgroup);
  for (auto operand : getOperands()) {
    if (mlir::isa<mlir::VectorType>(operand.getType())) {
      ++num_vector_operands;
    }

    if (auto ref_ty = mlir::dyn_cast<mlir::MemRefType>(operand.getType())) {
      if (ref_ty.getMemorySpace() == smem) {
        ++num_smem_ref_operands;
      }
    }
  }

  if (num_vector_operands != getInLayouts().size()) {
    return emitOpError(
        "Custom primitive must have a layout for each vector operand.");
  }

  if (num_smem_ref_operands != getInTransforms().size()) {
    return emitOpError(
        "Custom primitive must have transforms for each memref operand in "
        "smem.");
  }

  if (getResults().size() != getOutLayouts().size()) {
    return emitOpError("Custom primitive must have a layout for each result.");
  }

  return llvm::success();
}

llvm::LogicalResult BroadcastInDimOp::verify() {
  auto error = [this](auto... params) {
    return emitOpError(llvm::formatv(params...));
  };

  auto operand_type = mlir::cast<mlir::VectorType>(getOperand().getType());
  auto result_type = mlir::cast<mlir::VectorType>(getResult().getType());

  if (operand_type.getRank() == 0) {
    return error("The input vector must have rank > 0.");
  }

  if (operand_type.getRank() > result_type.getRank()) {
    return error(
        "The rank of the input vector must be smaller or equal to the rank "
        "of the result vector.");
  }

  if (operand_type.getRank() != getBroadcastDimensions().size()) {
    return error(
        "The size of the `broadcast_dimensions` attribute must be equal to "
        "the rank of the input vector.");
  }
  auto dims = llvm::to_vector(getBroadcastDimensions());
  for (int i = 0; i < dims.size(); ++i) {
    if (dims[i] < 0 || dims[i] >= result_type.getRank()) {
      return error(
          "The values in the `broadcast_dimensions` attribute must be in the "
          "range [0, result.shape.rank={0}).",
          result_type.getRank());
    }
    if (i > 0 && dims[i] <= dims[i - 1]) {
      return error(
          "The values in the `broadcast_dimensions` attribute must be strictly "
          "increasing.");
    }
  }

  return llvm::success();
}

void MosaicGPUDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "xla/mosaic/dialect/gpu/mosaic_gpu_types.cc.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "xla/mosaic/dialect/gpu/mosaic_gpu_attrdefs.cc.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "xla/mosaic/dialect/gpu/mosaic_gpu_ops.cc.inc"
      >();
}

}  // namespace mosaic_gpu
