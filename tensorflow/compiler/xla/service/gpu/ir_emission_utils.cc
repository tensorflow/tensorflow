/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <optional>
#include <vector>

#include "llvm/IR/IntrinsicsNVPTX.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/hlo_utils.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/service/gpu/target_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_type_conversion_util.h"

namespace xla {
namespace gpu {

namespace {

// Return whether the given shape is rank 2 excluding the batch dimensions.
bool IsRank2(const Shape& shape, int64_t batch_dimensions_size) {
  return shape.rank() == batch_dimensions_size + 2;
}

// Given a shape and a group of contiguous dimensions in the shape, returns
// a tuple of three values (major, middle, minor), where major is the size of
// the dimensions more major then the given dimensions, minor is the size of
// dimensions more minor then the given dimensions, and middle is the size of
// the given dimensions.
std::array<int64_t, 3> PartitionShapeByMiddleDimensions(
    const Shape& shape, absl::Span<const int64_t> dims_middle) {
  CHECK(LayoutUtil::AreDimensionsConsecutive(shape.layout(), dims_middle));
  std::array<int64_t, 3> values = {1, 1, 1};
  enum Segment { kMajor = 0, kMiddle = 1, kMinor = 2 };
  Segment cur_segment = kMinor;

  for (int64_t cur_dim : LayoutUtil::MinorToMajor(shape)) {
    if (cur_segment != kMajor) {
      // Handle change of segments.
      bool cur_dim_in_middle = absl::c_linear_search(dims_middle, cur_dim);
      if (cur_segment == kMinor) {
        if (cur_dim_in_middle) {
          cur_segment = kMiddle;
        }
      } else if (cur_segment == kMiddle) {
        if (!cur_dim_in_middle) {
          cur_segment = kMajor;
        }
      }
    }
    values[cur_segment] *= shape.dimensions(cur_dim);
  }
  return values;
}

Shape GetShapeFromTensorType(mlir::Value value) {
  constexpr char kDefaultLayoutAttrName[] = "xla_shape";

  mlir::Operation* op = value.getDefiningOp();
  CHECK(op);
  CHECK(value.getType().isa<mlir::TensorType>());
  Shape shape;
  if (auto attr = op->getAttrOfType<mlir::StringAttr>(kDefaultLayoutAttrName)) {
    shape = *xla::ParseShape(
        absl::string_view(attr.getValue().data(), attr.getValue().size()));
  } else {
    shape = TypeToShape(value.getType());
  }
  return shape;
}

}  // namespace

bool IsMatrixMultiplication(const HloInstruction& dot) {
  if (dot.opcode() != HloOpcode::kDot) {
    return false;
  }
  const Shape& lhs_shape = dot.operand(0)->shape();
  const Shape& rhs_shape = dot.operand(1)->shape();
  const DotDimensionNumbers& dim_numbers = dot.dot_dimension_numbers();

  PrimitiveType output_primitive_type = dot.shape().element_type();
  bool type_is_allowed =
      (output_primitive_type == F16 || output_primitive_type == BF16 ||
       output_primitive_type == F32 || output_primitive_type == F64 ||
       output_primitive_type == C64 || output_primitive_type == C128) ||
      (output_primitive_type == S32 && lhs_shape.element_type() == S8 &&
       lhs_shape.element_type() == S8);
  bool shapes_are_valid =
      type_is_allowed &&
      IsRank2(lhs_shape, dim_numbers.lhs_batch_dimensions_size()) &&
      IsRank2(rhs_shape, dim_numbers.lhs_batch_dimensions_size()) &&
      IsRank2(dot.shape(), dim_numbers.lhs_batch_dimensions_size()) &&
      !ShapeUtil::IsZeroElementArray(lhs_shape) &&
      !ShapeUtil::IsZeroElementArray(rhs_shape);

  if (!shapes_are_valid) {
    return false;
  }

  // The size of the reduction dimension should match. The shape inference
  // guarantees this invariant, so the check here is for programming
  // errors.
  CHECK_EQ(lhs_shape.dimensions(dim_numbers.lhs_contracting_dimensions(0)),
           rhs_shape.dimensions(dim_numbers.rhs_contracting_dimensions(0)));

  return true;
}

std::array<int64_t, 3> GetReductionTiling(
    const ReductionDimensions& reduction_dimensions,
    se::CudaComputeCapability cuda_compute_capability) {
  if (reduction_dimensions.is_row_reduction) {
    int64_t tile_z = std::min(reduction_dimensions.dimensions[0],
                              BatchedReductionRaceFreeBound());
    return {tile_z, 1, 16};
  }

  // Column reduction.
  return {1, 128, 1};
}

const char* const kCusolverCholeskyCallTarget = "__cusolver$cholesky";

bool IsCustomCallToCusolver(const HloInstruction& hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  const auto& target = hlo.custom_call_target();
  return target == kCusolverCholeskyCallTarget;
}

static ReductionDimensions GetReductionKindAndContiguousComponentsImpl(
    const Shape& input_shape, absl::Span<const int64_t> dims_to_reduce) {
  DimensionVector dims_to_keep;
  for (int64_t dim = 0; dim < input_shape.rank(); ++dim) {
    if (!absl::c_linear_search(dims_to_reduce, dim)) {
      dims_to_keep.push_back(dim);
    }
  }

  if (dims_to_keep.empty()) {
    return {/*is_row_reduction=*/true,
            {1, 1, ShapeUtil::ElementsIn(input_shape)}};
  }

  if (LayoutUtil::AreDimensionsConsecutive(input_shape.layout(),
                                           dims_to_keep)) {
    std::array<int64_t, 3> shape_partition =
        PartitionShapeByMiddleDimensions(input_shape, dims_to_keep);
    if (shape_partition[1] == 1) {
      return {/*is_row_reduction=*/true,
              {1, 1, shape_partition[0] * shape_partition[2]}};
    }
    if (shape_partition[2] == 1) {
      return {/*is_row_reduction=*/false,
              {1, shape_partition[0], shape_partition[1]}};
    }
    return {/*is_row_reduction=*/true, shape_partition};
  }

  std::array<int64_t, 3> shape_partition =
      PartitionShapeByMiddleDimensions(input_shape, dims_to_reduce);

  if (shape_partition[2] == 1) {
    return {/*is_row_reduction=*/true,
            {1, shape_partition[0], shape_partition[1]}};
  }
  return {/*is_row_reduction=*/false, shape_partition};
}

static bool IsUnnestedReductionFasterThanElemental(
    const ReductionDimensions& reduction_dimensions) {
  if (reduction_dimensions.is_row_reduction) {
    // For row reduction, the tile block is 1 x tile_size_x, and we are reducing
    // along tile_size_x which needs to be large enough to make the tiling
    // implementation efficient.
    // For very small reductions with a power-of-two size, we can fit multiple
    // reductions inside a single warp, which is more efficient than a loop.
    return (reduction_dimensions.dimensions[2] >= WarpSize()) ||
           ((WarpSize() % reduction_dimensions.dimensions[2]) == 0);
  }

  // For column reduction, the tile block is tile_size_y x tile_size_x, and we
  // are reducing along tile_size_y. Only tile_size_y needs to be
  // large enough to make the tiling implementation efficient.
  int64_t major_size = reduction_dimensions.dimensions[1];
  int64_t minor_size = reduction_dimensions.dimensions[2];

  // Rule generated by sweeping the search space of small column reductions.
  bool prefer_elemental_emitter =
      (major_size < WarpSize()) ||
      (major_size < 2 * WarpSize() && minor_size < WarpSize()) ||
      (major_size < 4 * WarpSize() && minor_size < 8) ||
      (major_size < 8 * WarpSize() && minor_size < 3);

  return !prefer_elemental_emitter;
}

// Whether we can/should use the unnested emitter for reduction.
static bool IsReductionFromOrToContiguousDimensionsImpl(
    const Shape& operand_shape, absl::Span<int64_t const> dims_to_reduce) {
  DimensionVector dims_to_keep;
  for (int64_t dim = 0; dim < operand_shape.dimensions().size(); ++dim) {
    if (!absl::c_linear_search(dims_to_reduce, dim)) {
      dims_to_keep.push_back(dim);
    }
  }

  // We support fast codegen for three cases:
  // 1) Row reduction: (K, R)
  // 2) Column reduction: (K, R, K)
  // 3) "Batched" row reduction: (R, K, R)
  return (LayoutUtil::AreDimensionsConsecutive(operand_shape.layout(),
                                               dims_to_keep) ||
          LayoutUtil::AreDimensionsConsecutive(operand_shape.layout(),
                                               dims_to_reduce)) &&
         IsUnnestedReductionFasterThanElemental(
             GetReductionKindAndContiguousComponentsImpl(operand_shape,
                                                         dims_to_reduce));
}

bool IsReductionFromOrToContiguousDimensions(const HloInstruction& reduce) {
  return reduce.opcode() == HloOpcode::kReduce &&
         IsReductionFromOrToContiguousDimensionsImpl(reduce.operand(0)->shape(),
                                                     reduce.dimensions());
}

bool IsReductionFromOrToContiguousDimensions(mlir::Operation* op) {
  auto reduce = mlir::dyn_cast<mlir::mhlo::ReduceOp>(op);
  if (!reduce) {
    return false;
  }

  mlir::Value first_operand = reduce.operands()[0];
  Shape operand_shape = GetShape(first_operand);

  llvm::SmallVector<int64_t> dimensions_to_reduce;
  for (const llvm::APInt& d : reduce.dimensions()) {
    dimensions_to_reduce.push_back(d.getZExtValue());
  }

  return IsReductionFromOrToContiguousDimensionsImpl(operand_shape,
                                                     dimensions_to_reduce);
}

bool IsInputFusibleSlices(mlir::Operation* unnested_hlo,
                          bool verify_no_strides) {
  auto fusion = mlir::dyn_cast<mlir::lmhlo::FusionOp>(unnested_hlo);
  if (!fusion) {
    return false;
  }

  auto is_non_strided = [](mlir::DenseIntElementsAttr strides) -> bool {
    return absl::c_all_of(
        strides, [](const llvm::APInt& stride) { return stride == 1; });
  };

  for (mlir::Value value : fusion.getFusionResults()) {
    auto slice =
        mlir::dyn_cast_or_null<mlir::mhlo::SliceOp>(value.getDefiningOp());
    if (!slice) {
      return false;
    }
    if (verify_no_strides && !is_non_strided(slice.strides())) {
      return false;
    }
  }
  return true;
}

ReductionDimensions GetReductionKindAndContiguousComponents(
    const HloInstruction& reduce) {
  return GetReductionKindAndContiguousComponentsImpl(reduce.operand(0)->shape(),
                                                     reduce.dimensions());
}

ReductionDimensions GetReductionKindAndContiguousComponents(
    mlir::Operation* reduce) {
  mlir::Value input = reduce->getOperand(0);
  Shape operand_shape = GetShape(input);
  llvm::SmallVector<int64_t> dimensions_to_reduce;
  for (const llvm::APInt& d :
       mlir::cast<mlir::mhlo::ReduceOp>(reduce).dimensions()) {
    dimensions_to_reduce.push_back(d.getZExtValue());
  }
  return GetReductionKindAndContiguousComponentsImpl(operand_shape,
                                                     dimensions_to_reduce);
}

// This emits a device-side call to
// "i32 vprintf(i8* fmt, arguments_type* arguments)" in the driver; see
// http://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html#system-calls
llvm::Value* EmitPrintf(absl::string_view fmt,
                        absl::Span<llvm::Value* const> arguments,
                        llvm::IRBuilder<>* builder) {
  std::vector<llvm::Type*> argument_types;

  // Variadic arguments implicit promotion [1] converts float to double,
  // and bool/char/short are converted to int.
  // [1] https://en.cppreference.com/w/cpp/language/variadic_arguments
  auto requires_int32_promotion = [](llvm::Type* type) {
    return type->isIntegerTy(/*BitWidth=*/1) ||
           type->isIntegerTy(/*BitWidth=*/8) ||
           type->isIntegerTy(/*BitWidth=*/16);
  };
  auto requires_double_promotion = [](llvm::Type* type) {
    return type->isFloatingPointTy();
  };

  for (auto argument : arguments) {
    llvm::Type* type = argument->getType();
    if (requires_double_promotion(type)) {
      argument_types.push_back(builder->getDoubleTy());
    } else if (requires_int32_promotion(type)) {
      argument_types.push_back(builder->getInt32Ty());
    } else {
      argument_types.push_back(type);
    }
  }
  auto* arguments_type = llvm::StructType::create(argument_types);
  llvm::Value* arguments_ptr = builder->CreateAlloca(arguments_type);
  for (size_t i = 0; i < arguments.size(); ++i) {
    llvm::Value* value = arguments[i];
    llvm::Type* type = value->getType();
    if (requires_double_promotion(type)) {
      value = builder->CreateFPCast(value, builder->getDoubleTy());
    } else if (requires_int32_promotion(type)) {
      value = builder->CreateIntCast(value, builder->getInt32Ty(),
                                     /*isSigned=*/true);
    }
    builder->CreateStore(
        value,
        builder->CreateGEP(arguments_type, arguments_ptr,
                           {builder->getInt64(0), builder->getInt32(i)}));
  }
  llvm::Type* ptr_ty = builder->getInt8Ty()->getPointerTo();
  return builder->CreateCall(
      builder->GetInsertBlock()->getParent()->getParent()->getOrInsertFunction(
          "vprintf",
          llvm::FunctionType::get(builder->getInt32Ty(), {ptr_ty, ptr_ty},
                                  /*isVarArg=*/false)),
      {builder->CreateGlobalStringPtr(llvm_ir::AsStringRef(fmt)),
       builder->CreatePointerCast(arguments_ptr, ptr_ty)});
}

// Helper function to emit call to AMDGPU shfl_down function.
llvm::Value* EmitAMDGPUShflDown(llvm::Value* value, llvm::Value* offset,
                                llvm::IRBuilder<>* b) {
  llvm::Module* module = b->GetInsertBlock()->getModule();
  CHECK_EQ(value->getType()->getPrimitiveSizeInBits(), 32);
  auto* i32_ty = b->getInt32Ty();
  llvm::FunctionCallee shfl_fn = module->getOrInsertFunction(
      llvm_ir::AsStringRef("__ockl_readuplane_i32"),
      llvm::FunctionType::get(/*Result=*/i32_ty, {i32_ty, i32_ty},
                              /*isVarArg=*/false));
  // AMDGPU device function requires first argument as i32.
  llvm::Value* result =
      b->CreateCall(shfl_fn, {b->CreateBitCast(value, i32_ty), offset});
  // AMDGPU device function always returns an i32 type.
  return b->CreateBitCast(result, value->getType());
}

// Helper function to emit call to NVPTX shfl_down intrinsic.
llvm::Value* EmitNVPTXShflDown(llvm::Value* value, llvm::Value* offset,
                               llvm::IRBuilder<>* b) {
  llvm::Module* module = b->GetInsertBlock()->getModule();
  llvm::Intrinsic::ID llvm_intrinsic_id;
  CHECK_EQ(value->getType()->getPrimitiveSizeInBits(), 32);
  if (value->getType()->isFloatTy()) {
    llvm_intrinsic_id = llvm::Intrinsic::nvvm_shfl_sync_down_f32;
  } else {
    llvm_intrinsic_id = llvm::Intrinsic::nvvm_shfl_sync_down_i32;
  }
  llvm::Function* intrinsic =
      llvm::Intrinsic::getDeclaration(module, llvm_intrinsic_id, {});
  return b->CreateCall(
      intrinsic, {b->getInt32(-1), value, offset, b->getInt32(WarpSize() - 1)});
}

llvm::Value* EmitFullWarpShuffleDown(llvm::Value* value, llvm::Value* offset,
                                     llvm::IRBuilder<>* builder) {
  int bit_width = value->getType()->getPrimitiveSizeInBits();
  llvm::Module* module = builder->GetInsertBlock()->getModule();
  llvm::Triple target_triple = llvm::Triple(module->getTargetTriple());

  // Special case for efficiency
  if (value->getType()->isFloatTy() && bit_width == 32) {
    if (target_triple.isNVPTX()) {
      return EmitNVPTXShflDown(value, offset, builder);
    } else if (target_triple.getArch() == llvm::Triple::amdgcn) {
      return EmitAMDGPUShflDown(value, offset, builder);
    } else {
      LOG(FATAL) << "Invalid triple " << target_triple.str();
    }
  }

  // We must split values wider than 32 bits as the "shfl" instruction operates
  // on 32-bit values.
  int num_segments = CeilOfRatio(bit_width, 32);
  llvm::Value* x = builder->CreateBitCast(
      builder->CreateZExt(
          builder->CreateBitCast(value, builder->getIntNTy(bit_width)),
          builder->getIntNTy(32 * num_segments)),
      llvm::VectorType::get(builder->getInt32Ty(), num_segments, false));
  for (int i = 0; i < num_segments; ++i) {
    llvm::Value* insert_val;
    if (target_triple.isNVPTX()) {
      insert_val = EmitNVPTXShflDown(builder->CreateExtractElement(x, i),
                                     offset, builder);
    } else if (target_triple.getArch() == llvm::Triple::amdgcn) {
      insert_val = EmitAMDGPUShflDown(builder->CreateExtractElement(x, i),
                                      offset, builder);
    } else {
      LOG(FATAL) << "Invalid triple " << target_triple.str();
    }
    x = builder->CreateInsertElement(x, insert_val, i);
  }
  return builder->CreateBitCast(
      builder->CreateTrunc(
          builder->CreateBitCast(x, builder->getIntNTy(32 * num_segments)),
          builder->getIntNTy(bit_width)),
      value->getType());
}

llvm::Value* IsBlock0Thread0(llvm::IRBuilder<>* b) {
  llvm::Value* is_thread0 = b->CreateICmpEQ(
      b->getInt32(0),
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kThreadIdx, {}, {}, b));

  llvm::Value* is_block0 = b->CreateICmpEQ(
      b->getInt32(0),
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kBlockIdx, {}, {}, b));
  return b->CreateAnd(is_thread0, is_block0);
}

bool IsFusedReductionOutputConsistent(const HloInstruction* inst,
                                      const HloInstruction* first_reduce) {
  if (IsReductionFromOrToContiguousDimensions(*inst)) {
    // Shapes, layouts and dimensions must be the same for all reduces
    // inside of this fusion.
    // TODO(tjoerg): Relax the shape constraint. The datatype does not matter.
    return ShapeUtil::Equal(first_reduce->shape(), inst->shape()) &&
           ShapeUtil::Equal(first_reduce->operand(0)->shape(),
                            inst->operand(0)->shape()) &&
           ShapeUtil::Equal(first_reduce->operand(1)->shape(),
                            inst->operand(1)->shape()) &&
           first_reduce->dimensions() == inst->dimensions();
  }
  return ShapeUtil::CompatibleIgnoringElementType(
             first_reduce->operand(0)->shape(), inst->shape()) &&
         LayoutUtil::Equal(first_reduce->operand(0)->shape().layout(),
                           inst->shape().layout());
}

// Given an LMHLO op, returns the operand index of the first output operand.
//
// Notice that an operand alised to an output isn't an output, even though in
// that case WritesMlirBuffer() returns true on that operand.
//
// An operand is !WritesMlirBuffer() || equals (aliases) to a later operand. An
// output is the opposite, being both WritesMlirBuffer() and does not equal to
// any later operand.
int PartitionLmhloOperandsAndOutputs(mlir::Operation* op) {
  CHECK(op->getDialect() == op->getContext()->getLoadedDialect("lmhlo"));

  int i;
  for (i = op->getOperands().size() - 1; i >= 0; i--) {
    const bool aliased =
        std::find(op->getOperands().begin() + i + 1, op->getOperands().end(),
                  op->getOperand(i)) != op->getOperands().end();
    if (!WritesMlirBuffer(op, op->getOperand(i)) || aliased) {
      break;
    }
  }
  return i + 1;
}

std::vector<mlir::Value> GetHloOperands(mlir::Operation* op) {
  if (auto fusion = mlir::dyn_cast<mlir::lmhlo::FusionOp>(op)) {
    return ToStdVector(fusion.getInputBuffers());
  }
  if (op->getDialect() == op->getContext()->getLoadedDialect("lmhlo")) {
    int output_start = PartitionLmhloOperandsAndOutputs(op);
    std::vector<mlir::Value> operands;
    operands.reserve(output_start);
    for (int i = 0; i < output_start; i++) {
      operands.push_back(op->getOperand(i));
    }
    return operands;
  }
  if (op->getDialect() == op->getContext()->getLoadedDialect("mhlo")) {
    return std::vector<mlir::Value>(op->getOperands().begin(),
                                    op->getOperands().end());
  }
  LOG(FATAL) << "Unexpected op: " << MlirToString(op);
}

std::vector<mlir::Value> GetHloOutputs(mlir::Operation* op) {
  if (auto fusion = mlir::dyn_cast<mlir::lmhlo::FusionOp>(op)) {
    return ToStdVector(fusion.getOutputBuffers());
  }
  if (op->getDialect() == op->getContext()->getLoadedDialect("lmhlo")) {
    int output_start = PartitionLmhloOperandsAndOutputs(op);
    std::vector<mlir::Value> outputs;
    for (int i = output_start; i < op->getNumOperands(); i++) {
      outputs.push_back(op->getOperand(i));
    }
    return outputs;
  }
  if (op->getDialect() == op->getContext()->getLoadedDialect("mhlo")) {
    return std::vector<mlir::Value>(op->getResults().begin(),
                                    op->getResults().end());
  }
  LOG(FATAL) << "Unexpected op: " << MlirToString(op);
}

bool WritesMlirBuffer(mlir::Operation* op, mlir::Value operand) {
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 2> effects;
  mlir::cast<mlir::MemoryEffectOpInterface>(op).getEffectsOnValue(operand,
                                                                  effects);
  return absl::c_any_of(
      effects, [](const mlir::MemoryEffects::EffectInstance& instance) {
        return mlir::isa<mlir::MemoryEffects::Write>(instance.getEffect());
      });
}

static int64_t GetMemRefSizeInBytes(mlir::MemRefType type) {
  // For i1 memrefs, the underlying allocation is 8 bits.
  if (type.getElementType().isInteger(/*width=*/1)) {
    return type.getNumElements();
  } else {
    return type.cast<mlir::ShapedType>().getSizeInBits() / CHAR_BIT;
  }
}

static int64_t GetAllocationIndex(mlir::BlockArgument func_arg,
                                  std::string* constant_name) {
  auto func_op =
      mlir::cast<mlir::func::FuncOp>(func_arg.getParentRegion()->getParentOp());
  if (constant_name) {
    if (auto constant_name_attr = func_op.getArgAttrOfType<mlir::StringAttr>(
            func_arg.getArgNumber(), "lmhlo.constant_name")) {
      *constant_name = constant_name_attr.getValue().str();
    }
  }
  return func_arg.getArgNumber();
}

StatusOr<BufferAllocation::Slice> GetAllocationSlice(
    mlir::Value v, absl::Span<const BufferAllocation> allocations,
    std::string* constant_name) {
  if (constant_name) {
    constant_name->clear();
  }

  int64_t size = GetMemRefSizeInBytes(v.getType().cast<mlir::MemRefType>());

  // We match the following patterns here:
  //  base := ViewOp(arg) | get_global_memref (global_memref) | arg
  //  root := base | MemRefReinterpretCastOp(base)

  if (auto cast = mlir::dyn_cast_or_null<mlir::memref::ReinterpretCastOp>(
          v.getDefiningOp())) {
    v = cast.getViewSource();
  }
  if (auto view =
          mlir::dyn_cast_or_null<mlir::memref::ViewOp>(v.getDefiningOp())) {
    TF_RET_CHECK(view.getSource().isa<mlir::BlockArgument>());

    return BufferAllocation::Slice(
        &allocations[GetAllocationIndex(
            view.getSource().cast<mlir::BlockArgument>(), constant_name)],
        mlir::cast<mlir::arith::ConstantOp>(view.getByteShift().getDefiningOp())
            .getValue()
            .cast<mlir::IntegerAttr>()
            .getValue()
            .getSExtValue(),
        size);
  }
  if (auto get_global = mlir::dyn_cast_or_null<mlir::memref::GetGlobalOp>(
          v.getDefiningOp())) {
    auto module = get_global->getParentOfType<mlir::ModuleOp>();
    if (constant_name) {
      *constant_name = get_global.getName().str();
    }
    auto global = mlir::cast<mlir::memref::GlobalOp>(
        module.lookupSymbol(get_global.getName()));
    int64_t index =
        global->getAttrOfType<mlir::IntegerAttr>("lmhlo.alloc").getInt();
    return BufferAllocation::Slice(&allocations[index], 0,
                                   allocations[index].size());
  }
  if (auto arg = v.dyn_cast<mlir::BlockArgument>()) {
    return BufferAllocation::Slice(
        &allocations[GetAllocationIndex(arg, constant_name)], 0, size);
  }

  return Unimplemented(
      "Operand has to be in the form of ViewOp(arg) or "
      "StaticMemRefCastOp(ViewOp(arg)) or arg");
}

bool CanEmitFusedDynamicUpdateSliceInPlaceForGpu(
    mlir::lmhlo::FusionOp fusion,
    absl::Span<const BufferAllocation> allocations) {
  auto results = fusion.getFusionResults();
  if (results.size() != 1) {
    return false;
  }
  auto dus = mlir::dyn_cast<mlir::mhlo::DynamicUpdateSliceOp>(
      results[0].getDefiningOp());
  if (!dus) {
    return false;
  }

  auto output_buffers = fusion.getOutputBuffers();
  CHECK_EQ(1, output_buffers.size());
  auto parameter = mlir::dyn_cast<mlir::bufferization::ToTensorOp>(
      dus.operand().getDefiningOp());

  if (!parameter) {
    return false;
  }

  auto maybe_lhs = GetAllocationSlice(parameter.getMemref(), allocations);
  auto maybe_rhs = GetAllocationSlice(output_buffers[0], allocations);
  return maybe_lhs.ok() && maybe_rhs.ok() && *maybe_lhs == *maybe_rhs;
}

Shape GetShape(mlir::Value value) {
  if (value.getType().isa<mlir::MemRefType>()) {
    return TypeToShape(value.getType());
  } else if (value.getType().isa<mlir::TensorType>()) {
    return GetShapeFromTensorType(value);
  } else if (value.getType().isa<mlir::TupleType>()) {
    return TypeToShape(value.getType());
  }
  LOG(FATAL) << "Unexpected value type to get shape for";
  return {};
}

bool ReductionIsRaceFree(const ReductionDimensions& reduction_dimensions,
                         const std::array<int64_t, 3>& reduction_tiling) {
  return (reduction_dimensions.is_row_reduction &&
          reduction_dimensions.dimensions[2] <=
              MinThreadsXRowReduction() * reduction_tiling[2] &&
          reduction_dimensions.dimensions[0] <=
              BatchedReductionRaceFreeBound()) ||
         (!reduction_dimensions.is_row_reduction &&
          reduction_dimensions.dimensions[1] <=
              WarpSize() * reduction_tiling[1]);
}

// A recursive function to inspect the users of a parameter to determine
// whether it's safe for a parameter to participate in a shared-memory
// transpose.
//
// Consider a fusion parameter P for which we might want to use a shmem
// transpose.  If we do, we use a GPU thread block to preload a tile of P with
// indices [z, y..y+31, x..x+31] to compute an output tile with the same indices
// cooperatively, where z, y, x are the indices for the normalized input/output
// tensor (see the document for FindTranspose021 for the definition of
// normalized tensor for 0-2-1 transpose). This shmem transpose implementation
// requires that the computation of the output tile only read elements within
// the preload tile. If this is not true, we can't use a shmem transpose for P.
//
// If the computation of output element [z, y, x] only requires the element of
// P with the same indices, the shmem transpose implementation can be applied
// to P safely. This is a sufficient but not necessary condition. We check all
// the transitive users of P to see if we can find a user that may cause an
// exception to the situation. If such a user is not found, we conclude that P
// is safe for shmem transpose.
//
// This is trivially true for elementwise operations and some "data-movement"
// ops like kTuple. However, it's not true for operations that can change the
// dimensions of the inputs (e.g. pad, slice) and bitcast operation.
// For example:
//
// fused_computation {
//   param_0 = f32[64,64]{1,0} parameter(0)
//   ROOT bitcast = f32[64,64]{0,1} bitcast(param_0)
// }
// The output element at logical address [0, 63] depends on the input element
// at logical address [63, 0], which would not be within the shared-memory
// block.
//
// TODO(bixia): In order to extend this for kInput fusion, that is reduction
// with transpose, we only need to end the use-chain checking with the input of
// a reduce operations. In this case, the above description on "output" apply
// to the result of such a use-chain, which provides the input to the reduce
// operation.
static bool IsInstructionSafeForShmemTransposeRec(
    const HloInstruction* hlo,
    absl::flat_hash_map<const HloInstruction*, bool>* memoized) {
  auto it = memoized->find(hlo);
  if (it != memoized->end()) {
    return it->second;
  }
  static const auto* const to_traverse = new absl::flat_hash_set<HloOpcode>{
      HloOpcode::kGetTupleElement, HloOpcode::kMap, HloOpcode::kParameter,
      HloOpcode::kTuple};

  bool is_safe = [&] {
    if (hlo->IsElementwise() || to_traverse->contains(hlo->opcode())) {
      return absl::c_all_of(hlo->users(), [&](const HloInstruction* user) {
        return IsInstructionSafeForShmemTransposeRec(user, memoized);
      });
    } else if (hlo->opcode() == HloOpcode::kGetDimensionSize) {
      // The result of the operation doesn't rely on the content of the
      // tensor. As such, there is no need to further inspect its users.
      return true;
    }
    return false;
  }();

  memoized->emplace(hlo, is_safe);
  return is_safe;
}

static bool IsInstructionSafeForShmemTranspose(const HloInstruction* hlo) {
  absl::flat_hash_map<const HloInstruction*, bool> mapping;
  return IsInstructionSafeForShmemTransposeRec(hlo, &mapping);
}

// Given a group of input parameters that are 0-2-1 transpose of the outputs
// of a fusion kernel, returns the input parameters that are safe for the
// shared memory transpose implementation.
//
// When a tile based shared memory transpose is used to implement an input
// with 0-2-1 transpose, we preload a tile of the input elements [z, y..y+31,
// x..x+31] to compute the output tile elements of the same indices.
// Preloading the input tile this way is only safe when the computation of the
// output tile elements do not need any input element outside the preloaded
// tile. We inspect all the transitive users of the input parameter up to the
// fusion root instruction to see if we can find any instruction that can make
// preloading the input tile unsafe.
static std::vector<int64_t> FilterInputsForShmemTranspose(
    const HloComputation* fused_computation, std::vector<int64_t> input_ids) {
  std::vector<int64_t> filtered_input_ids;
  for (int64_t input_id : input_ids) {
    const HloInstruction* instr =
        fused_computation->parameter_instruction(input_id);
    if (IsInstructionSafeForShmemTranspose(instr)) {
      filtered_input_ids.push_back(input_id);
    } else {
      VLOG(10) << "Input not safe for shmem transpose " << instr->ToString();
    }
  }
  return filtered_input_ids;
}

// Whether code emission supports shared memory transposes for one or more
// `input_ids`.
bool ShmemTransposeSupportedForInputs(const HloInstruction& instr,
                                      std::vector<int64_t> input_ids) {
  if (instr.opcode() == HloOpcode::kFusion) {
    return !FilterInputsForShmemTranspose(
                instr.fused_instructions_computation(), input_ids)
                .empty();
  }
  // Needs to be kept in sync with `IsInstructionSafeForShmemTranspose` above.
  return instr.IsElementwise() ||
         instr.opcode() == HloOpcode::kGetDimensionSize;
}

static std::optional<TransposeDimsAndParams> FindTranspose021DimsAndParameters(
    const std::vector<Shape>& operand_shapes, const Shape& output_shape) {
  std::vector<int64_t> params_012;
  std::optional<Vector3> reduced_dims_021;
  for (int64_t operand_idx = 0; operand_idx < operand_shapes.size();
       ++operand_idx) {
    std::optional<Vector3> find_transpose_result =
        ShapeUtil::FindTranspose021(operand_shapes[operand_idx], output_shape);
    if (!find_transpose_result.has_value()) {
      continue;
    } else if (!reduced_dims_021.has_value()) {
      reduced_dims_021 = *find_transpose_result;
    } else if (!absl::c_equal(*reduced_dims_021, *find_transpose_result)) {
      // There is more than one possible transpose. Instead of picking one
      // transpose, we simply give up here.
      VLOG(3) << "021 transpose not matched; More than one possible "
                 "transposition of parameters: "
              << VectorString(*reduced_dims_021) << " and "
              << VectorString(*find_transpose_result);
      return std::nullopt;
    }
    params_012.push_back(operand_idx);
  }
  if (!reduced_dims_021.has_value()) {
    return std::nullopt;
  }
  return TransposeDimsAndParams{*reduced_dims_021, params_012};
}

std::optional<TransposeDimsAndParams> Match021Transpose(
    const HloComputation* fused_computation) {
  // If a dimensions is smaller than this, untiled transposition may be more
  // efficient.
  static const int64_t kMinDimensionToTransposeTiled = 16;

  const HloInstruction* root = fused_computation->root_instruction();
  if (root->shape().IsTuple()) {
    // TODO(cheshire): Why we are not pattern-matching other outputs?
    root = root->operand(0);
  }
  const Shape& output_shape = root->shape();

  std::vector<Shape> param_shapes;
  absl::c_for_each(fused_computation->parameter_instructions(),
                   [&](const HloInstruction* param) {
                     param_shapes.push_back(param->shape());
                   });
  std::optional<TransposeDimsAndParams> reduced_dims_and_params_021 =
      FindTranspose021DimsAndParameters(param_shapes, output_shape);

  if (!reduced_dims_and_params_021.has_value()) {
    VLOG(3) << "021 transposition not found on instruction "
            << root->ToString();
    return std::nullopt;
  }
  std::vector<int64_t> params_012 = reduced_dims_and_params_021->params;
  const Vector3& reduced_dims_021 = reduced_dims_and_params_021->dims;

  if (reduced_dims_021.at(1) < kMinDimensionToTransposeTiled ||
      reduced_dims_021.at(2) < kMinDimensionToTransposeTiled) {
    VLOG(3) << "021 transpose not matched: dimensions of transposition "
            << root->ToString() << " are too small";
    return std::nullopt;
  }

  params_012 = FilterInputsForShmemTranspose(fused_computation, params_012);
  if (params_012.empty()) {
    VLOG(3) << "021 transpose on " << root->ToString()
            << "not matched: no inputs matched after filtering "
               "for shmem access";
    return std::nullopt;
  }

  // Each of our shared memory tiles has 32*33 elements (so ~4kb, if the
  // elements are of size 4 bytes), and CUDA has an architectural limit of
  // 48kb shared memory per SM.  (This is increased to 96kb in Volta, but we
  // don't use this, in part because it eats into our L1 cache space.)
  //
  // For correctness we need to ensure that we don't make more than 48kb worth
  // of shmem tiles per block.  And for performance, we'd probably like to use
  // significantly less, so that we can fit more than one block at a time on a
  // gpu core.
  //
  // We say without benchmarks that we want at least 3 threads/block,
  // corresponding to 3 shmem tiles if the elements are 32 bits wide.  We
  // choose which params get the shmem transpose treatment arbitrarily; it's
  // not clear if there's a Right Choice.
  //
  // This is only sound if tiled transposes are the only place where we use
  // shared memory in fusions.  If in the future other fusible ops use shared
  // memory, we'll have to adjust this heuristic.
  constexpr int kMinBlocksPerCore = 3;
  constexpr int64_t kShmemPerCore = 48 * 1024;
  int64_t shmem_used = 0;
  for (int64_t i = 0; i < params_012.size(); ++i) {
    const Shape& operand_shape =
        fused_computation->parameter_instruction(params_012[i])->shape();
    shmem_used +=
        32 * 33 *
        ShapeUtil::ByteSizeOfPrimitiveType(operand_shape.element_type());

    if (kMinBlocksPerCore * shmem_used > kShmemPerCore) {
      // Erase this element and everything after it from params_012.
      params_012.resize(i);
      break;
    }
  }

  if (params_012.empty()) {
    VLOG(3)
        << "021 transpose on :" << root->ToString()
        << " not matched: no inputs matched after filtering for shmem budget";
    return std::nullopt;
  }

  return TransposeDimsAndParams{reduced_dims_021, params_012};
}

}  // namespace gpu
}  // namespace xla
