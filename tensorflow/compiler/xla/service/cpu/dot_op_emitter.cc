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

#include "tensorflow/compiler/xla/service/cpu/dot_op_emitter.h"

#include <memory>
#include <vector>

#include "external/llvm/include/llvm/IR/BasicBlock.h"
#include "external/llvm/include/llvm/IR/Instructions.h"
#include "external/llvm/include/llvm/IR/Module.h"
#include "external/llvm/include/llvm/IR/Value.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_runtime_flags.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

using llvm_ir::SetToFirstInsertPoint;

namespace cpu {

DotOpEmitter::DotOpEmitter(const HloInstruction& dot, bool transpose_lhs,
                           bool transpose_rhs,
                           const llvm_ir::IrArray& target_array,
                           const llvm_ir::IrArray& lhs_array,
                           const llvm_ir::IrArray& rhs_array,
                           llvm::Value* executable_run_options_value,
                           llvm::IRBuilder<>* ir_builder)
    : dot_(dot),
      transpose_lhs_(transpose_lhs),
      transpose_rhs_(transpose_rhs),
      target_array_(target_array),
      lhs_array_(lhs_array),
      rhs_array_(rhs_array),
      executable_run_options_value_(executable_run_options_value),
      ir_builder_(ir_builder) {}

/* static */ tensorflow::Status DotOpEmitter::EmitDotOperation(
    const HloInstruction& dot, bool transpose_lhs, bool transpose_rhs,
    const llvm_ir::IrArray& target_array, const llvm_ir::IrArray& lhs_array,
    const llvm_ir::IrArray& rhs_array,
    llvm::Value* executable_run_options_value, llvm::IRBuilder<>* ir_builder) {
  PrimitiveType type = target_array.GetShape().element_type();
  TF_RET_CHECK(F32 == type || F64 == type);
  DotOpEmitter dot_emitter(dot, transpose_lhs, transpose_rhs, target_array,
                           lhs_array, rhs_array, executable_run_options_value,
                           ir_builder);
  return dot_emitter.Emit();
}

bool DotOpEmitter::ShapesAreLegalForRuntimeDot() const { return true; }

tensorflow::Status DotOpEmitter::Emit() {
  // The dot operation performs a sum of products over dimension 0 of the left
  // hand side operand and dimension 1 of the right hand side operand.
  //
  // Let the shapes of lhs and rhs be defined as below:
  //
  //   lhs = [L{n-1} x L{n-2} x ... L{0}]
  //   rhs = [R{m-1} x R{m-2} x ... R{0}]
  //
  // The sum-of-products dimension in the lhs has size L{0} and the dimension in
  // the rhs has size R{1}. Necessarily, then:
  //
  //   L{0} == R{1}
  //
  // The output of the operation has the following shape:
  //
  //   output = [L{n-1} x L{n-2} x ... L{1} x R{m-1} x R{m-2} x ... R{2} x R{0}]
  //
  // To perform the operation we construct a loop nest with one for-loop for
  // each dimension of the output. Inside this loop nest is another for-loop
  // which performs the sum-of-products (the reduction loop) before storing
  // the result in the output buffer.

  const Shape& lhs_shape = lhs_array_.GetShape();
  const Shape& rhs_shape = rhs_array_.GetShape();

  if (ShapeUtil::IsScalar(lhs_shape) || ShapeUtil::IsScalar(rhs_shape)) {
    // If the operands are scalar, don't emit any loops.
    TF_RET_CHECK(ShapeUtil::IsScalar(lhs_shape) &&
                 ShapeUtil::IsScalar(rhs_shape));
    return EmitScalarDot();
  }

  if (PotentiallyImplementedAsEigenDot(dot_)) {
    return EmitCallToRuntime();
  }

  // Reduce along dimension 0 of the LHS and 1 of the RHS. Vectors are a special
  // case where the reduction dimension is 0 for both LHS and RHS. This results
  // in a vector dot product producing a scalar.
  int64 lhs_reduction_dimension = 0;
  if (ShapeUtil::Rank(lhs_shape) >= 2) {
    lhs_reduction_dimension =
        ShapeUtil::GetDimensionNumber(lhs_shape, transpose_lhs_ ? -2 : -1);
  }
  int64 rhs_reduction_dimension = 0;
  if (ShapeUtil::Rank(rhs_shape) >= 2) {
    rhs_reduction_dimension =
        ShapeUtil::GetDimensionNumber(rhs_shape, transpose_rhs_ ? -1 : -2);
  }

  // Verify the reduction dimension in the two operands are the same size.
  TF_RET_CHECK(lhs_shape.dimensions(lhs_reduction_dimension) ==
               rhs_shape.dimensions(rhs_reduction_dimension));

  // Create loop nests which loop through the LHS operand dimensions and the RHS
  // operand dimensions. The reduction dimension of the LHS and RHS are handled
  // in a separate innermost loop which performs the sum of products.
  llvm_ir::ForLoopNest loop_nest(ir_builder_);
  llvm_ir::IrArray::Index lhs_index = EmitOperandArrayLoopNest(
      &loop_nest, lhs_array_, lhs_reduction_dimension, "lhs");
  llvm_ir::IrArray::Index rhs_index = EmitOperandArrayLoopNest(
      &loop_nest, rhs_array_, rhs_reduction_dimension, "rhs");

  // Create the loop which does the sum of products reduction.
  std::unique_ptr<llvm_ir::ForLoop> reduction_loop = loop_nest.AddLoop(
      0, lhs_shape.dimensions(lhs_reduction_dimension), "reduction");

  // The final entry in the rhs and lhs indexes is the indvar of the
  // reduction loop.
  lhs_index[lhs_reduction_dimension] = reduction_loop->GetIndVarValue();
  rhs_index[rhs_reduction_dimension] = reduction_loop->GetIndVarValue();

  // For computing the sum of products we alloca a single location to store the
  // dot product result as we accumulate it within the reduction loop. After the
  // reduction loop we load the result and store into the output array.

  // Function entry basic block.
  // - Emit alloca for accumulator
  llvm::Function* func = reduction_loop->GetPreheaderBasicBlock()->getParent();
  SetToFirstInsertPoint(&func->getEntryBlock(), ir_builder_);
  llvm::Type* accum_type = target_array_.GetElementLlvmType();
  llvm::Value* accum_address = ir_builder_->CreateAlloca(
      accum_type, /*ArraySize=*/nullptr, "accum_address");

  // Preheader basic block of reduction loop:
  // - Initialize accumulator to zero.
  llvm::BasicBlock* preheader_bb = reduction_loop->GetPreheaderBasicBlock();
  ir_builder_->SetInsertPoint(preheader_bb->getTerminator());

  ir_builder_->CreateStore(llvm::ConstantFP::get(accum_type, 0.0),
                           accum_address);

  // Body basic block of reduction loop:
  // - Load elements from lhs and rhs array.
  // - Multiply lhs-element and rhs-element.
  // - Load accumulator and add to product.
  // - Store sum back into accumulator.
  SetToFirstInsertPoint(reduction_loop->GetBodyBasicBlock(), ir_builder_);

  llvm::Value* lhs_element =
      lhs_array_.EmitReadArrayElement(lhs_index, ir_builder_);
  llvm::Value* rhs_element =
      rhs_array_.EmitReadArrayElement(rhs_index, ir_builder_);

  llvm::Value* product = ir_builder_->CreateFMul(lhs_element, rhs_element);
  llvm::Value* accum = ir_builder_->CreateLoad(accum_address);
  llvm::Value* updated_accum = ir_builder_->CreateFAdd(accum, product);
  ir_builder_->CreateStore(updated_accum, accum_address);

  // Exit basic block of reduction loop.
  // - Load accumulator value (the result).
  // - Store into output array.
  SetToFirstInsertPoint(reduction_loop->GetExitBasicBlock(), ir_builder_);

  llvm::Value* result = ir_builder_->CreateLoad(accum_address);

  // Create index into target address. The target index is the concatenation of
  // the rhs and lhs indexes with the reduction dimensions removed. The terms
  // from the rhs index are the lower dimensions in the index so we add them
  // first.
  llvm_ir::IrArray::Index target_index;
  for (int dimension = 0; dimension < lhs_index.size(); ++dimension) {
    if (dimension != lhs_reduction_dimension) {
      target_index.push_back(lhs_index[dimension]);
    }
  }
  for (int dimension = 0; dimension < rhs_index.size(); ++dimension) {
    if (dimension != rhs_reduction_dimension) {
      target_index.push_back(rhs_index[dimension]);
    }
  }

  target_array_.EmitWriteArrayElement(target_index, result, ir_builder_);

  // Set the IR builder insert point to the exit basic block of the outer most
  // loop.
  ir_builder_->SetInsertPoint(loop_nest.GetOuterLoopExitBasicBlock());

  return tensorflow::Status::OK();
}

tensorflow::Status DotOpEmitter::EmitScalarDot() {
  // A scalar dot is just a scalar multiply.
  llvm::Value* lhs_value =
      lhs_array_.EmitReadArrayElement(/*index=*/{}, ir_builder_);
  llvm::Value* rhs_value =
      rhs_array_.EmitReadArrayElement(/*index=*/{}, ir_builder_);
  llvm::Value* result = ir_builder_->CreateFMul(lhs_value, rhs_value);
  target_array_.EmitWriteArrayElement(/*index=*/{}, result, ir_builder_);
  return tensorflow::Status::OK();
}

tensorflow::Status DotOpEmitter::EmitCallToRuntime() {
  DCHECK(ShapesAreLegalForRuntimeDot());

  // The signature of the Eigen runtime matmul function is:
  //
  //   (void)(void* run_options, float* out, float* lhs, float* rhs,
  //          int64 m, int64 n, int64 k, int32 transpose_lhs,
  //          int32 transpose_rhs);
  // The two transpose_... parameters are actually booleans, but we use int32
  // to avoid target-dependent calling convention details.

  legacy_flags::CpuRuntimeFlags* flags = legacy_flags::GetCpuRuntimeFlags();
  bool multi_threaded = flags->xla_cpu_multi_thread_eigen;
  PrimitiveType type = target_array_.GetShape().element_type();
  llvm::Type* float_type;
  const char* fn_name;
  switch (type) {
    case F32:
      fn_name = multi_threaded
                    ? runtime::kEigenMatmulF32SymbolName
                    : runtime::kEigenSingleThreadedMatmulF32SymbolName;
      float_type = ir_builder_->getFloatTy();
      break;
    case F64:
      fn_name = multi_threaded
                    ? runtime::kEigenMatmulF64SymbolName
                    : runtime::kEigenSingleThreadedMatmulF64SymbolName;
      float_type = ir_builder_->getDoubleTy();
      break;
    default:
      return Unimplemented("Invalid type %s for dot operation",
                           PrimitiveType_Name(type).c_str());
  }

  llvm::Type* float_ptr_type = float_type->getPointerTo();
  llvm::Type* int64_type = ir_builder_->getInt64Ty();
  llvm::Type* int32_type = ir_builder_->getInt32Ty();
  llvm::Type* int8_ptr_type = ir_builder_->getInt8Ty()->getPointerTo();
  llvm::FunctionType* matmul_type = llvm::FunctionType::get(
      ir_builder_->getVoidTy(),
      {int8_ptr_type, float_ptr_type, float_ptr_type, float_ptr_type,
       int64_type, int64_type, int64_type, int32_type, int32_type},
      /*isVarArg=*/false);

  llvm::Function* function = ir_builder_->GetInsertBlock()->getParent();
  llvm::Module* module = function->getParent();

  llvm::Function* matmul_func = llvm::cast<llvm::Function>(
      module->getOrInsertFunction(fn_name, matmul_type));
  matmul_func->setCallingConv(llvm::CallingConv::C);
  matmul_func->setDoesNotThrow();
  matmul_func->setOnlyAccessesArgMemory();

  // The Eigen runtime function expects column-major layout. If the matrices are
  // row major, then use the following identity to compute the product:
  //
  //   (A x B)^T = B^T x A^T
  //
  // Effectively this involves swapping the 'lhs' with 'rhs' and 'm' with 'n'.

  const Shape& lhs_shape = lhs_array_.GetShape();
  const Shape& rhs_shape = rhs_array_.GetShape();
  int64 m = lhs_shape.dimensions(transpose_lhs_ ? 1 : 0);
  int64 k = lhs_shape.dimensions(transpose_lhs_ ? 0 : 1);
  int64 n = rhs_shape.dimensions(transpose_rhs_ ? 0 : 1);
  const llvm_ir::IrArray* lhs = &lhs_array_;
  const llvm_ir::IrArray* rhs = &rhs_array_;
  bool transpose_lhs = transpose_lhs_;
  bool transpose_rhs = transpose_rhs_;

  bool is_column_major = lhs_shape.layout().minor_to_major(0) == 0;
  if (!is_column_major) {
    std::swap(m, n);
    std::swap(lhs, rhs);
    std::swap(transpose_lhs, transpose_rhs);
  }

  ir_builder_->CreateCall(
      matmul_func,
      {ir_builder_->CreateBitCast(executable_run_options_value_, int8_ptr_type),
       ir_builder_->CreateBitCast(target_array_.GetBasePointer(),
                                  float_ptr_type),
       ir_builder_->CreateBitCast(lhs->GetBasePointer(), float_ptr_type),
       ir_builder_->CreateBitCast(rhs->GetBasePointer(), float_ptr_type),
       ir_builder_->getInt64(m), ir_builder_->getInt64(n),
       ir_builder_->getInt64(k), ir_builder_->getInt32(transpose_lhs),
       ir_builder_->getInt32(transpose_rhs)});
  return tensorflow::Status::OK();
}

llvm_ir::IrArray::Index DotOpEmitter::EmitOperandArrayLoopNest(
    llvm_ir::ForLoopNest* loop_nest, const llvm_ir::IrArray& operand_array,
    int64 reduction_dimension, tensorflow::StringPiece name_suffix) {
  // Prepares the dimension list we will use to emit the loop nest. Outermost
  // loops are added first. Add loops in major-to-minor order, and skip the
  // reduction dimension.
  std::vector<int64> dimensions;
  const Shape& shape = operand_array.GetShape();
  for (int i = shape.layout().minor_to_major_size() - 1; i >= 0; --i) {
    int64 dimension = shape.layout().minor_to_major(i);
    if (dimension != reduction_dimension) {
      dimensions.push_back(dimension);
    }
  }

  // Create loop nest with one for-loop for each dimension of the
  // output.
  llvm_ir::IrArray::Index index =
      loop_nest->AddLoopsForShapeOnDimensions(shape, dimensions, name_suffix);
  // Verify every dimension except the reduction dimension was set in the index.
  for (int dimension = 0; dimension < index.size(); ++dimension) {
    if (dimension == reduction_dimension) {
      DCHECK_EQ(nullptr, index[dimension]);
    } else {
      DCHECK_NE(nullptr, index[dimension]);
    }
  }
  return index;
}

}  // namespace cpu
}  // namespace xla
