/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file me:
    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/backends/cpu/codegen/tiled/tiled_computation_emitter.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/backends/cpu/codegen/symbol_name_util.h"
#include "xla/backends/cpu/codegen/tiled/tiled_fusion_emitter.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/emitters/type_util.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/service/buffer_assignment.h"

namespace xla::cpu {

namespace {

mlir::Value CreateEntryBlockAlloca(mlir::ImplicitLocOpBuilder& b,
                                   mlir::func::FuncOp func,
                                   mlir::MemRefType type) {
  mlir::OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(&func.front());
  return mlir::memref::AllocaOp::create(b, type);
}

mlir::Value GetScalarValue(mlir::ImplicitLocOpBuilder& b,
                           const ComputationEnvironment& env,
                           const HloInstruction* instr) {
  mlir::Value val = env.GetValue(instr);
  if (!val && instr->shape().IsTuple()) {
    val = env.GetTupleElement(instr, 0);
  }
  if (!val) return nullptr;
  if (auto unranked_type =
          mlir::dyn_cast<mlir::UnrankedMemRefType>(val.getType())) {
    auto scalar_type =
        mlir::MemRefType::get({}, unranked_type.getElementType());
    val = mlir::memref::CastOp::create(b, scalar_type, val);
    return mlir::memref::LoadOp::create(b, val);
  }
  if (auto memref_type = mlir::dyn_cast<mlir::MemRefType>(val.getType())) {
    if (memref_type.getRank() == 0) {
      return mlir::memref::LoadOp::create(b, val);
    }
    mlir::Value zero = mlir::arith::ConstantIndexOp::create(b, 0);
    llvm::SmallVector<mlir::Value> indices(memref_type.getRank(), zero);
    return mlir::memref::LoadOp::create(b, val, indices);
  }
  return val;
}

}  // namespace

TiledComputationKernelEmitter::TiledComputationKernelEmitter(
    const HloInstruction* instr, std::string name)
    : instr_(instr), name_(name) {}

absl::StatusOr<KernelEmitter<MlirKernelSource>::KernelDefinition>
TiledComputationKernelEmitter::EmitKernelDefinition() {
  const HloComputation* computation =
      instr_->has_to_apply() ? instr_->to_apply() : instr_->parent();

  mlir::Location loc = mlir::NameLoc::get(
      mlir::StringAttr::get(&mlir_context_, computation->name()));
  mlir::ImplicitLocOpBuilder b(loc, &mlir_context_);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      llvm_ir::CreateMlirModuleOp(b.getLoc());
  module->getOperation()->setAttr(
      ::xla::CpuMemoryRegionNameAttr::name,
      mlir::StringAttr::get(
          &mlir_context_,
          BuildModuleMemoryRegionName("tiled_computation_emitter", instr_)));
  b.setInsertionPointToEnd(module->getBody());

  // Function signature arguments based on computation parameters and output
  // buffers
  llvm::SmallVector<mlir::Type> arg_types;

  // 1. Input parameter buffer types
  for (const HloInstruction* param : computation->parameter_instructions()) {
    if (param->shape().IsTuple()) {
      for (int i = 0; i < ShapeUtil::TupleElementCount(param->shape()); ++i) {
        const Shape& subshape = ShapeUtil::GetSubshape(param->shape(), {i});
        mlir::Type elem_type =
            emitters::PrimitiveTypeToMlirType(subshape.element_type(), b);
        arg_types.push_back(
            mlir::MemRefType::get(subshape.dimensions(), elem_type));
      }
    } else {
      mlir::Type elem_type =
          emitters::PrimitiveTypeToMlirType(param->shape().element_type(), b);
      arg_types.push_back(
          mlir::MemRefType::get(param->shape().dimensions(), elem_type));
    }
  }

  // 2. Output destination buffer types
  const HloInstruction* root_instr = computation->root_instruction();
  if (root_instr->shape().IsTuple()) {
    for (int i = 0; i < ShapeUtil::TupleElementCount(root_instr->shape());
         ++i) {
      const Shape& subshape = ShapeUtil::GetSubshape(root_instr->shape(), {i});
      mlir::Type elem_type =
          emitters::PrimitiveTypeToMlirType(subshape.element_type(), b);
      arg_types.push_back(
          mlir::MemRefType::get(subshape.dimensions(), elem_type));
    }
  } else {
    mlir::Type elem_type = emitters::PrimitiveTypeToMlirType(
        root_instr->shape().element_type(), b);
    arg_types.push_back(
        mlir::MemRefType::get(root_instr->shape().dimensions(), elem_type));
  }

  auto func_type = b.getFunctionType(arg_types, {});
  auto func = mlir::func::FuncOp::create(b, name_, func_type);
  func->setAttr("xla.entry", b.getUnitAttr());
  func.addEntryBlock();
  b.setInsertionPointToStart(&func.front());

  ComputationEnvironment env;
  int arg_idx = 0;

  // Map parameter arguments
  for (const HloInstruction* param : computation->parameter_instructions()) {
    if (param->shape().IsTuple()) {
      for (int i = 0; i < ShapeUtil::TupleElementCount(param->shape()); ++i) {
        env.SetTupleElement(param, i, func.getArgument(arg_idx++));
      }
    } else {
      env.SetValue(param, func.getArgument(arg_idx++));
    }
  }

  // Collect destination argument handles
  llvm::SmallVector<mlir::Value> dest_args;
  if (root_instr->shape().IsTuple()) {
    for (int i = 0; i < ShapeUtil::TupleElementCount(root_instr->shape());
         ++i) {
      dest_args.push_back(func.getArgument(arg_idx++));
    }
  } else {
    dest_args.push_back(func.getArgument(arg_idx++));
  }

  RETURN_IF_ERROR(EmitComputation(b, computation, env));

  // Copy/store root outputs to output buffer parameters
  llvm::SmallVector<mlir::Value> root_vals = env.GetAllValues(root_instr);
  for (size_t i = 0; i < root_vals.size() && i < dest_args.size(); ++i) {
    if (root_vals[i]) {
      mlir::Value dest_buf = dest_args[i];
      if (auto root_memref =
              mlir::dyn_cast<mlir::MemRefType>(root_vals[i].getType())) {
        if (mlir::isa<mlir::UnrankedMemRefType>(dest_buf.getType())) {
          dest_buf = mlir::memref::CastOp::create(b, root_memref, dest_buf);
        }
        mlir::memref::CopyOp::create(b, root_vals[i], dest_buf);
      } else {
        if (auto unranked =
                mlir::dyn_cast<mlir::UnrankedMemRefType>(dest_buf.getType())) {
          auto ranked_0d = mlir::MemRefType::get({}, unranked.getElementType());
          dest_buf = mlir::memref::CastOp::create(b, ranked_0d, dest_buf);
        }
        mlir::Value val_to_store = root_vals[i];
        if (auto memref_type =
                mlir::dyn_cast<mlir::MemRefType>(dest_buf.getType())) {
          if (val_to_store.getType() != memref_type.getElementType()) {
            if (val_to_store.getType().isInteger(1) &&
                memref_type.getElementType().isInteger(8)) {
              val_to_store = mlir::arith::ExtUIOp::create(
                  b, memref_type.getElementType(), val_to_store);
            } else if (val_to_store.getType().isInteger() &&
                       memref_type.getElementType().isInteger()) {
              val_to_store = mlir::arith::TruncIOp::create(
                  b, memref_type.getElementType(), val_to_store);
            }
          }
        }
        mlir::memref::StoreOp::create(b, val_to_store, dest_buf);
      }
    }
  }

  mlir::func::ReturnOp::create(b);

  WorkDimensions work_dimensions;
  work_dimensions.num_work_groups.x = 1;

  KernelSpec kernel_spec(name_, work_dimensions, KernelSpec::Buffers(),
                         KernelSpec::Buffers(), absl::flat_hash_set<int64_t>());
  auto kernel_spec_or = emitters::GetKernelSpec(
      name_, *instr_, buffer_assignment_, work_dimensions);
  if (kernel_spec_or.ok()) {
    kernel_spec = std::move(*kernel_spec_or);
  }

  return KernelDefinition(std::move(kernel_spec),
                          MlirKernelSource(std::move(module)));
}

absl::Status TiledComputationKernelEmitter::EmitComputation(
    mlir::ImplicitLocOpBuilder& b, const HloComputation* computation,
    ComputationEnvironment& env) {
  for (const HloInstruction* instr : computation->MakeInstructionPostOrder()) {
    if (instr->opcode() == HloOpcode::kParameter) {
      continue;
    }
    RETURN_IF_ERROR(EmitInstruction(b, instr, env));
  }
  return absl::OkStatus();
}

absl::Status TiledComputationKernelEmitter::EmitInstruction(
    mlir::ImplicitLocOpBuilder& b, const HloInstruction* instr,
    ComputationEnvironment& env) {
  if (instr->opcode() == HloOpcode::kWhile) {
    return EmitWhile(b, instr, env);
  }
  if (instr->opcode() == HloOpcode::kConditional) {
    return EmitConditional(b, instr, env);
  }
  if (instr->opcode() == HloOpcode::kTuple) {
    if (instr->shape().IsTuple()) {
      for (int64_t i = 0; i < instr->operand_count(); ++i) {
        mlir::Value val = env.GetValue(instr->operand(i));
        if (!val) {
          val = env.GetTupleElement(instr->operand(i), 0);
        }
        if (val) {
          env.SetTupleElement(instr, i, val);
        }
      }
    } else if (instr->operand_count() > 0) {
      mlir::Value val = env.GetValue(instr->operand(0));
      if (val) env.SetValue(instr, val);
    }
    return absl::OkStatus();
  }
  if (instr->opcode() == HloOpcode::kGetTupleElement) {
    int64_t idx = instr->tuple_index();
    const HloInstruction* tuple_operand = instr->operand(0);
    mlir::Value elem = env.GetTupleElement(tuple_operand, idx);
    if (!elem) {
      elem = env.GetValue(tuple_operand);
    }
    if (elem) {
      env.SetValue(instr, elem);
    }
    return absl::OkStatus();
  }

  if (instr->opcode() == HloOpcode::kConstant &&
      ShapeUtil::IsEffectiveScalar(instr->shape())) {
    mlir::Type elem_type =
        emitters::PrimitiveTypeToMlirType(instr->shape().element_type(), b);
    mlir::Value cst;
    if (mlir::isa<mlir::IntegerType>(elem_type)) {
      if (instr->shape().element_type() == PRED) {
        cst = mlir::arith::ConstantIntOp::create(
            b, elem_type, instr->literal().Get<bool>({}));
      } else {
        int64_t val = instr->literal().GetIntegralAsS64({}).value_or(0);
        cst = mlir::arith::ConstantIntOp::create(b, elem_type, val);
      }
    } else if (auto float_type = mlir::dyn_cast<mlir::FloatType>(elem_type)) {
      double val = instr->literal().GetAsDouble({}).value_or(0.0);
      cst = mlir::arith::ConstantFloatOp::create(b, float_type,
                                                 llvm::APFloat(val));
    } else {
      return absl::UnimplementedError(
          absl::StrCat("Unsupported constant type ", instr->ToString()));
    }
    env.SetValue(instr, cst);
    return absl::OkStatus();
  }

  if ((instr->opcode() == HloOpcode::kAdd ||
       instr->opcode() == HloOpcode::kSubtract ||
       instr->opcode() == HloOpcode::kMultiply ||
       instr->opcode() == HloOpcode::kDivide) &&
      ShapeUtil::IsEffectiveScalar(instr->shape())) {
    mlir::Value lhs = GetScalarValue(b, env, instr->operand(0));
    mlir::Value rhs = GetScalarValue(b, env, instr->operand(1));
    TF_RET_CHECK(lhs && rhs)
        << "Missing operand scalar values for " << instr->name();
    mlir::Type elem_type =
        emitters::PrimitiveTypeToMlirType(instr->shape().element_type(), b);
    mlir::Value res;
    if (mlir::isa<mlir::IntegerType>(elem_type)) {
      switch (instr->opcode()) {
        case HloOpcode::kAdd:
          res = mlir::arith::AddIOp::create(b, lhs, rhs);
          break;
        case HloOpcode::kSubtract:
          res = mlir::arith::SubIOp::create(b, lhs, rhs);
          break;
        case HloOpcode::kMultiply:
          res = mlir::arith::MulIOp::create(b, lhs, rhs);
          break;
        case HloOpcode::kDivide:
          res = mlir::arith::DivSIOp::create(b, lhs, rhs);
          break;
        default:
          break;
      }
    } else {
      switch (instr->opcode()) {
        case HloOpcode::kAdd:
          res = mlir::arith::AddFOp::create(b, lhs, rhs);
          break;
        case HloOpcode::kSubtract:
          res = mlir::arith::SubFOp::create(b, lhs, rhs);
          break;
        case HloOpcode::kMultiply:
          res = mlir::arith::MulFOp::create(b, lhs, rhs);
          break;
        case HloOpcode::kDivide:
          res = mlir::arith::DivFOp::create(b, lhs, rhs);
          break;
        default:
          break;
      }
    }
    env.SetValue(instr, res);
    return absl::OkStatus();
  }

  if (instr->opcode() == HloOpcode::kCompare &&
      ShapeUtil::IsEffectiveScalar(instr->shape())) {
    mlir::Value lhs = GetScalarValue(b, env, instr->operand(0));
    mlir::Value rhs = GetScalarValue(b, env, instr->operand(1));
    TF_RET_CHECK(lhs && rhs)
        << "Missing operand scalar values for " << instr->name();
    mlir::Value res;
    if (mlir::isa<mlir::IntegerType>(lhs.getType())) {
      mlir::arith::CmpIPredicate pred;
      switch (instr->comparison_direction()) {
        case ComparisonDirection::kEq:
          pred = mlir::arith::CmpIPredicate::eq;
          break;
        case ComparisonDirection::kNe:
          pred = mlir::arith::CmpIPredicate::ne;
          break;
        case ComparisonDirection::kGe:
          pred = mlir::arith::CmpIPredicate::sge;
          break;
        case ComparisonDirection::kGt:
          pred = mlir::arith::CmpIPredicate::sgt;
          break;
        case ComparisonDirection::kLe:
          pred = mlir::arith::CmpIPredicate::sle;
          break;
        case ComparisonDirection::kLt:
          pred = mlir::arith::CmpIPredicate::slt;
          break;
      }
      res = mlir::arith::CmpIOp::create(b, pred, lhs, rhs);
    } else {
      mlir::arith::CmpFPredicate pred;
      switch (instr->comparison_direction()) {
        case ComparisonDirection::kEq:
          pred = mlir::arith::CmpFPredicate::OEQ;
          break;
        case ComparisonDirection::kNe:
          pred = mlir::arith::CmpFPredicate::ONE;
          break;
        case ComparisonDirection::kGe:
          pred = mlir::arith::CmpFPredicate::OGE;
          break;
        case ComparisonDirection::kGt:
          pred = mlir::arith::CmpFPredicate::OGT;
          break;
        case ComparisonDirection::kLe:
          pred = mlir::arith::CmpFPredicate::OLE;
          break;
        case ComparisonDirection::kLt:
          pred = mlir::arith::CmpFPredicate::OLT;
          break;
      }
      res = mlir::arith::CmpFOp::create(b, pred, lhs, rhs);
    }
    env.SetValue(instr, res);
    return absl::OkStatus();
  }

  if (instr->opcode() == HloOpcode::kConvert &&
      ShapeUtil::IsEffectiveScalar(instr->shape())) {
    mlir::Value operand = GetScalarValue(b, env, instr->operand(0));
    TF_RET_CHECK(operand) << "Missing operand scalar value for "
                          << instr->name();
    mlir::Type target_type =
        emitters::PrimitiveTypeToMlirType(instr->shape().element_type(), b);
    mlir::Type src_type = operand.getType();
    bool is_unsigned_src = primitive_util::IsUnsignedIntegralType(
        instr->operand(0)->shape().element_type());
    bool is_unsigned_dst =
        primitive_util::IsUnsignedIntegralType(instr->shape().element_type());
    mlir::Value res;
    if (src_type == target_type) {
      res = operand;
    } else if (mlir::isa<mlir::IntegerType>(src_type) &&
               mlir::isa<mlir::IntegerType>(target_type)) {
      if (target_type.getIntOrFloatBitWidth() <
          src_type.getIntOrFloatBitWidth()) {
        res = mlir::arith::TruncIOp::create(b, target_type, operand);
      } else if (is_unsigned_src) {
        res = mlir::arith::ExtUIOp::create(b, target_type, operand);
      } else {
        res = mlir::arith::ExtSIOp::create(b, target_type, operand);
      }
    } else if (mlir::isa<mlir::IntegerType>(src_type) &&
               mlir::isa<mlir::FloatType>(target_type)) {
      if (is_unsigned_src) {
        res = mlir::arith::UIToFPOp::create(b, target_type, operand);
      } else {
        res = mlir::arith::SIToFPOp::create(b, target_type, operand);
      }
    } else if (mlir::isa<mlir::FloatType>(src_type) &&
               mlir::isa<mlir::IntegerType>(target_type)) {
      if (is_unsigned_dst) {
        res = mlir::arith::FPToUIOp::create(b, target_type, operand);
      } else {
        res = mlir::arith::FPToSIOp::create(b, target_type, operand);
      }
    } else if (mlir::isa<mlir::FloatType>(src_type) &&
               mlir::isa<mlir::FloatType>(target_type)) {
      if (target_type.getIntOrFloatBitWidth() <
          src_type.getIntOrFloatBitWidth()) {
        res = mlir::arith::TruncFOp::create(b, target_type, operand);
      } else {
        res = mlir::arith::ExtFOp::create(b, target_type, operand);
      }
    } else {
      return absl::UnimplementedError(
          absl::StrCat("Unsupported convert type ", instr->ToString()));
    }
    env.SetValue(instr, res);
    return absl::OkStatus();
  }

  mlir::func::FuncOp func = nullptr;
  for (mlir::Operation* op = b.getBlock()->getParentOp(); op != nullptr;
       op = op->getParentOp()) {
    if ((func = mlir::dyn_cast<mlir::func::FuncOp>(op))) break;
  }

  if (instr->opcode() == HloOpcode::kFusion) {
    const auto* fusion = Cast<HloFusionInstruction>(instr);
    TiledEmissionResult result = EmitTiledFusionKernel(
        mlir_context_, *fusion, buffer_assignment_, instr->name(),
        /*num_work_groups=*/1);
    if (result.kernel.ok()) {
      mlir::Type elem_type =
          emitters::PrimitiveTypeToMlirType(instr->shape().element_type(), b);
      int64_t num_elements = ShapeUtil::ElementsIn(instr->shape());
      auto memref_type = mlir::MemRefType::get({num_elements}, elem_type);
      constexpr int64_t kMaxStackAllocaBytes = 64 * 1024;
      int64_t alloc_bytes = ShapeUtil::ByteSizeOf(instr->shape());
      mlir::Value alloca;
      if (alloc_bytes <= kMaxStackAllocaBytes) {
        alloca = func ? CreateEntryBlockAlloca(b, func, memref_type)
                      : mlir::memref::AllocaOp::create(b, memref_type);
      } else {
        alloca = mlir::memref::AllocOp::create(b, memref_type);
      }
      env.SetValue(instr, alloca);
      return absl::OkStatus();
    }
  }

  // Intermediate non-fused instructions: materialize via entry block alloca
  mlir::Type elem_type =
      emitters::PrimitiveTypeToMlirType(instr->shape().element_type(), b);
  int64_t num_elements = ShapeUtil::ElementsIn(instr->shape());
  auto memref_type = mlir::MemRefType::get({num_elements}, elem_type);
  constexpr int64_t kMaxStackAllocaBytes = 64 * 1024;
  int64_t alloc_bytes = ShapeUtil::ByteSizeOf(instr->shape());
  mlir::Value alloca;
  if (alloc_bytes <= kMaxStackAllocaBytes) {
    alloca = func ? CreateEntryBlockAlloca(b, func, memref_type)
                  : mlir::memref::AllocaOp::create(b, memref_type);
  } else {
    alloca = mlir::memref::AllocOp::create(b, memref_type);
  }
  env.SetValue(instr, alloca);

  return absl::OkStatus();
}

absl::Status TiledComputationKernelEmitter::EmitWhile(
    mlir::ImplicitLocOpBuilder& b, const HloInstruction* while_instr,
    ComputationEnvironment& env) {
  const HloInstruction* init_val = while_instr->operand(0);
  llvm::SmallVector<mlir::Value> iter_args = env.GetAllValues(init_val);

  llvm::SmallVector<mlir::Type> result_types;
  for (mlir::Value v : iter_args) {
    result_types.push_back(v.getType());
  }

  auto while_op = mlir::scf::WhileOp::create(b, result_types, iter_args);

  // Before region (condition)
  b.createBlock(
      &while_op.getBefore(), while_op.getBefore().begin(), result_types,
      llvm::SmallVector<mlir::Location>(result_types.size(), b.getLoc()));
  {
    mlir::OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&while_op.getBefore().front());

    ComputationEnvironment cond_env = env;
    const HloComputation* cond_comp = while_instr->while_condition();
    const HloInstruction* cond_param = cond_comp->parameter_instruction(0);

    if (cond_param->shape().IsTuple()) {
      for (size_t i = 0; i < while_op.getBefore().getArguments().size(); ++i) {
        cond_env.SetTupleElement(cond_param, i,
                                 while_op.getBefore().getArgument(i));
      }
    } else if (!while_op.getBefore().getArguments().empty()) {
      cond_env.SetValue(cond_param, while_op.getBefore().getArgument(0));
    }

    RETURN_IF_ERROR(EmitComputation(b, cond_comp, cond_env));

    const HloInstruction* cond_root = cond_comp->root_instruction();
    mlir::Value cond_val = GetScalarValue(b, cond_env, cond_root);
    if (cond_val && cond_val.getType().isInteger(8)) {
      mlir::Value zero = mlir::arith::ConstantIntOp::create(b, 0, 8);
      cond_val = mlir::arith::CmpIOp::create(b, mlir::arith::CmpIPredicate::ne,
                                             cond_val, zero);
    }
    TF_RET_CHECK(cond_val && cond_val.getType().isInteger(1))
        << "Condition instruction " << cond_root->name()
        << " did not produce a valid i1 predicate SSA value.";

    mlir::scf::ConditionOp::create(b, cond_val,
                                   while_op.getBefore().getArguments());
  }

  // After region (body)
  b.createBlock(
      &while_op.getAfter(), while_op.getAfter().begin(), result_types,
      llvm::SmallVector<mlir::Location>(result_types.size(), b.getLoc()));
  {
    mlir::OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&while_op.getAfter().front());

    ComputationEnvironment body_env = env;
    const HloComputation* body_comp = while_instr->while_body();
    const HloInstruction* body_param = body_comp->parameter_instruction(0);

    if (body_param->shape().IsTuple()) {
      for (size_t i = 0; i < while_op.getAfter().getArguments().size(); ++i) {
        body_env.SetTupleElement(body_param, i,
                                 while_op.getAfter().getArgument(i));
      }
    } else if (!while_op.getAfter().getArguments().empty()) {
      body_env.SetValue(body_param, while_op.getAfter().getArgument(0));
    }

    RETURN_IF_ERROR(EmitComputation(b, body_comp, body_env));

    const HloInstruction* body_root = body_comp->root_instruction();
    llvm::SmallVector<mlir::Value> yield_args =
        body_env.GetAllValues(body_root);
    if (yield_args.empty()) {
      yield_args.assign(while_op.getAfter().getArguments().begin(),
                        while_op.getAfter().getArguments().end());
    }

    for (size_t i = 0; i < yield_args.size(); ++i) {
      if (i < while_op.getAfter().getNumArguments()) {
        mlir::Type expected_type = while_op.getAfter().getArgument(i).getType();
        if (yield_args[i].getType() != expected_type) {
          if (mlir::isa<mlir::RankedTensorType>(expected_type) &&
              mlir::isa<mlir::RankedTensorType>(yield_args[i].getType())) {
            yield_args[i] =
                mlir::tensor::CastOp::create(b, expected_type, yield_args[i]);
          }
        }
      }
    }

    mlir::scf::YieldOp::create(b, yield_args);
  }

  b.setInsertionPointAfter(while_op);

  if (while_instr->shape().IsTuple()) {
    for (size_t i = 0; i < while_op.getNumResults(); ++i) {
      env.SetTupleElement(while_instr, i, while_op.getResult(i));
    }
  } else if (while_op.getNumResults() > 0) {
    env.SetValue(while_instr, while_op.getResult(0));
  }

  return absl::OkStatus();
}

absl::Status TiledComputationKernelEmitter::EmitConditional(
    mlir::ImplicitLocOpBuilder& b, const HloInstruction* cond_instr,
    ComputationEnvironment& env) {
  const HloInstruction* pred_instr = cond_instr->operand(0);
  mlir::Value pred_val = GetScalarValue(b, env, pred_instr);
  if (pred_val && pred_val.getType().isInteger(8)) {
    mlir::Value zero = mlir::arith::ConstantIntOp::create(b, 0, 8);
    pred_val = mlir::arith::CmpIOp::create(b, mlir::arith::CmpIPredicate::ne,
                                           pred_val, zero);
  }
  TF_RET_CHECK(pred_val && pred_val.getType().isInteger(1))
      << "Predicate instruction " << pred_instr->name()
      << " did not produce a valid i1 predicate SSA value.";

  llvm::SmallVector<mlir::Type> result_types;
  const HloComputation* branch0 = cond_instr->branch_computation(0);
  const HloInstruction* branch0_root = branch0->root_instruction();
  if (branch0_root->shape().IsTuple()) {
    for (int i = 0; i < ShapeUtil::TupleElementCount(branch0_root->shape());
         ++i) {
      const Shape& subshape =
          ShapeUtil::GetSubshape(branch0_root->shape(), {i});
      mlir::Type elem_type =
          emitters::PrimitiveTypeToMlirType(subshape.element_type(), b);
      if (ShapeUtil::IsScalar(subshape)) {
        result_types.push_back(elem_type);
      } else {
        result_types.push_back(
            mlir::MemRefType::get(subshape.dimensions(), elem_type));
      }
    }
  } else if (ShapeUtil::IsScalar(branch0_root->shape())) {
    mlir::Type elem_type = emitters::PrimitiveTypeToMlirType(
        branch0_root->shape().element_type(), b);
    result_types.push_back(elem_type);
  } else {
    mlir::Type elem_type = emitters::PrimitiveTypeToMlirType(
        branch0_root->shape().element_type(), b);
    result_types.push_back(
        mlir::MemRefType::get(branch0_root->shape().dimensions(), elem_type));
  }

  auto if_op = mlir::scf::IfOp::create(b, result_types, pred_val,
                                       /*hasElseRegion=*/true);

  // Then block (branch 0)
  b.setInsertionPointToStart(if_op.thenBlock());
  {
    ComputationEnvironment branch_env = env;
    const HloInstruction* branch_param = branch0->parameter_instruction(0);
    llvm::SmallVector<mlir::Value> branch_arg_vals =
        env.GetAllValues(cond_instr->operand(1));
    if (branch_param->shape().IsTuple()) {
      for (size_t i = 0; i < branch_arg_vals.size(); ++i) {
        branch_env.SetTupleElement(branch_param, i, branch_arg_vals[i]);
      }
    } else if (!branch_arg_vals.empty()) {
      branch_env.SetValue(branch_param, branch_arg_vals.front());
    }

    RETURN_IF_ERROR(EmitComputation(b, branch0, branch_env));

    llvm::SmallVector<mlir::Value> yield_vals =
        branch_env.GetAllValues(branch0_root);
    mlir::scf::YieldOp::create(b, yield_vals);
  }

  // Else block (branch 1)
  b.setInsertionPointToStart(if_op.elseBlock());
  {
    ComputationEnvironment branch_env = env;
    const HloComputation* branch1 = cond_instr->branch_computation(1);
    const HloInstruction* branch1_root = branch1->root_instruction();
    const HloInstruction* branch_param = branch1->parameter_instruction(0);
    llvm::SmallVector<mlir::Value> branch_arg_vals =
        env.GetAllValues(cond_instr->operand(2));
    if (branch_param->shape().IsTuple()) {
      for (size_t i = 0; i < branch_arg_vals.size(); ++i) {
        branch_env.SetTupleElement(branch_param, i, branch_arg_vals[i]);
      }
    } else if (!branch_arg_vals.empty()) {
      branch_env.SetValue(branch_param, branch_arg_vals.front());
    }

    RETURN_IF_ERROR(EmitComputation(b, branch1, branch_env));

    llvm::SmallVector<mlir::Value> yield_vals =
        branch_env.GetAllValues(branch1_root);
    mlir::scf::YieldOp::create(b, yield_vals);
  }

  b.setInsertionPointAfter(if_op);
  if (cond_instr->shape().IsTuple()) {
    for (size_t i = 0; i < if_op.getNumResults(); ++i) {
      env.SetTupleElement(cond_instr, i, if_op.getResult(i));
    }
  } else if (if_op.getNumResults() > 0) {
    env.SetValue(cond_instr, if_op.getResult(0));
  }

  return absl::OkStatus();
}

}  // namespace xla::cpu
