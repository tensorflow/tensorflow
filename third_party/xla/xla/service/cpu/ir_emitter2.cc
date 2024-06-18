/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/cpu/ir_emitter2.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "xla/cpu_function_runtime.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/dot_op_emitter.h"
#include "xla/service/cpu/elemental_math_emitter.h"
#include "xla/service/cpu/ir_emitter.h"
#include "xla/service/cpu/parallel_loop_emitter.h"
#include "xla/service/cpu/shape_partition.h"
#include "xla/service/elemental_ir_emitter.h"
#include "xla/service/llvm_ir/fused_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla::cpu {
namespace {

// We do not materialize buffers for tuples at run time, and work only with leaf
// arrays. These are the helper functions to flatten HLO instruction parameters
// and results into a list of leaf shapes.

static std::vector<Shape> FlattenedParameters(const HloInstruction* instr) {
  std::vector<Shape> parameters;
  for (auto* operand : instr->operands()) {
    for (auto& indexed : ShapeUtil::GetLeafShapes(operand->shape())) {
      parameters.push_back(indexed.shape);
    }
  }
  return parameters;
}

static std::vector<Shape> FlattenedResults(const HloInstruction* instr) {
  std::vector<Shape> results;
  for (auto& indexed : ShapeUtil::GetLeafShapes(instr->shape())) {
    results.push_back(indexed.shape);
  }
  return results;
}

// Following struct types correspond to HostKernel C API.
// See: xla/stream_executor/host/host_kernel_c_api.h

static llvm::StructType* Dim3StructTy(llvm::LLVMContext& ctx,
                                      std::string_view name) {
  auto* i64 = llvm::IntegerType::getInt64Ty(ctx);
  return llvm::StructType::create(name, i64, i64, i64);
}

static llvm::StructType* KernelThreadDimTy(llvm::LLVMContext& ctx) {
  return Dim3StructTy(ctx, "SE_HOST_KernelThreadDim");
}

static llvm::StructType* KernelThreadTy(llvm::LLVMContext& ctx) {
  return Dim3StructTy(ctx, "SE_HOST_KernelThread");
}

static llvm::StructType* KernelArgTy(llvm::LLVMContext& ctx) {
  auto* ptr = llvm::PointerType::getUnqual(ctx);
  auto* i64 = llvm::IntegerType::getInt64Ty(ctx);
  return llvm::StructType::create("SE_HOST_KernelArg", ptr, i64);
}

static llvm::StructType* KernelCallFrameTy(llvm::LLVMContext& ctx) {
  auto* ptr = llvm::PointerType::getUnqual(ctx);
  auto* i64 = llvm::IntegerType::getInt64Ty(ctx);
  return llvm::StructType::create("SE_HOST_KernelCallFrame", ptr, ptr, i64,
                                  ptr);
}

static llvm::FunctionType* KernelFunctionTy(llvm::LLVMContext& ctx) {
  return llvm::FunctionType::get(llvm::PointerType::getUnqual(ctx),
                                 llvm::PointerType::getUnqual(ctx),
                                 /*isVarArg=*/false);
}

}  // namespace

//===----------------------------------------------------------------------===//
// ElementalIrEmitter
//===----------------------------------------------------------------------===//

class IrEmitter2::ElementalIrEmitter : public xla::ElementalIrEmitter {
 public:
  ElementalIrEmitter(llvm::Module* module, llvm::IRBuilder<>* b,
                     const HloModule* hlo_module, IrEmitter* nested_ir_emitter,
                     bool fast_min_max)
      : xla::ElementalIrEmitter(module, b),
        hlo_module_(hlo_module),
        nested_ir_emitter_(nested_ir_emitter),
        fast_min_max_(fast_min_max) {}

 protected:
  absl::StatusOr<llvm::Value*> EmitAtan2(PrimitiveType prim_type,
                                         llvm::Value* lhs, llvm::Value* rhs,
                                         absl::string_view) override {
    return xla::cpu::EmitAtan2(module(), *b(), prim_type, lhs, rhs);
  }

  absl::StatusOr<llvm::Value*> EmitTanh(PrimitiveType prim_type,
                                        llvm::Value* value) override {
    return xla::cpu::EmitTanh(module(), *b(), prim_type, value);
  }

  absl::StatusOr<llvm::Value*> EmitErf(PrimitiveType prim_type,
                                       llvm::Value* value) override {
    return xla::cpu::EmitErf(module(), *b(), prim_type, value);
  }

  absl::StatusOr<std::vector<llvm::Value*>> EmitThreadLocalCall(
      const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
      absl::string_view name, bool is_reducer) override {
    // Module must be scheduled to emit thread local computation.
    if (!hlo_module_ || !hlo_module_->has_schedule()) {
      return absl::InternalError(
          "HLO module must be scheduled to emit thread local computation.");
    }

    // Create a nested function for thread local computation(s) if it is not
    // already created. Nested functions are created with internal linkage.
    auto emit_computation = [&](const HloComputation* computation) {
      if (!nested_ir_emitter_->is_computation_emitted(*computation,
                                                      is_reducer)) {
        VLOG(2) << "Emit nested computation: " << computation->name();
        TF_RETURN_IF_ERROR(
            nested_ir_emitter_
                ->EmitComputation(
                    const_cast<HloComputation*>(computation), name, false,
                    hlo_module_->schedule()
                        .sequence(computation)
                        .instructions(),
                    /*allow_reassociation=*/is_reducer,
                    /*function_attributes=*/{llvm::Attribute::AlwaysInline})
                .status());
      }
      return absl::OkStatus();
    };

    // We emit all embedded computations reachable through the `callee` to
    // support nested thread local call, i.e., nested map computations.
    for (HloComputation* embedded : callee.MakeEmbeddedComputationsList()) {
      if (embedded->IsFusionComputation()) continue;
      TF_RETURN_IF_ERROR(emit_computation(embedded));
    }
    TF_RETURN_IF_ERROR(emit_computation(&callee));

    // Add a thread local call to the nested computation.
    VLOG(2) << "Emit thread local call to: " << callee.name();
    nested_ir_emitter_->b()->SetInsertPoint(b()->GetInsertPoint());
    auto values = nested_ir_emitter_->EmitThreadLocalCall(
        callee, parameters, name, is_reducer, /*in_compute_function=*/false);

    return values;
  }

  bool fast_min_max() override { return fast_min_max_; }

 private:
  const HloModule* hlo_module_;
  IrEmitter* nested_ir_emitter_;
  bool fast_min_max_;
};

//===----------------------------------------------------------------------===//
// IrEmitter2
//===----------------------------------------------------------------------===//

IrEmitter2::IrEmitter2(const HloModule& hlo_module, llvm::Module* module,
                       IrEmitter* nested_ir_emitter)
    : hlo_module_(hlo_module),
      module_(module),
      nested_ir_emitter_(nested_ir_emitter),
      call_frame_ty_(KernelCallFrameTy(module_->getContext())),
      thread_dims_ty_(KernelThreadDimTy(module_->getContext())),
      thread_ty_(KernelThreadTy(module_->getContext())),
      arg_ty_(KernelArgTy(module_->getContext())) {}

bool IrEmitter2::fast_min_max() const {
  return hlo_module_.config().debug_options().xla_cpu_enable_fast_min_max();
}

absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitElementalHostKernel(
    const HloInstruction* instr) {
  VLOG(2) << "Emit elemental host kernel: " << instr->name();

  KernelPrototype kernel_prototype = EmitKernelPrototype(instr);

  llvm::IRBuilder<> b(module_->getContext());
  b.SetInsertPoint(kernel_prototype.function->getEntryBlock().getTerminator());

  ElementalIrEmitter::HloToElementGeneratorMap operand_to_generator;
  for (int64_t i = 0; i < instr->operand_count(); ++i) {
    const HloInstruction* operand = instr->operand(i);
    operand_to_generator[operand] = [&, i](const llvm_ir::IrArray::Index& idx) {
      return kernel_prototype.arguments[i].EmitReadArrayElement(idx, &b);
    };
  }

  ElementalIrEmitter elemental_emitter(module_, &b, &hlo_module_,
                                       nested_ir_emitter_, fast_min_max());
  llvm_ir::ElementGenerator element_generator =
      elemental_emitter.MakeElementGenerator(instr, operand_to_generator);

  TF_ASSIGN_OR_RETURN(
      se::ThreadDim thread_dims,
      EmitElementalLoops(b, instr, kernel_prototype, element_generator));

  return kernels_.emplace_back(KernelInfo{
      kernel_prototype.function->getName().str(), se::BlockDim(), thread_dims});
}

absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitFusionHostKernel(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emit fusion host kernel: " << fusion->name();

  // In XLA:CPU output fusion can only be a fusion into dot operation.
  if (fusion->fusion_kind() == HloInstruction::FusionKind::kOutput) {
    return EmitDotFusionHostKernel(fusion);
  }

  if (fusion->fusion_kind() != HloInstruction::FusionKind::kLoop) {
    return Internal("Unsupported loop fusion kind for instruction: %s",
                    fusion->ToString());
  }

  KernelPrototype kernel_prototype = EmitKernelPrototype(fusion);

  llvm::IRBuilder<> b(module_->getContext());
  b.SetInsertPoint(kernel_prototype.function->getEntryBlock().getTerminator());

  ElementalIrEmitter elemental_emitter(module_, &b, &hlo_module_,
                                       nested_ir_emitter_, fast_min_max());
  FusedIrEmitter fused_emitter(elemental_emitter);

  for (int i = 0; i < fusion->operand_count(); i++) {
    fused_emitter.BindGenerator(
        *fusion->fused_parameter(i), [&, i](llvm_ir::IrArray::Index idx) {
          return kernel_prototype.arguments[i].EmitReadArrayElement(idx, &b);
        });
  }

  TF_ASSIGN_OR_RETURN(
      auto element_generator,
      fused_emitter.GetGenerator(*fusion->fused_expression_root()));

  TF_ASSIGN_OR_RETURN(
      se::ThreadDim thread_dims,
      EmitElementalLoops(b, fusion, kernel_prototype, element_generator));

  return kernels_.emplace_back(KernelInfo{
      kernel_prototype.function->getName().str(), se::BlockDim(), thread_dims});
}

absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitReductionHostKernel(
    const HloInstruction* instr) {
  VLOG(2) << "Emit reduction host kernel: " << instr->name();

  // TODO(ezhulenev): Port vectorized reduction emitter from IrEmitter.
  return EmitElementalHostKernel(instr);
}

// Dot (fusion) host kernel only supports strategies that emit LLVM IR.
static bool IsDotCodegenStrategy(DotImplementationStrategy strategy) {
  static std::array<DotImplementationStrategy, 3> kDotCodegenStrategies = {
      DotImplementationStrategy::kNaiveLlvmIr,
      DotImplementationStrategy::kTiledLlvmIrGemm,
      DotImplementationStrategy::kTiledLlvmIrGemv,
  };

  return absl::c_find(kDotCodegenStrategies, strategy) !=
         kDotCodegenStrategies.end();
}

absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitDotHostKernel(
    const HloInstruction* instr) {
  VLOG(2) << "Emit dot host kernel: " << instr->name();

  DotImplementationStrategy strategy = GetDotImplementationStrategy(
      hlo_module_.config(), *instr,
      nested_ir_emitter_->target_machine_features());

  if (!IsDotCodegenStrategy(strategy)) {
    return Internal("Unsupported dot implementation strategy");
  }

  KernelPrototype kernel_prototype = EmitKernelPrototype(instr);

  llvm::IRBuilder<> b(module_->getContext());
  b.SetInsertPoint(kernel_prototype.function->getEntryBlock().getTerminator());

  llvm_ir::IrArray lhs_array = kernel_prototype.arguments[0];
  llvm_ir::IrArray rhs_array = kernel_prototype.arguments[1];
  llvm_ir::IrArray target_array = kernel_prototype.results[0];

  TF_RETURN_IF_ERROR(EmitDotOperation(
      *instr, target_array, lhs_array, rhs_array,
      /*addend_array=*/nullptr, /*executable_run_options_value=*/nullptr, &b,
      hlo_module_.config(), nested_ir_emitter_->target_machine_features(),
      /*allow_runtime_calls=*/false));

  return kernels_.emplace_back(
      KernelInfo{kernel_prototype.function->getName().str(), se::BlockDim(),
                 se::ThreadDim()});
}

absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitDotFusionHostKernel(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emit dot fusion host kernel: " << fusion->name();

  // Dot fusion only supports adding a side input to the dot product.
  const HloInstruction* add = fusion->fused_expression_root();
  if (add->opcode() != HloOpcode::kAdd) {
    return Internal("Dot fusion supports only `add` root instruction");
  }

  // Check that fusion root has a single dot operand.
  bool is_dot_operand0 = add->operand(0)->opcode() == HloOpcode::kDot;
  bool is_dot_operand1 = add->operand(1)->opcode() == HloOpcode::kDot;
  if (is_dot_operand0 == is_dot_operand1) {
    return Internal("Dot fusion root instruction must have single dot operand");
  }

  int64_t dot_op_index = is_dot_operand0 ? 0 : 1;
  int64_t addend_op_index = 1 - dot_op_index;

  const HloInstruction* dot = add->operand(dot_op_index);

  // Check that we can emit LLVM IR for this dot operation.
  DotImplementationStrategy strategy = GetDotImplementationStrategy(
      hlo_module_.config(), *dot,
      nested_ir_emitter_->target_machine_features());

  if (!IsDotCodegenStrategy(strategy)) {
    return Internal("Unsupported dot implementation strategy");
  }

  // Indices of fusion parameters that are used as dot operands and result.
  int64_t dot_lhs_pnum = dot->operand(0)->parameter_number();
  int64_t dot_rhs_pnum = dot->operand(1)->parameter_number();
  int64_t addend_pnum = add->operand(addend_op_index)->parameter_number();

  KernelPrototype kernel_prototype = EmitKernelPrototype(fusion);

  llvm::IRBuilder<> b(module_->getContext());
  b.SetInsertPoint(kernel_prototype.function->getEntryBlock().getTerminator());

  llvm_ir::IrArray lhs_array = kernel_prototype.arguments[dot_lhs_pnum];
  llvm_ir::IrArray rhs_array = kernel_prototype.arguments[dot_rhs_pnum];
  llvm_ir::IrArray addend_array = kernel_prototype.arguments[addend_pnum];
  llvm_ir::IrArray target_array = kernel_prototype.results[0];

  TF_RETURN_IF_ERROR(EmitDotOperation(
      *dot, target_array, lhs_array, rhs_array, &addend_array,
      /*executable_run_options_value=*/nullptr, &b, hlo_module_.config(),
      nested_ir_emitter_->target_machine_features(),
      /*allow_runtime_calls=*/false));

  return kernels_.emplace_back(
      KernelInfo{kernel_prototype.function->getName().str(), se::BlockDim(),
                 se::ThreadDim()});
}

//===----------------------------------------------------------------------===//
// Building HostKernel prototypes.
//===----------------------------------------------------------------------===//

IrEmitter2::KernelThreadDims IrEmitter2::EmitKernelThreadDims(
    llvm::IRBuilder<>& b, llvm::Value* call_frame) {
  auto* td_gep = b.CreateStructGEP(call_frame_ty_, call_frame, 0, "tdims_gep");
  auto* tdims = b.CreateLoad(b.getPtrTy(), td_gep, "tdims");
  auto* x_gep = b.CreateStructGEP(thread_dims_ty_, tdims, 0, "tdim_x_gep");
  auto* y_gep = b.CreateStructGEP(thread_dims_ty_, tdims, 1, "tdim_y_gep");
  auto* z_gep = b.CreateStructGEP(thread_dims_ty_, tdims, 2, "tdim_z_gep");

  return {b.CreateLoad(b.getInt64Ty(), x_gep, "tdim_x"),
          b.CreateLoad(b.getInt64Ty(), y_gep, "tdim_y"),
          b.CreateLoad(b.getInt64Ty(), z_gep, "tdim_z")};
}

IrEmitter2::KernelThread IrEmitter2::EmitKernelThread(llvm::IRBuilder<>& b,
                                                      llvm::Value* call_frame) {
  auto* t_gep = b.CreateStructGEP(call_frame_ty_, call_frame, 1, "tid_gep");
  auto* tids = b.CreateLoad(b.getPtrTy(), t_gep, "tids");
  auto* x_gep = b.CreateStructGEP(thread_ty_, tids, 0, "tid_x_gep");
  auto* y_gep = b.CreateStructGEP(thread_ty_, tids, 1, "tid_y_gep");
  auto* z_gep = b.CreateStructGEP(thread_ty_, tids, 2, "tid_z_gep");

  return {b.CreateLoad(b.getInt64Ty(), x_gep, "tid_x"),
          b.CreateLoad(b.getInt64Ty(), y_gep, "tid_y"),
          b.CreateLoad(b.getInt64Ty(), z_gep, "tid_z")};
}

llvm_ir::IrArray IrEmitter2::EmitKernelArgument(llvm::IRBuilder<>& b,
                                                llvm::Value* call_frame,
                                                int64_t index,
                                                const Shape& shape) {
  llvm::Type* ptr = llvm::PointerType::get(b.getContext(), 0);
  std::string name = absl::StrCat("arg", index);

  auto* args_gep = b.CreateStructGEP(call_frame_ty_, call_frame, 3, "args_gep");
  auto* args = b.CreateLoad(ptr, args_gep, "args");
  auto* data_gep = b.CreateConstGEP2_32(arg_ty_, args, index, 0, name + "_gep");
  auto* data = b.CreateLoad(ptr, data_gep, name);

  // All buffers passed to host kernels are expected to be properly aligned,
  // emit metadata to allow LLVM to use that information for optimization.
  llvm_ir::SetAlignmentMetadataForLoad(data, cpu_function_runtime::MinAlign());

  return llvm_ir::IrArray(data, llvm_ir::ShapeToIrType(shape, module_), shape);
}

IrEmitter2::KernelPrototype IrEmitter2::EmitKernelPrototype(
    std::string_view name, absl::Span<const Shape> arguments,
    absl::Span<const Shape> results) {
  VLOG(3) << "Emit kernel prototype: " << name
          << ", #arguments=" << arguments.size()
          << ", #results=" << results.size();
  for (const Shape& argument : arguments) {
    VLOG(3) << "  argument: " << argument.ToString(true);
  }
  for (const Shape& result : results) {
    VLOG(3) << "  result: " << result.ToString(true);
  }

  llvm::LLVMContext& ctx = module_->getContext();
  llvm::IRBuilder<> b(ctx);

  // Create a kernel function with HostKernel API.
  llvm::Function* function = llvm::dyn_cast<llvm::Function>(
      module_->getOrInsertFunction(name, KernelFunctionTy(ctx)).getCallee());
  function->setCallingConv(llvm::CallingConv::C);
  function->setDoesNotThrow();

  // Set prefer-vector-width attribute to allow LLVM to use wider vector
  // registers (by default LLVM uses at most 256-bit registers).
  const DebugOptions& debug_options = hlo_module_.config().debug_options();
  function->addFnAttr(
      "prefer-vector-width",
      absl::StrCat(debug_options.xla_cpu_prefer_vector_width()));

  // Always keep a frame pointer for the host kernel so we can see them in all
  // performance profiling tools.
  function->addFnAttr("frame-pointer", "all");

  // Create an entry basic block and set insert point to the end of it.
  b.SetInsertPoint(llvm::BasicBlock::Create(ctx, "", function));

  llvm::Value* call_frame = function->getArg(0);
  // Build thread coordinates from the call frame.
  KernelThreadDims kernel_thread_dims = EmitKernelThreadDims(b, call_frame);
  KernelThread kernel_thread = EmitKernelThread(b, call_frame);

  int64_t idx = 0;

  // IrArrays for the parameters.
  std::vector<llvm_ir::IrArray> ir_arguments;
  for (const Shape& argument : arguments) {
    ir_arguments.push_back(EmitKernelArgument(b, call_frame, idx++, argument));
  }

  // IrArrays for the results.
  std::vector<llvm_ir::IrArray> ir_results;
  for (const Shape& result : results) {
    ir_results.push_back(EmitKernelArgument(b, call_frame, idx++, result));
  }

  // Return null pointer to signal success as we do not support error handling
  // in the compiled host kernel.
  b.CreateRet(
      llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(ctx)));

  return KernelPrototype{function, kernel_thread_dims, kernel_thread,
                         std::move(ir_arguments), std::move(ir_results)};
}

IrEmitter2::KernelPrototype IrEmitter2::EmitKernelPrototype(
    const HloInstruction* instr) {
  return EmitKernelPrototype(instr->name(), FlattenedParameters(instr),
                             FlattenedResults(instr));
}

std::optional<IrEmitter2::ParallelConfig> IrEmitter2::GetParallelConfig(
    const HloInstruction* instr) {
  // Check if the instruction is marked for parallel execution.
  auto backend_config = instr->backend_config<BackendConfig>();
  if (!backend_config.ok() ||
      backend_config->outer_dimension_partitions().empty()) {
    return std::nullopt;
  }

  ParallelConfig config;
  config.outer_dimension_partitions.assign(
      backend_config->outer_dimension_partitions().begin(),
      backend_config->outer_dimension_partitions().end());

  return config;
}

IrEmitter2::ParallelPartitionBounds IrEmitter2::EmitParallelPartitionBounds(
    llvm::IRBuilder<>& b, const KernelPrototype& kernel_prototype,
    const ParallelConfig& parallel_config, const Shape& shape,
    std::string_view name) {
  ShapePartitionIterator it(shape, parallel_config.outer_dimension_partitions);

  size_t num_parallel_dimensions =
      parallel_config.outer_dimension_partitions.size();

  // Create a constant array of all partition bounds. We will be indexing into
  // this array using block and thread dimension indices passed in a call frame.
  //
  // Type: [#partitions x [#outer_dimensions x [lower_bound, upper_bound]]]
  //
  llvm::ArrayType* dim_bounds_ty = llvm::ArrayType::get(b.getInt64Ty(), 2);
  llvm::ArrayType* partition_bounds_ty =
      llvm::ArrayType::get(dim_bounds_ty, num_parallel_dimensions);
  llvm::ArrayType* parallel_bounds_ty =
      llvm::ArrayType::get(partition_bounds_ty, it.GetTotalPartitionCount());

  // Build a nested array of partition bounds from shape partition iterator.
  std::vector<llvm::Constant*> partition_bounds;
  for (int64_t i = 0; i < it.GetTotalPartitionCount(); ++i) {
    std::vector<llvm::Constant*> dim_counts;
    for (auto [lower, size] : it.GetPartition(i)) {
      dim_counts.push_back(llvm::ConstantArray::get(
          dim_bounds_ty, {b.getInt64(lower), b.getInt64(lower + size)}));
    }
    partition_bounds.push_back(
        llvm::ConstantArray::get(partition_bounds_ty, dim_counts));
  }

  llvm::Constant* parallel_bounds =
      llvm::ConstantArray::get(parallel_bounds_ty, partition_bounds);

  llvm::Module* module = b.GetInsertBlock()->getParent()->getParent();
  llvm::GlobalVariable* parallel_bounds_global = new llvm::GlobalVariable(
      /*M=*/*module,
      /*Ty=*/parallel_bounds_ty,
      /*isConstant=*/true,
      /*Linkage=*/llvm::GlobalValue::PrivateLinkage,
      /*Initializer=*/parallel_bounds,
      /*Name=*/absl::StrCat(name, "_parallel_bounds"));

  // Construct IR to load bounds for all parallel dimensions.
  ParallelPartitionBounds bounds;
  for (size_t i = 0; i < num_parallel_dimensions; ++i) {
    llvm::Value* partition = kernel_prototype.thread.x;
    llvm::Value* parallel_dim = b.getInt32(i);

    llvm::Value* lower_gep = b.CreateInBoundsGEP(
        parallel_bounds_ty, parallel_bounds_global,
        {b.getInt32(0), partition, parallel_dim, b.getInt32(0)},
        absl::StrCat("lo_dim_", i, "_gep"));

    llvm::Value* upper_gep = b.CreateInBoundsGEP(
        parallel_bounds_ty, parallel_bounds_global,
        {b.getInt32(0), partition, parallel_dim, b.getInt32(1)},
        absl::StrCat("up_dim_", i, "_gep"));

    bounds.emplace_back(
        b.CreateLoad(b.getInt64Ty(), lower_gep, absl::StrCat("lo_dim_", i)),
        b.CreateLoad(b.getInt64Ty(), upper_gep, absl::StrCat("up_dim_", i)));
  }

  return bounds;
}

absl::StatusOr<se::ThreadDim> IrEmitter2::EmitElementalLoops(
    llvm::IRBuilder<>& b, const HloInstruction* instr,
    const KernelPrototype& kernel_prototype,
    const llvm_ir::ElementGenerator& element_generator) {
  // We can emit loops for instruction with multiple results only if it is a
  // fusion, reduce or reduce window.
  bool multiple_results = kernel_prototype.results.size() > 1;
  bool support_multiple_results = instr->opcode() == HloOpcode::kFusion ||
                                  instr->opcode() == HloOpcode::kReduce ||
                                  instr->opcode() == HloOpcode::kReduceWindow;

  auto parallel_config = GetParallelConfig(instr);
  bool has_parallel_config = parallel_config.has_value();

  if (multiple_results && !support_multiple_results) {
    return Internal(
        "Multi-output host kernels are not supported for %s instruction",
        HloOpcodeString(instr->opcode()));
  }

  // TODO(ezhulenev): Support multiple results for parallel loops.
  if (multiple_results) {
    TF_RETURN_IF_ERROR(
        llvm_ir::LoopEmitter(element_generator, kernel_prototype.results, &b)
            .EmitLoop(llvm_ir::IrName(instr)));
    return se::ThreadDim();
  }

  const llvm_ir::IrArray& result = kernel_prototype.results.front();

  // Emit a loop for a single parallel partition with dynamic bounds computed
  // from thread index.
  if (has_parallel_config) {
    ParallelPartitionBounds parallel_bounds = EmitParallelPartitionBounds(
        b, kernel_prototype, *parallel_config, instr->shape(), instr->name());
    TF_RETURN_IF_ERROR(
        ParallelLoopEmitter(element_generator, result, &parallel_bounds, &b)
            .EmitLoop(llvm_ir::IrName(instr)));
    return se::ThreadDim(ShapePartitionAssigner::GetTotalPartitionCount(
        parallel_config->outer_dimension_partitions));
  }

  // Emit a whole loop for the instruction.
  TF_RETURN_IF_ERROR(llvm_ir::LoopEmitter(element_generator, result, &b)
                         .EmitLoop(llvm_ir::IrName(instr)));
  return se::ThreadDim();
}

}  // namespace xla::cpu
