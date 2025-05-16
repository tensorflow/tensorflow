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

#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CodeGen.h"
#include "xla/backends/cpu/codegen/symbol_name_util.h"
#include "xla/cpu_function_runtime.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {

namespace {

class MemoryDependencyAnalyzer {
 public:
  MemoryDependencyAnalyzer(
      llvm::LLVMContext& context, absl::string_view name,
      absl::Span<const KernelApiIrBuilder::KernelParameter> results)
      : context_(context), mb_(context) {
    // Create an alias domain for the host kernel function.
    llvm::MDNode* domain = mb_.createAliasScopeDomain(
        absl::StrFormat("XLA host kernel %s AA domain", name));

    result_slices_.reserve(results.size());
    for (const KernelApiIrBuilder::KernelParameter& result : results) {
      result_slices_.insert(result.slice);

      // Skip result buffers that are aliased with entry parameters as we don't
      // know if they can alias with any other buffers.
      if (result.slice.allocation()->is_parameter_aliased_with_output()) {
        continue;
      }
      alias_scopes_[result.slice] = mb_.createAliasScope(
          absl::StrFormat("result slice: %s", result.slice.ToString()), domain);
    }
  }

  // Returns alias scope for the given buffer slice.
  llvm::MDNode* GetAliasScope(BufferAllocation::Slice slice) {
    auto it = alias_scopes_.find(slice);
    return it == alias_scopes_.end() ? nullptr
                                     : llvm::MDNode::get(context_, it->second);
  };

  // Construct !noalias metadata for buffer slice.
  llvm::MDNode* GetNoAlias(BufferAllocation::Slice slice) {
    llvm::SmallVector<llvm::Metadata*> scopes;
    for (const auto& [alias_slice, alias_scope] : alias_scopes_) {
      if (!slice.OverlapsWith(alias_slice)) {
        scopes.push_back(alias_scope);
      }
    }
    return scopes.empty() ? nullptr : llvm::MDNode::get(context_, scopes);
  };

  bool ResultsOverlapWithSlice(BufferAllocation::Slice slice) {
    for (const BufferAllocation::Slice& result_slice : result_slices_) {
      if (result_slice.OverlapsWith(slice)) {
        return true;
      }
    }
    return false;
  }

 private:
  llvm::LLVMContext& context_;
  llvm::MDBuilder mb_;

  absl::btree_map<BufferAllocation::Slice, llvm::MDNode*> alias_scopes_;
  absl::flat_hash_set<BufferAllocation::Slice> result_slices_;
};

// Following struct types correspond to XLA:CPU Kernel C API.
// See: xla/backends/cpu/runtime/kernel_c_api.h

llvm::StructType* Dim3StructTy(llvm::LLVMContext& ctx, absl::string_view name) {
  llvm::IntegerType* i64 = llvm::IntegerType::getInt64Ty(ctx);
  return llvm::StructType::create(name, i64, i64, i64);
}

llvm::StructType* KernelWorkgroupDimTy(llvm::LLVMContext& ctx) {
  return Dim3StructTy(ctx, "XLA_CPU_WorkgroupDim");
}

llvm::StructType* KernelWorkgroupTy(llvm::LLVMContext& ctx) {
  return Dim3StructTy(ctx, "XLA_CPU_WorkgroupId");
}

llvm::StructType* KernelArgTy(llvm::LLVMContext& ctx) {
  llvm::PointerType* ptr = llvm::PointerType::getUnqual(ctx);
  llvm::IntegerType* i64 = llvm::IntegerType::getInt64Ty(ctx);
  return llvm::StructType::create("XLA_CPU_KernelArg", ptr, i64);
}

llvm::StructType* KernelCallFrameTy(llvm::LLVMContext& ctx) {
  llvm::PointerType* ptr = llvm::PointerType::getUnqual(ctx);
  llvm::IntegerType* i64 = llvm::IntegerType::getInt64Ty(ctx);
  return llvm::StructType::create("XLA_CPU_KernelCallFrame", ptr, ptr, i64,
                                  ptr);
}

llvm::FunctionType* KernelFunctionTy(llvm::LLVMContext& ctx) {
  return llvm::FunctionType::get(llvm::PointerType::getUnqual(ctx),
                                 llvm::PointerType::getUnqual(ctx),
                                 /*isVarArg=*/false);
}

// Check that all kernel arguments are coming from non-overlapping slices. It
// is fine to pass same slice as different arguments. This property is not
// used anywhere during the codegen, it acts mostly as a sanity check for
// the buffer assignment. In the future we might emit better aliasing metadata
// based on this property.
absl::Status VerifyKernelArgumentsNonOverlapping(
    absl::Span<const KernelApiIrBuilder::KernelParameter> arguments) {
  for (size_t i = 0; i < arguments.size(); ++i) {
    for (size_t j = i + 1; j < arguments.size(); ++j) {
      const KernelApiIrBuilder::KernelParameter& a = arguments[i];
      const KernelApiIrBuilder::KernelParameter& b = arguments[j];

      if (a.slice != b.slice && a.slice.OverlapsWith(b.slice)) {
        return Internal(
            "Kernel arguments must not overlap: result #%d (%s) overlaps "
            "with result #%d (%s)",
            i, a.slice.ToString(), j, b.slice.ToString());
      }
    }
  }

  return absl::OkStatus();
}

// Check that all kernel results are unique and coming from non-overlapping
// slices.
absl::Status VerifyKernelResultsNonOverlapping(
    absl::Span<const KernelApiIrBuilder::KernelParameter> results) {
  for (size_t i = 0; i < results.size(); ++i) {
    for (size_t j = i + 1; j < results.size(); ++j) {
      const KernelApiIrBuilder::KernelParameter& a = results[i];
      const KernelApiIrBuilder::KernelParameter& b = results[j];

      if (a.slice.OverlapsWith(b.slice)) {
        return Internal(
            "Kernel results must not overlap: result #%d (%s) overlaps "
            "with result #%d (%s)",
            i, a.slice.ToString(), j, b.slice.ToString());
      }
    }
  }

  return absl::OkStatus();
}

// Check that results do not overlap with arguments, or if they do, they must
// be the same as one of the arguments, which can happen for inplace kernels.
absl::Status VerifyKernelResultsNonOverlappingWithArguments(
    absl::Span<const KernelApiIrBuilder::KernelParameter> arguments,
    absl::Span<const KernelApiIrBuilder::KernelParameter> results) {
  for (size_t i = 0; i < results.size(); ++i) {
    for (size_t j = 0; j < arguments.size(); ++j) {
      const KernelApiIrBuilder::KernelParameter& result = results[i];
      const KernelApiIrBuilder::KernelParameter& argument = arguments[j];

      if (result.slice.OverlapsWith(argument.slice) &&
          result.slice != argument.slice) {
        return Internal(
            "Kernel results must not partially overlap with arguments: result "
            "#%d (%s) overlaps with argument #%d (%s)",
            i, result.slice.ToString(), j, argument.slice.ToString());
      }
    }
  }

  return absl::OkStatus();
}

absl::Status VerifyKernelParameters(
    absl::Span<const KernelApiIrBuilder::KernelParameter> arguments,
    absl::Span<const KernelApiIrBuilder::KernelParameter> results) {
  // IMPORTANT: Buffer slice non-overlapping property checked below does not
  // necessarily mean that the buffers do not alias. Parameter allocations
  // might have different index but at run time might be backed by the same
  // memory (or aliased memory). We conservatively do not emit noalias metadata
  // for buffers coming from parameter allocations.

  TF_RETURN_IF_ERROR(VerifyKernelArgumentsNonOverlapping(arguments));
  TF_RETURN_IF_ERROR(VerifyKernelResultsNonOverlapping(results));
  TF_RETURN_IF_ERROR(
      VerifyKernelResultsNonOverlappingWithArguments(arguments, results));

  return absl::OkStatus();
}

absl::StatusOr<BufferAllocation::Slice> GetUniqueSlice(
    const BufferAssignment* buffer_assignment,
    const HloInstruction* instruction, const ShapeIndex& index) {
  return buffer_assignment->GetUniqueSlice(instruction, index);
}

}  // namespace

absl::StatusOr<std::vector<KernelApiIrBuilder::KernelParameter>>
KernelApiIrBuilder::GetKernelArgumentsParameters(
    const HloInstruction* instruction,
    const BufferAssignment* buffer_assignment) {
  std::vector<KernelParameter> arguments;

  for (HloInstruction* operand : instruction->operands()) {
    for (auto& indexed : ShapeUtil::GetLeafShapes(operand->shape())) {
      TF_ASSIGN_OR_RETURN(
          BufferAllocation::Slice slice,
          GetUniqueSlice(buffer_assignment, operand, indexed.index));
      arguments.push_back(KernelParameter{indexed.shape, slice});
    }
  }
  return arguments;
}

absl::StatusOr<std::vector<KernelApiIrBuilder::KernelParameter>>
KernelApiIrBuilder::GetKernelResultsParameters(
    const HloInstruction* instruction,
    const BufferAssignment* buffer_assignment) {
  std::vector<KernelParameter> results;
  for (auto& indexed : ShapeUtil::GetLeafShapes(instruction->shape())) {
    TF_ASSIGN_OR_RETURN(
        BufferAllocation::Slice slice,
        GetUniqueSlice(buffer_assignment, instruction, indexed.index));
    results.push_back(KernelParameter{indexed.shape, slice});
  }
  return results;
}

auto KernelApiIrBuilder::Options::FromHloModuleConfig(
    const HloModuleConfig& config) -> Options {
  return KernelApiIrBuilder::Options{
      config.debug_options().xla_llvm_enable_invariant_load_metadata(),
      config.debug_options().xla_cpu_prefer_vector_width(),
      config.debug_options()
          .xla_cpu_generate_unique_c_style_kernel_entry_points()};
}

KernelApiIrBuilder::KernelApiIrBuilder(llvm::LLVMContext& context,
                                       Options options,
                                       BufferValidation buffer_validation)
    : context_(context),
      options_(std::move(options)),
      buffer_validation_(buffer_validation) {
  workgroup_dim_ty_ = KernelWorkgroupDimTy(context_);
  workgroup_id_ty_ = KernelWorkgroupTy(context_);
  arg_ty_ = KernelArgTy(context_);
  call_frame_ty_ = KernelCallFrameTy(context_);
  kernel_function_ty_ = KernelFunctionTy(context_);
}

auto KernelApiIrBuilder::EmitKernelPrototype(
    llvm::Module& module, const HloInstruction* instr,
    const BufferAssignment* buffer_assignment, absl::string_view suffix)
    -> absl::StatusOr<KernelPrototype> {
  TF_ASSIGN_OR_RETURN(std::vector<KernelParameter> arguments,
                      GetKernelArgumentsParameters(instr, buffer_assignment));
  TF_ASSIGN_OR_RETURN(std::vector<KernelParameter> results,
                      GetKernelResultsParameters(instr, buffer_assignment));

  TF_ASSIGN_OR_RETURN(std::string name, GetKernelName(instr, suffix));

  return EmitKernelPrototype(module, name, arguments, results);
}

auto KernelApiIrBuilder::EmitKernelPrototype(
    llvm::Module& module, absl::string_view name,
    absl::Span<const KernelParameter> arguments,
    absl::Span<const KernelParameter> results)
    -> absl::StatusOr<KernelPrototype> {
  CHECK(&module.getContext() == &context_) << "Module context mismatch";

  VLOG(3) << "Emit kernel prototype: " << name
          << ", #arguments=" << arguments.size()
          << ", #results=" << results.size();
  for (const KernelParameter& argument : arguments) {
    VLOG(3) << "  argument: " << argument.shape.ToString(true) << " in "
            << argument.slice.ToString();
  }
  for (const KernelParameter& result : results) {
    VLOG(3) << "  result: " << result.shape.ToString(true) << " in "
            << result.slice.ToString();
  }

  if (buffer_validation_ == BufferValidation::kDisjoint) {
    TF_RETURN_IF_ERROR(VerifyKernelParameters(arguments, results));
  }

  MemoryDependencyAnalyzer memory_dependency_analyzer(context_, name, results);

  llvm::IRBuilder<> b(context_);

  // Create a kernel function with HostKernel API.
  llvm::Function* function = EmitKernelFunction(module, name);

  // Create an entry basic block and set insert point to the end of it.
  b.SetInsertPoint(llvm::BasicBlock::Create(context_, "", function));

  llvm::Value* call_frame = function->getArg(0);

  // Build workgroup coordinates from the call frame.
  KernelApiIrBuilder::WorkgroupDim kernel_workgroup_dim =
      EmitKernelWorkgroupDim(b, call_frame);
  KernelApiIrBuilder::WorkgroupId kernel_workgroup_id =
      EmitKernelWorkgroupId(b, call_frame);

  int64_t idx = 0;

  // A set of invariant (read-only) buffer indices, feeded in the loop array in
  // the next section.
  absl::flat_hash_set<int64_t> invariant_arguments;

  // IrArrays for the parameters.
  std::vector<llvm_ir::IrArray> ir_arguments;
  for (int64_t i = 0; i < arguments.size(); ++i) {
    const KernelParameter& argument = arguments[i];
    auto ir_argument = EmitKernelArgument(b, call_frame, idx++, argument.shape);
    if (auto* noalias = memory_dependency_analyzer.GetNoAlias(argument.slice)) {
      ir_argument.AddNoaliasMetadata(noalias);
    }

    // If a buffer slice is not a part of result set, then it must be invariant
    // (read-only).
    if (!memory_dependency_analyzer.ResultsOverlapWithSlice(argument.slice)) {
      ir_argument.MarkInvariantOverWholeProgram(&context_);
      invariant_arguments.insert(i);
    }

    ir_arguments.push_back(std::move(ir_argument));
  }

  // IrArrays for the results.
  std::vector<llvm_ir::IrArray> ir_results;
  for (const KernelParameter& result : results) {
    auto ir_result = EmitKernelArgument(b, call_frame, idx++, result.shape);
    if (auto* noalias = memory_dependency_analyzer.GetNoAlias(result.slice)) {
      ir_result.AddNoaliasMetadata(noalias);
    }
    if (auto* alias_scope =
            memory_dependency_analyzer.GetAliasScope(result.slice)) {
      ir_result.AddAliasScopeMetadata(alias_scope);
    }
    ir_results.push_back(std::move(ir_result));
  }

  // Return null pointer to signal success as we do not support error handling
  // in the compiled host kernel.
  llvm::BasicBlock* return_block =
      llvm::BasicBlock::Create(context_, "return", function);

  b.CreateBr(return_block);

  b.SetInsertPoint(return_block);
  b.CreateRet(
      llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(context_)));

  absl::InlinedVector<BufferAllocation::Slice, 8> argument_buffers;
  for (const KernelParameter& argument : arguments) {
    argument_buffers.push_back(argument.slice);
  }

  absl::InlinedVector<BufferAllocation::Slice, 8> result_buffers;
  for (const KernelParameter& result : results) {
    result_buffers.push_back(result.slice);
  }

  return KernelPrototype{function,
                         return_block,
                         kernel_workgroup_dim,
                         kernel_workgroup_id,
                         std::move(ir_arguments),
                         std::move(ir_results),
                         std::move(invariant_arguments),
                         std::move(argument_buffers),
                         std::move(result_buffers)};
}

absl::StatusOr<std::string> KernelApiIrBuilder::GetKernelName(
    const HloInstruction* instr, absl::string_view suffix) const {
  if (options_.generate_unique_c_style_kernel_entry_points) {
    return ConvertToCName(
        absl::StrCat(instr->GetModule()->name(), "_", instr->name(), suffix));
  }
  return absl::StrCat(instr->name(), suffix);
}

std::unique_ptr<llvm::Module> KernelApiIrBuilder::CreateModule(
    absl::string_view name, llvm::LLVMContext& context) {
  constexpr absl::string_view kXlaModuleIdentifier = "__compute_module";
  return std::make_unique<llvm::Module>(
      absl::StrCat(kXlaModuleIdentifier, "_", name), context);
}

auto KernelApiIrBuilder::EmitKernelWorkgroupDim(llvm::IRBuilderBase& builder,
                                                llvm::Value* call_frame)
    -> WorkgroupDim {
  llvm::Value* td_gep =
      builder.CreateStructGEP(call_frame_ty_, call_frame, 0, "wdims_gep");
  llvm::Value* wdims = builder.CreateLoad(builder.getPtrTy(), td_gep, "wdims");
  llvm::Value* x_gep =
      builder.CreateStructGEP(workgroup_dim_ty_, wdims, 0, "wdim_x_gep");
  llvm::Value* y_gep =
      builder.CreateStructGEP(workgroup_dim_ty_, wdims, 1, "wdim_y_gep");
  llvm::Value* z_gep =
      builder.CreateStructGEP(workgroup_dim_ty_, wdims, 2, "wdim_z_gep");

  return {builder.CreateLoad(builder.getInt64Ty(), x_gep, "wdim_x"),
          builder.CreateLoad(builder.getInt64Ty(), y_gep, "wdim_y"),
          builder.CreateLoad(builder.getInt64Ty(), z_gep, "wdim_z")};
}

auto KernelApiIrBuilder::EmitKernelWorkgroupId(llvm::IRBuilderBase& builder,
                                               llvm::Value* call_frame)
    -> WorkgroupId {
  llvm::Value* t_gep =
      builder.CreateStructGEP(call_frame_ty_, call_frame, 1, "wid_gep");
  llvm::LoadInst* wids = builder.CreateLoad(builder.getPtrTy(), t_gep, "wids");
  llvm::Value* x_gep =
      builder.CreateStructGEP(workgroup_id_ty_, wids, 0, "wid_x_gep");
  llvm::Value* y_gep =
      builder.CreateStructGEP(workgroup_id_ty_, wids, 1, "wid_y_gep");
  llvm::Value* z_gep =
      builder.CreateStructGEP(workgroup_id_ty_, wids, 2, "wid_z_gep");

  return {builder.CreateLoad(builder.getInt64Ty(), x_gep, "wid_x"),
          builder.CreateLoad(builder.getInt64Ty(), y_gep, "wid_y"),
          builder.CreateLoad(builder.getInt64Ty(), z_gep, "wid_z")};
}

llvm_ir::IrArray KernelApiIrBuilder::EmitKernelArgument(
    llvm::IRBuilderBase& builder, llvm::Value* call_frame, int64_t index,
    const Shape& shape) {
  llvm::LLVMContext& ctx = builder.getContext();

  llvm::Type* ptr = llvm::PointerType::get(ctx, 0);
  std::string name = absl::StrCat("arg", index);

  llvm::Value* args_gep =
      builder.CreateStructGEP(call_frame_ty_, call_frame, 3, "args_gep");
  llvm::LoadInst* args = builder.CreateLoad(ptr, args_gep, "args");
  llvm::Value* data_gep =
      builder.CreateConstGEP2_32(arg_ty_, args, index, 0, name + "_gep");
  llvm::LoadInst* data = builder.CreateLoad(ptr, data_gep, name);

  // All buffers passed to host kernels are expected to be properly aligned,
  // emit metadata to allow LLVM to use that information for optimization.
  llvm_ir::SetAlignmentMetadataForLoad(data, cpu_function_runtime::MinAlign());

  // All buffers pointers passed to host kernels are expected to be
  // dereferenceable.
  const llvm::Module* llvm_module = builder.GetInsertBlock()->getModule();
  const llvm::DataLayout& data_layout = llvm_module->getDataLayout();
  int64_t pointer_size = data_layout.getTypeStoreSize(builder.getPtrTy());
  int64_t byte_size = ShapeUtil::ByteSizeOf(shape, pointer_size);
  llvm_ir::SetDereferenceableMetadataForLoad(data, byte_size);

  // All buffers pointers passed to host kernels are expected to be invariant
  // over the whole program. Note the metadata is attached only to loading
  // buffer pointers, not to loading actual buffers.
  if (options_.enable_invariant_load_metadata) {
    data->setMetadata(llvm::LLVMContext::MD_invariant_load,
                      llvm::MDNode::get(data->getContext(), /*MDs=*/{}));
  }

  return llvm_ir::IrArray(data, llvm_ir::ShapeToIrType(shape, ctx), shape);
}

llvm::Function* KernelApiIrBuilder::EmitKernelFunction(llvm::Module& module,
                                                       absl::string_view name) {
  llvm::Function* function = llvm::Function::Create(
      kernel_function_ty_, llvm::GlobalValue::ExternalLinkage, name, module);

  SetKernelFunctionAttributes(function);
  return function;
}

void KernelApiIrBuilder::SetKernelFunctionAttributes(llvm::Function* function) {
  // We use external linkage because we'll be resolving this function from the
  // XLA runtime.
  function->setCallingConv(llvm::CallingConv::C);

  // Generate unwind information so that GDB can crawl through the stack frames
  // created by the JIT compiled code.
  function->setUWTableKind(llvm::UWTableKind::Default);

  // Set prefer-vector-width attribute to allow LLVM to use wider vector
  // registers (by default LLVM uses at most 256-bit registers).
  function->addFnAttr("prefer-vector-width",
                      absl::StrCat(options_.prefer_vector_width));

  // Always keep a frame pointer for the host kernel so we can see them in all
  // performance profiling tools.
  function->addFnAttr("frame-pointer", "all");
}

}  // namespace xla::cpu
