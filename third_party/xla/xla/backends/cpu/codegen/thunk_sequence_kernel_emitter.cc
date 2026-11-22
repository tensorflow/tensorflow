/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/cpu/codegen/thunk_sequence_kernel_emitter.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/runtime/call_thunk.h"
#include "xla/backends/cpu/runtime/conditional_thunk.h"
#include "xla/backends/cpu/runtime/copy_thunk.h"
#include "xla/backends/cpu/runtime/kernel_thunk.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_executor.h"
#include "xla/backends/cpu/runtime/while_thunk.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/work_group.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/decision.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"

namespace xla::cpu {
namespace {

// Module flag carrying per kernel LLVM pipeline options (see ir_compiler.cc).
constexpr absl::string_view kBackendExtraOptionsFlag =
    "xla_backend_extra_options";

Decision CanEmitThunks(const ThunkSequence& thunks,
                       const absl::flat_hash_set<std::string>& kernels) {
  for (const auto& thunk : thunks) {
    switch (thunk->kind()) {
      case Thunk::Kind::kKernel: {
        absl::string_view name =
            absl::down_cast<const KernelThunkBase&>(*thunk).kernel_name();
        if (!kernels.contains(name)) {
          return Decision::Forbid(absl::StrCat("No source for kernel ", name));
        }
        break;
      }
      case Thunk::Kind::kCopy: {
        const auto& copy = absl::down_cast<const CopyThunk&>(*thunk);
        if (copy.src_shape() != copy.dst_shape()) {
          return Decision::Forbid("Copy changes the layout");
        }
        if (copy.src_buffer() != copy.dst_buffer() &&
            copy.src_buffer().OverlapsWith(copy.dst_buffer())) {
          return Decision::Forbid("Copy buffers overlap");
        }
        break;
      }
      case Thunk::Kind::kWhile:
      case Thunk::Kind::kConditional:
      case Thunk::Kind::kCall:
        for (const auto& [name, nested] : thunk->nested_thunks()) {
          if (Decision d = CanEmitThunks(*nested, kernels); !d) {
            return d;
          }
        }
        break;
      default:
        return Decision::Forbid(
            absl::StrCat("Unsupported thunk kind ", thunk->kind()));
    }
  }
  return Decision::Allow();
}

// Removes the module flag `name` and returns its comma separated values.
std::vector<std::string> TakeListModuleFlag(llvm::Module& module,
                                            absl::string_view name) {
  std::vector<std::string> values;
  llvm::NamedMDNode* flags = module.getModuleFlagsMetadata();
  if (flags == nullptr) {
    return values;
  }
  std::vector<llvm::MDNode*> kept;
  for (llvm::MDNode* flag : flags->operands()) {
    auto* key = llvm::dyn_cast<llvm::MDString>(flag->getOperand(1));
    if (key == nullptr || key->getString() != llvm::StringRef(name)) {
      kept.push_back(flag);
      continue;
    }
    if (auto* value = llvm::dyn_cast<llvm::MDString>(flag->getOperand(2))) {
      values = absl::StrSplit(value->getString().str(), ',', absl::SkipEmpty());
    }
  }
  flags->clearOperands();
  for (llvm::MDNode* flag : kept) {
    flags->addOperand(flag);
  }
  return values;
}

std::vector<KernelApiIrBuilder::KernelParameter> ToParameters(
    const KernelSpec::Buffers& buffers) {
  std::vector<KernelApiIrBuilder::KernelParameter> parameters;
  parameters.reserve(buffers.size());
  for (const ShapedSlice& buffer : buffers) {
    parameters.push_back({buffer.shape, buffer.slice});
  }
  return parameters;
}

struct ParameterRegion {
  BufferAllocation::Slice slice;
  llvm::Value* ptr;
};

// Emits the thunks of a sequence as control flow at the insert point of `b`.
// Call frames do not change between launches except for the workgroup id, so
// they are initialized once on kernel entry by `setup`.
class SequenceEmitter {
 public:
  SequenceEmitter(
      llvm::Module& module, llvm::IRBuilderBase& setup, llvm::IRBuilderBase& b,
      const KernelApiIrBuilder& api,
      absl::btree_map<BufferAllocation::Index, std::vector<ParameterRegion>>
          regions,
      llvm::BasicBlock* error_block, llvm::PHINode* error)
      : module_(module),
        setup_(setup),
        b_(b),
        api_(api),
        regions_(std::move(regions)),
        error_block_(error_block),
        error_(error) {}

  absl::Status EmitSequence(const ThunkSequence& thunks) {
    for (const auto& thunk : thunks) {
      ABSL_RETURN_IF_ERROR(EmitThunk(*thunk));
    }
    return absl::OkStatus();
  }

 private:
  absl::Status EmitThunk(const Thunk& thunk) {
    switch (thunk.kind()) {
      case Thunk::Kind::kKernel:
        return EmitKernel(absl::down_cast<const KernelThunkBase&>(thunk));
      case Thunk::Kind::kCopy:
        return EmitCopy(absl::down_cast<const CopyThunk&>(thunk));
      case Thunk::Kind::kWhile:
        return EmitWhile(absl::down_cast<const WhileThunk&>(thunk));
      case Thunk::Kind::kConditional:
        return EmitConditional(absl::down_cast<const ConditionalThunk&>(thunk));
      case Thunk::Kind::kCall:
        return EmitSequence(absl::down_cast<const CallThunk&>(thunk)
                                .called_executor()
                                .thunk_sequence());
      default:
        return Internal("Unsupported thunk kind %v", thunk.kind());
    }
  }

  llvm::BasicBlock* NewBlock(absl::string_view name) {
    return llvm::BasicBlock::Create(b_.getContext(), name,
                                    b_.GetInsertBlock()->getParent());
  }

  absl::StatusOr<llvm::Value*> Buffer(const BufferAllocation::Slice& slice) {
    if (slice.empty()) {
      if (empty_ == nullptr) {
        llvm::AllocaInst* alloca = setup_.CreateAlloca(
            setup_.getInt8Ty(), setup_.getInt64(MinAlign()), "empty");
        alloca->setAlignment(llvm::Align(MinAlign()));
        empty_ = alloca;
      }
      return empty_;
    }
    if (auto it = cache_.find(slice); it != cache_.end()) {
      return it->second;
    }
    auto regions_it = regions_.find(slice.index());
    if (regions_it != regions_.end()) {
      const std::vector<ParameterRegion>& alloc_regions = regions_it->second;
      auto it =
          absl::c_upper_bound(alloc_regions, slice.offset(),
                              [](int64_t offset, const ParameterRegion& r) {
                                return offset < r.slice.offset();
                              });
      if (it != alloc_regions.begin()) {
        --it;
        if (it->slice.Contains(slice)) {
          int64_t delta = slice.offset() - it->slice.offset();
          llvm::Value* ptr = delta == 0 ? it->ptr
                                        : setup_.CreateInBoundsPtrAdd(
                                              it->ptr, setup_.getInt64(delta));
          cache_.emplace(slice, ptr);
          return ptr;
        }
      }
    }
    return Internal("Slice %s is not inside a kernel parameter",
                    slice.ToString());
  }

  // Emits `body(i)` for i in [0, trip_count) and leaves the insert point after
  // the loop. A single trip emits the body inline.
  absl::Status EmitCountedLoop(
      int64_t trip_count, absl::string_view name,
      absl::FunctionRef<absl::Status(llvm::Value*)> body) {
    if (trip_count <= 0) {
      return absl::OkStatus();
    }
    if (trip_count == 1) {
      return body(b_.getInt64(0));
    }
    llvm::BasicBlock* preheader = b_.GetInsertBlock();
    llvm::BasicBlock* loop = NewBlock(name);
    llvm::BasicBlock* exit = NewBlock(absl::StrCat(name, "_exit"));
    b_.CreateBr(loop);
    b_.SetInsertPoint(loop);
    llvm::PHINode* i = b_.CreatePHI(b_.getInt64Ty(), 2, name);
    i->addIncoming(b_.getInt64(0), preheader);
    ABSL_RETURN_IF_ERROR(body(i));
    llvm::Value* next = b_.CreateAdd(i, b_.getInt64(1));
    i->addIncoming(next, b_.GetInsertBlock());
    b_.CreateCondBr(b_.CreateICmpSLT(next, b_.getInt64(trip_count)), loop,
                    exit);
    b_.SetInsertPoint(exit);
    return absl::OkStatus();
  }

  absl::StatusOr<llvm::Function*> DeclareKernel(absl::string_view name) {
    if (llvm::Function* function = module_.getFunction(name)) {
      if (!function->isDeclaration()) {
        return Internal("Nested kernel %s collides with enclosing kernel",
                        name);
      }
      return function;
    }
    return llvm::Function::Create(api_.kernel_function_type(),
                                  llvm::GlobalValue::ExternalLinkage, name,
                                  module_);
  }

  // Calls the kernel once per workgroup, in the order of Kernel::Launch, and
  // returns its error, if any, from the enclosing kernel.
  absl::Status EmitKernel(const KernelThunkBase& thunk) {
    ABSL_ASSIGN_OR_RETURN(llvm::Function * kernel,
                     DeclareKernel(thunk.kernel_name()));

    int64_t num_args =
        thunk.arguments_buffers().size() + thunk.results_buffers().size();
    llvm::StructType* arg_type = api_.kernel_arg_type();
    llvm::ArrayType* args_type = llvm::ArrayType::get(arg_type, num_args);
    llvm::StructType* frame_type = api_.call_frame_type();
    llvm::StructType* dims_type = api_.num_workgroups_type();
    llvm::StructType* id_type = api_.workgroup_id_type();

    llvm::Value* args = setup_.CreateAlloca(args_type, nullptr, "args");
    llvm::Value* dims = setup_.CreateAlloca(dims_type, nullptr, "num_wg");
    llvm::Value* id = setup_.CreateAlloca(id_type, nullptr, "wg_id");
    llvm::Value* frame = setup_.CreateAlloca(frame_type, nullptr, "frame");

    int64_t index = 0;
    for (absl::Span<const ShapedSlice> buffers :
         {thunk.arguments_buffers(), thunk.results_buffers()}) {
      for (const ShapedSlice& buffer : buffers) {
        ABSL_ASSIGN_OR_RETURN(llvm::Value * data, Buffer(buffer.slice));
        llvm::Value* arg =
            setup_.CreateConstGEP2_32(args_type, args, 0, index++);
        setup_.CreateStore(data, setup_.CreateStructGEP(arg_type, arg, 0));
        setup_.CreateStore(setup_.getInt64(buffer.slice.size()),
                           setup_.CreateStructGEP(arg_type, arg, 1));
      }
    }

    const NumWorkGroups& n = thunk.num_workgroups();
    const std::array<uint64_t, 3> num_workgroups = {n.x, n.y, n.z};
    for (int d = 0; d < 3; ++d) {
      setup_.CreateStore(setup_.getInt64(num_workgroups[d]),
                         setup_.CreateStructGEP(dims_type, dims, d));
    }
    setup_.CreateStore(dims, setup_.CreateStructGEP(frame_type, frame, 0));
    setup_.CreateStore(id, setup_.CreateStructGEP(frame_type, frame, 1));
    setup_.CreateStore(setup_.getInt64(num_args),
                       setup_.CreateStructGEP(frame_type, frame, 2));
    setup_.CreateStore(args, setup_.CreateStructGEP(frame_type, frame, 3));

    auto call = [&](llvm::Value* x, llvm::Value* y, llvm::Value* z) {
      b_.CreateStore(x, b_.CreateStructGEP(id_type, id, 0));
      b_.CreateStore(y, b_.CreateStructGEP(id_type, id, 1));
      b_.CreateStore(z, b_.CreateStructGEP(id_type, id, 2));
      llvm::Value* error = b_.CreateCall(kernel, {frame});
      llvm::BasicBlock* ok = NewBlock("kernel_ok");
      b_.CreateCondBr(b_.CreateIsNull(error), ok, error_block_);
      error_->addIncoming(error, b_.GetInsertBlock());
      b_.SetInsertPoint(ok);
      return absl::OkStatus();
    };
    return EmitCountedLoop(n.z, "wg_z", [&](llvm::Value* z) {
      return EmitCountedLoop(n.y, "wg_y", [&](llvm::Value* y) {
        return EmitCountedLoop(n.x, "wg_x",
                               [&](llvm::Value* x) { return call(x, y, z); });
      });
    });
  }

  absl::Status EmitCopy(const CopyThunk& thunk) {
    TF_RET_CHECK(thunk.src_shape() == thunk.dst_shape());
    int64_t size = ShapeUtil::ByteSizeOf(thunk.src_shape());
    if (size == 0 || thunk.src_buffer() == thunk.dst_buffer()) {
      return absl::OkStatus();
    }
    TF_RET_CHECK(!thunk.src_buffer().OverlapsWith(thunk.dst_buffer()));
    ABSL_ASSIGN_OR_RETURN(llvm::Value * src, Buffer(thunk.src_buffer()));
    ABSL_ASSIGN_OR_RETURN(llvm::Value * dst, Buffer(thunk.dst_buffer()));
    b_.CreateMemCpy(dst, llvm::MaybeAlign(1), src, llvm::MaybeAlign(1), size);
    return absl::OkStatus();
  }

  // Matches WhileThunk: a known trip count runs the body without evaluating
  // the condition.
  absl::Status EmitWhile(const WhileThunk& thunk) {
    const ThunkSequence& body = thunk.body_executor().thunk_sequence();
    if (std::optional<int64_t> trip_count = thunk.trip_count()) {
      return EmitCountedLoop(*trip_count, "trip",
                             [&](llvm::Value*) { return EmitSequence(body); });
    }

    ABSL_ASSIGN_OR_RETURN(llvm::Value * pred, Buffer(thunk.cond_buffer()));
    llvm::BasicBlock* cond_block = NewBlock("while_cond");
    llvm::BasicBlock* body_block = NewBlock("while_body");
    llvm::BasicBlock* exit = NewBlock("while_exit");
    b_.CreateBr(cond_block);
    b_.SetInsertPoint(cond_block);
    ABSL_RETURN_IF_ERROR(EmitSequence(thunk.cond_executor().thunk_sequence()));
    llvm::Value* cond = b_.CreateICmpNE(
        b_.CreateLoad(b_.getInt8Ty(), pred, "pred"), b_.getInt8(0));
    b_.CreateCondBr(cond, body_block, exit);
    b_.SetInsertPoint(body_block);
    ABSL_RETURN_IF_ERROR(EmitSequence(body));
    b_.CreateBr(cond_block);
    b_.SetInsertPoint(exit);
    return absl::OkStatus();
  }

  // Matches ConditionalThunk: a pred index selects branch 0 when true, and an
  // out of range s32 index selects the last branch.
  absl::Status EmitConditional(const ConditionalThunk& thunk) {
    const std::vector<ThunkExecutor>& branches = thunk.branch_executors();
    const BufferAllocation::Slice& index_slice = thunk.branch_index_buffer();
    ABSL_ASSIGN_OR_RETURN(llvm::Value * index_ptr, Buffer(index_slice));

    llvm::Value* index;
    if (index_slice.size() == sizeof(bool)) {
      TF_RET_CHECK(branches.size() == 2);
      llvm::Value* pred = b_.CreateICmpNE(
          b_.CreateLoad(b_.getInt8Ty(), index_ptr, "pred"), b_.getInt8(0));
      index = b_.CreateSelect(pred, b_.getInt32(0), b_.getInt32(1));
    } else {
      TF_RET_CHECK(index_slice.size() == sizeof(int32_t));
      TF_RET_CHECK(!branches.empty());
      index = b_.CreateLoad(b_.getInt32Ty(), index_ptr, "branch_index");
    }

    std::vector<llvm::BasicBlock*> blocks;
    blocks.reserve(branches.size());
    for (int64_t i = 0; i < branches.size(); ++i) {
      blocks.push_back(NewBlock(absl::StrCat("branch", i)));
    }
    llvm::BasicBlock* exit = NewBlock("cond_exit");
    llvm::SwitchInst* dispatch = b_.CreateSwitch(index, blocks.back());
    for (int64_t i = 0; i + 1 < branches.size(); ++i) {
      dispatch->addCase(b_.getInt32(i), blocks[i]);
    }
    for (int64_t i = 0; i < branches.size(); ++i) {
      b_.SetInsertPoint(blocks[i]);
      ABSL_RETURN_IF_ERROR(EmitSequence(branches[i].thunk_sequence()));
      b_.CreateBr(exit);
    }
    b_.SetInsertPoint(exit);
    return absl::OkStatus();
  }

  llvm::Module& module_;
  llvm::IRBuilderBase& setup_;
  llvm::IRBuilderBase& b_;
  const KernelApiIrBuilder& api_;
  absl::btree_map<BufferAllocation::Index, std::vector<ParameterRegion>>
      regions_;
  absl::flat_hash_map<BufferAllocation::Slice, llvm::Value*> cache_;
  llvm::Value* empty_ = nullptr;
  llvm::BasicBlock* error_block_;
  llvm::PHINode* error_;
};

}  // namespace

Decision ThunkSequenceKernelEmitter::CanEmit(
    const ThunkSequence& thunks,
    const absl::flat_hash_set<std::string>& kernels,
    const KernelSpec::Buffers& arguments, const KernelSpec::Buffers& results) {
  if (Decision d = CanEmitThunks(thunks, kernels); !d) {
    return d;
  }

  struct Parameter {
    BufferAllocation::Slice slice;
    bool written;
  };
  std::vector<Parameter> parameters;
  parameters.reserve(arguments.size() + results.size());
  for (const ShapedSlice& buffer : arguments) {
    if (buffer.slice.empty()) {
      return Decision::Forbid("Empty parameter slice");
    }
    parameters.push_back({buffer.slice, /*written=*/false});
  }
  for (const ShapedSlice& buffer : results) {
    if (buffer.slice.empty()) {
      return Decision::Forbid("Empty parameter slice");
    }
    parameters.push_back({buffer.slice, /*written=*/true});
  }
  absl::c_sort(parameters, [](const Parameter& a, const Parameter& b) {
    return a.slice < b.slice;
  });
  for (int64_t i = 1; i < parameters.size(); ++i) {
    if (parameters[i - 1].slice.OverlapsWith(parameters[i].slice)) {
      return Decision::Forbid(
          absl::StrCat("Kernel parameters ", parameters[i - 1].slice.ToString(),
                       " and ", parameters[i].slice.ToString(), " overlap"));
    }
  }

  for (const BufferUse& use : thunks.buffer_uses()) {
    if (use.slice().empty()) {
      continue;
    }
    auto it = absl::c_upper_bound(
        parameters, use.slice(),
        [](const BufferAllocation::Slice& s, const Parameter& p) {
          if (s.index() != p.slice.index()) {
            return s.index() < p.slice.index();
          }
          return s.offset() < p.slice.offset();
        });
    if (it == parameters.begin() || !(--it)->slice.Contains(use.slice())) {
      return Decision::Forbid(
          absl::StrCat("Slice ", use.slice().ToString(),
                       " is not inside a kernel parameter"));
    }
    if (use.access() == BufferUse::MemoryAccess::kWrite && !it->written) {
      return Decision::Forbid(absl::StrCat(
          "Slice ", use.slice().ToString(),
          " is written but lies in argument parameter ", it->slice.ToString()));
    }
  }
  return Decision::Allow();
}

absl::Status ThunkSequenceKernelEmitter::LinkKernels(
    llvm::Module& module, absl::Span<const KernelModule> kernels) {
  llvm::LLVMContext& context = module.getContext();
  absl::btree_set<std::string> backend_options;
  for (std::string& option :
       TakeListModuleFlag(module, kBackendExtraOptionsFlag)) {
    backend_options.insert(std::move(option));
  }

  // Kernels called by the module. Others, such as the condition of a loop
  // with a known trip count, are never run and are not linked.
  absl::flat_hash_set<std::string> called;
  for (const llvm::Function& function : module) {
    if (function.isDeclaration() && !function.isIntrinsic()) {
      called.insert(function.getName().str());
    }
  }

  llvm::Linker linker(module);
  for (const KernelModule& kernel : kernels) {
    if (!called.contains(kernel.name)) {
      continue;
    }
    // Nested kernels were compiled in their own contexts, and the linker needs
    // them in the context of `module`.
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<llvm::Module> nested,
        llvm_ir::CloneModuleToContext(*kernel.module, context,
                                      kernel.module->getModuleIdentifier()));
    llvm::Function* entry = nested->getFunction(kernel.name);
    if (entry == nullptr || entry->isDeclaration()) {
      return Internal("Module %s does not define kernel %s",
                      nested->getModuleIdentifier(), kernel.name);
    }

    // Private symbols of two kernels must not collide.
    for (llvm::GlobalValue& value : nested->global_values()) {
      if (!value.isDeclaration() && value.getName() != kernel.name &&
          !value.getName().starts_with("llvm.")) {
        value.setLinkage(llvm::GlobalValue::InternalLinkage);
      }
    }
    // Options apply to whole modules, so the enclosing kernel gets the union.
    // They only tune LLVM passes, so applying them more broadly is safe.
    for (std::string& option :
         TakeListModuleFlag(*nested, kBackendExtraOptionsFlag)) {
      backend_options.insert(std::move(option));
    }
    // The enclosing module names the memory region of all its kernels.
    if (llvm::NamedMDNode* region =
            nested->getNamedMetadata(kMemoryRegionNameMetadataName)) {
      nested->eraseNamedMetadata(region);
    }

    nested->setDataLayout(module.getDataLayout());
    nested->setTargetTriple(module.getTargetTriple());
    if (linker.linkInModule(std::move(nested))) {
      return Internal("Failed to link kernel %s into %s", kernel.name,
                      module.getModuleIdentifier());
    }
    module.getFunction(kernel.name)
        ->setLinkage(llvm::GlobalValue::InternalLinkage);
    called.erase(kernel.name);
  }

  if (!called.empty()) {
    std::vector<std::string> missing(called.begin(), called.end());
    absl::c_sort(missing);
    return Internal("Kernels called by %s but not linked: %s",
                    module.getModuleIdentifier(), absl::StrJoin(missing, ", "));
  }

  if (!backend_options.empty()) {
    module.addModuleFlag(
        llvm::Module::Error, kBackendExtraOptionsFlag,
        llvm::MDString::get(context, absl::StrJoin(backend_options, ",")));
  }

  std::string errors;
  llvm::raw_string_ostream errors_stream(errors);
  if (llvm::verifyModule(module, &errors_stream)) {
    return Internal("Invalid LLVM IR after linking kernels into %s: %s",
                    module.getModuleIdentifier(), errors);
  }
  return absl::OkStatus();
}

ThunkSequenceKernelEmitter::ThunkSequenceKernelEmitter(
    std::string name, std::string memory_region_name,
    KernelSpec::Buffers arguments, KernelSpec::Buffers results,
    const ThunkSequence* thunks, const TargetMachineFeatures* target_machine,
    KernelApiIrBuilder::Options options)
    : kernel_name_(std::move(name)),
      memory_region_name_(std::move(memory_region_name)),
      arguments_(std::move(arguments)),
      results_(std::move(results)),
      thunks_(thunks),
      target_machine_(target_machine),
      options_(std::move(options)) {}

absl::StatusOr<ThunkSequenceKernelEmitter::KernelDefinition>
ThunkSequenceKernelEmitter::EmitKernelDefinition() {
  auto context = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> module = KernelApiIrBuilder::CreateModule(
      absl::StrCat(kernel_name_, "_module"), *context, target_machine_);

  KernelApiIrBuilder api(*context, options_,
                         KernelApiIrBuilder::BufferValidation::kDisjoint);
  ABSL_ASSIGN_OR_RETURN(
      KernelApiIrBuilder::KernelPrototype prototype,
      api.EmitKernelPrototype(*module, kernel_name_, ToParameters(arguments_),
                              ToParameters(results_), memory_region_name_));

  llvm::Type* ptr_ty = llvm::PointerType::getUnqual(*context);
  std::vector<llvm::Type*> param_types(arguments_.size() + results_.size(),
                                       ptr_ty);
  // Leave the body unnamed until nested kernels are declared, so a nested
  // kernel named `<kernel_name>_body` declares its own symbol first.
  llvm::Function* body_fn = llvm::Function::Create(
      llvm::FunctionType::get(ptr_ty, param_types, /*isVarArg=*/false),
      llvm::GlobalValue::InternalLinkage, /*Name=*/"", *module);
  body_fn->setAttributes(llvm::AttributeList::get(
      *context, llvm::AttributeList::FunctionIndex,
      llvm::AttrBuilder(*context,
                        prototype.function->getAttributes().getFnAttrs())));

  // Regions are never empty, and buffers live for the whole launch. Like the
  // entry's loads of the region pointers, the body parameters promise buffer
  // assignment's alignment and slice sizes. LLVM uses these attributes in
  // places where it ignores load metadata, for example to hoist loads.
  std::vector<llvm::Value*> body_args;
  body_args.reserve(arguments_.size() + results_.size());
  absl::btree_map<BufferAllocation::Index, std::vector<ParameterRegion>>
      regions;
  auto add_region = [&](int64_t param_idx, const ShapedSlice& region,
                        llvm::Value* entry_ptr, bool is_readonly) {
    body_args.push_back(entry_ptr);
    body_fn->addParamAttr(param_idx, llvm::Attribute::NoUndef);
    body_fn->addParamAttr(param_idx, llvm::Attribute::NoFree);
    body_fn->addParamAttr(param_idx, llvm::Attribute::NonNull);
    body_fn->addParamAttr(param_idx, llvm::Attribute::getWithAlignment(
                                         *context, llvm::Align(MinAlign())));
    body_fn->addDereferenceableParamAttr(param_idx, region.slice.size());
    body_fn->addParamAttr(param_idx, llvm::Attribute::NoAlias);
    if (is_readonly) {
      body_fn->addParamAttr(param_idx, llvm::Attribute::ReadOnly);
    }
    llvm::Argument* arg = body_fn->getArg(param_idx);
    arg->setName(absl::StrCat("region", param_idx));
    regions[region.slice.index()].push_back({region.slice, arg});
  };
  int64_t param_idx = 0;
  for (int64_t i = 0; i < arguments_.size(); ++i) {
    add_region(param_idx++, arguments_[i],
               prototype.arguments[i].GetBasePointer(), /*is_readonly=*/true);
  }
  for (int64_t i = 0; i < results_.size(); ++i) {
    add_region(param_idx++, results_[i], prototype.results[i].GetBasePointer(),
               /*is_readonly=*/false);
  }
  for (auto& [index, alloc_regions] : regions) {
    absl::c_sort(alloc_regions,
                 [](const ParameterRegion& a, const ParameterRegion& b) {
                   return a.slice.offset() < b.slice.offset();
                 });
  }

  llvm::BasicBlock* proto_entry = &prototype.function->getEntryBlock();
  proto_entry->getTerminator()->eraseFromParent();
  prototype.return_block->eraseFromParent();
  llvm::IRBuilder<> proto_builder(proto_entry);
  llvm::Value* body_result = proto_builder.CreateCall(body_fn, body_args);
  proto_builder.CreateRet(body_result);

  llvm::BasicBlock* entry =
      llvm::BasicBlock::Create(*context, "entry", body_fn);
  llvm::BasicBlock* body = llvm::BasicBlock::Create(*context, "body", body_fn);
  llvm::BasicBlock* return_block =
      llvm::BasicBlock::Create(*context, "return", body_fn);
  llvm::BasicBlock* error_block =
      llvm::BasicBlock::Create(*context, "error", body_fn);

  llvm::IRBuilder<> setup(entry);
  llvm::Instruction* entry_br = setup.CreateBr(body);
  setup.SetInsertPoint(entry_br);

  llvm::IRBuilder<> b(body);

  llvm::IRBuilder<> return_builder(return_block);
  return_builder.CreateRet(llvm::ConstantPointerNull::get(ptr_ty));

  llvm::IRBuilder<> error_builder(error_block);
  llvm::PHINode* error = error_builder.CreatePHI(ptr_ty, 0, "error");
  error_builder.CreateRet(error);

  SequenceEmitter emitter(*module, setup, b, api, std::move(regions),
                          error_block, error);
  ABSL_RETURN_IF_ERROR(emitter.EmitSequence(*thunks_));
  body_fn->setName(absl::StrCat(kernel_name_, "_body"));
  b.CreateBr(return_block);

  if (error->getNumIncomingValues() == 0) {
    error_block->eraseFromParent();
  }

  std::string errors;
  llvm::raw_string_ostream errors_stream(errors);
  if (llvm::verifyModule(*module, &errors_stream)) {
    return Internal("Invalid LLVM IR for kernel %s: %s", kernel_name_, errors);
  }

  KernelSpec spec(kernel_name_, NumWorkGroups(), arguments_, results_,
                  std::move(prototype.invariant_arguments));
  return KernelDefinition(
      std::move(spec), LlvmKernelSource(std::move(context), std::move(module)));
}

}  // namespace xla::cpu
