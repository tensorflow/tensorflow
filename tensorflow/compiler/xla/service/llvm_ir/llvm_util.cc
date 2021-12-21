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

#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "absl/base/casts.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_options.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace llvm_ir {

namespace {

// Note, this function is only useful in an insertion context; in a global
// (e.g. constants) context it will CHECK fail.
llvm::Module* ModuleFromIRBuilder(llvm::IRBuilder<>* b) {
  auto block = CHECK_NOTNULL(b->GetInsertBlock());
  auto fn = CHECK_NOTNULL(block->getParent());
  auto module = CHECK_NOTNULL(fn->getParent());
  return module;
}

}  // namespace

std::unique_ptr<llvm::Module> DropConstantInitializers(
    const llvm::Module& module) {
  std::unique_ptr<llvm::Module> cloned_module = CloneModule(module);
  for (llvm::GlobalVariable& global_var : cloned_module->globals()) {
    global_var.setInitializer(nullptr);
    global_var.setLinkage(llvm::GlobalValue::LinkageTypes::ExternalLinkage);
  }
  return cloned_module;
}

std::string DumpModuleToString(const llvm::Module& module) {
  std::string buffer_string;
  llvm::raw_string_ostream ostream(buffer_string);
  module.print(ostream, nullptr);
  ostream.flush();
  return buffer_string;
}

llvm::CallInst* EmitCallToIntrinsic(
    llvm::Intrinsic::ID intrinsic_id, absl::Span<llvm::Value* const> operands,
    absl::Span<llvm::Type* const> overloaded_types, llvm::IRBuilder<>* b,
    absl::string_view name) {
  llvm::Module* module = ModuleFromIRBuilder(b);
  llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(
      module, intrinsic_id, AsArrayRef(overloaded_types));
  return b->CreateCall(intrinsic, AsArrayRef(operands), name.data());
}

llvm::Value* EmitFloatMax(llvm::Value* lhs_value, llvm::Value* rhs_value,
                          llvm::IRBuilder<>* b, bool enable_fast_min_max,
                          absl::string_view name) {
  if (b->getFastMathFlags().noNaNs() || enable_fast_min_max) {
    auto cmp = b->CreateFCmpUGE(lhs_value, rhs_value);
    return b->CreateSelect(cmp, lhs_value, rhs_value, name.data());
  } else {
    auto cmp_ge = b->CreateFCmpOGE(lhs_value, rhs_value);
    auto lhs_is_nan = b->CreateFCmpUNE(lhs_value, lhs_value);
    auto sel_lhs = b->CreateOr(cmp_ge, lhs_is_nan);
    return b->CreateSelect(sel_lhs, lhs_value, rhs_value, name.data());
  }
}

llvm::Value* EmitFloatMin(llvm::Value* lhs_value, llvm::Value* rhs_value,
                          llvm::IRBuilder<>* b, bool enable_fast_min_max,
                          absl::string_view name) {
  if (b->getFastMathFlags().noNaNs() || enable_fast_min_max) {
    auto cmp = b->CreateFCmpULE(lhs_value, rhs_value);
    return b->CreateSelect(cmp, lhs_value, rhs_value, name.data());
  } else {
    auto cmp_le = b->CreateFCmpOLE(lhs_value, rhs_value);
    auto lhs_is_nan = b->CreateFCmpUNE(lhs_value, lhs_value);
    auto sel_lhs = b->CreateOr(cmp_le, lhs_is_nan);
    return b->CreateSelect(sel_lhs, lhs_value, rhs_value, name.data());
  }
}

llvm::Value* EmitBufferIndexingGEP(llvm::Value* array, llvm::Value* index,
                                   llvm::IRBuilder<>* b) {
  llvm::Type* array_type = array->getType();
  CHECK(array_type->isPointerTy());
  llvm::PointerType* array_type_as_pointer =
      llvm::cast<llvm::PointerType>(array_type);
  VLOG(2) << "EmitBufferIndexingGEP with type="
          << llvm_ir::DumpToString(*array_type)
          << " array=" << llvm_ir::DumpToString(*array)
          << " index=" << llvm_ir::DumpToString(*index);

  return b->CreateInBoundsGEP(
      array_type_as_pointer->getElementType(), array,
      llvm::isa<llvm::GlobalVariable>(array)
          ? llvm::ArrayRef<llvm::Value*>({b->getInt64(0), index})
          : index);
}

llvm::Value* EmitBufferIndexingGEP(llvm::Value* array, int64_t index,
                                   llvm::IRBuilder<>* b) {
  return EmitBufferIndexingGEP(array, b->getInt64(index), b);
}

llvm::Type* PrimitiveTypeToIrType(PrimitiveType element_type,
                                  llvm::Module* module) {
  switch (element_type) {
    case PRED:
    case S8:
    case U8:
      return llvm::Type::getInt8Ty(module->getContext());
    case S16:
    case U16:
    case BF16:
      // For BF16 we just need some type that is 16 bits wide so that it will
      // take up the right amount of space in memory. LLVM does not have a BF16
      // type (the LLVM half type is IEEE 16 bit floating point, not bfloat), so
      // we can't map it directly to an LLVM type. We will not map a BF16
      // addition to an addition on this type (int16_t) - this is just the type
      // used for storage.
      return llvm::Type::getInt16Ty(module->getContext());
    case F16:
      return llvm::Type::getHalfTy(module->getContext());
    case S32:
    case U32:
      return llvm::Type::getInt32Ty(module->getContext());
    case S64:
    case U64:
      return llvm::Type::getInt64Ty(module->getContext());
    case F32:
      return llvm::Type::getFloatTy(module->getContext());
    case F64:
      return llvm::Type::getDoubleTy(module->getContext());
    case C64: {
      auto cplx_t =
          llvm::StructType::getTypeByName(module->getContext(), "complex64");
      if (cplx_t == nullptr) {
        // C++ standard dictates the memory layout of std::complex is contiguous
        // real followed by imaginary. C++11 section 26.4 [complex.numbers]:
        // If z is an lvalue expression of type cv std::complex<T> then the
        // expression reinterpret_cast<cv T(&)[2]>(z) shall be well-formed,
        // reinterpret_cast<cv T(&)[2]>(z)[0] shall designate the real part of
        // z, and reinterpret_cast<cv T(&)[2]>(z)[1] shall designate the
        // imaginary part of z.
        return llvm::StructType::create(
            {llvm::Type::getFloatTy(module->getContext()),
             llvm::Type::getFloatTy(module->getContext())},
            "complex64", /*isPacked=*/true);
      }
      return cplx_t;
    }
    case C128: {
      auto cplx_t =
          llvm::StructType::getTypeByName(module->getContext(), "complex128");
      if (cplx_t == nullptr) {
        return llvm::StructType::create(
            {llvm::Type::getDoubleTy(module->getContext()),
             llvm::Type::getDoubleTy(module->getContext())},
            "complex128", /*isPacked=*/true);
      }
      return cplx_t;
    }  // A Tuple contains an array of pointers. Use i8*.
    case TUPLE:
    // An Opaque is like a void*, use i8*.
    case OPAQUE_TYPE:
      return llvm::Type::getInt8PtrTy(module->getContext());
    case TOKEN:
      // Tokens do not have a physical representation, but the compiler needs
      // some placeholder type, so use int8_t*.
      return llvm::Type::getInt8PtrTy(module->getContext());
    default:
      LOG(FATAL) << "unsupported type " << element_type;
  }
}

int GetSizeInBits(llvm::Type* type) {
  const llvm::StructType* struct_ty = llvm::dyn_cast<llvm::StructType>(type);
  if (struct_ty) {
    CHECK(struct_ty->isPacked());
    int bits = 0;
    for (auto element_type : struct_ty->elements()) {
      bits += GetSizeInBits(element_type);
    }
    return bits;
  }
  int bits = type->getPrimitiveSizeInBits();
  CHECK_GT(bits, 0) << "type is not sized";
  return bits;
}

llvm::Type* ShapeToIrType(const Shape& shape, llvm::Module* module) {
  llvm::Type* result_type = PrimitiveTypeToIrType(shape.element_type(), module);
  if (shape.IsTuple()) {
    // A tuple buffer is an array of pointers.
    result_type = llvm::ArrayType::get(result_type, shape.tuple_shapes_size());
  } else if (shape.IsArray()) {
    for (int64_t dimension : LayoutUtil::MinorToMajor(shape)) {
      result_type =
          llvm::ArrayType::get(result_type, shape.dimensions(dimension));
    }
  }
  return result_type;
}

StatusOr<llvm::Value*> EncodeSelfDescribingShapeConstant(const Shape& shape,
                                                         int32_t* shape_size,
                                                         llvm::IRBuilder<>* b) {
  std::string encoded_shape = shape.SerializeAsString();
  if (encoded_shape.size() > std::numeric_limits<int32_t>::max()) {
    return InternalError("Encoded shape size exceeded int32_t size limit.");
  }
  *shape_size = static_cast<int32_t>(encoded_shape.size());
  return b->CreateGlobalStringPtr(encoded_shape);
}

llvm::Constant* ConvertLiteralToIrConstant(const Literal& literal,
                                           llvm::Module* module) {
  const char* data = static_cast<const char*>(literal.untyped_data());
  CHECK_EQ(module->getDataLayout().isLittleEndian(),
           tensorflow::port::kLittleEndian);
  return llvm::ConstantDataArray::getString(
      module->getContext(), llvm::StringRef(data, literal.size_bytes()),
      /*AddNull=*/false);
}

llvm::GlobalVariable* AllocateSharedMemoryTile(llvm::Module* module,
                                               llvm::Type* tile_type,
                                               absl::string_view name) {
  // Both AMDGPU and NVPTX use the same address space for shared memory.
  const int kGPUSharedMemoryAddrSpace = 3;
  return new llvm::GlobalVariable(
      *module, tile_type,
      /*isConstant=*/false, llvm::GlobalValue::PrivateLinkage,
      llvm::UndefValue::get(tile_type), AsStringRef(name), nullptr,
      llvm::GlobalValue::NotThreadLocal, kGPUSharedMemoryAddrSpace);
}

llvm::AllocaInst* EmitAllocaAtFunctionEntry(llvm::Type* type,
                                            absl::string_view name,
                                            llvm::IRBuilder<>* b,
                                            int alignment) {
  return EmitAllocaAtFunctionEntryWithCount(type, nullptr, name, b, alignment);
}

llvm::AllocaInst* EmitAllocaAtFunctionEntryWithCount(llvm::Type* type,
                                                     llvm::Value* element_count,
                                                     absl::string_view name,
                                                     llvm::IRBuilder<>* b,
                                                     int alignment) {
  llvm::IRBuilder<>::InsertPointGuard guard(*b);
  llvm::Function* function = b->GetInsertBlock()->getParent();
  b->SetInsertPoint(&function->getEntryBlock(),
                    function->getEntryBlock().getFirstInsertionPt());
  llvm::AllocaInst* alloca =
      b->CreateAlloca(type, element_count, AsStringRef(name));
  if (alignment != 0) {
    alloca->setAlignment(llvm::Align(alignment));
  }
  return alloca;
}

llvm::BasicBlock* CreateBasicBlock(llvm::BasicBlock* insert_before,
                                   absl::string_view name,
                                   llvm::IRBuilder<>* b) {
  return llvm::BasicBlock::Create(
      /*Context=*/b->getContext(),
      /*Name=*/AsStringRef(name),
      /*Parent=*/b->GetInsertBlock()->getParent(),
      /*InsertBefore*/ insert_before);
}

LlvmIfData EmitIfThenElse(llvm::Value* condition, absl::string_view name,
                          llvm::IRBuilder<>* b, bool emit_else) {
  llvm_ir::LlvmIfData if_data;
  if_data.if_block = b->GetInsertBlock();
  if_data.true_block =
      CreateBasicBlock(nullptr, absl::StrCat(name, "-true"), b);
  if_data.false_block =
      emit_else ? CreateBasicBlock(nullptr, absl::StrCat(name, "-false"), b)
                : nullptr;

  // Add a terminator to the if block, if necessary.
  if (if_data.if_block->getTerminator() == nullptr) {
    b->SetInsertPoint(if_data.if_block);
    if_data.after_block =
        CreateBasicBlock(nullptr, absl::StrCat(name, "-after"), b);
    b->CreateBr(if_data.after_block);
  } else {
    if_data.after_block = if_data.if_block->splitBasicBlock(
        b->GetInsertPoint(), absl::StrCat(name, "-after"));
  }

  // Our basic block should now end with an unconditional branch.  Remove it;
  // we're going to replace it with a conditional branch.
  if_data.if_block->getTerminator()->eraseFromParent();

  b->SetInsertPoint(if_data.if_block);
  b->CreateCondBr(condition, if_data.true_block,
                  emit_else ? if_data.false_block : if_data.after_block);

  b->SetInsertPoint(if_data.true_block);
  b->CreateBr(if_data.after_block);

  if (emit_else) {
    b->SetInsertPoint(if_data.false_block);
    b->CreateBr(if_data.after_block);
  }

  b->SetInsertPoint(if_data.after_block,
                    if_data.after_block->getFirstInsertionPt());

  return if_data;
}

llvm::Value* EmitComparison(llvm::CmpInst::Predicate predicate,
                            llvm::Value* lhs_value, llvm::Value* rhs_value,
                            llvm::IRBuilder<>* b, absl::string_view name) {
  llvm::Value* comparison_result;
  if (lhs_value->getType()->isIntegerTy()) {
    comparison_result =
        b->CreateICmp(predicate, lhs_value, rhs_value, name.data());
  } else {
    comparison_result =
        b->CreateFCmp(predicate, lhs_value, rhs_value, name.data());
  }
  // comparison_result is i1, but the NVPTX codegen incorrectly lowers i1
  // arrays. So we extend it to i8 so that it's addressable.
  return b->CreateZExt(comparison_result, llvm_ir::PrimitiveTypeToIrType(
                                              PRED, ModuleFromIRBuilder(b)));
}

// Internal helper that is called from emitted code to log an int64_t value with
// a tag.
static void LogS64(const char* tag, int64_t value) {
  LOG(INFO) << tag << " (int64_t): " << value;
}

void EmitLogging(const char* tag, llvm::Value* value, llvm::IRBuilder<>* b) {
  llvm::FunctionType* log_function_type = llvm::FunctionType::get(
      b->getVoidTy(), {b->getInt64Ty(), b->getInt64Ty()}, /*isVarArg=*/false);
  b->CreateCall(log_function_type,
                b->CreateIntToPtr(b->getInt64(absl::bit_cast<int64_t>(&LogS64)),
                                  log_function_type->getPointerTo()),
                {b->getInt64(absl::bit_cast<int64_t>(tag)), value});
}

void SetAlignmentMetadataForLoad(llvm::LoadInst* load, uint64_t alignment) {
  llvm::LLVMContext& context = load->getContext();
  llvm::Type* int64_ty = llvm::Type::getInt64Ty(context);
  llvm::Constant* alignment_constant =
      llvm::ConstantInt::get(int64_ty, alignment);
  llvm::MDBuilder metadata_builder(context);
  auto* alignment_metadata =
      metadata_builder.createConstant(alignment_constant);
  load->setMetadata(llvm::LLVMContext::MD_align,
                    llvm::MDNode::get(context, alignment_metadata));
}

void SetDereferenceableMetadataForLoad(llvm::LoadInst* load,
                                       uint64_t dereferenceable_bytes) {
  llvm::LLVMContext& context = load->getContext();
  llvm::Type* int64_ty = llvm::Type::getInt64Ty(context);
  llvm::Constant* dereferenceable_bytes_constant =
      llvm::ConstantInt::get(int64_ty, dereferenceable_bytes);
  llvm::MDBuilder metadata_builder(context);
  auto* dereferenceable_bytes_metadata =
      metadata_builder.createConstant(dereferenceable_bytes_constant);
  load->setMetadata(llvm::LLVMContext::MD_dereferenceable,
                    llvm::MDNode::get(context, dereferenceable_bytes_metadata));
}

llvm::Instruction* AddRangeMetadata(int64_t lower, int64_t upper,
                                    llvm::Instruction* inst) {
  llvm::LLVMContext& context = inst->getParent()->getContext();
  llvm::IntegerType* i32 = llvm::Type::getInt32Ty(context);
  inst->setMetadata(
      llvm::LLVMContext::MD_range,
      llvm::MDNode::get(
          context,
          {llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, lower)),
           llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, upper))}));
  return inst;
}

std::string IrName(absl::string_view a) {
  std::string s(a);
  s.erase(std::remove(s.begin(), s.end(), '%'), s.end());
  return s;
}

std::string IrName(absl::string_view a, absl::string_view b) {
  if (!a.empty() && !b.empty()) {
    return IrName(absl::StrCat(a, ".", b));
  }
  return IrName(absl::StrCat(a, b));
}

std::string IrName(const HloInstruction* a, absl::string_view b) {
  return IrName(a->name(), b);
}

std::string SanitizeFunctionName(std::string function_name) {
  // The backend with the strictest requirements on function names is NVPTX, so
  // we sanitize to its requirements.
  //
  // A slightly stricter version of the NVPTX requirements is that names match
  // /[a-zA-Z_$][a-zA-Z0-9_$]*/, with the exception that the names "_" and "$"
  // are illegal.

  // Sanitize chars in function_name.
  std::transform(function_name.begin(), function_name.end(),
                 function_name.begin(), [](char c) {
                   if (('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') ||
                       ('0' <= c && c <= '9') || c == '_' || c == '$') {
                     return c;
                   }
                   return '_';
                 });

  // Ensure the name isn't empty.
  if (function_name.empty()) {
    function_name = "__unnamed";
  }

  // Ensure the name doesn't start with a number.
  if (!function_name.empty() && function_name[0] >= '0' &&
      function_name[0] <= '9') {
    function_name.insert(function_name.begin(), '_');
  }

  // Ensure the name isn't "_" or "$".
  if (function_name == "_" || function_name == "$") {
    function_name += '_';
  }

  return function_name;
}

void SetToFirstInsertPoint(llvm::BasicBlock* blk, llvm::IRBuilder<>* builder) {
  builder->SetInsertPoint(blk, blk->getFirstInsertionPt());
}

void SetToLastInsertPoint(llvm::BasicBlock* blk, llvm::IRBuilder<>* builder) {
  if (llvm::Instruction* terminator = blk->getTerminator()) {
    builder->SetInsertPoint(terminator);
  } else {
    builder->SetInsertPoint(blk);
  }
}

llvm::Value* CreateRor(llvm::Value* rotand, llvm::Value* rotor,
                       llvm::IRBuilder<>* builder) {
  auto size = rotand->getType()->getPrimitiveSizeInBits();
  auto size_value = builder->getIntN(size, size);
  auto mod = [=](llvm::Value* x) { return builder->CreateURem(x, size_value); };
  return builder->CreateOr(
      builder->CreateShl(rotand, mod(builder->CreateSub(size_value, rotor))),
      builder->CreateLShr(rotand, mod(rotor)));
}

int64_t ByteSizeOf(const Shape& shape, const llvm::DataLayout& data_layout) {
  unsigned pointer_size = data_layout.getPointerSize();
  return ShapeUtil::ByteSizeOf(shape, pointer_size);
}

llvm::FastMathFlags GetCpuFastMathFlags(const HloModuleConfig& module_config) {
  llvm::FastMathFlags flags;
  const auto& options = module_config.debug_options();
  if (!options.xla_cpu_enable_fast_math()) {
    return flags;
  }
  // Fast implies AllowReassoc, NoInfs, NoNaNs, NoSignedZeros, AllowReciprocal,
  // AllowContract, and ApproxFunc.
  flags.setFast();
  flags.setNoNaNs(!options.xla_cpu_fast_math_honor_nans());
  flags.setNoInfs(!options.xla_cpu_fast_math_honor_infs());
  flags.setAllowReciprocal(!options.xla_cpu_fast_math_honor_division());
  flags.setApproxFunc(!options.xla_cpu_fast_math_honor_functions());
  return flags;
}

std::map<int, llvm::MDNode*> MergeMetadata(
    llvm::LLVMContext* context, const std::map<int, llvm::MDNode*>& a,
    const std::map<int, llvm::MDNode*>& b) {
  // We should extend this as needed to deal with other kinds of metadata like
  // !dereferenceable and !range.

  std::map<int, llvm::MDNode*> result;
  for (auto kind_md_pair : a) {
    if (kind_md_pair.first == llvm::LLVMContext::MD_alias_scope) {
      llvm::SmallVector<llvm::Metadata*, 8> union_of_scopes;
      llvm::SmallPtrSet<llvm::Metadata*, 8> scope_set;
      for (const auto& scope_a : kind_md_pair.second->operands()) {
        scope_set.insert(llvm::cast<llvm::MDNode>(scope_a.get()));
        union_of_scopes.push_back(llvm::cast<llvm::MDNode>(scope_a.get()));
      }
      auto it = b.find(kind_md_pair.first);
      if (it != b.end()) {
        for (const auto& scope_b : it->second->operands()) {
          if (!scope_set.count(llvm::cast<llvm::MDNode>(scope_b.get()))) {
            union_of_scopes.push_back(llvm::cast<llvm::MDNode>(scope_b.get()));
          }
        }
      }
      result[llvm::LLVMContext::MD_alias_scope] =
          llvm::MDNode::get(*context, union_of_scopes);
    } else if (kind_md_pair.first == llvm::LLVMContext::MD_noalias) {
      llvm::SmallVector<llvm::Metadata*, 8> intersection_of_scopes;
      llvm::SmallPtrSet<llvm::Metadata*, 8> scope_set;
      for (const auto& scope_a : kind_md_pair.second->operands()) {
        scope_set.insert(llvm::cast<llvm::MDNode>(scope_a.get()));
      }
      auto it = b.find(kind_md_pair.first);
      if (it != b.end()) {
        for (const auto& scope_b : it->second->operands()) {
          if (scope_set.count(llvm::cast<llvm::MDNode>(scope_b))) {
            intersection_of_scopes.push_back(llvm::cast<llvm::MDNode>(scope_b));
          }
        }
      }
      if (!intersection_of_scopes.empty()) {
        result[llvm::LLVMContext::MD_noalias] =
            llvm::MDNode::get(*context, intersection_of_scopes);
      }
    }
  }
  return result;
}

static Status CreateAndWriteStringToFile(const std::string& directory_name,
                                         const std::string& file_name,
                                         const std::string& text) {
  std::unique_ptr<tensorflow::WritableFile> f;
  TF_RETURN_IF_ERROR(
      tensorflow::Env::Default()->RecursivelyCreateDir(directory_name));
  TF_RETURN_IF_ERROR(
      tensorflow::Env::Default()->NewWritableFile(file_name, &f));
  TF_RETURN_IF_ERROR(f->Append(text));
  TF_RETURN_IF_ERROR(f->Close());
  return Status::OK();
}

void DumpIrIfEnabled(const HloModule& hlo_module,
                     const llvm::Module& llvm_module, bool optimized,
                     absl::string_view filename_suffix) {
  const auto& debug_opts = hlo_module.config().debug_options();
  if (!DumpingEnabledForHloModule(hlo_module)) {
    return;
  }
  // We can end up compiling different modules with the same name when using
  // XlaJitCompiledCpuFunction::Compile.  Avoid overwriting IR files previously
  // dumped from the same process in such cases.
  std::string suffix =
      absl::StrCat("ir-", optimized ? "with" : "no", "-opt",
                   filename_suffix.empty() ? "" : ".", filename_suffix);
  DumpToFileInDirOrStdout(hlo_module, "", absl::StrCat(suffix, ".ll"),
                          DumpModuleToString(llvm_module));

  // For some models the embedded constants can be huge, so also dump the module
  // with the constants stripped to get IR that is easier to manipulate.  Skip
  // this if we're dumping to stdout; there's no point in duplicating everything
  // when writing to the terminal.
  if (!DumpingToStdout(debug_opts)) {
    DumpToFileInDir(hlo_module, "", absl::StrCat(suffix, "-noconst.ll"),
                    DumpModuleToString(*DropConstantInitializers(llvm_module)));
  }
}

llvm::Function* CreateCpuFunction(llvm::FunctionType* function_type,
                                  llvm::GlobalValue::LinkageTypes linkage,
                                  const HloModuleConfig& module_config,
                                  absl::string_view name,
                                  llvm::Module* module) {
  llvm::Function* function =
      llvm::Function::Create(function_type, linkage, AsStringRef(name), module);
  function->setCallingConv(llvm::CallingConv::C);
  function->addFnAttr("no-frame-pointer-elim", "false");

  // Generate unwind information so that GDB can crawl through the stack frames
  // created by the JIT compiled code.
  function->setHasUWTable();

  // Tensorflow always flushes denormals to zero, let LLVM know that flushing
  // denormals is safe. This allows vectorization using ARM's neon instruction
  // set.
  function->addFnAttr("denormal-fp-math", "preserve-sign");

  // Add the optimize attribute to the function if optimizing for size. This
  // controls internal behavior of some optimization passes (e.g. loop
  // unrolling).
  if (cpu::options::OptimizeForSizeRequested(module_config)) {
    function->addFnAttr(llvm::Attribute::OptimizeForSize);
  }

  return function;
}

std::pair<llvm::Value*, llvm::Value*> UMulLowHigh32(llvm::IRBuilder<>* b,
                                                    llvm::Value* src0,
                                                    llvm::Value* src1) {
  CHECK_EQ(src0->getType()->getPrimitiveSizeInBits(), 32);
  CHECK_EQ(src1->getType()->getPrimitiveSizeInBits(), 32);
  llvm::Type* int64_ty = b->getInt64Ty();
  src0 = b->CreateZExt(src0, int64_ty);
  src1 = b->CreateZExt(src1, int64_ty);
  return SplitInt64ToInt32s(b, b->CreateMul(src0, src1));
}

std::pair<llvm::Value*, llvm::Value*> SplitInt64ToInt32s(
    llvm::IRBuilder<>* b, llvm::Value* value_64bits) {
  CHECK_EQ(value_64bits->getType()->getPrimitiveSizeInBits(), 64);
  llvm::Type* int32_ty = b->getInt32Ty();
  llvm::Value* low_32bits = b->CreateTrunc(value_64bits, int32_ty);
  llvm::Value* high_32bits =
      b->CreateTrunc(b->CreateLShr(value_64bits, 32), int32_ty);
  return std::make_pair(low_32bits, high_32bits);
}

unsigned GetGlobalMemoryAddressSpace() { return 1; }

llvm::GlobalVariable* GetOrCreateVariableForRngState(llvm::Module* module,
                                                     llvm::IRBuilder<>* b) {
  static const char* kRngStateVariableName = "rng_state";
  llvm::GlobalVariable* state_ptr =
      module->getNamedGlobal(kRngStateVariableName);
  if (!state_ptr) {
    llvm::Type* state_type = b->getInt128Ty();
    // Use a non-zero initial value as zero state can cause the result of the
    // first random number generation not passing the chi-square test. The
    // values used here are arbitrarily chosen, any non-zero values should be
    // fine.
    state_ptr = new llvm::GlobalVariable(
        /*M=*/*module,
        /*Ty=*/state_type,
        /*isConstant=*/false,
        /*Linkage=*/llvm::GlobalValue::PrivateLinkage,
        /*Initializer=*/llvm::ConstantInt::get(b->getInt128Ty(), 0x7012395ull),
        /*Name=*/kRngStateVariableName,
        /*InsertBefore=*/nullptr,
        /*TLMode=*/llvm::GlobalValue::NotThreadLocal,
        /*AddressSpace=*/GetGlobalMemoryAddressSpace(),
        /*isExternallyInitialized=*/false);
  }
  return state_ptr;
}

llvm::Value* RngGetAndUpdateState(uint64_t delta, llvm::Module* module,
                                  llvm::IRBuilder<>* builder) {
  llvm::GlobalVariable* state_ptr =
      GetOrCreateVariableForRngState(module, builder);
  llvm::LoadInst* state_value_old = builder->CreateLoad(
      state_ptr->getType()->getPointerElementType(), state_ptr, "load_state");
  llvm::Value* state_value_new = builder->CreateAdd(
      state_value_old,
      llvm::ConstantInt::get(state_value_old->getType(), delta));
  builder->CreateStore(state_value_new, state_ptr);
  return state_value_old;
}

llvm::BasicBlock* EmitReturnBlock(llvm::IRBuilder<>* b) {
  llvm::Function* function = b->GetInsertBlock()->getParent();
  llvm::Module* module = b->GetInsertBlock()->getModule();
  llvm::IRBuilder<>::InsertPointGuard guard(*b);
  llvm::BasicBlock* early_return =
      llvm::BasicBlock::Create(/*Context=*/module->getContext(),
                               /*Name=*/"early_return",
                               /*Parent=*/function);
  b->SetInsertPoint(early_return);
  b->CreateRetVoid();
  return early_return;
}

void EmitEarlyReturn(llvm::Value* condition, llvm::IRBuilder<>* b,
                     llvm::BasicBlock* return_block) {
  if (!return_block) {
    return_block = EmitReturnBlock(b);
  }

  llvm::BasicBlock* continued;

  // Implicitly check whtether we are already at the end of unterminated block.
  if (b->GetInsertBlock()->getTerminator() == nullptr) {
    // If we are generating code into an incomplete basic block we can just
    // create a new basic block to jump to after our conditional branch.
    continued = llvm_ir::CreateBasicBlock(/*insert_before=*/nullptr,
                                          /*name=*/"", b);
  } else {
    // If we are generating code into a basic block that already has code, we
    // need to split that block so as to not disturb the existing code.
    auto original = b->GetInsertBlock();
    continued = original->splitBasicBlock(b->GetInsertPoint());
    // Remove the auto-generated unconditional branch to replace with our
    // conditional branch.
    original->getTerminator()->eraseFromParent();
    b->SetInsertPoint(original);
  }

  b->CreateCondBr(condition, continued, return_block);
  b->SetInsertPoint(continued, continued->getFirstInsertionPt());
}

}  // namespace llvm_ir
}  // namespace xla
