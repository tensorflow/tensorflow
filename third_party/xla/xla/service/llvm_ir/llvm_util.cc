/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/llvm_ir/llvm_util.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/cpu/cpu_options.h"
#include "xla/service/dump.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/byte_order.h"
#include "tsl/platform/logging.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace xla {
namespace llvm_ir {

namespace {

// This works for most llvm / mlir types. This also accepts a const pointer to
// objects which have a const print() method.
template <typename T>
std::string DumpToStringTempl(T* entity) {
  CHECK_NE(entity, nullptr);

  std::string s;
  llvm::raw_string_ostream ostream(s);
  ostream << *entity;
  return s;
}

// Note, this function is only useful in an insertion context; in a global
// (e.g. constants) context it will CHECK fail.
llvm::Module* ModuleFromIRBuilder(llvm::IRBuilderBase* b) {
  auto block = CHECK_NOTNULL(b->GetInsertBlock());
  auto fn = CHECK_NOTNULL(block->getParent());
  auto module = CHECK_NOTNULL(fn->getParent());
  return module;
}

}  // namespace

std::string DumpToString(const llvm::Module* module) {
  return DumpToStringTempl(module);
}

std::string DumpToString(const llvm::Type* type) {
  return DumpToStringTempl(type);
}

std::string DumpToString(const llvm::Value* value) {
  return DumpToStringTempl(value);
}

std::string DumpToString(mlir::Operation* operation) {
  return DumpToStringTempl(operation);
}

std::string DumpToString(mlir::Type type) { return DumpToStringTempl(&type); }

std::string DumpToString(mlir::Value value) {
  return DumpToStringTempl(&value);
}

llvm::CallInst* EmitCallToIntrinsic(
    llvm::Intrinsic::ID intrinsic_id, absl::Span<llvm::Value* const> operands,
    absl::Span<llvm::Type* const> overloaded_types, llvm::IRBuilderBase* b,
    absl::string_view name) {
  llvm::Module* module = ModuleFromIRBuilder(b);
  llvm::Function* intrinsic = llvm::Intrinsic::getOrInsertDeclaration(
      module, intrinsic_id, AsArrayRef(overloaded_types));
  return b->CreateCall(intrinsic, AsArrayRef(operands), name.data());
}

llvm::Value* EmitFloatMax(llvm::Value* lhs_value, llvm::Value* rhs_value,
                          llvm::IRBuilderBase* b, bool enable_fast_min_max,
                          absl::string_view name) {
  if (b->getFastMathFlags().noNaNs() || enable_fast_min_max) {
    auto cmp = b->CreateFCmpUGE(lhs_value, rhs_value);
    return b->CreateSelect(cmp, lhs_value, rhs_value, name.data());
  }
  return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::maximum,
                                      {lhs_value, rhs_value},
                                      {lhs_value->getType()}, b);
}

llvm::Value* EmitFloatMin(llvm::Value* lhs_value, llvm::Value* rhs_value,
                          llvm::IRBuilderBase* b, bool enable_fast_min_max,
                          absl::string_view name) {
  if (b->getFastMathFlags().noNaNs() || enable_fast_min_max) {
    auto cmp = b->CreateFCmpULE(lhs_value, rhs_value);
    return b->CreateSelect(cmp, lhs_value, rhs_value, name.data());
  }
  return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::minimum,
                                      {lhs_value, rhs_value},
                                      {lhs_value->getType()}, b);
}

llvm::Value* EmitBufferIndexingGEP(llvm::Value* array, llvm::Type* element_type,
                                   llvm::Value* index, llvm::IRBuilderBase* b) {
  llvm::Type* array_type = array->getType();
  CHECK(array_type->isPointerTy());
  VLOG(2) << "EmitBufferIndexingGEP with type="
          << llvm_ir::DumpToString(array_type)
          << " array=" << llvm_ir::DumpToString(array)
          << " index=" << llvm_ir::DumpToString(index);

  return b->CreateInBoundsGEP(
      element_type, array,
      llvm::isa<llvm::GlobalVariable>(array)
          ? llvm::ArrayRef<llvm::Value*>({b->getInt64(0), index})
          : index);
}

llvm::Value* EmitBufferIndexingGEP(llvm::Value* array, llvm::Type* element_type,
                                   int64_t index, llvm::IRBuilderBase* b) {
  return EmitBufferIndexingGEP(array, element_type, b->getInt64(index), b);
}

llvm::Type* PrimitiveTypeToIrType(PrimitiveType element_type,
                                  llvm::LLVMContext& context) {
  switch (element_type) {
    case S2:
    case U2:
      return llvm::Type::getIntNTy(context, 2);
    case S4:
    case U4:
      return llvm::Type::getIntNTy(context, 4);
    case PRED:
    case S8:
    case U8:
      return llvm::Type::getInt8Ty(context);
    case S16:
    case U16:
      return llvm::Type::getInt16Ty(context);
    case F8E5M2:
    case F8E5M2FNUZ:
    case F8E4M3:
    case F8E4M3FN:
    case F8E4M3B11FNUZ:
    case F8E4M3FNUZ:
    case F8E3M4:
      // We represent F8 as an int since there is no LLVM F8 dtype.
      return llvm::Type::getInt8Ty(context);
    case BF16:
      return llvm::Type::getBFloatTy(context);
    case F16:
      return llvm::Type::getHalfTy(context);
    case S32:
    case U32:
      return llvm::Type::getInt32Ty(context);
    case S64:
    case U64:
      return llvm::Type::getInt64Ty(context);
    case F32:
      return llvm::Type::getFloatTy(context);
    case F64:
      return llvm::Type::getDoubleTy(context);
    case C64: {
      auto cplx_t = llvm::StructType::getTypeByName(context, "complex64");
      if (cplx_t == nullptr) {
        // C++ standard dictates the memory layout of std::complex is contiguous
        // real followed by imaginary. C++11 section 26.4 [complex.numbers]:
        // If z is an lvalue expression of type cv std::complex<T> then the
        // expression reinterpret_cast<cv T(&)[2]>(z) shall be well-formed,
        // reinterpret_cast<cv T(&)[2]>(z)[0] shall designate the real part of
        // z, and reinterpret_cast<cv T(&)[2]>(z)[1] shall designate the
        // imaginary part of z.
        return llvm::StructType::create(
            {llvm::Type::getFloatTy(context), llvm::Type::getFloatTy(context)},
            "complex64", /*isPacked=*/true);
      }
      return cplx_t;
    }
    case C128: {
      auto cplx_t = llvm::StructType::getTypeByName(context, "complex128");
      if (cplx_t == nullptr) {
        return llvm::StructType::create({llvm::Type::getDoubleTy(context),
                                         llvm::Type::getDoubleTy(context)},
                                        "complex128", /*isPacked=*/true);
      }
      return cplx_t;
    }  // A Tuple contains an array of pointers. Use i8*.
    case TUPLE:
    // An Opaque is like a void*, use i8*.
    case OPAQUE_TYPE:
      return llvm::PointerType::getUnqual(context);
    case TOKEN:
      // Tokens do not have a physical representation, but the compiler needs
      // some placeholder type, so use int8_t*.
      return llvm::PointerType::getUnqual(context);
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

llvm::Type* ShapeToIrType(const Shape& shape, llvm::LLVMContext& context) {
  llvm::Type* result_type =
      PrimitiveTypeToIrType(shape.element_type(), context);
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

absl::StatusOr<llvm::Value*> EncodeSelfDescribingShapeConstant(
    const Shape& shape, int32_t* shape_size, llvm::IRBuilderBase* b) {
  std::string encoded_shape = shape.SerializeAsString();
  if (encoded_shape.size() > std::numeric_limits<int32_t>::max()) {
    return Internal("Encoded shape size exceeded int32_t size limit.");
  }
  *shape_size = static_cast<int32_t>(encoded_shape.size());
  return b->CreateGlobalStringPtr(encoded_shape);
}

llvm::Constant* ConvertLiteralToIrConstant(const Literal& literal,
                                           llvm::Module* module) {
  const char* data = static_cast<const char*>(literal.untyped_data());
  int64_t size_bytes = literal.size_bytes();
  CHECK_EQ(module->getDataLayout().isLittleEndian(), tsl::port::kLittleEndian);
  std::vector<char> packed_data;
  if (primitive_util::IsSubByteNonPredType(literal.shape().element_type())) {
    auto bit_width = primitive_util::BitWidth(literal.shape().element_type());
    int elements_per_byte = 8 / bit_width;
    packed_data.resize(CeilOfRatio<int64_t>(size_bytes, elements_per_byte));
    PackIntN(bit_width, absl::MakeSpan(data, size_bytes),
             absl::MakeSpan(packed_data));
    data = packed_data.data();
    size_bytes = packed_data.size();
  }
  return llvm::ConstantDataArray::getString(module->getContext(),
                                            llvm::StringRef(data, size_bytes),
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

SharedMemoryTile AllocateSharedMemoryTile(
    llvm::Module* module, llvm::Type* element_type,
    absl::Span<int64_t const> dimensions_major_to_minor,
    absl::string_view buffer_name) {
  llvm::Type* ty = element_type;
  for (auto dim : llvm::reverse(dimensions_major_to_minor)) {
    ty = llvm::ArrayType::get(ty, dim);
  }
  return SharedMemoryTile{
      llvm_ir::AllocateSharedMemoryTile(module, ty, buffer_name), element_type};
}

static std::vector<llvm::Value*> IndexWith0(
    absl::Span<llvm::Value* const> index, llvm::IRBuilderBase* b) {
  std::vector<llvm::Value*> index_with_0{
      llvm::ConstantInt::get(index.front()->getType(), 0)};
  absl::c_copy(index, std::back_inserter(index_with_0));
  return index_with_0;
}

llvm::Value* SharedMemoryTile::Address(absl::Span<llvm::Value* const> index,
                                       llvm::IRBuilderBase* b) const {
  llvm::Value* gep = b->CreateInBoundsGEP(base_ptr_->getValueType(), base_ptr_,
                                          IndexWith0(index, b));
  // __shared__ memory uses a different address space, so we cast it
  // to global address space before writing or reading.
  return b->CreateAddrSpaceCast(gep,
                                llvm::PointerType::get(b->getContext(), 0));
};

llvm::Value* SharedMemoryTile::Load(absl::Span<llvm::Value* const> index,
                                    llvm::IRBuilderBase* b) const {
  auto* load_type = llvm::GetElementPtrInst::getIndexedType(
      base_ptr_->getValueType(), IndexWith0(index, b));
  return b->CreateLoad(load_type, Address(index, b));
}

llvm::StoreInst* SharedMemoryTile::Store(llvm::Value* value,
                                         absl::Span<llvm::Value* const> index,
                                         llvm::IRBuilderBase* b) const {
  return b->CreateStore(value, Address(index, b));
}

llvm::AllocaInst* EmitAllocaAtFunctionEntry(llvm::Type* type,
                                            absl::string_view name,
                                            llvm::IRBuilderBase* b,
                                            int alignment) {
  return EmitAllocaAtFunctionEntryWithCount(type, nullptr, name, b, alignment);
}

llvm::AllocaInst* EmitAllocaAtFunctionEntryWithCount(llvm::Type* type,
                                                     llvm::Value* element_count,
                                                     absl::string_view name,
                                                     llvm::IRBuilderBase* b,
                                                     int alignment) {
  llvm::IRBuilderBase::InsertPointGuard guard(*b);
  llvm::Function* function = b->GetInsertBlock()->getParent();
  b->SetInsertPoint(&function->getEntryBlock(),
                    function->getEntryBlock().getFirstInsertionPt());
  llvm::Module* module = b->GetInsertBlock()->getModule();
  // Explicitly set local addrspace for SPIR backend.
  llvm::Triple target(module->getTargetTriple());
  int addrspace = target.isSPIR() || target.isAMDGPU() ? 5 : 0;
  llvm::AllocaInst* alloca =
      b->CreateAlloca(type, addrspace, element_count, AsStringRef(name));
  if (alignment != 0) {
    alloca->setAlignment(llvm::Align(alignment));
  }
  return alloca;
}

llvm::BasicBlock* CreateBasicBlock(llvm::BasicBlock* insert_before,
                                   absl::string_view name,
                                   llvm::IRBuilderBase* b) {
  return llvm::BasicBlock::Create(
      /*Context=*/b->getContext(),
      /*Name=*/AsStringRef(name),
      /*Parent=*/b->GetInsertBlock()->getParent(),
      /*InsertBefore*/ insert_before);
}

LlvmIfData EmitIfThenElse(llvm::Value* condition, absl::string_view name,
                          llvm::IRBuilderBase* b, bool emit_else) {
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
                            llvm::IRBuilderBase* b, absl::string_view name) {
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
  return b->CreateZExt(comparison_result,
                       llvm_ir::PrimitiveTypeToIrType(PRED, b->getContext()));
}

// Internal helper that is called from emitted code to log an int64_t value with
// a tag.
static void LogS64(const char* tag, int64_t value) {
  LOG(INFO) << tag << " (int64_t): " << value;
}

void EmitLogging(const char* tag, llvm::Value* value, llvm::IRBuilderBase* b) {
  llvm::FunctionType* log_function_type = llvm::FunctionType::get(
      b->getVoidTy(), {b->getInt64Ty(), b->getInt64Ty()}, /*isVarArg=*/false);
  b->CreateCall(log_function_type,
                b->CreateIntToPtr(b->getInt64(absl::bit_cast<int64_t>(&LogS64)),
                                  b->getPtrTy()),
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

llvm::Instruction* AddRangeMetadata(int32_t lower, int32_t upper,
                                    llvm::Instruction* inst,
                                    llvm::Module* module) {
  if (llvm::Triple(module->getTargetTriple()).isSPIR()) {
    return inst;
  }
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

mlir::OwningOpRef<mlir::ModuleOp> CreateMlirModuleOp(
    mlir::Location loc, std::optional<llvm::StringRef> name) {
  return mlir::OwningOpRef<mlir::ModuleOp>(
      /*ALLOW_MLIR_MODULE_OP_CREATE*/ mlir::ModuleOp::create(std::move(loc),
                                                             std::move(name)));
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

void SetToFirstInsertPoint(llvm::BasicBlock* blk,
                           llvm::IRBuilderBase* builder) {
  builder->SetInsertPoint(blk, blk->getFirstInsertionPt());
}

void SetToLastInsertPoint(llvm::BasicBlock* blk, llvm::IRBuilderBase* builder) {
  if (llvm::Instruction* terminator = blk->getTerminator()) {
    builder->SetInsertPoint(terminator);
  } else {
    builder->SetInsertPoint(blk);
  }
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

void DumpIrIfEnabled(const HloModule& hlo_module,
                     const llvm::Module& llvm_module, bool optimized,
                     absl::string_view filename_suffix) {
  if (!DumpingEnabledForHloModule(hlo_module)) {
    return;
  }
  tsl::profiler::ScopedAnnotation annotation([&] {
    return absl::StrFormat("XlaDumpLlvmIr:#module=%s,program_id=%d#",
                           hlo_module.name(), hlo_module.unique_id());
  });
  // We can end up compiling different modules with the same name when using
  // XlaJitCompiledCpuFunction::Compile.  Avoid overwriting IR files previously
  // dumped from the same process in such cases.
  std::string suffix =
      absl::StrCat("ir-", optimized ? "with" : "no", "-opt",
                   filename_suffix.empty() ? "" : ".", filename_suffix);
  DumpToFileInDirOrStdout(hlo_module, "", absl::StrCat(suffix, ".ll"),
                          DumpToString(&llvm_module));
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
  function->setUWTableKind(llvm::UWTableKind::Default);

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

unsigned GetGlobalMemoryAddressSpace() { return 1; }

llvm::GlobalVariable* GetOrCreateVariableForRngState(llvm::Module* module,
                                                     llvm::IRBuilderBase* b) {
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
                                  llvm::IRBuilderBase* builder) {
  llvm::GlobalVariable* state_ptr =
      GetOrCreateVariableForRngState(module, builder);
  llvm::LoadInst* state_value_old =
      builder->CreateLoad(state_ptr->getValueType(), state_ptr, "load_state");
  llvm::Value* state_value_new = builder->CreateAdd(
      state_value_old,
      llvm::ConstantInt::get(state_value_old->getType(), delta));
  builder->CreateStore(state_value_new, state_ptr);
  return state_value_old;
}

llvm::BasicBlock* EmitReturnBlock(llvm::IRBuilderBase* b) {
  llvm::Function* function = b->GetInsertBlock()->getParent();
  llvm::Module* module = b->GetInsertBlock()->getModule();
  llvm::IRBuilderBase::InsertPointGuard guard(*b);
  llvm::BasicBlock* early_return =
      llvm::BasicBlock::Create(/*Context=*/module->getContext(),
                               /*Name=*/"early_return",
                               /*Parent=*/function);
  b->SetInsertPoint(early_return);
  b->CreateRetVoid();
  return early_return;
}

void EmitEarlyReturn(llvm::Value* condition, llvm::IRBuilderBase* b,
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
