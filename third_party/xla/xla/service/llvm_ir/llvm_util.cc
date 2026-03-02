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
#include <cstddef>
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
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FloatingPointMode.h"
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
#include "llvm/Support/TypeSize.h"
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
#include "xla/tsl/platform/byte_order.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace xla {
namespace llvm_ir {

namespace {

// Note, this function is only useful in an insertion context; in a global
// (e.g. constants) context it will CHECK fail.
llvm::Module* ModuleFromIRBuilder(llvm::IRBuilderBase* b) {
  auto block = CHECK_NOTNULL(b->GetInsertBlock());
  auto fn = CHECK_NOTNULL(block->getParent());
  auto module = CHECK_NOTNULL(fn->getParent());
  return module;
}

PrimitiveType PrimitiveTypeFromIrIntegerType(
    llvm::IntegerType* type, bool default_to_signed_for_integers) {
  // PRED (boolean) is typically a 1-bit integer.
  if (type->getBitWidth() == 1) {
    return PRED;
  }

  // LLVM's llvm::IntegerType (e.g., i8, i32) does not distinguish between
  // signed and unsigned types by itself. The interpretation (signed/unsigned)
  // depends on the operations using these types (e.g., sdiv vs. udiv).
  // The 'default_to_signed_for_integers' flag helps make a choice here.
  switch (type->getBitWidth()) {
    case 8:
      return default_to_signed_for_integers ? S8 : U8;
    case 16:
      return default_to_signed_for_integers ? S16 : U16;
    case 32:
      return default_to_signed_for_integers ? S32 : U32;
    case 64:
      return default_to_signed_for_integers ? S64 : U64;
    default:
      return PRIMITIVE_TYPE_INVALID;
  }
}

std::optional<PrimitiveType> PrimitiveComplexTypeFromIrStructType(
    llvm::StructType* struct_type) {
  // XLA C64 is typically represented as an LLVM struct {float, float}.
  // XLA C128 is typically represented as an LLVM struct {double, double}.
  if (struct_type->getNumElements() == 2) {
    llvm::Type* el_type0 = struct_type->getElementType(0);
    llvm::Type* el_type1 = struct_type->getElementType(1);
    if (el_type0->isFloatTy() && el_type1->isFloatTy()) {
      return C64;  // Complex64
    }
    if (el_type0->isDoubleTy() && el_type1->isDoubleTy()) {
      return C128;  // Complex128
    }
  }
  return std::nullopt;
}

}  // namespace

llvm::CallInst* EmitCallToIntrinsic(
    llvm::Intrinsic::ID intrinsic_id, absl::Span<llvm::Value* const> operands,
    absl::Span<llvm::Type* const> overloaded_types, llvm::IRBuilderBase* b,
    absl::string_view name) {
  llvm::Module* module = ModuleFromIRBuilder(b);
  llvm::Function* intrinsic = llvm::Intrinsic::getOrInsertDeclaration(
      module, intrinsic_id, AsArrayRef(overloaded_types));
  return b->CreateCall(intrinsic, AsArrayRef(operands), AsStringRef(name));
}

llvm::Value* EmitFloatMax(llvm::Value* lhs_value, llvm::Value* rhs_value,
                          llvm::IRBuilderBase* b, bool enable_fast_min_max,
                          absl::string_view name) {
  if (b->getFastMathFlags().noNaNs() || enable_fast_min_max) {
    auto cmp = b->CreateFCmpUGE(lhs_value, rhs_value);
    return b->CreateSelect(cmp, lhs_value, rhs_value, AsStringRef(name));
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
    return b->CreateSelect(cmp, lhs_value, rhs_value, AsStringRef(name));
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
    case F4E2M1FN:
      return llvm::Type::getIntNTy(context, 4);
    case F8E5M2:
    case F8E5M2FNUZ:
    case F8E4M3:
    case F8E4M3FN:
    case F8E4M3B11FNUZ:
    case F8E4M3FNUZ:
    case F8E3M4:
    case F8E8M0FNU:
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

PrimitiveType PrimitiveTypeFromIrType(llvm::Type* type,
                                      bool default_to_signed_for_integers) {
  if (!type) {
    return PRIMITIVE_TYPE_INVALID;
  }

  // If it's a vector type, XLA PrimitiveType refers to the element type.
  // So, we get the underlying element type for further checks.
  if (type->isVectorTy()) {
    type = llvm::cast<llvm::VectorType>(type)->getElementType();
  }

  // Floating-point types
  if (type->isHalfTy()) {
    return F16;
  }
  if (type->isBFloatTy()) {
    return BF16;
  }
  if (type->isFloatTy()) {
    return F32;
  }
  if (type->isDoubleTy()) {
    return F64;
  }

  if (type->isIntegerTy()) {
    return PrimitiveTypeFromIrIntegerType(llvm::cast<llvm::IntegerType>(type),
                                          default_to_signed_for_integers);
  }

  if (type->isStructTy()) {
    if (auto result = PrimitiveComplexTypeFromIrStructType(
            llvm::cast<llvm::StructType>(type))) {
      return *result;
    }
  }

  if (type->isPointerTy()) {
    return OPAQUE_TYPE;
  }

  return PRIMITIVE_TYPE_INVALID;
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
    result_type =
        llvm::ArrayType::get(result_type, shape.tuple_shapes().size());
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
  const std::string encoded_shape = shape.ToProto().SerializeAsString();
  if (encoded_shape.size() > std::numeric_limits<int32_t>::max()) {
    return Internal("Encoded shape size exceeded int32_t size limit.");
  }
  *shape_size = static_cast<int32_t>(encoded_shape.size());
  return b->CreateGlobalString(encoded_shape);
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
        b->CreateICmp(predicate, lhs_value, rhs_value, AsStringRef(name));
  } else {
    comparison_result =
        b->CreateFCmp(predicate, lhs_value, rhs_value, AsStringRef(name));
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
  absl::c_transform(function_name, function_name.begin(), [](char c) {
    if (absl::ascii_isalnum(c) || c == '_' || c == '$') {
      return c;
    }
    return '_';
  });

  // Ensure the name isn't empty.
  if (function_name.empty()) {
    function_name = "__unnamed";
  }

  // Ensure the name doesn't start with a number.
  if (!function_name.empty() && absl::ascii_isdigit(function_name[0])) {
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
      absl::StrCat(filename_suffix, filename_suffix.empty() ? "" : ".", "ir-",
                   optimized ? "with" : "no", "-opt");
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

absl::StatusOr<llvm::Value*> EmitReducePrecisionIR(
    PrimitiveType src_ty, llvm::Value* x, int64_t dest_exponent_bits,
    int64_t dest_mantissa_bits, bool quiet_nans, llvm::IRBuilderBase* b) {
  using llvm::APInt;

  if (!primitive_util::IsFloatingPointType(src_ty)) {
    return Unimplemented(
        "ReducePrecision cannot accept non-floating-point type %s.",
        PrimitiveType_Name(src_ty));
  }

  // Integer and float types for casting and constant generation.
  llvm::Type* const value_type = x->getType();
  llvm::Type* const float_scalar_type = value_type->getScalarType();
  const int nbits = float_scalar_type->getPrimitiveSizeInBits();
  llvm::Type* int_work_type = b->getIntNTy(nbits);
  unsigned width = 1;
  if (auto* vec_ty = llvm::dyn_cast<llvm::FixedVectorType>(value_type)) {
    width = vec_ty->getNumElements();
    int_work_type = llvm::VectorType::get(int_work_type,
                                          llvm::ElementCount::getFixed(width));
  }

  // Helper to create a splatted vector constant. If the input is scalar, this
  // will just produce a scalar ConstantInt.
  auto int_const = [&](const APInt& val) -> llvm::Constant* {
    return llvm::ConstantInt::get(int_work_type, val);
  };

  // SignificandWidth includes the implicit extra bit.
  int src_mantissa_bits = primitive_util::SignificandWidth(src_ty) - 1;
  int src_exponent_bits = nbits - 1 - src_mantissa_bits;

  // Cast the input value to an integer for bitwise manipulation.
  llvm::Value* x_as_int = b->CreateBitCast(x, int_work_type);

  // Clear the sign bit, it does not participate in rounding and we will restore
  // it later.
  APInt sign_bit_mask(nbits, 1);
  sign_bit_mask <<= nbits - 1;
  llvm::Value* x_abs_bits = b->CreateAnd(x_as_int, int_const(~sign_bit_mask));

  APInt exp_bits_mask(nbits, 1);
  exp_bits_mask = ((exp_bits_mask << src_exponent_bits) - 1)
                  << src_mantissa_bits;
  auto x_is_nan = b->CreateICmpUGT(x_abs_bits, int_const(exp_bits_mask));

  if (dest_mantissa_bits < src_mantissa_bits) {
    // Last remaining mantissa bit.
    APInt last_mantissa_bit_mask(nbits, 1);
    last_mantissa_bit_mask <<= src_mantissa_bits - dest_mantissa_bits;

    // Compute rounding bias for round-to-nearest with ties to even.  This is
    // equal to a base value of 0111... plus one bit if the last remaining
    // mantissa bit is 1.
    APInt base_rounding_bias = last_mantissa_bit_mask.lshr(1) - 1;
    llvm::Value* x_last_mantissa_bit =
        b->CreateLShr(b->CreateAnd(x_as_int, int_const(last_mantissa_bit_mask)),
                      (src_mantissa_bits - dest_mantissa_bits));
    llvm::Value* x_rounding_bias =
        b->CreateAdd(x_last_mantissa_bit, int_const(base_rounding_bias));

    // Add rounding bias, and mask out truncated bits.  Note that the case
    // where adding the rounding bias overflows into the exponent bits is
    // correct; the non-masked mantissa bits will all be zero, and the
    // exponent will be incremented by one.
    APInt truncation_mask = ~(last_mantissa_bit_mask - 1);
    llvm::Value* x_rounded = b->CreateAdd(x_as_int, x_rounding_bias);
    x_rounded = b->CreateAnd(x_rounded, int_const(truncation_mask));
    if (quiet_nans) {
      x_as_int = b->CreateSelect(x_is_nan, x_as_int, x_rounded);
    } else {
      x_as_int = x_rounded;
    }
  }

  if (dest_exponent_bits < src_exponent_bits) {
    // An exponent of 2^(n-1)-1 -- that is, 0111... with the zero in the most-
    // significant bit -- is equal to 1.0f for all exponent sizes.  Adding
    // 2^(n-1)-1 to this gives us the highest non-infinite exponent for a bit-
    // size of n, and subtracting 2^(n-1)-1 from this gives us the lowest'
    // exponent (corresponding to 0.0f).
    //
    // Thus, the f32 exponent corresponding to the highest non-infinite
    // exponent for a bit size of n is (2^7-1) + 2^(n-1)-1, and the f32
    // exponent corresponding to the lowest exponent for a bit size of n is
    // (2^7-1) - 2^(n-1)-1.
    //
    // Note that we have already checked that exponents_bits >= 1.
    APInt exponent_bias(nbits, 1);
    exponent_bias = (exponent_bias << (src_exponent_bits - 1)) - 1;

    APInt reduced_exponent_bias(nbits, 1);
    reduced_exponent_bias =
        (reduced_exponent_bias << (dest_exponent_bits - 1)) - 1;

    APInt reduced_max_exponent = exponent_bias + reduced_exponent_bias;
    APInt reduced_min_exponent = exponent_bias - reduced_exponent_bias;

    // Do we overflow or underflow?
    llvm::Value* x_exponent = b->CreateAnd(x_as_int, int_const(exp_bits_mask));
    llvm::Value* x_overflows = b->CreateICmpUGT(
        x_exponent, int_const(reduced_max_exponent << src_mantissa_bits));
    llvm::Value* x_underflows = b->CreateICmpULE(
        x_exponent, int_const(reduced_min_exponent << src_mantissa_bits));

    // Compute appropriately-signed values of zero and infinity.
    llvm::Value* x_signed_zero =
        b->CreateAnd(x_as_int, int_const(sign_bit_mask));
    llvm::Value* x_signed_inf =
        b->CreateOr(x_signed_zero, int_const(exp_bits_mask));

    // Force to zero or infinity if overflow or underflow.  (Note that this
    // truncates all denormal values to zero, rather than rounding them.)
    x_as_int = b->CreateSelect(x_overflows, x_signed_inf, x_as_int);
    x_as_int = b->CreateSelect(x_underflows, x_signed_zero, x_as_int);
  }

  // Cast the result back to a floating-point type.
  llvm::Value* result = b->CreateBitCast(x_as_int, value_type);

  // Correct result for NaN inputs.
  //
  // The exponent handling will "normalize" NaN values to infinities, which is
  // undesirable (except in the case with no mantissa bits, in which case it
  // is mandatory).  This logic also handles cases where mantissa-rounding
  // causes a NaN's mantissa to overflow into the exponent bits, which would
  // otherwise create an erroneous zero value.

  if (dest_mantissa_bits > 0) {
    if (quiet_nans) {
      APInt qnan_mask(nbits, 1);
      qnan_mask <<= src_mantissa_bits - 1;
      llvm::Value* x_with_qnan_bit_set =
          b->CreateOr(x_as_int, int_const(qnan_mask));
      x_with_qnan_bit_set = b->CreateBitCast(x_with_qnan_bit_set, value_type);
      result = b->CreateSelect(x_is_nan, x_with_qnan_bit_set, result);
    } else {
      result = b->CreateSelect(x_is_nan, x, result);
    }
  } else {
    result = b->CreateSelect(x_is_nan,
                             llvm::ConstantFP::getInfinity(value_type), result);
  }

  return result;
}

llvm::Value* HandleHalfwayPointsFxToF8(
    PrimitiveType fx_type, int f8_exponent_bits, int f8_mantissa_bits,
    int f8_bias, llvm::Value* fx_abs_bits, llvm::Value* f8_bits,
    std::optional<size_t> vector_width, llvm::IRBuilderBase* b) {
  using llvm::APFloat;
  using llvm::APInt;
  using llvm::Value;
  CHECK(fx_type == F16 || fx_type == F32 || fx_type == F64);

  const llvm::fltSemantics* fx_semantics;
  llvm::Type* fx_float_type = PrimitiveTypeToIrType(fx_type, b->getContext());
  llvm::Type* scale_factor_type = fx_float_type;

  if (fx_type == F16) {
    // Scale factor can be > 2^17, which overflows F16.
    fx_semantics = &llvm::APFloat::IEEEsingle();
    scale_factor_type = b->getFloatTy();
  } else if (fx_type == F32) {
    fx_semantics = &llvm::APFloat::IEEEsingle();
  } else if (fx_type == F64) {
    fx_semantics = &llvm::APFloat::IEEEdouble();
  } else {
    LOG(FATAL) << "Unsupported FX type: " << fx_type;
  }

  // Get the input/output types, accounting for vectors.
  llvm::Type* ix_type = fx_abs_bits->getType();
  llvm::Type* i8_type = f8_bits->getType();

  if (vector_width.has_value()) {
    auto vec_width = llvm::ElementCount::getFixed(*vector_width);
    fx_float_type = llvm::VectorType::get(fx_float_type, vec_width);
    scale_factor_type = llvm::VectorType::get(scale_factor_type, vec_width);
  }
  llvm::RoundingMode rm = llvm::RoundingMode::NearestTiesToEven;

  auto fp_const = [&](APFloat val) {
    bool losesInfo;
    val.convert(*fx_semantics, rm, &losesInfo);
    return llvm::ConstantFP::get(scale_factor_type, val);
  };

  const int num_subnormal_steps = 1 << f8_mantissa_bits;
  const int smallest_normal_exp = 1 - f8_bias;
  const int quantum_exponent = smallest_normal_exp - f8_mantissa_bits;

  // Create the scaling factor constant: 2^(-quantum_exponent)
  // e.g., for B11 (quantum_exp = -13), this is 2^13, or 8192.0.
  APFloat scale_apfloat = scalbn(APFloat(1.0), -quantum_exponent, rm);

  // Create the upper boundary constant: (num_steps - 0.5) * quantum
  // This is the halfway point for the *largest* subnormal step (e.g., 7.5 *
  // q).
  APFloat quantum = scalbn(APFloat(1.0), quantum_exponent, rm);

  APFloat num_steps_apfloat(static_cast<double>(num_subnormal_steps));
  APFloat half_apfloat(0.5);

  APFloat upper_bound_apfloat = num_steps_apfloat;
  upper_bound_apfloat.subtract(half_apfloat, rm);
  upper_bound_apfloat.multiply(quantum, rm);

  Value* scale_factor = fp_const(scale_apfloat);
  Value* upper_bound_constant = fp_const(upper_bound_apfloat);
  Value* input_float = b->CreateBitCast(fx_abs_bits, fx_float_type);
  input_float = b->CreateFPExt(input_float, scale_factor_type);

  // Check if the input is below the subnormal range boundary.
  // Anything >= 7.5q (for an M3 format) is a normal number and should
  // use the default 'f8_bits' value passed into this function.
  Value* is_subnormal_candidate =
      b->CreateFCmpOLT(input_float, upper_bound_constant);

  // --- Subnormal Path ---
  // Apply the rounding formula: i = round_to_even(input_float * scale_factor)
  Value* scaled = b->CreateFMul(input_float, scale_factor);

  // Use llvm.nearbyint, which rounds to the nearest integer using
  // ties-to-even.
  llvm::Module* module = b->GetInsertBlock()->getModule();
  llvm::Function* nearbyint = llvm::Intrinsic::getOrInsertDeclaration(
      module, llvm::Intrinsic::nearbyint, {scaled->getType()});
  Value* rounded = b->CreateCall(nearbyint, scaled);

  // Convert the rounded float result to its integer bucket index.
  Value* int_bucket = b->CreateFPToUI(rounded, ix_type);

  // Truncate the index (which is i32 or i64) down to our final i8 value.
  Value* subnormal_result = b->CreateTrunc(int_bucket, i8_type);

  // --- Final Select ---
  // If it was a subnormal candidate, use our calculated result.
  // Otherwise, use the original 'f8_bits' value (the default normal/inf/nan).
  Value* final_result =
      b->CreateSelect(is_subnormal_candidate, subnormal_result, f8_bits);

  return final_result;
}

}  // namespace llvm_ir
}  // namespace xla
