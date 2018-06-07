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

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Operator.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/casts.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace llvm_ir {

namespace {

// Note, this function is only useful in an insertion context; in a global
// (e.g. constants) context it will CHECK fail.
llvm::Module* ModuleFromIRBuilder(llvm::IRBuilder<>* ir_builder) {
  auto block = CHECK_NOTNULL(ir_builder->GetInsertBlock());
  auto fn = CHECK_NOTNULL(block->getParent());
  auto module = CHECK_NOTNULL(fn->getParent());
  return module;
}

}  // namespace

string AsString(const std::string& str) {
  return string(str.data(), str.length());
}

llvm::StringRef AsStringRef(tensorflow::StringPiece str) {
  return llvm::StringRef(str.data(), str.size());
}

std::unique_ptr<llvm::Module> DropConstantInitializers(
    const llvm::Module& module) {
  std::unique_ptr<llvm::Module> cloned_module = CloneModule(module);
  for (llvm::GlobalVariable& global_var : cloned_module->globals()) {
    global_var.setInitializer(nullptr);
    global_var.setLinkage(llvm::GlobalValue::LinkageTypes::ExternalLinkage);
  }
  return cloned_module;
}

string DumpModuleToString(const llvm::Module& module) {
  std::string buffer_string;
  llvm::raw_string_ostream ostream(buffer_string);
  module.print(ostream, nullptr);
  ostream.flush();
  return AsString(buffer_string);
}

llvm::Value* EmitCallToIntrinsic(
    llvm::Intrinsic::ID intrinsic_id,
    tensorflow::gtl::ArraySlice<llvm::Value*> operands,
    tensorflow::gtl::ArraySlice<llvm::Type*> overloaded_types,
    llvm::IRBuilder<>* ir_builder) {
  llvm::Module* module = ModuleFromIRBuilder(ir_builder);
  llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(
      module, intrinsic_id, AsArrayRef(overloaded_types));
  return ir_builder->CreateCall(intrinsic, AsArrayRef(operands));
}

llvm::Value* EmitFloatMax(llvm::Value* lhs_value, llvm::Value* rhs_value,
                          llvm::IRBuilder<>* ir_builder) {
  if (ir_builder->getFastMathFlags().noNaNs()) {
    auto cmp = ir_builder->CreateFCmpUGE(lhs_value, rhs_value);
    return ir_builder->CreateSelect(cmp, lhs_value, rhs_value);
  } else {
    auto cmp_ge = ir_builder->CreateFCmpOGE(lhs_value, rhs_value);
    auto lhs_is_nan = ir_builder->CreateFCmpUNE(lhs_value, lhs_value);
    auto sel_lhs = ir_builder->CreateOr(cmp_ge, lhs_is_nan);
    return ir_builder->CreateSelect(sel_lhs, lhs_value, rhs_value);
  }
}

llvm::Value* EmitFloatMin(llvm::Value* lhs_value, llvm::Value* rhs_value,
                          llvm::IRBuilder<>* ir_builder) {
  if (ir_builder->getFastMathFlags().noNaNs()) {
    auto cmp = ir_builder->CreateFCmpULE(lhs_value, rhs_value);
    return ir_builder->CreateSelect(cmp, lhs_value, rhs_value);
  } else {
    auto cmp_le = ir_builder->CreateFCmpOLE(lhs_value, rhs_value);
    auto lhs_is_nan = ir_builder->CreateFCmpUNE(lhs_value, lhs_value);
    auto sel_lhs = ir_builder->CreateOr(cmp_le, lhs_is_nan);
    return ir_builder->CreateSelect(sel_lhs, lhs_value, rhs_value);
  }
}

llvm::Value* EmitBufferIndexingGEP(llvm::Value* array, llvm::Value* index,
                                   llvm::IRBuilder<>* ir_builder) {
  llvm::Type* array_type = array->getType();
  CHECK(array_type->isPointerTy());
  llvm::PointerType* array_type_as_pointer =
      llvm::cast<llvm::PointerType>(array_type);
  VLOG(2) << "EmitBufferIndexingGEP with type="
          << llvm_ir::DumpToString(*array_type)
          << " array=" << llvm_ir::DumpToString(*array)
          << " index=" << llvm_ir::DumpToString(*index);

  return ir_builder->CreateInBoundsGEP(
      array_type_as_pointer->getElementType(), array,
      llvm::isa<llvm::GlobalVariable>(array)
          ? llvm::ArrayRef<llvm::Value*>({ir_builder->getInt64(0), index})
          : index);
}

llvm::Value* EmitBufferIndexingGEP(llvm::Value* array, int64 index,
                                   llvm::IRBuilder<>* ir_builder) {
  return EmitBufferIndexingGEP(array, ir_builder->getInt64(index), ir_builder);
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
      // addition to an addition on this type (int16) - this is just the type
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
      auto cplx_t = module->getTypeByName("complex64");
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
    // A Tuple contains an array of pointers. Use i8*.
    case TUPLE:
    // An Opaque is like a void*, use i8*.
    case OPAQUE:
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
  if (ShapeUtil::IsTuple(shape)) {
    // A tuple buffer is an array of pointers.
    result_type = llvm::ArrayType::get(result_type, shape.tuple_shapes_size());
  } else if (ShapeUtil::IsArray(shape)) {
    for (int64 dimension : LayoutUtil::MinorToMajor(shape)) {
      result_type =
          llvm::ArrayType::get(result_type, shape.dimensions(dimension));
    }
  }
  return result_type;
}

StatusOr<llvm::Value*> EncodeSelfDescribingShapeConstant(
    const Shape& shape, int32* shape_size, llvm::IRBuilder<>* ir_builder) {
  string encoded_shape = shape.SerializeAsString();
  if (encoded_shape.size() > std::numeric_limits<int32>::max()) {
    return InternalError("Encoded shape size exceeded int32 size limit.");
  }
  *shape_size = static_cast<int32>(encoded_shape.size());
  return ir_builder->CreateGlobalStringPtr(llvm_ir::AsStringRef(encoded_shape));
}

StatusOr<Shape> DecodeSelfDescribingShapeConstant(const void* shape_ptr,
                                                  int32 size_bytes) {
  Shape shape;
  TF_RET_CHECK(shape.ParseFromArray(shape_ptr, size_bytes));
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(shape));
  return shape;
}

namespace {

// Recursively construct a multidimensional LLVM constant which represents the
// given literal. The minor-to-major dimension ordering in the constant matches
// that of the literal. For example, given a [2 x 3 x 4] Literal (dimension 0
// has size 4, dimension 1 has size 3, etc) of primitive type F32 with a
// minor_to_major value of [2, 1, 0] (column major), a LLVM constant of type
// [4 x [3 x [2 x float]] will be returned.
//
// multi_index is a multidimensional index into the array. dimension_index is an
// index into the minor_to_major field in the literal shape. This determines
// which dimension is iterated over in this level of the recursion. Dimensions
// are iterated from most major down to most minor (highest dimension_index
// value down to zero).
llvm::Constant* LiteralToConstant(const Literal& literal, int64 dimension_index,
                                  std::vector<int64>* multi_index,
                                  llvm::Module* module) {
  const Shape& shape = literal.shape();
  llvm::Type* ir_element_type =
      llvm_ir::PrimitiveTypeToIrType(shape.element_type(), module);
  if (dimension_index == -1) {
    // Base case of the recursion. Index into the data field of the protobuf
    // with the multi index.
    llvm::Constant* value;
    switch (shape.element_type()) {
      case PRED:
        value = llvm::ConstantInt::get(ir_element_type,
                                       literal.Get<bool>(*multi_index));
        break;
      case U8:
        value = llvm::ConstantInt::get(ir_element_type,
                                       literal.Get<uint8>(*multi_index));
        break;
      case S32:
        value = llvm::ConstantInt::get(ir_element_type,
                                       literal.Get<int32>(*multi_index));
        break;
      case U32:
        value = llvm::ConstantInt::get(ir_element_type,
                                       literal.Get<uint32>(*multi_index));
        break;
      case S64:
        value = llvm::ConstantInt::get(ir_element_type,
                                       literal.Get<int64>(*multi_index));
        break;
      case U64:
        value = llvm::ConstantInt::get(ir_element_type,
                                       literal.Get<uint64>(*multi_index));
        break;
      case F32:
        value = llvm::ConstantFP::get(ir_element_type,
                                      literal.Get<float>(*multi_index));
        break;
      case BF16:
        value = llvm::ConstantInt::get(
            ir_element_type,
            tensorflow::bit_cast<uint16>(literal.Get<bfloat16>(*multi_index)));
        break;
      case F16:
        value = llvm::ConstantFP::get(
            ir_element_type,
            static_cast<float>(literal.Get<half>(*multi_index)));
        break;
      case F64:
        value = llvm::ConstantFP::get(ir_element_type,
                                      literal.Get<double>(*multi_index));
        break;
      case C64: {
        complex64 x = literal.Get<complex64>(*multi_index);
        value = llvm::ConstantStruct::get(
            static_cast<llvm::StructType*>(ir_element_type),
            llvm::ConstantFP::get(llvm_ir::PrimitiveTypeToIrType(F32, module),
                                  x.real()),
            llvm::ConstantFP::get(llvm_ir::PrimitiveTypeToIrType(F32, module),
                                  x.imag()));
        break;
      }
      default:
        LOG(FATAL) << "unsupported type " << shape.element_type();
    }
    return value;
  }

  // The dimension index starts at the one less than the rank of the array and
  // decrements with each recursive call. We want to iterate through the
  // dimensions in major-to-minor order as we recurse so just index into
  // minor_to_major to get the dimension number for this level of the recursion.
  int64 dimension = LayoutUtil::Minor(shape.layout(), dimension_index);

  // Recursively call LiteralToConstant to construct subarrays for the
  // more-minor dimensions. Gather the subarrays into a vector for bundling into
  // a new (higher-dimensional) ConstantArray.
  std::vector<llvm::Constant*> elements;
  for (int64 i = 0; i < shape.dimensions(dimension); ++i) {
    (*multi_index)[dimension] = i;
    elements.push_back(
        LiteralToConstant(literal, dimension_index - 1, multi_index, module));
  }

  llvm::Type* element_type;
  if (elements.empty()) {
    element_type = ir_element_type;
    for (int i = 0; i < dimension_index; ++i) {
      int64 index = LayoutUtil::Minor(shape.layout(), i);
      element_type =
          llvm::ArrayType::get(element_type, shape.dimensions(index));
    }
  } else {
    element_type = elements[0]->getType();
  }
  llvm::ArrayType* aggregate_type =
      llvm::ArrayType::get(element_type, shape.dimensions(dimension));
  return llvm::ConstantArray::get(aggregate_type, elements);
}

template <typename T>
llvm::Constant* GetConstantDataArray(const Literal& literal,
                                     llvm::Module* module) {
  const T* data = static_cast<const T*>(literal.untyped_data());
  int64 num_elements = literal.size_bytes() / sizeof(T);
  return llvm::ConstantDataArray::get(module->getContext(),
                                      llvm::makeArrayRef(data, num_elements));
}

}  // namespace

llvm::Constant* ConvertLiteralToIrConstant(const Literal& literal,
                                           llvm::Module* module) {
  const Shape& shape = literal.shape();
  // TODO(b/29904935): We can get rid of this switch by exposing a
  // ConstantDataArray factory method that takes a llvm::Type and a StringRef.
  switch (shape.element_type()) {
    case U64:
      return GetConstantDataArray<uint64>(literal, module);
    case U32:
      return GetConstantDataArray<uint32>(literal, module);
    case U8:
      return GetConstantDataArray<uint8>(literal, module);
    case S64:
      return GetConstantDataArray<int64>(literal, module);
    case S32:
      return GetConstantDataArray<int32>(literal, module);
    case F64:
      return GetConstantDataArray<double>(literal, module);
    case F32:
      return GetConstantDataArray<float>(literal, module);
    case BF16:
    case F16:
      return GetConstantDataArray<uint16>(literal, module);
    case PRED:
      return GetConstantDataArray<bool>(literal, module);
    // TODO(b/29904935): Also use ConstantDataArray for complex numbers.
    case C64: {
      int64 dimensions = ShapeUtil::Rank(shape);
      std::vector<int64> multi_index(dimensions, 0);
      return LiteralToConstant(literal, /*dimension_index=*/dimensions - 1,
                               &multi_index, module);
    }
    default:
      LOG(FATAL) << "unsupported type " << shape.element_type();
  }
}

llvm::AllocaInst* EmitAllocaAtFunctionEntry(llvm::Type* type,
                                            tensorflow::StringPiece name,
                                            llvm::IRBuilder<>* ir_builder,
                                            int alignment) {
  return EmitAllocaAtFunctionEntryWithCount(type, nullptr, name, ir_builder,
                                            alignment);
}

llvm::AllocaInst* EmitAllocaAtFunctionEntryWithCount(
    llvm::Type* type, llvm::Value* element_count, tensorflow::StringPiece name,
    llvm::IRBuilder<>* ir_builder, int alignment) {
  llvm::IRBuilder<>::InsertPoint insert_point = ir_builder->saveIP();
  llvm::Function* function = ir_builder->GetInsertBlock()->getParent();
  ir_builder->SetInsertPoint(&function->getEntryBlock(),
                             function->getEntryBlock().getFirstInsertionPt());
  llvm::AllocaInst* alloca =
      ir_builder->CreateAlloca(type, element_count, AsStringRef(name));
  if (alignment != 0) {
    alloca->setAlignment(alignment);
  }
  ir_builder->restoreIP(insert_point);
  return alloca;
}

llvm::BasicBlock* CreateBasicBlock(llvm::BasicBlock* insert_before,
                                   tensorflow::StringPiece name,
                                   llvm::IRBuilder<>* ir_builder) {
  return llvm::BasicBlock::Create(
      /*Context=*/ir_builder->getContext(),
      /*Name=*/AsStringRef(name),
      /*Parent=*/ir_builder->GetInsertBlock()->getParent(),
      /*InsertBefore*/ insert_before);
}

LlvmIfData EmitIfThenElse(llvm::Value* condition, tensorflow::StringPiece name,
                          llvm::IRBuilder<>* ir_builder, bool emit_else) {
  llvm_ir::LlvmIfData if_data;
  if_data.if_block = ir_builder->GetInsertBlock();
  if_data.true_block = CreateBasicBlock(
      nullptr, tensorflow::strings::StrCat(name, "-true"), ir_builder);
  if_data.false_block =
      emit_else ? CreateBasicBlock(nullptr,
                                   tensorflow::strings::StrCat(name, "-false"),
                                   ir_builder)
                : nullptr;

  // Add a terminator to the if block, if necessary.
  if (if_data.if_block->getTerminator() == nullptr) {
    ir_builder->SetInsertPoint(if_data.if_block);
    if_data.after_block = CreateBasicBlock(
        nullptr, tensorflow::strings::StrCat(name, "-after"), ir_builder);
    ir_builder->CreateBr(if_data.after_block);
  } else {
    if_data.after_block = if_data.if_block->splitBasicBlock(
        ir_builder->GetInsertPoint(),
        AsStringRef(tensorflow::strings::StrCat(name, "-after")));
  }

  // Our basic block should now end with an unconditional branch.  Remove it;
  // we're going to replace it with a conditional branch.
  if_data.if_block->getTerminator()->eraseFromParent();

  ir_builder->SetInsertPoint(if_data.if_block);
  ir_builder->CreateCondBr(
      condition, if_data.true_block,
      emit_else ? if_data.false_block : if_data.after_block);

  ir_builder->SetInsertPoint(if_data.true_block);
  ir_builder->CreateBr(if_data.after_block);

  if (emit_else) {
    ir_builder->SetInsertPoint(if_data.false_block);
    ir_builder->CreateBr(if_data.after_block);
  }

  ir_builder->SetInsertPoint(if_data.after_block,
                             if_data.after_block->getFirstInsertionPt());

  return if_data;
}

llvm::Value* EmitComparison(llvm::CmpInst::Predicate predicate,
                            llvm::Value* lhs_value, llvm::Value* rhs_value,
                            llvm::IRBuilder<>* ir_builder) {
  llvm::Value* comparison_result;
  if (lhs_value->getType()->isIntegerTy()) {
    comparison_result = ir_builder->CreateICmp(predicate, lhs_value, rhs_value);
  } else {
    comparison_result = ir_builder->CreateFCmp(predicate, lhs_value, rhs_value);
  }
  // comparison_result is i1, but the NVPTX codegen incorrectly lowers i1
  // arrays. So we extend it to i8 so that it's addressable.
  return ir_builder->CreateZExt(
      comparison_result,
      llvm_ir::PrimitiveTypeToIrType(PRED, ModuleFromIRBuilder(ir_builder)));
}

// Internal helper that is called from emitted code to log an int64 value with a
// tag.
static void LogS64(const char* tag, int64 value) {
  LOG(INFO) << tag << " (int64): " << value;
}

void EmitLogging(const char* tag, llvm::Value* value,
                 llvm::IRBuilder<>* ir_builder) {
  llvm::FunctionType* log_function_type = llvm::FunctionType::get(
      ir_builder->getVoidTy(),
      {ir_builder->getInt64Ty(), ir_builder->getInt64Ty()}, /*isVarArg=*/false);
  ir_builder->CreateCall(
      log_function_type,
      ir_builder->CreateIntToPtr(
          ir_builder->getInt64(tensorflow::bit_cast<int64>(&LogS64)),
          log_function_type->getPointerTo()),
      {ir_builder->getInt64(tensorflow::bit_cast<int64>(tag)), value});
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

llvm::Instruction* AddRangeMetadata(int64 lower, int64 upper,
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

string IrName(string a) {
  a.erase(std::remove(a.begin(), a.end(), '%'), a.end());
  return a;
}

string IrName(tensorflow::StringPiece a, tensorflow::StringPiece b) {
  if (!a.empty() && !b.empty()) {
    return IrName(tensorflow::strings::StrCat(a, ".", b));
  }
  return IrName(tensorflow::strings::StrCat(a, b));
}

string IrName(const HloInstruction* a, tensorflow::StringPiece b) {
  return IrName(a->name(), b);
}

string SanitizeFunctionName(string function_name) {
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

int64 ByteSizeOf(const Shape& shape, const llvm::DataLayout& data_layout) {
  unsigned pointer_size = data_layout.getPointerSize();
  return ShapeUtil::ByteSizeOf(shape, pointer_size);
}

llvm::FastMathFlags GetFastMathFlags(bool fast_math_enabled) {
  llvm::FastMathFlags flags;
  if (fast_math_enabled) {
    // Fast implies AllowReassoc, NoInfs, NoNaNs, NoSignedZeros,
    // AllowReciprocal, AllowContract, and ApproxFunc.
    flags.setFast();
  }
  return flags;
}

void SetTargetOptions(bool fast_math_enabled,
                      llvm::TargetOptions* target_options) {
  // In LLVM backend flags, UnsafeFPMath does not explicitly imply
  // NoInfs, etc.
  target_options->UnsafeFPMath = fast_math_enabled;
  target_options->NoInfsFPMath = fast_math_enabled;
  target_options->NoNaNsFPMath = fast_math_enabled;
  target_options->NoSignedZerosFPMath = fast_math_enabled;
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

static string GetProcessUniqueIrFileName(tensorflow::StringPiece prefix) {
  static tensorflow::mutex mu(tensorflow::LINKER_INITIALIZED);
  static NameUniquer* uniquer = new NameUniquer(/*separator=*/"-");

  tensorflow::mutex_lock lock(mu);
  return uniquer->GetUniqueName(prefix);
}

static Status CreateAndWriteStringToFile(const string& directory_name,
                                         const string& file_name,
                                         const string& text) {
  std::unique_ptr<tensorflow::WritableFile> f;
  TF_RETURN_IF_ERROR(
      tensorflow::Env::Default()->RecursivelyCreateDir(directory_name));
  TF_RETURN_IF_ERROR(
      tensorflow::Env::Default()->NewWritableFile(file_name, &f));
  TF_RETURN_IF_ERROR(f->Append(text));
  TF_RETURN_IF_ERROR(f->Close());
  return Status::OK();
}

Status DumpIRToDirectory(const string& directory_name,
                         const string& hlo_module_name,
                         const llvm::Module& llvm_module, bool optimized) {
  // We can end up compiling different modules with the same name when using
  // XlaJitCompiledCpuFunction::Compile.  Avoid overwriting IR files previously
  // dumped from the same process in such cases.
  string unique_and_safe_file_name = GetProcessUniqueIrFileName(
      tensorflow::strings::StrCat("ir-", SanitizeFileName(hlo_module_name), "-",
                                  optimized ? "with" : "no", "-opt"));

  string ir_file_name = tensorflow::io::JoinPath(
      directory_name,
      tensorflow::strings::StrCat(unique_and_safe_file_name, ".ll"));

  // For some models the embedded constants can be huge, so also dump the module
  // with the constants stripped to get IR that is easier to manipulate.
  string ir_no_constant_initializers_file_name = tensorflow::io::JoinPath(
      directory_name,
      tensorflow::strings::StrCat(unique_and_safe_file_name, "-noconst.ll"));

  TF_RETURN_IF_ERROR(CreateAndWriteStringToFile(
      directory_name, ir_file_name, DumpModuleToString(llvm_module)));
  return CreateAndWriteStringToFile(
      directory_name, ir_no_constant_initializers_file_name,
      DumpModuleToString(*DropConstantInitializers(llvm_module)));
}

llvm::Function* CreateFunction(llvm::FunctionType* function_type,
                               llvm::GlobalValue::LinkageTypes linkage,
                               bool enable_fast_math, bool optimize_for_size,
                               tensorflow::StringPiece name,
                               llvm::Module* module) {
  llvm::Function* function =
      llvm::Function::Create(function_type, linkage, AsStringRef(name), module);
  function->setCallingConv(llvm::CallingConv::C);
  function->addFnAttr("no-frame-pointer-elim", "false");

  if (enable_fast_math) {
    function->addFnAttr("unsafe-fp-math", "true");
    function->addFnAttr("no-infs-fp-math", "true");
    function->addFnAttr("no-nans-fp-math", "true");
    function->addFnAttr("no-signed-zeros-fp-math", "true");
  }

  // Add the optize attribute to the function if optimizing for size. This
  // controls internal behavior of some optimization passes (e.g. loop
  // unrolling).
  if (optimize_for_size) {
    function->addFnAttr(llvm::Attribute::OptimizeForSize);
  }

  return function;
}

void InitializeLLVMCommandLineOptions(const HloModuleConfig& config) {
  auto options = config.debug_options().xla_backend_extra_options();
  if (!options.empty()) {
    std::vector<string> fake_argv_storage;
    fake_argv_storage.push_back("");
    for (const auto& it : options) {
      // Skip options the XLA backend itself consumes.
      if (!tensorflow::str_util::StartsWith(it.first, "xla_")) {
        if (it.second.empty()) {
          fake_argv_storage.push_back(it.first);
        } else {
          fake_argv_storage.push_back(it.first + "=" + it.second);
        }
      }
    }

    VLOG(2) << "Passing argv to LLVM:";
    std::vector<const char*> fake_argv;
    for (const auto& s : fake_argv_storage) {
      fake_argv.push_back(s.c_str());
      VLOG(2) << s;
    }
    llvm::cl::ParseCommandLineOptions(fake_argv.size(), &fake_argv[0]);
  }
}

}  // namespace llvm_ir
}  // namespace xla
