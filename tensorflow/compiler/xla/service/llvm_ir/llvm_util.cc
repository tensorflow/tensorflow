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
#include <vector>

#include "external/llvm/include/llvm/IR/MDBuilder.h"
#include "external/llvm/include/llvm/IR/Operator.h"
#include "external/llvm/include/llvm/Target/TargetOptions.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/legacy_flags/llvm_util_flags.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/casts.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace llvm_ir {

string AsString(const std::string& str) {
  return string(str.data(), str.length());
}

llvm::StringRef AsStringRef(tensorflow::StringPiece str) {
  return llvm::StringRef(str.data(), str.size());
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
  std::vector<llvm::Type*> types;
  for (auto type : overloaded_types) {
    types.push_back(type);
  }
  llvm::Module* module = ir_builder->GetInsertBlock()->getParent()->getParent();
  llvm::Function* intrinsic =
      llvm::Intrinsic::getDeclaration(module, intrinsic_id, types);
  std::vector<llvm::Value*> operands_vec;
  for (auto operand : operands) {
    operands_vec.push_back(operand);
  }
  return ir_builder->CreateCall(intrinsic, operands_vec);
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
                                  llvm::IRBuilder<>* ir_builder) {
  switch (element_type) {
    case PRED:
    case S8:
    case U8:
      return ir_builder->getInt8Ty();
    case S16:
    case U16:
      return ir_builder->getInt16Ty();
    case S32:
    case U32:
      return ir_builder->getInt32Ty();
    case S64:
    case U64:
      return ir_builder->getInt64Ty();
    case F32:
      return ir_builder->getFloatTy();
    case F64:
      return ir_builder->getDoubleTy();
    // A Tuple contains an array of pointers. Use i8*.
    case TUPLE:
    // An Opaque is like a void*, use i8*.
    case OPAQUE:
      return ir_builder->getInt8PtrTy();
    default:
      LOG(FATAL) << "unsupported type " << element_type;
  }
}

llvm::Type* ShapeToIrType(const Shape& shape, llvm::IRBuilder<>* ir_builder) {
  llvm::Type* result_type =
      PrimitiveTypeToIrType(shape.element_type(), ir_builder);
  if (ShapeUtil::IsTuple(shape)) {
    // A tuple buffer is an array of pointers.
    result_type = llvm::ArrayType::get(result_type, shape.tuple_shapes_size());
  } else {
    for (int64 dimension : shape.layout().minor_to_major()) {
      result_type =
          llvm::ArrayType::get(result_type, shape.dimensions(dimension));
    }
  }
  return result_type;
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
                                  llvm::IRBuilder<>* ir_builder) {
  const Shape& shape = literal.shape();
  llvm::Type* ir_element_type =
      llvm_ir::PrimitiveTypeToIrType(shape.element_type(), ir_builder);
  if (dimension_index == -1) {
    // Base case of the recursion. Index into the data field of the protobuf
    // with the multi index.
    llvm::Constant* value;
    switch (shape.element_type()) {
      case PRED:
        value = llvm::ConstantInt::get(
            ir_element_type, LiteralUtil::Get<bool>(literal, *multi_index));
        break;
      case U8:
        value = llvm::ConstantInt::get(
            ir_element_type, LiteralUtil::Get<uint8>(literal, *multi_index));
        break;
      case S32:
        value = llvm::ConstantInt::get(
            ir_element_type, LiteralUtil::Get<int32>(literal, *multi_index));
        break;
      case U32:
        value = llvm::ConstantInt::get(
            ir_element_type, LiteralUtil::Get<uint32>(literal, *multi_index));
        break;
      case S64:
        value = llvm::ConstantInt::get(
            ir_element_type, LiteralUtil::Get<int64>(literal, *multi_index));
        break;
      case U64:
        value = llvm::ConstantInt::get(
            ir_element_type, LiteralUtil::Get<uint64>(literal, *multi_index));
        break;
      case F32:
        value = llvm::ConstantFP::get(
            ir_element_type, LiteralUtil::Get<float>(literal, *multi_index));
        break;
      case F64:
        value = llvm::ConstantFP::get(
            ir_element_type, LiteralUtil::Get<double>(literal, *multi_index));
        break;
      default:
        LOG(FATAL) << "unsupported type " << shape.element_type();
    }
    return value;
  }

  // The dimension index starts at the one less than the rank of the array and
  // decrements with each recursive call. We want to iterate through the
  // dimensions in major-to-minor order as we recurse so just index into
  // minor_to_major to get the dimension number for this level of the recursion.
  int64 dimension = shape.layout().minor_to_major(dimension_index);

  // Recursively call LiteralToConstant to construct subarrays for the
  // more-minor dimensions. Gather the subarrays into a vector for bundling into
  // a new (higher-dimensional) ConstantArray.
  std::vector<llvm::Constant*> elements;
  for (int64 i = 0; i < shape.dimensions(dimension); ++i) {
    (*multi_index)[dimension] = i;
    elements.push_back(LiteralToConstant(literal, dimension_index - 1,
                                         multi_index, ir_builder));
  }

  llvm::Type* element_type;
  if (elements.empty()) {
    element_type = ir_element_type;
    for (int i = 0; i < dimension_index; ++i) {
      int64 index = shape.layout().minor_to_major(i);
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

}  // namespace

llvm::Constant* ConvertLiteralToIrConstant(const Literal& literal,
                                           llvm::IRBuilder<>* ir_builder) {
  std::vector<int64> multi_index(ShapeUtil::Rank(literal.shape()), 0);
  llvm::Constant* value = LiteralToConstant(
      literal, /*dimension_index=*/ShapeUtil::Rank(literal.shape()) - 1,
      &multi_index, ir_builder);
  return value;
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

  // There is no reason this function cannot work without a
  // terminator, that is just a different case that has not been
  // implemented yet. It is a different case because splitBasicBlock
  // requires a terminator.
  CHECK_NE(nullptr, if_data.if_block->getTerminator());
  if_data.after_block = if_data.if_block->splitBasicBlock(
      ir_builder->GetInsertPoint(),
      AsStringRef(tensorflow::strings::StrCat(name, "-after")));

  // splitBasicBlock inserts an unconditional terminator that we have
  // to remove as we want a conditional branch there.
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
      comparison_result, llvm_ir::PrimitiveTypeToIrType(PRED, ir_builder));
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

void SetTbaaForInstruction(llvm::Instruction* instruction, Shape shape,
                           bool is_pointer_to) {
  legacy_flags::LlvmUtilFlags* flags = legacy_flags::GetLlvmUtilFlags();
  if (!flags->xla_emit_tbaa) {
    return;
  }

  llvm::MDBuilder metadata_builder(instruction->getContext());
  llvm::MDNode* root = metadata_builder.createTBAARoot("XLA TBAA");
  string type_name;
  if (is_pointer_to) {
    type_name += "pointer-to ";
  }
  // Scalars do not have layout which makes it permissible to omit an explicit
  // layout.  To make sure that equivalent scalar shapes have the same TBAA,
  // remove the (meaningless) explicit layout if one is present.
  if (ShapeUtil::Rank(shape) == 0) {
    LayoutUtil::ClearLayout(&shape);
  } else {
    CHECK(shape.has_layout());
  }
  type_name += shape.ShortDebugString();
  llvm::MDNode* tbaa_node =
      metadata_builder.createTBAANode(llvm_ir::AsStringRef(type_name), root);
  instruction->setMetadata(llvm::LLVMContext::MD_tbaa,
                           metadata_builder.createTBAAStructTagNode(
                               tbaa_node, tbaa_node, /*Offset=*/0));
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

string SanitizeIrName(string function_name) {
  // Replace some characters that cannot occur in LLVM names with '_'
  std::replace(function_name.begin(), function_name.end(), '.', '_');
  std::replace(function_name.begin(), function_name.end(), '%', '_');
  std::replace(function_name.begin(), function_name.end(), '-', '_');
  return function_name;
}

void SetToFirstInsertPoint(llvm::BasicBlock* blk, llvm::IRBuilder<>* builder) {
  builder->SetInsertPoint(blk, blk->getFirstInsertionPt());
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

llvm::FastMathFlags GetFastMathFlags(const HloModuleConfig& config) {
  llvm::FastMathFlags flags;
  if (!config.fast_math_disabled()) {
    // UnsafeAlgebra implies NoInfs, NoNaNs, NoSignedZeros, and AllowReciprocal.
    flags.setUnsafeAlgebra();
  }
  return flags;
}

void SetTargetOptions(const HloModuleConfig& config,
                      llvm::TargetOptions* target_options) {
  bool fast = !config.fast_math_disabled();
  // In LLVM backend flags, UnsafeFPMath does not explicitly imply
  // NoInfs, etc.
  target_options->UnsafeFPMath = fast;
  target_options->NoInfsFPMath = fast;
  target_options->NoNaNsFPMath = fast;
  target_options->NoSignedZerosFPMath = fast;
  target_options->LessPreciseFPMADOption = fast;
}

}  // namespace llvm_ir
}  // namespace xla
