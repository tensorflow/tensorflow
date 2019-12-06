//===-- LLVMOps.td - LLVM IR dialect op definition file ----*- tablegen -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This is the LLVM IR operation definition file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVMIR_OPS
#define LLVMIR_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

// Base class for LLVM operations.  All operations get an "llvm." prefix in
// their name automatically.  LLVM operations have either zero or one result,
// this class is specialized below for both cases and should not be used
// directly.
class LLVM_Op<string mnemonic, list<OpTrait> traits = []> :
    LLVM_OpBase<LLVM_Dialect, mnemonic, traits> {
}

class LLVM_Builder<string builder> {
  string llvmBuilder = builder;
}

def LLVM_OneResultOpBuilder : OpBuilder<
  "Builder *, OperationState &result, Type resultType, "
  "ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes = {}",
  [{
    if (resultType) result.addTypes(resultType);
    result.addOperands(operands);
    for (auto namedAttr : attributes) {
      result.addAttribute(namedAttr.first, namedAttr.second);
    }
  }]>;

def LLVM_ZeroResultOpBuilder : OpBuilder<
  "Builder *, OperationState &result, ArrayRef<Value *> operands, "
  "ArrayRef<NamedAttribute> attributes = {}",
  [{
    result.addOperands(operands);
    for (auto namedAttr : attributes) {
      result.addAttribute(namedAttr.first, namedAttr.second);
    }
  }]>;

class LLVM_TwoBuilders<OpBuilder b1, OpBuilder b2> {
  list<OpBuilder> builders = [b1, b2];
}

// Base class for LLVM operations with one result.
class LLVM_OneResultOp<string mnemonic, list<OpTrait> traits = []> :
    LLVM_Op<mnemonic, traits>, Results<(outs LLVM_Type:$res)> {
  let builders = [LLVM_OneResultOpBuilder];
}

// Compatibility builder that takes an instance of wrapped llvm::VoidType
// to indicate no result.
def LLVM_VoidResultTypeOpBuilder : OpBuilder<
  "Builder *builder, OperationState &result, Type resultType, "
  "ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes = {}",
  [{
    auto llvmType = resultType.dyn_cast<LLVM::LLVMType>(); (void)llvmType;
    assert(llvmType && "result must be an LLVM type");
    assert(llvmType.getUnderlyingType() &&
            llvmType.getUnderlyingType()->isVoidTy() &&
            "for zero-result operands, only 'void' is accepted as result type");
    build(builder, result, operands, attributes);
  }]>;

// Base class for LLVM operations with zero results.
class LLVM_ZeroResultOp<string mnemonic, list<OpTrait> traits = []> :
    LLVM_Op<mnemonic, traits>, Results<(outs)>,
    LLVM_TwoBuilders<LLVM_VoidResultTypeOpBuilder, LLVM_ZeroResultOpBuilder>;

// Base class for LLVM terminator operations.  All terminator operations have
// zero results and an optional list of successors.
class LLVM_TerminatorOp<string mnemonic, list<OpTrait> traits = []> :
    LLVM_Op<mnemonic, !listconcat(traits, [Terminator])>,
    Arguments<(ins Variadic<LLVM_Type>:$args)>, Results<(outs)> {
  let builders = [
    OpBuilder<
      "Builder *, OperationState &result, "
      "ArrayRef<Value *> properOperands, "
      "ArrayRef<Block *> destinations, "
      "ArrayRef<ArrayRef<Value *>> operands, "
      "ArrayRef<NamedAttribute> attributes = {}",
      [{
        result.addOperands(properOperands);
        for (auto kvp : llvm::zip(destinations, operands)) {
          result.addSuccessor(std::get<0>(kvp), std::get<1>(kvp));
        }
        for (auto namedAttr : attributes) {
          result.addAttribute(namedAttr.first, namedAttr.second);
        }
      }]
    >,
    OpBuilder<
      "Builder *builder, OperationState &result, "
      "ArrayRef<Value *> properOperands, "
      "ArrayRef<Block *> destinations, "
      "ArrayRef<NamedAttribute> attributes = {}",
    [{
        SmallVector<ArrayRef<Value *>, 2> operands(destinations.size(), {});
        build(builder, result, properOperands,
            destinations, operands, attributes);
      }]
    >,
  ];
}

// Class for arithmetic binary operations.
class LLVM_ArithmeticOp<string mnemonic, string builderFunc,
                        list<OpTrait> traits = []> :
    LLVM_OneResultOp<mnemonic,
           !listconcat([NoSideEffect, SameOperandsAndResultType], traits)>,
    Arguments<(ins LLVM_Type:$lhs, LLVM_Type:$rhs)>,
    LLVM_Builder<"$res = builder." # builderFunc # "($lhs, $rhs);"> {
  let parser = [{ return impl::parseOneResultSameOperandTypeOp(parser, result); }];
  let printer = [{ mlir::impl::printOneResultOp(this->getOperation(), p); }];
}
class LLVM_UnaryArithmeticOp<string mnemonic, string builderFunc,
                        list<OpTrait> traits = []> :
    LLVM_OneResultOp<mnemonic,
           !listconcat([NoSideEffect, SameOperandsAndResultType], traits)>,
    Arguments<(ins LLVM_Type:$operand)>,
    LLVM_Builder<"$res = builder." # builderFunc # "($operand);"> {
  let parser = [{ return impl::parseOneResultSameOperandTypeOp(parser, result); }];
  let printer = [{ mlir::impl::printOneResultOp(this->getOperation(), p); }];
}

// Integer binary operations.
def LLVM_AddOp : LLVM_ArithmeticOp<"add", "CreateAdd", [Commutative]>;
def LLVM_SubOp : LLVM_ArithmeticOp<"sub", "CreateSub">;
def LLVM_MulOp : LLVM_ArithmeticOp<"mul", "CreateMul", [Commutative]>;
def LLVM_UDivOp : LLVM_ArithmeticOp<"udiv", "CreateUDiv">;
def LLVM_SDivOp : LLVM_ArithmeticOp<"sdiv", "CreateSDiv">;
def LLVM_URemOp : LLVM_ArithmeticOp<"urem", "CreateURem">;
def LLVM_SRemOp : LLVM_ArithmeticOp<"srem", "CreateSRem">;
def LLVM_AndOp : LLVM_ArithmeticOp<"and", "CreateAnd">;
def LLVM_OrOp : LLVM_ArithmeticOp<"or", "CreateOr">;
def LLVM_XOrOp : LLVM_ArithmeticOp<"xor", "CreateXor">;
def LLVM_ShlOp : LLVM_ArithmeticOp<"shl", "CreateShl">;
def LLVM_LShrOp : LLVM_ArithmeticOp<"lshr", "CreateLShr">;
def LLVM_AShrOp : LLVM_ArithmeticOp<"ashr", "CreateAShr">;

// Predicate for integer comparisons.
def ICmpPredicateEQ  : I64EnumAttrCase<"eq", 0>;
def ICmpPredicateNE  : I64EnumAttrCase<"ne", 1>;
def ICmpPredicateSLT : I64EnumAttrCase<"slt", 2>;
def ICmpPredicateSLE : I64EnumAttrCase<"sle", 3>;
def ICmpPredicateSGT : I64EnumAttrCase<"sgt", 4>;
def ICmpPredicateSGE : I64EnumAttrCase<"sge", 5>;
def ICmpPredicateULT : I64EnumAttrCase<"ult", 6>;
def ICmpPredicateULE : I64EnumAttrCase<"ule", 7>;
def ICmpPredicateUGT : I64EnumAttrCase<"ugt", 8>;
def ICmpPredicateUGE : I64EnumAttrCase<"uge", 9>;
def ICmpPredicate : I64EnumAttr<
    "ICmpPredicate",
    "llvm.icmp comparison predicate",
    [ICmpPredicateEQ, ICmpPredicateNE, ICmpPredicateSLT, ICmpPredicateSLE,
     ICmpPredicateSGT, ICmpPredicateSGE, ICmpPredicateULT, ICmpPredicateULE,
     ICmpPredicateUGT, ICmpPredicateUGE]> {
  let cppNamespace = "::mlir::LLVM";
}

// Other integer operations.
def LLVM_ICmpOp : LLVM_OneResultOp<"icmp", [NoSideEffect]>,
                  Arguments<(ins ICmpPredicate:$predicate, LLVM_Type:$lhs,
                             LLVM_Type:$rhs)> {
  let llvmBuilder = [{
    $res = builder.CreateICmp(getLLVMCmpPredicate($predicate), $lhs, $rhs);
  }];
  let builders = [OpBuilder<
    "Builder *b, OperationState &result, ICmpPredicate predicate, Value *lhs, "
    "Value *rhs", [{
      LLVMDialect *dialect = &lhs->getType().cast<LLVMType>().getDialect();
      build(b, result, LLVMType::getInt1Ty(dialect),
            b->getI64IntegerAttr(static_cast<int64_t>(predicate)), lhs, rhs);
    }]>];
  let parser = [{ return parseCmpOp<ICmpPredicate>(parser, result); }];
  let printer = [{ printICmpOp(p, *this); }];
}

// Predicate for float comparisons
def FCmpPredicateFALSE  : I64EnumAttrCase<"_false", 0>;
def FCmpPredicateOEQ    : I64EnumAttrCase<"oeq", 1>;
def FCmpPredicateOGT    : I64EnumAttrCase<"ogt", 2>;
def FCmpPredicateOGE    : I64EnumAttrCase<"oge", 3>;
def FCmpPredicateOLT    : I64EnumAttrCase<"olt", 4>;
def FCmpPredicateOLE    : I64EnumAttrCase<"ole", 5>;
def FCmpPredicateONE    : I64EnumAttrCase<"one", 6>;
def FCmpPredicateORD    : I64EnumAttrCase<"ord", 7>;
def FCmpPredicateUEQ    : I64EnumAttrCase<"ueq", 8>;
def FCmpPredicateUGT    : I64EnumAttrCase<"ugt", 9>;
def FCmpPredicateUGE    : I64EnumAttrCase<"uge", 10>;
def FCmpPredicateULT    : I64EnumAttrCase<"ult", 11>;
def FCmpPredicateULE    : I64EnumAttrCase<"ule", 12>;
def FCmpPredicateUNE    : I64EnumAttrCase<"une", 13>;
def FCmpPredicateUNO    : I64EnumAttrCase<"uno", 14>;
def FCmpPredicateTRUE   : I64EnumAttrCase<"_true", 15>;

def FCmpPredicate : I64EnumAttr<
    "FCmpPredicate",
    "llvm.fcmp comparison predicate",
    [FCmpPredicateFALSE, FCmpPredicateOEQ, FCmpPredicateOGT, FCmpPredicateOGE,
     FCmpPredicateOLT, FCmpPredicateOLE, FCmpPredicateONE, FCmpPredicateORD,
     FCmpPredicateUEQ, FCmpPredicateUGT, FCmpPredicateUGE, FCmpPredicateULT,
     FCmpPredicateULE, FCmpPredicateUNE, FCmpPredicateUNO, FCmpPredicateTRUE
    ]> {
  let cppNamespace = "::mlir::LLVM";
}

// Other integer operations.
def LLVM_FCmpOp : LLVM_OneResultOp<"fcmp", [NoSideEffect]>,
                  Arguments<(ins FCmpPredicate:$predicate, LLVM_Type:$lhs,
                             LLVM_Type:$rhs)> {
  let llvmBuilder = [{
    $res = builder.CreateFCmp(getLLVMCmpPredicate($predicate), $lhs, $rhs);
  }];
  let builders = [OpBuilder<
    "Builder *b, OperationState &result, FCmpPredicate predicate, Value *lhs, "
    "Value *rhs", [{
      LLVMDialect *dialect = &lhs->getType().cast<LLVMType>().getDialect();
      build(b, result, LLVMType::getInt1Ty(dialect),
            b->getI64IntegerAttr(static_cast<int64_t>(predicate)), lhs, rhs);
    }]>];
  let parser = [{ return parseCmpOp<FCmpPredicate>(parser, result); }];
  let printer = [{ printFCmpOp(p, *this); }];
}

// Floating point binary operations.
def LLVM_FAddOp : LLVM_ArithmeticOp<"fadd", "CreateFAdd">;
def LLVM_FSubOp : LLVM_ArithmeticOp<"fsub", "CreateFSub">;
def LLVM_FMulOp : LLVM_ArithmeticOp<"fmul", "CreateFMul">;
def LLVM_FDivOp : LLVM_ArithmeticOp<"fdiv", "CreateFDiv">;
def LLVM_FRemOp : LLVM_ArithmeticOp<"frem", "CreateFRem">;
def LLVM_FNegOp : LLVM_UnaryArithmeticOp<"fneg", "CreateFNeg">;

// Memory-related operations.
def LLVM_AllocaOp :
    LLVM_OneResultOp<"alloca">,
    Arguments<(ins LLVM_Type:$arraySize, OptionalAttr<I64Attr>:$alignment)> {
  string llvmBuilder = [{
    auto *alloca = builder.CreateAlloca(
      $_resultType->getPointerElementType(), $arraySize);
    if ($alignment.hasValue()) {
      auto align = $alignment.getValue().getZExtValue();
      if (align != 0)
        alloca->setAlignment(llvm::MaybeAlign(align));
    }
    $res = alloca;
  }];
  let builders = [OpBuilder<
    "Builder *b, OperationState &result, Type resultType, Value *arraySize, "
    "unsigned alignment",
    [{
      if (alignment == 0)
        return build(b, result, resultType, arraySize, IntegerAttr());
      build(b, result, resultType, arraySize, b->getI64IntegerAttr(alignment));
  }]>];
  let parser = [{ return parseAllocaOp(parser, result); }];
  let printer = [{ printAllocaOp(p, *this); }];
  let verifier = [{
    if (alignment().hasValue()) {
      auto align = alignment().getValue().getSExtValue();
      if (align < 0)
        return emitOpError("expected positive alignment");
    }
    return success();
  }];
}
def LLVM_GEPOp : LLVM_OneResultOp<"getelementptr", [NoSideEffect]>,
                 Arguments<(ins LLVM_Type:$base, Variadic<LLVM_Type>:$indices)>,
                 LLVM_Builder<"$res = builder.CreateGEP($base, $indices);"> {
  let parser = [{ return parseGEPOp(parser, result); }];
  let printer = [{ printGEPOp(p, *this); }];
}
def LLVM_LoadOp : LLVM_OneResultOp<"load">, Arguments<(ins LLVM_Type:$addr)>,
                  LLVM_Builder<"$res = builder.CreateLoad($addr);"> {
  let builders = [OpBuilder<
    "Builder *b, OperationState &result, Value *addr",
    [{
      auto type = addr->getType().cast<LLVM::LLVMType>().getPointerElementTy();
      build(b, result, type, addr);
    }]>];
  let parser = [{ return parseLoadOp(parser, result); }];
  let printer = [{ printLoadOp(p, *this); }];
}
def LLVM_StoreOp : LLVM_ZeroResultOp<"store">,
                   Arguments<(ins LLVM_Type:$value, LLVM_Type:$addr)>,
                   LLVM_Builder<"builder.CreateStore($value, $addr);"> {
  let parser = [{ return parseStoreOp(parser, result); }];
  let printer = [{ printStoreOp(p, *this); }];
}

// Casts.
class LLVM_CastOp<string mnemonic, string builderFunc,
                  list<OpTrait> traits = []> :
    LLVM_OneResultOp<mnemonic,
           !listconcat([NoSideEffect], traits)>,
    Arguments<(ins LLVM_Type:$arg)>,
    LLVM_Builder<"$res = builder." # builderFunc # "($arg, $_resultType);"> {
  let parser = [{ return mlir::impl::parseCastOp(parser, result); }];
  let printer = [{ mlir::impl::printCastOp(this->getOperation(), p); }];
}
def LLVM_BitcastOp : LLVM_CastOp<"bitcast", "CreateBitCast">;
def LLVM_AddrSpaceCastOp : LLVM_CastOp<"addrspacecast", "CreateAddrSpaceCast">;
def LLVM_IntToPtrOp : LLVM_CastOp<"inttoptr", "CreateIntToPtr">;
def LLVM_PtrToIntOp : LLVM_CastOp<"ptrtoint", "CreatePtrToInt">;
def LLVM_SExtOp : LLVM_CastOp<"sext", "CreateSExt">;
def LLVM_ZExtOp : LLVM_CastOp<"zext", "CreateZExt">;
def LLVM_TruncOp : LLVM_CastOp<"trunc", "CreateTrunc">;
def LLVM_SIToFPOp : LLVM_CastOp<"sitofp", "CreateSIToFP">;
def LLVM_UIToFPOp : LLVM_CastOp<"uitofp", "CreateUIToFP">;
def LLVM_FPToSIOp : LLVM_CastOp<"fptosi", "CreateFPToSI">;
def LLVM_FPToUIOp : LLVM_CastOp<"fptoui", "CreateFPToUI">;
def LLVM_FPExtOp : LLVM_CastOp<"fpext", "CreateFPExt">;
def LLVM_FPTruncOp : LLVM_CastOp<"fptrunc", "CreateFPTrunc">;

// Call-related operations.
def LLVM_CallOp : LLVM_Op<"call">,
                  Arguments<(ins OptionalAttr<FlatSymbolRefAttr>:$callee,
                             Variadic<LLVM_Type>)>,
                  Results<(outs Variadic<LLVM_Type>)>,
                  LLVM_TwoBuilders<LLVM_OneResultOpBuilder,
                                   LLVM_ZeroResultOpBuilder> {
  let verifier = [{
    if (getNumResults() > 1)
      return emitOpError("must have 0 or 1 result");
    return success();
  }];
  let parser = [{ return parseCallOp(parser, result); }];
  let printer = [{ printCallOp(p, *this); }];
}
def LLVM_ExtractElementOp : LLVM_OneResultOp<"extractelement", [NoSideEffect]>,
                          Arguments<(ins LLVM_Type:$vector,
                                     LLVM_Type:$position)> {
  string llvmBuilder = [{
    $res = builder.CreateExtractElement($vector, $position);
  }];
  let builders = [OpBuilder<
    "Builder *b, OperationState &result, Value *vector, Value *position,"
    "ArrayRef<NamedAttribute> attrs = {}">];
  let parser = [{ return parseExtractElementOp(parser, result); }];
  let printer = [{ printExtractElementOp(p, *this); }];
}
def LLVM_ExtractValueOp : LLVM_OneResultOp<"extractvalue", [NoSideEffect]>,
                          Arguments<(ins LLVM_Type:$container,
                                     ArrayAttr:$position)> {
  string llvmBuilder = [{
    $res = builder.CreateExtractValue($container, extractPosition($position));
  }];
  let parser = [{ return parseExtractValueOp(parser, result); }];
  let printer = [{ printExtractValueOp(p, *this); }];
}
def LLVM_InsertElementOp : LLVM_OneResultOp<"insertelement", [NoSideEffect]>,
                         Arguments<(ins LLVM_Type:$vector, LLVM_Type:$value,
                                    LLVM_Type:$position)> {
  string llvmBuilder = [{
    $res = builder.CreateInsertElement($vector, $value, $position);
  }];
  let parser = [{ return parseInsertElementOp(parser, result); }];
  let printer = [{ printInsertElementOp(p, *this); }];
}
def LLVM_InsertValueOp : LLVM_OneResultOp<"insertvalue", [NoSideEffect]>,
                         Arguments<(ins LLVM_Type:$container, LLVM_Type:$value,
                                    ArrayAttr:$position)> {
  string llvmBuilder = [{
    $res = builder.CreateInsertValue($container, $value,
                                     extractPosition($position));
  }];
  let builders = [OpBuilder<
    "Builder *b, OperationState &result, Value *container, Value *value, "
    "ArrayAttr position",
    [{
      build(b, result, container->getType(), container, value, position);
    }]>];
  let parser = [{ return parseInsertValueOp(parser, result); }];
  let printer = [{ printInsertValueOp(p, *this); }];
}
def LLVM_ShuffleVectorOp
    : LLVM_OneResultOp<"shufflevector", [NoSideEffect]>,
      Arguments<(ins LLVM_Type:$v1, LLVM_Type:$v2, I32ArrayAttr:$mask)>,
      LLVM_Builder<
      "$res = builder.CreateShuffleVector($v1, $v2, extractPosition($mask));"> {
  let builders = [OpBuilder<
    "Builder *b, OperationState &result, Value *v1, Value *v2, "
    "ArrayAttr mask, ArrayRef<NamedAttribute> attrs = {}">];
  let verifier = [{
    auto wrappedVectorType1 = v1()->getType().cast<LLVM::LLVMType>();
    auto wrappedVectorType2 = v2()->getType().cast<LLVM::LLVMType>();
    if (!wrappedVectorType2.getUnderlyingType()->isVectorTy())
      return emitOpError("expected LLVM IR Dialect vector type for operand #2");
    if (wrappedVectorType1.getVectorElementType() !=
        wrappedVectorType2.getVectorElementType())
      return emitOpError("expected matching LLVM IR Dialect element types");
    return success();
  }];
  let parser = [{ return parseShuffleVectorOp(parser, result); }];
  let printer = [{ printShuffleVectorOp(p, *this); }];
}

// Misc operations.
def LLVM_SelectOp
    : LLVM_OneResultOp<"select", [NoSideEffect]>,
      Arguments<(ins LLVM_Type:$condition, LLVM_Type:$trueValue,
                 LLVM_Type:$falseValue)>,
      LLVM_Builder<
          "$res = builder.CreateSelect($condition, $trueValue, $falseValue);"> {
  let builders = [OpBuilder<
    "Builder *b, OperationState &result, Value *condition, Value *lhs, "
    "Value *rhs", [{
      build(b, result, lhs->getType(), condition, lhs, rhs);
    }]>];
  let parser = [{ return parseSelectOp(parser, result); }];
  let printer = [{ printSelectOp(p, *this); }];
}

// Terminators.
def LLVM_BrOp : LLVM_TerminatorOp<"br", []> {
  let parser = [{ return parseBrOp(parser, result); }];
  let printer = [{ printBrOp(p, *this); }];
}
def LLVM_CondBrOp : LLVM_TerminatorOp<"cond_br", []> {
  let verifier = [{
    if (getNumSuccessors() != 2)
      return emitOpError("expected exactly two successors");
    return success();
  }];
  let parser = [{ return parseCondBrOp(parser, result); }];
  let printer = [{ printCondBrOp(p, *this); }];
}
def LLVM_ReturnOp : LLVM_TerminatorOp<"return", []> {
  string llvmBuilder = [{
    if ($_numOperands != 0)
      builder.CreateRet($args[0]);
    else
      builder.CreateRetVoid();
  }];

  let verifier = [{
    if (getNumOperands() > 1)
      return emitOpError("expects at most 1 operand");
    return success();
  }];

  let parser = [{ return parseReturnOp(parser, result); }];
  let printer = [{ printReturnOp(p, *this); }];
}
def LLVM_UnreachableOp : LLVM_TerminatorOp<"unreachable", []> {
  string llvmBuilder = [{ builder.CreateUnreachable(); }];
  let parser = [{ return success(); }];
  let printer = [{ p << getOperationName(); }];
}

////////////////////////////////////////////////////////////////////////////////
// Auxiliary operations (do not appear in LLVM IR but necessary for the dialect
// to work correctly).
////////////////////////////////////////////////////////////////////////////////

// Linkage attribute is used on functions and globals. The order follows that of
// https://llvm.org/docs/LangRef.html#linkage-types. The names are equivalent to
// visible names in the IR rather than to enum values names in llvm::GlobalValue
// since the latter is easier to change.
def LinkagePrivate             : I64EnumAttrCase<"Private", 0>;
def LinkageInternal            : I64EnumAttrCase<"Internal", 1>;
def LinkageAvailableExternally : I64EnumAttrCase<"AvailableExternally", 2>;
def LinkageLinkonce            : I64EnumAttrCase<"Linkonce", 3>;
def LinkageWeak                : I64EnumAttrCase<"Weak", 4>;
def LinkageCommon              : I64EnumAttrCase<"Common", 5>;
def LinkageAppending           : I64EnumAttrCase<"Appending", 6>;
def LinkageExternWeak          : I64EnumAttrCase<"ExternWeak", 7>;
def LinkageLinkonceODR         : I64EnumAttrCase<"LinkonceODR", 8>;
def LinkageWeakODR             : I64EnumAttrCase<"WeakODR", 9>;
def LinkageExternal            : I64EnumAttrCase<"External", 10>;
def Linkage : I64EnumAttr<
    "Linkage",
    "LLVM linkage types",
    [LinkagePrivate, LinkageInternal, LinkageAvailableExternally,
     LinkageLinkonce, LinkageWeak, LinkageCommon, LinkageAppending,
     LinkageExternWeak, LinkageLinkonceODR, LinkageWeakODR, LinkageExternal]> {
  let cppNamespace = "::mlir::LLVM";
}


def LLVM_AddressOfOp
    : LLVM_OneResultOp<"mlir.addressof">,
      Arguments<(ins FlatSymbolRefAttr:$global_name)> {
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, LLVMType resType, "
              "StringRef name, ArrayRef<NamedAttribute> attrs = {}", [{
      result.addAttribute("global_name", builder->getSymbolRefAttr(name));
      result.addAttributes(attrs);
      result.addTypes(resType);}]>,

    OpBuilder<"Builder *builder, OperationState &result, GlobalOp global, "
              "ArrayRef<NamedAttribute> attrs = {}", [{
      build(builder, result,
            global.getType().getPointerTo(global.addr_space().getZExtValue()),
            global.sym_name(), attrs);}]>
  ];

  let extraClassDeclaration = [{
    /// Return the llvm.mlir.global operation that defined the value referenced
    /// here.
    GlobalOp getGlobal();
  }];

  let printer = "printAddressOfOp(p, *this);";
  let parser = "return parseAddressOfOp(parser, result);";
  let verifier = "return ::verify(*this);";
}

def LLVM_GlobalOp
    : LLVM_ZeroResultOp<"mlir.global",
                        [IsolatedFromAbove,
                         SingleBlockImplicitTerminator<"ReturnOp">, Symbol]>,
      Arguments<(ins TypeAttr:$type, UnitAttr:$constant, StrAttr:$sym_name,
                 Linkage:$linkage,
                 OptionalAttr<AnyAttr>:$value,
                 DefaultValuedAttr<NonNegativeI32Attr, "0">:$addr_space)> {
  let summary = "LLVM dialect global.";
  let description = [{
    Can contain an optional initializer region or attribute for simple
    initializers.

    Examples:
      // Initialized using an attribute.
      llvm.mlir.global @a("abc") : !llvm<"[3 x i8]">
      // Initialized using a region.
      llvm.mlir.global constant @b() : !llvm<"i32*"> {
        %0 = llvm.constant(0 : i32) : !llvm.i32
        %1 = llvm.inttoptr %0 : !llvm.i32 to !llvm<"i32*">
        llvm.return %1 : !llvm<"i32*">
      }
  }];
  let regions = (region AnyRegion:$initializer);

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, LLVMType type, "
              "bool isConstant, Linkage linkage, StringRef name, "
              "Attribute value, ArrayRef<NamedAttribute> attrs = {}">
  ];

  let extraClassDeclaration = [{
    /// Return the LLVM type of the global.
    LLVMType getType() {
      return type().cast<LLVMType>();
    }
    /// Return the initializer attribute if it exists, or a null attribute.
    Attribute getValueOrNull() {
      return value().getValueOr(Attribute());
    }
    /// Return the initializer region. This may be empty, but if it is not it
    /// terminates in an `llvm.return` op with the initializer value.
    Region &getInitializerRegion() {
      return getOperation()->getRegion(0);
    }
    /// Return the initializer block. If the initializer region is empty this
    /// is nullptr. If it is not nullptr, it terminates with an `llvm.return`
    /// op with the initializer value.
    Block *getInitializerBlock() {
      return getInitializerRegion().empty() ?
        nullptr : &getInitializerRegion().front();
    }
  }];

  let printer = "printGlobalOp(p, *this);";
  let parser = "return parseGlobalOp(parser, result);";
  let verifier = "return ::verify(*this);";
}

def LLVM_LLVMFuncOp
    : LLVM_ZeroResultOp<"func",
                        [NativeOpTrait<"IsIsolatedFromAbove">,
                         NativeOpTrait<"FunctionLike">, Symbol]>,
      Arguments<(ins DefaultValuedAttr<Linkage,
                                       "Linkage::External">:$linkage)> {
  let summary = "LLVM dialect function, has wrapped LLVM IR function type";

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, StringRef name, "
              "LLVMType type, LLVM::Linkage linkage = LLVM::Linkage::External, "
              "ArrayRef<NamedAttribute> attrs = {}, "
              "ArrayRef<NamedAttributeList> argAttrs = {}">
  ];

  let extraClassDeclaration = [{
    // Add an entry block to an empty function, and set up the block arguments
    // to match the signature of the function.
    Block *addEntryBlock();

    LLVMType getType() {
      return getAttrOfType<TypeAttr>(getTypeAttrName())
          .getValue().cast<LLVMType>();
    }
    bool isVarArg() {
      return getType().getUnderlyingType()->isFunctionVarArg();
    }

    // Hook for OpTrait::FunctionLike, returns the number of function arguments.
    // Depends on the type attribute being correct as checked by verifyType.
    unsigned getNumFuncArguments();

    // Hook for OpTrait::FunctionLike, returns the number of function results.
    // Depends on the type attribute being correct as checked by verifyType.
    unsigned getNumFuncResults();

    // Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    // attribute is present.  This can check for preconditions of the
    // getNumArguments hook not failing.
    LogicalResult verifyType();
  }];

  let verifier = [{ return ::verify(*this); }];
  let printer = [{ printLLVMFuncOp(p, *this); }];
  let parser = [{ return parseLLVMFuncOp(parser, result); }];
}

def LLVM_NullOp
    : LLVM_OneResultOp<"mlir.null", [NoSideEffect]>,
      LLVM_Builder<"$res = llvm::ConstantPointerNull::get("
                   "    llvm::cast<llvm::PointerType>($_resultType));"> {
  let parser = [{ return parseNullOp(parser, result); }];
  let printer = [{ printNullOp(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
}

def LLVM_UndefOp : LLVM_OneResultOp<"mlir.undef", [NoSideEffect]>,
                   LLVM_Builder<"$res = llvm::UndefValue::get($_resultType);"> {
  let parser = [{ return parseUndefOp(parser, result); }];
  let printer = [{ printUndefOp(p, *this); }];
}
def LLVM_ConstantOp
    : LLVM_OneResultOp<"mlir.constant", [NoSideEffect]>,
      Arguments<(ins AnyAttr:$value)>,
      LLVM_Builder<"$res = getLLVMConstant($_resultType, $value, $_location);">
{
  let parser = [{ return parseConstantOp(parser, result); }];
  let printer = [{ printConstantOp(p, *this); }];
}

// Operations that correspond to LLVM intrinsics. With MLIR operation set being
// extendable, there is no reason to introduce a hard boundary between "core"
// operations and intrinsics. However, we systematically prefix them with
// "intr." to avoid potential name clashes.

def LLVM_FMulAddOp : LLVM_Op<"intr.fmuladd", [NoSideEffect]>,
                   Arguments<(ins LLVM_Type:$a, LLVM_Type:$b, LLVM_Type:$c)>,
                   Results<(outs LLVM_Type:$res)> {
  let llvmBuilder = [{
    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module, llvm::Intrinsic::fmuladd,
        {$a->getType(), $b->getType(), $c->getType()});
    $res = builder.CreateCall(fn, {$a, $b, $c});
  }];
}

def LLVM_ExpOp : LLVM_Op<"intr.exp", [NoSideEffect]>,
                   Arguments<(ins LLVM_Type:$in)>,
                   Results<(outs LLVM_Type:$res)> {
  let llvmBuilder = [{
    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module, llvm::Intrinsic::exp, {$in->getType()});
    $res = builder.CreateCall(fn, {$in});
  }];
}
#endif // LLVMIR_OPS
