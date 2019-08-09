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

#ifdef LLVMIR_OPS
#else
#define LLVMIR_OPS

include "mlir/LLVMIR/LLVMOpBase.td"

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
  "Builder *, OperationState *result, Type resultType, "
  "ArrayRef<Value *> operands, ArrayRef<NamedAttribute> attributes = {}",
  [{
    if (resultType) result->addTypes(resultType);
    result->addOperands(operands);
    for (auto namedAttr : attributes) {
      result->addAttribute(namedAttr.first, namedAttr.second);
    }
  }]>;

def LLVM_ZeroResultOpBuilder : OpBuilder<
  "Builder *, OperationState *result, ArrayRef<Value *> operands, "
  "ArrayRef<NamedAttribute> attributes = {}",
  [{
    result->addOperands(operands);
    for (auto namedAttr : attributes) {
      result->addAttribute(namedAttr.first, namedAttr.second);
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
  "Builder *builder, OperationState *result, Type resultType, "
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
  let builders = [OpBuilder<
    "Builder *, OperationState *result, "
    "ArrayRef<Value *> properOperands, "
    "ArrayRef<Block *> destinations, "
    "ArrayRef<ArrayRef<Value *>> operands = {}, "
    "ArrayRef<NamedAttribute> attributes = {}",
    [{
      result->addOperands(properOperands);
      for (auto kvp : llvm::zip(destinations, operands)) {
        result->addSuccessor(std::get<0>(kvp), std::get<1>(kvp));
      }
      for (auto namedAttr : attributes) {
        result->addAttribute(namedAttr.first, namedAttr.second);
      }
    }]
  >];
}

// Class for arithmetic binary operations.
class LLVM_ArithmeticOp<string mnemonic, string builderFunc,
                        list<OpTrait> traits = []> :
    LLVM_OneResultOp<mnemonic,
           !listconcat([NoSideEffect, SameOperandsAndResultType], traits)>,
    Arguments<(ins LLVM_Type:$lhs, LLVM_Type:$rhs)>,
    LLVM_Builder<"$res = builder." # builderFunc # "($lhs, $rhs);"> {
  let parser = [{ return impl::parseBinaryOp(parser, result); }];
  let printer = [{ mlir::impl::printBinaryOp(this->getOperation(), p); }];
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
  let cppNamespace = "mlir::LLVM";

  let returnType = "ICmpPredicate";
  let convertFromStorage =
      "static_cast<" # returnType # ">($_self.getValue().getZExtValue())";
}

// Other integer operations.
def LLVM_ICmpOp : LLVM_OneResultOp<"icmp", [NoSideEffect]>,
                  Arguments<(ins ICmpPredicate:$predicate, LLVM_Type:$lhs,
                             LLVM_Type:$rhs)> {
  let llvmBuilder = [{
    $res = builder.CreateICmp(getLLVMCmpPredicate($predicate), $lhs, $rhs);
  }];
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
  let cppNamespace = "mlir::LLVM";

  let returnType = "FCmpPredicate";
  let convertFromStorage =
      "static_cast<" # returnType # ">($_self.getValue().getZExtValue())";
}

// Other integer operations.
def LLVM_FCmpOp : LLVM_OneResultOp<"fcmp", [NoSideEffect]>,
                  Arguments<(ins FCmpPredicate:$predicate, LLVM_Type:$lhs,
                             LLVM_Type:$rhs)> {
  let llvmBuilder = [{
    $res = builder.CreateFCmp(getLLVMCmpPredicate($predicate), $lhs, $rhs);
  }];
  let parser = [{ return parseCmpOp<FCmpPredicate>(parser, result); }];
  let printer = [{ printFCmpOp(p, *this); }];
}

// Floating point binary operations.
def LLVM_FAddOp : LLVM_ArithmeticOp<"fadd", "CreateFAdd">;
def LLVM_FSubOp : LLVM_ArithmeticOp<"fsub", "CreateFSub">;
def LLVM_FMulOp : LLVM_ArithmeticOp<"fmul", "CreateFMul">;
def LLVM_FDivOp : LLVM_ArithmeticOp<"fdiv", "CreateFDiv">;
def LLVM_FRemOp : LLVM_ArithmeticOp<"frem", "CreateFRem">;

// Memory-related operations.
def LLVM_AllocaOp : LLVM_OneResultOp<"alloca">,
                    Arguments<(ins LLVM_Type:$arraySize)> {
  string llvmBuilder = [{
    $res = builder.CreateAlloca($_resultType->getPointerElementType(),
                                $arraySize);
  }];
  let parser = [{ return parseAllocaOp(parser, result); }];
  let printer = [{ printAllocaOp(p, *this); }];
}
def LLVM_GEPOp : LLVM_OneResultOp<"getelementptr", [NoSideEffect]>,
                 Arguments<(ins LLVM_Type:$base, Variadic<LLVM_Type>:$indices)>,
                 LLVM_Builder<"$res = builder.CreateGEP($base, $indices);"> {
  let parser = [{ return parseGEPOp(parser, result); }];
  let printer = [{ printGEPOp(p, *this); }];
}
def LLVM_LoadOp : LLVM_OneResultOp<"load">, Arguments<(ins LLVM_Type:$addr)>,
                  LLVM_Builder<"$res = builder.CreateLoad($addr);"> {
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
def LLVM_IntToPtrOp : LLVM_CastOp<"inttoptr", "CreateIntToPtr">;
def LLVM_PtrToIntOp : LLVM_CastOp<"ptrtoint", "CreatePtrToInt">;
def LLVM_SExtOp : LLVM_CastOp<"sext", "CreateSExt">;
def LLVM_ZExtOp : LLVM_CastOp<"zext", "CreateZExt">;
def LLVM_TruncOp : LLVM_CastOp<"trunc", "CreateTrunc">;
def LLVM_SIToFPOp : LLVM_CastOp<"sitofp", "CreateSIToFP">;

// Call-related operations.
def LLVM_CallOp : LLVM_Op<"call">,
                  Arguments<(ins OptionalAttr<SymbolRefAttr>:$callee,
                             // TODO(b/133216756): fix test failure and
                             // change to LLVM_Type
                             Variadic<AnyType>)>,
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
def LLVM_ExtractValueOp : LLVM_OneResultOp<"extractvalue", [NoSideEffect]>,
                          Arguments<(ins LLVM_Type:$container,
                                     ArrayAttr:$position)> {
  string llvmBuilder = [{
    $res = builder.CreateExtractValue($container, extractPosition($position));
  }];
  let parser = [{ return parseExtractValueOp(parser, result); }];
  let printer = [{ printExtractValueOp(p, *this); }];
}
def LLVM_InsertValueOp : LLVM_OneResultOp<"insertvalue", [NoSideEffect]>,
                         Arguments<(ins LLVM_Type:$container, LLVM_Type:$value,
                                    ArrayAttr:$position)> {
  string llvmBuilder = [{
    $res = builder.CreateInsertValue($container, $value,
                                     extractPosition($position));
  }];
  let parser = [{ return parseInsertValueOp(parser, result); }];
  let printer = [{ printInsertValueOp(p, *this); }];
}

// Misc operations.
def LLVM_SelectOp
    : LLVM_OneResultOp<"select", [NoSideEffect]>,
      Arguments<(ins LLVM_Type:$condition, LLVM_Type:$trueValue,
                 LLVM_Type:$falseValue)>,
      LLVM_Builder<
          "$res = builder.CreateSelect($condition, $trueValue, $falseValue);"> {
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
  let printer = [{ *p << getOperationName(); }];
}

// Pseudo-operations (do not appear in LLVM IR but necessary for the dialect to
// work correctly).
def LLVM_GlobalOp
    : LLVM_ZeroResultOp<"global">,
      Arguments<(ins TypeAttr:$type, UnitAttr:$constant, StrAttr:$sym_name,
                 AnyAttr:$value)> {

  let builders = [
    OpBuilder<"Builder *builder, OperationState *result, LLVMType type, "
              "bool isConstant, StringRef name, Attribute value, "
              "ArrayRef<NamedAttribute> attrs = {}">
  ];

  let extraClassDeclaration = [{
    /// Return the LLVM type of the global.
    LLVMType getType() {
      return type().cast<LLVMType>();
    }
  }];

  let printer = "printGlobalOp(p, *this);";
  let parser = "return parseGlobalOp(parser, result);";
  let verifier = "return ::verify(*this);";
}

def LLVM_LLVMFuncOp : LLVM_ZeroResultOp<"func",
      [NativeOpTrait<"IsIsolatedFromAbove">, NativeOpTrait<"FunctionLike">]> {
  let summary = "LLVM dialect function, has wrapped LLVM IR function type";

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<"Builder *builder, OperationState *result, StringRef name, "
              "LLVMType type, ArrayRef<NamedAttribute> attrs, "
              "ArrayRef<NamedAttributeList> argAttrs = {}">
  ];

  let extraClassDeclaration = [{
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

    // Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    // attribute is present.  This can check for preconditions of the
    // getNumArguments hook not failing.
    LogicalResult verifyType();
  }];

  let verifier = [{ return ::verify(*this); }];
  let printer = [{ printLLVMFuncOp(p, *this); }];
  let parser = [{
    return impl::parseFunctionLikeOp(parser, result, /*allowVariadic=*/true,
                                     buildLLVMFunctionType);
  }];
}

def LLVM_UndefOp : LLVM_OneResultOp<"undef", [NoSideEffect]>,
                   LLVM_Builder<"$res = llvm::UndefValue::get($_resultType);"> {
  let parser = [{ return parseUndefOp(parser, result); }];
  let printer = [{ printUndefOp(p, *this); }];
}
def LLVM_ConstantOp
    : LLVM_OneResultOp<"constant", [NoSideEffect]>,
      Arguments<(ins AnyAttr:$value)>,
      LLVM_Builder<"$res = getLLVMConstant($_resultType, $value, $_location);">
{
  let parser = [{ return parseConstantOp(parser, result); }];
  let printer = [{ printConstantOp(p, *this); }];
}

#endif // LLVMIR_OPS
