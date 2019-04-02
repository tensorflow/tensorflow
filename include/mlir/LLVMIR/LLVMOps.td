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

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

// LLVM IR type wrapped in MLIR.
def LLVM_Type : Type<CPred<"{0}.isa<::mlir::LLVM::LLVMType>()">,
                     "LLVM dialect type">;

// Base class for LLVM operations.  All operations get an "llvm." prefix in
// their name automatically.  LLVM operations have either zero or one result,
// this class is specialized below for both cases and should not be used
// directly.
class LLVM_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<!strconcat("llvm.", mnemonic), traits> {
  // A pattern for constructing the LLVM IR Instruction (or other Value) that
  // corresponds to this op.  This pattern can use `builder` to refer to an
  // `llvm::IRBuilder<>` instance, $-names of arguments and results and the
  // following special variable names:
  //   - $_resultType - substituted with the LLVM IR type of the result;
  //   - $_numOperands - substituted with the number of operands (including
  //                     the variadic ones);
  //   - $_hasResult - substituted with a check that a variadic-result op does
  //                   have a result (LLVM ops can have 0 or 1 result);
  //   - $_location - mlir::Location object of the instruction.
  // Additionally, `$$` can be used to produce the dollar character.
  string llvmBuilder = "";
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
    auto llvmType = resultType.dyn_cast<LLVM::LLVMType>();
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
           !listconcat([NoSideEffect, SameValueType], traits)>,
    Arguments<(ins LLVM_Type:$lhs, LLVM_Type:$rhs)>,
    LLVM_Builder<"$res = builder." # builderFunc # "($lhs, $rhs);"> {
  let parser = [{ return impl::parseBinaryOp(parser, result); }];
  let printer = [{ printLLVMBinaryOp(p, this->getOperation()); }];
}

// Integer binary operations.
def LLVM_AddOp : LLVM_ArithmeticOp<"add", "CreateAdd", [Commutative]>;
def LLVM_SubOp : LLVM_ArithmeticOp<"sub", "CreateSub">;
def LLVM_MulOp : LLVM_ArithmeticOp<"mul", "CreateMul", [Commutative]>;
def LLVM_UDivOp : LLVM_ArithmeticOp<"udiv", "CreateUDiv">;
def LLVM_SDivOp : LLVM_ArithmeticOp<"sdiv", "CreateSDiv">;
def LLVM_URemOp : LLVM_ArithmeticOp<"urem", "CreateURem">;
def LLVM_SRemOp : LLVM_ArithmeticOp<"srem", "CreateSRem">;

// Other integer operations.
def LLVM_ICmpOp : LLVM_OneResultOp<"icmp", [NoSideEffect]>,
                  Arguments<(ins I32Attr:$predicate, LLVM_Type:$lhs,
                             LLVM_Type:$rhs)> {
  let llvmBuilder = [{
    $res = builder.CreateICmp(getLLVMCmpPredicate(
        static_cast<CmpIPredicate>($predicate.getZExtValue())), $lhs, $rhs);
  }];
  let parser = [{ return parseICmpOp(parser, result); }];
  let printer = [{ printICmpOp(p, *this); }];
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
    $res = builder.CreateAlloca($_resultType, $arraySize);
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
def LLVM_BitcastOp
    : LLVM_OneResultOp<"bitcast", [NoSideEffect]>,
      Arguments<(ins LLVM_Type:$arg)>,
      LLVM_Builder<"$res = builder.CreateBitCast($arg, $_resultType);"> {
  let parser = [{ return parseBitcastOp(parser, result); }];
  let printer = [{ printBitcastOp(p, *this); }];
}

// Call-related operations.
def LLVM_CallOp : LLVM_Op<"call">,
                  Arguments<(ins OptionalAttr<FunctionAttr>:$callee,
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
def LLVM_BrOp : LLVM_TerminatorOp<"br", [NoSideEffect]> {
  let parser = [{ return parseBrOp(parser, result); }];
  let printer = [{ printBrOp(p, *this); }];
}
def LLVM_CondBrOp : LLVM_TerminatorOp<"cond_br", [NoSideEffect]> {
  let verifier = [{
    if (getNumSuccessors() != 2)
      return emitOpError("expected exactly two successors");
    return success();
  }];
  let parser = [{ return parseCondBrOp(parser, result); }];
  let printer = [{ printCondBrOp(p, *this); }];
}
def LLVM_ReturnOp : LLVM_TerminatorOp<"return", [NoSideEffect]> {
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

// Pseudo-operations (do not appear in LLVM IR but necessary for the dialect to
// work correctly).
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
