//===- SPIRVGLSLOps.td - GLSL extended insts spec file -----*- tablegen -*-===//
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
// This is the op definition spec of GLSL extension ops.
//
//===----------------------------------------------------------------------===//

#ifdef SPIRV_GLSL_OPS
#else
#define SPIRV_GLSL_OPS

#ifdef SPIRV_BASE
#else
include "mlir/Dialect/SPIRV/SPIRVBase.td"
#endif // SPIRV_BASE

//===----------------------------------------------------------------------===//
// SPIR-V GLSL 4.50 opcode specification.
//===----------------------------------------------------------------------===//

// Base class for all GLSL ops.
class SPV_GLSLOp<string mnemonic, int opcode, list<OpTrait> traits = []> :
  SPV_ExtInstOp<mnemonic, "GLSL", "GLSL.std.450", opcode, traits>;

// Base class for GLSL unary ops.
class SPV_GLSLUnaryOp<string mnemonic, int opcode, Type resultType,
                      Type operandType, list<OpTrait> traits = []> :
  SPV_GLSLOp<mnemonic, opcode, !listconcat([NoSideEffect], traits)> {

  let arguments = (ins
    SPV_ScalarOrVectorOf<operandType>:$operand
  );

  let results = (outs
    SPV_ScalarOrVectorOf<resultType>:$result
  );

  let parser = [{ return parseUnaryOp(parser, result); }];

  let printer = [{ return printUnaryOp(getOperation(), p); }];

  let verifier = [{ return success(); }];
}

// Base class for GLSL Unary arithmatic ops where return type matches
// the operand type.
class SPV_GLSLUnaryArithmaticOp<string mnemonic, int opcode, Type type,
                                list<OpTrait> traits = []> :
  SPV_GLSLUnaryOp<mnemonic, opcode, type, type, traits>;

// Base class for GLSL binary ops.
class SPV_GLSLBinaryOp<string mnemonic, int opcode, Type resultType,
                        Type operandType, list<OpTrait> traits = []> :
  SPV_GLSLOp<mnemonic, opcode, !listconcat([NoSideEffect], traits)> {

  let arguments = (ins
    SPV_ScalarOrVectorOf<operandType>:$lhs,
    SPV_ScalarOrVectorOf<operandType>:$rhs
  );

  let results = (outs
    SPV_ScalarOrVectorOf<resultType>:$result
  );

  let parser = [{ return impl::parseBinaryOp(parser, result); }];

  let printer = [{ return impl::printBinaryOp(getOperation(), p); }];

  let verifier = [{ return success(); }];
}

// Base class for GLSL Binary arithmatic ops where operand types and
// return type matches.
class SPV_GLSLBinaryArithmaticOp<string mnemonic, int opcode, Type type,
                                 list<OpTrait> traits = []> :
  SPV_GLSLBinaryOp<mnemonic, opcode, type, type, traits>;

// -----

def SPV_GLSLExpOp : SPV_GLSLUnaryArithmaticOp<"Exp", 27, FloatOfWidths<[16, 32]>> {
  let summary = "Exponentiation of Operand 1";

  let description = [{
    Result is the natural exponentiation of x; e^x.

    The operand x must be a scalar or vector whose component type is
    16-bit or 32-bit floating-point.

    Result Type and the type of x must be the same type. Results are
    computed per component.";

    ### Custom assembly format
    ``` {.ebnf}
    restricted-float-scalar-type ::=  `f16` | `f32`
    restricted-float-scalar-vector-type ::=
      restricted-float-scalar-type |
      `vector<` integer-literal `x` restricted-float-scalar-type `>`
    exp-op ::= ssa-id `=` `spv.GLSL.Exp` ssa-use `:`
               restricted-float-scalar-vector-type
    ```
    For example:

    ```
    %2 = spv.GLSL.Exp %0 : f32
    %3 = spv.GLSL.Exp %1 : vector<3xf16>
    ```
  }];
}

// -----

def SPV_GLSLFMaxOp : SPV_GLSLBinaryArithmaticOp<"FMax", 40, SPV_Float> {
  let summary = "Return maximum of two floating-point operands";

  let description = [{
    Result is y if x < y; otherwise result is x. Which operand is the
    result is undefined if one of the operands is a NaN.

    The operands must all be a scalar or vector whose component type
    is floating-point.

    Result Type and the type of all operands must be the same
    type. Results are computed per component.

    ### Custom assembly format
    ``` {.ebnf}
    fmax-op ::= ssa-id `=` `spv.GLSL.FMax` ssa-use `:`
                restricted-float-scalar-vector-type
    ```
    For example:

    ```
    %2 = spv.GLSL.FMax %0, %1 : f32
    %3 = spv.GLSL.FMax %0, %1 : vector<3xf16>
    ```
  }];
}

// -----

#endif // SPIRV_GLSL_OPS
