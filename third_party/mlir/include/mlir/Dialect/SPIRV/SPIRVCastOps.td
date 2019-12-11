//===-- SPIRVCastOps.td - MLIR SPIR-V Cast Ops -------*- tablegen -*-------===//
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
// This file contains cast ops for the SPIR-V dialect. It corresponds
// to "3.32.11. Convertion Instructions" of the SPIR-V specification.
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_CAST_OPS
#define SPIRV_CAST_OPS

include "mlir/Dialect/SPIRV/SPIRVBase.td"

class SPV_CastOp<string mnemonic, Type resultType, Type operandType,
                 list<OpTrait> traits = []> :
      SPV_Op<mnemonic,
             !listconcat(traits,
                         [NoSideEffect, SameOperandsAndResultShape])> {
  let arguments = (ins
    SPV_ScalarOrVectorOf<operandType>:$operand
  );

  let results = (outs
    SPV_ScalarOrVectorOf<resultType>:$result
  );

  let parser = [{ return mlir::impl::parseCastOp(parser, result); }];
  let printer = [{ mlir::impl::printCastOp(this->getOperation(), p); }];
  let verifier = [{ return verifyCastOp(this->getOperation()); }];
}

// -----

def SPV_BitcastOp : SPV_Op<"Bitcast", [NoSideEffect]> {
  let summary = "Bit pattern-preserving type conversion.";

  let description = [{
    Result Type must be an OpTypePointer, or a scalar or vector of
    numerical-type.

    Operand must have a type of OpTypePointer, or a scalar or vector of
    numerical-type. It must be a different type than Result Type.

    If either Result Type or Operand is a pointer, the other must be a
    pointer (diverges from the SPIR-V spec).

    If Result Type has a different number of components than Operand, the
    total number of bits in Result Type must equal the total number of bits
    in Operand. Let L be the type, either Result Type or Operand’s type,
    that has the larger number of components. Let S be the other type, with
    the smaller number of components. The number of components in L must be
    an integer multiple of the number of components in S. The first
    component (that is, the only or lowest-numbered component) of S maps to
    the first components of L, and so on,  up to the last component of S
    mapping to the last components of L. Within this mapping, any single
    component of S (mapping to multiple components of L) maps its lower-
    ordered bits to the lower-numbered components of L.

    ### Custom assembly form

    ```
    bitcast-op ::= ssa-id `=` `spv.Bitcast` ssa-use
                   `:` operand-type `to` result-type
    ```

    For example:

    ```
    %1 = spv.Bitcast %0 : f32 to i32
    %1 = spv.Bitcast %0 : vector<2xf32> to i64
    %1 = spv.Bitcast %0 : !spv.ptr<f32, Function> to !spv.ptr<i32, Function>
    ```
  }];

  let arguments = (ins
    SPV_ScalarOrVectorOrPtr:$operand
  );

  let results = (outs
    SPV_ScalarOrVectorOrPtr:$result
  );

  let parser = [{ return mlir::impl::parseCastOp(parser, result); }];
  let printer = [{ mlir::impl::printCastOp(this->getOperation(), p); }];

  let hasCanonicalizer = 1;
}

// -----

def SPV_ConvertFToSOp : SPV_CastOp<"ConvertFToS", SPV_Integer, SPV_Float, []> {
  let summary = [{
    Convert value numerically from floating point to signed integer, with
    round toward 0.0.
  }];

  let description = [{
    Result Type must be a scalar or vector of integer type.

    Float Value must be a scalar or vector of floating-point type.  It must
    have the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form

    ```
    convert-f-to-s-op ::= ssa-id `=` `spv.ConvertFToSOp` ssa-use
                          `:` operand-type `to` result-type
    ```

    For example:

    ```
    %1 = spv.ConvertFToS %0 : f32 to i32
    %3 = spv.ConvertFToS %2 : vector<3xf32> to vector<3xi32>
    ```
  }];
}

// -----

def SPV_ConvertFToUOp : SPV_CastOp<"ConvertFToU", SPV_Integer, SPV_Float, []> {
  let summary = [{
    Convert value numerically from floating point to unsigned integer, with
    round toward 0.0.
  }];

  let description = [{
    Result Type must be a scalar or vector of integer type, whose Signedness
    operand is 0.

    Float Value must be a scalar or vector of floating-point type.  It must
    have the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form

    ```
    convert-f-to-u-op ::= ssa-id `=` `spv.ConvertFToUOp` ssa-use
                          `:` operand-type `to` result-type
    ```

    For example:

    ```
    %1 = spv.ConvertFToU %0 : f32 to i32
    %3 = spv.ConvertFToU %2 : vector<3xf32> to vector<3xi32>
    ```
  }];
}

// -----

def SPV_ConvertSToFOp : SPV_CastOp<"ConvertSToF", SPV_Float, SPV_Integer, []> {
  let summary = [{
    Convert value numerically from signed integer to floating point.
  }];

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

    Signed Value must be a scalar or vector of integer type.  It must have
    the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form

    ```
    convert-s-to-f-op ::= ssa-id `=` `spv.ConvertSToFOp` ssa-use
                          `:` operand-type `to` result-type
    ```

    For example:

    ```
    %1 = spv.ConvertSToF %0 : i32 to f32
    %3 = spv.ConvertSToF %2 : vector<3xi32> to vector<3xf32>
    ```
  }];
}

// -----

def SPV_ConvertUToFOp : SPV_CastOp<"ConvertUToF", SPV_Float, SPV_Integer, []> {
  let summary = [{
    Convert value numerically from unsigned integer to floating point.
  }];

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

    Unsigned Value must be a scalar or vector of integer type.  It must have
    the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form

    ```
    convert-u-to-f-op ::= ssa-id `=` `spv.ConvertUToFOp` ssa-use
                          `:` operand-type `to` result-type
    ```

    For example:

    ```
    %1 = spv.ConvertUToF %0 : i32 to f32
    %3 = spv.ConvertUToF %2 : vector<3xi32> to vector<3xf32>
    ```
  }];
}

// -----

def SPV_FConvertOp : SPV_CastOp<"FConvert", SPV_Float, SPV_Float, []> {
  let summary = [{
    Convert value numerically from one floating-point width to another
    width.
  }];

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

    Float Value must be a scalar or vector of floating-point type.  It must
    have the same number of components as Result Type.  The component width
    cannot equal the component width in Result Type.

     Results are computed per component.

    ### Custom assembly form

    ```
    f-convert-op ::= ssa-id `=` `spv.FConvertOp` ssa-use
                     `:` operand-type `to` result-type
    ```

    For example:

    ```
    %1 = spv.FConvertOp %0 : f32 to f64
    %3 = spv.FConvertOp %2 : vector<3xf32> to vector<3xf64>
    ```
  }];

  let verifier = [{ return verifyCastOp(this->getOperation(), false); }];
}

// -----

def SPV_SConvertOp : SPV_CastOp<"SConvert", SPV_Integer, SPV_Integer, []> {
  let summary = [{
    Convert signed width.  This is either a truncate or a sign extend.
  }];

  let description = [{
    Result Type must be a scalar or vector of integer type.

    Signed Value must be a scalar or vector of integer type.  It must have
    the same number of components as Result Type.  The component width
    cannot equal the component width in Result Type.

     Results are computed per component.

    ### Custom assembly form

    ```
    s-convert-op ::= ssa-id `=` `spv.SConvertOp` ssa-use
                     `:` operand-type `to` result-type
    ```

    For example:

    ```
    %1 = spv.SConvertOp %0 : i32 to i64
    %3 = spv.SConvertOp %2 : vector<3xi32> to vector<3xi64>
    ```
  }];

  let verifier = [{ return verifyCastOp(this->getOperation(), false); }];
}

// -----

def SPV_UConvertOp : SPV_CastOp<"UConvert", SPV_Integer, SPV_Integer, []> {
  let summary = [{
    Convert unsigned width. This is either a truncate or a zero extend.
  }];

  let description = [{
    Result Type must be a scalar or vector of integer type, whose Signedness
    operand is 0.

    Unsigned Value must be a scalar or vector of integer type.  It must have
    the same number of components as Result Type.  The component width
    cannot equal the component width in Result Type.

     Results are computed per component.

    ### Custom assembly form

    ```
    u-convert-op ::= ssa-id `=` `spv.UConvertOp` ssa-use
                 `:` operand-type `to` result-type
    ```

    For example:

    ```
    %1 = spv.UConvertOp %0 : i32 to i64
    %3 = spv.UConvertOp %2 : vector<3xi32> to vector<3xi64>
    ```
  }];

  let verifier = [{ return verifyCastOp(this->getOperation(), false); }];
}

#endif // SPIRV_CAST_OPS
