//===-- op_base.td - Base op definition file ---------------*- tablegen -*-===//
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
// This is the base operation definition file.
//
//===----------------------------------------------------------------------===//

// Base class for all types.
class Type {
  string builder = ?;
}

// Scalar types.
def F32 : Type {
  let builder = "getF32Type()";
}
class I<int width> : Type {
  let builder = "getIntegerType(" # width # ")";
  int intWidth = width;
}

// Vector types.
class Vector<Type t, list<int> dims> : Type {
  Type elementType = t;
  list<int> dimensions = dims;
}

// Tensor type.
// This represents a generic tensor without constraints on elemental type,
// rank, size.
def Tensor : Type;

def String : Type;

// The operands of an op.
class Operands<list<Type> types> {
  list<Type> operandTypes = types;
}

// The result types of an op.
class Results<list<Type> types> {
  list<Type> returnTypes = types;
}

// Add an attribute to the generated op class.
class Attr<Type t> {
  Type type = t;

  // Define the storage and return type in the subclass.
  code storageType = ?;
  code returnType = ?;

  // Define converter method to convert from the storage type to the return
  // type. For example, the FloatAttr storage is APFloat, while the returned
  // primitive type could be float. Similiarly a enum can be stored as a int
  // but returned as an enum class.
  //
  // Format: {0} will be expanded to the attribute's value. So
  // 'return {0}.convertToFloat();' for 'FloatAttr val' will expand to
  // 'return getAttrOfType<FloatAttr>("val").getValue().convertToFloat();'
  code convertFromStorage = "return {0};";
}

class BoolAttr : Attr<I<1>> {
 let storageType = [{ BoolAttr }];
 let returnType = [{ bool }];
}
class F32Attr : Attr<F32> {
 let storageType = [{ FloatAttr }];
 let returnType = [{ float }];
 let convertFromStorage = [{ return {0}.convertToFloat(); }];
}
class I32Attr : Attr<I<32>> {
 let storageType = [{ IntegerAttr }];
 let returnType = [{ int }];
 let convertFromStorage = [{ return {0}.getSExtValue(); }];
}
class StrAttr : Attr<String> {
 let storageType = [{ StringAttr }];
 let returnType = [{ StringRef }];
}

// DerivedAttr are attributes whose value is computed from properties
// of the operation. They do not require additional storage and are
// materialized as needed.
def DerivedAttrBody : Type;
class DerivedAttr<code ReturnType, code Body> : Attr<DerivedAttrBody> {
  let returnType = ReturnType;
  code body = Body;
}

// Class representing a Trait (defined in a C++ file that needs to be included
// before the generated op definitions).
class Traits<list<string> Traits> {
  list<string> traits = Traits;
}

class OpProperty;

//===----------------------------------------------------------------------===//
// Op Properties.
//
// Note: These are hard coded into mlir-op-gen.
//
def Commutative   : OpProperty;   // X op Y == Y op X
def NoSideEffect  : OpProperty;   // Sets 'HasNoSideEffects'.

// Base class for Ops.
class Op<string mnemonic, list<OpProperty> props = []> {
  // The mnemonic of the Op.
  string name = mnemonic;

  // One-line human-readable description of what the Op does.
  string summary = ?;

  // Additional, longer human-readable description of what the Op does.
  string description = ?;

  // The list of operands of the Op. Default 0 operands.
  list<Type> operandTypes = [];

  // The list of return types of the Op. Default no return type set.
  list<Type> returnTypes = [];

  // Attribute getters can be added to the op by adding an Attr member
  // with the name and type of the attribute. E.g., adding int attribute
  // with name "value" and type "i32":
  //   I32Attr value;

  // Define the hooks used for building, parsing, printing, verification.
  // Custom builder.
  code builder = ?;

  // Custom parser.
  code parser = ?;

  // Custom printer.
  code printer = ?;

  // Custom verifier.
  code verifier = ?;

  // Whether this op has associated canonicalization patterns.
  // TODO(b/120163349): figure out a better way to write canonicalization
  // patterns in TableGen rules directly instead of using this marker
  // and C++ implementations.
  bit hasCanonicalizationPatterns = 0b0;

  // Op properties.
  list<OpProperty> properties = props;
}

class BinaryOp<string mnemonic, list<OpProperty> props> :
    Op<mnemonic, props>, Operands<[Tensor, Tensor]>, Results<[Tensor]> {

  // TODO(jpienaar): To autogen the builder the type of the result needs to be
  // determined from the operands. That would (beyond trivial cases) require
  // type propagation information.
  let builder = [{
    static void build(Builder *builder, OperationState *result, SSAValue *lhs,
	              SSAValue *rhs) {
      return impl::buildBinaryOp(builder, result, lhs, rhs);
    }
  }];
}

class TernaryOp<string mnemonic, list<OpProperty> props> :
    Op<mnemonic, props>, Operands<[Tensor, Tensor, Tensor]>;

class UnaryOp<string mnemonic, list<OpProperty> props> :
    Op<mnemonic, props>, Operands<[Tensor]>, Results<[Tensor]> {
  let builder = [{
    static void build(Builder *builder, OperationState *result,
                      SSAValue *arg) {
      result->addOperands({arg});
      result->types.push_back(arg->getType());
    }
  }];
}
