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

//===----------------------------------------------------------------------===//
// Types.
//===----------------------------------------------------------------------===//

// Base class for all types.
class Type;

// Integer types.
class I<int width> : Type {
  int bitwidth = width;
}
def I1  : I<1>;
def I32 : I<32>;

// Floating point types.
class F<int width> : Type {
  int bitwidth = width;
}
def F32 : F<32>;

// Vector types.
class Vector<Type t, list<int> dims> : Type {
  Type elementType = t;
  list<int> dimensions = dims;
}

// Tensor type.
// This represents a generic tensor without constraints on elemental type,
// rank, size.
def Tensor : Type;

// String type.
def String : Type;

def DerivedAttrBody : Type;

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

// Base class for all attributes.
class Attr<Type t> {
  Type type = t;

  code storageType = ?; // The backing mlir::Attribute type
  code returnType = ?;  // The underlying C++ value type

  // Define converter method to convert from the storage type to the return
  // type. For example, an enum can be stored as an int but returned as an
  // enum class.
  //
  // Format: {0} will be expanded to the attribute's value. So
  // '{0}.convertToFloat()' for 'FloatAttr val' will expand to
  // 'getAttrOfType<FloatAttr>("val").getValue().convertToFloat()'.
  code convertFromStorage = "{0}";
}

class BoolAttr : Attr<I1> {
  let storageType = [{ BoolAttr }];
  let returnType = [{ bool }];
}
class F32Attr : Attr<F32> {
  let storageType = [{ FloatAttr }];
  let returnType = [{ float }];
  let convertFromStorage = [{ {0}.convertToFloat() }];
}
class I32Attr : Attr<I32> {
  let storageType = [{ IntegerAttr }];
  let returnType = [{ int }];
  let convertFromStorage = [{ {0}.getSExtValue() }];
}
class StrAttr : Attr<String> {
  let storageType = [{ StringAttr }];
  let returnType = [{ StringRef }];
}

// DerivedAttr are attributes whose value is computed from properties
// of the operation. They do not require additional storage and are
// materialized as needed.
class DerivedAttr<code ReturnType, code Body> : Attr<DerivedAttrBody> {
  let returnType = ReturnType;
  code body = Body;
}

//===----------------------------------------------------------------------===//
// Op Properties
//===----------------------------------------------------------------------===//

class OpProperty;

//
// Note: These are hard coded into mlir-op-gen.
//
def Commutative   : OpProperty;   // X op Y == Y op X
def NoSideEffect  : OpProperty;   // op has no side effect

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

// Marker used to identify the operand list for an op.
def ins;

// Base class for all ops.
class Op<string mnemonic, list<OpProperty> props = []> {
  // The mnemonic of the op.
  string opName = mnemonic;

  // One-line human-readable description of what the op does.
  string summary = ?;

  // Additional, longer human-readable description of what the op does.
  string description = ?;

  // Dag containting the operands of the op. Default 0 operands.
  dag operands = (ins);

  // The list of return types of the op. Default no return type set.
  list<Type> returnTypes = [];

  // Attribute getters can be added to the op by adding an Attr member
  // with the name and type of the attribute. E.g., adding int attribute
  // with name "value" and type "i32":
  //   I32Attr value;

  // Define the hooks used for building, parsing, printing, verification.

  // Custom builder.
  // If a derived class/def does not override this, then two default builders
  // are generated, with the following signatures:
  //
  //   static void build(Builder* builder, OperationState* result,
  //                     Type resultType0, Type resultType1, ...,
  //                     SSAValue* arg0, SSAValue* arg1, ...,
  //                     Attribute <attr0-name>, Attribute <attr1-name>, ...);
  //
  //   * where the attributes follow the same declaration order as in the op.
  //
  //   static void build(Builder* builder, OperationState* result,
  //                     ArrayRef<Type> resultTypes,
  //                     ArrayRef<SSAValue*> args,
  //                     ArrayRef<NamedAttribute> attributes);
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

// The operand types of an op.
class Operands<list<Type> types> {
  // TODO(jpienaar): Switch operands to use named operand instead of using list.
  dag operands = !dag(ins, types, ?);
}

// The result types of an op.
class Results<list<Type> types> {
  list<Type> returnTypes = types;
}

// The traits of an op.
// Traits are defined in C++ and need to be included for the generated
// op definitions.
class Traits<list<string> Traits> {
  list<string> traits = Traits;
}

class UnaryOp<string mnemonic, list<OpProperty> props> :
    Op<mnemonic, props>, Operands<[Tensor]>, Results<[Tensor]>;

class BinaryOp<string mnemonic, list<OpProperty> props> :
    Op<mnemonic, props>, Operands<[Tensor, Tensor]>, Results<[Tensor]>;

class TernaryOp<string mnemonic, list<OpProperty> props> :
    Op<mnemonic, props>, Operands<[Tensor, Tensor, Tensor]>;
