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

#ifdef OP_BASE
#else
#define OP_BASE

//===----------------------------------------------------------------------===//
// Predicates.
//===----------------------------------------------------------------------===//

// Singular predicate condition.
class PredAtom<code call, bit neg = 0> {
  // The function to invoke to compute the predicate.
  code predCall = call;

  // Whether the predicate result should be negated.
  bit negated = neg;
}

// Predicate atoms in conjunctive normal form. The inner list consists
// of PredAtoms, one of which in the list must hold, while all the outer
// most conditions must hold. Conceptually
//   all_of(outer_conditions, any_of(inner_conditions)).
class PredCNF<list<list<PredAtom>> conds> {
  list<list<PredAtom>> conditions = conds;
}

def IsVectorTypePred : PredAtom<"{0}.isa<VectorType>()">;

def IsTensorTypePred : PredAtom<"{0}.isa<TensorType>()">;

// For a TensorType, verify that it is a statically shaped tensor.
def IsStaticShapeTensorTypePred :
  PredAtom<"{0}.cast<TensorType>().hasStaticShape()">;

//===----------------------------------------------------------------------===//
// Types.
//===----------------------------------------------------------------------===//

// Base class for all types.
class Type {
  // The builder call to invoke (if specified) to construct the Type.
  // Format: this will be affixed to the builder.
  code builderCall = ?;
  // The predicates that this type satisfies.
  // Format: {0} will be expanded to the type.
  PredCNF predicate = ?;
}

// Integer types.
class I<int width> : Type {
  int bitwidth = width;
  let builderCall = "getIntegerType(" # bitwidth # ")";
  let predicate = PredCNF<[[PredAtom<"{0}.isInteger(" # bitWidth # ")">]]>;
}
def I1  : I<1>;
def I32 : I<32>;

// Floating point types.
class F<int width> : Type {
  int bitwidth = width;
}
def F32 : F<32> {
  let builderCall = "getF32Type()";
  let predicate = PredCNF<[[PredAtom<"{0}.isF32()">]]>;
}

// A container type is a type that has another type embedded within it.
class ContainerType<Type etype, PredCNF containerPred> : Type {
  // The type of elements in the container.
  Type elementType = etype;

  // Call to retrieve.
  code getElementTypeCall = ?;

  let predicate = PredCNF<
    !foldl(
      // Initialize with the predicate of the container.
      containerPred.conditions,
      // Add constraints of the element type. This uses TableGen foldl (fold
      // left) to iterate over the rules of the element type's predicates,
      // expanding '{0}' which correspond to the type of the element to
      // getElementTypeCall of the container type so that the
      // predicates of the element type are applied to the elements of
      // the container.
      elementType.predicate.conditions, a, b,
      !listconcat(a, [!foldl([]<PredAtom>, b, c, d,
        !listconcat(c, [PredAtom<
          !subst("{0}", !cast<string>(getElementTypeCall),
                        !cast<string>(d.predCall))>]
        ))]
      )
    )
  >;
}

// Vector types.
class Vector<Type t, list<int> dims> : ContainerType<t, PredCNF<[
    [IsVectorTypePred],
    // Match dims. Construct an ArrayRef with the elements of `dims` by folding
    // over the list.
    [PredAtom<"{0}.cast<VectorType>().getShape() == ArrayRef{{" #
      !foldl("", dims, sum, element, sum #
       !if(!empty(sum), "", ",") # !cast<string>(element)) # "}">]
    ]>> {
  list<int> dimensions = dims;
  let getElementTypeCall = "{0}.cast<VectorType>().getElementType()";
}

// Tensor type.

// This represents a generic tensor without constraints on elemental type,
// rank, size. As there is no constraint on elemental type, derive from Type
// directly instead of ContainerType.
def Tensor : Type {
  let predicate = PredCNF<[[IsTensorTypePred]]>;
}

// A tensor with static shape but no other constraints. Note: as
// Tensor is a def this doesn't derive from it, but reuses the predicate
// that must hold for it to be a tensor.
def StaticShapeTensor : Type {
  let predicate = PredCNF<
    !listconcat(Tensor.predicate.conditions, [[IsStaticShapeTensorTypePred]])
  >;
}

// For typed tensors.
class TypedTensor<Type t> : ContainerType<t, Tensor.predicate> {
  let getElementTypeCall = "{0}.cast<TensorType>().getElementType()";
}

def F32Tensor : TypedTensor<F32>;

// String type.
def String : Type;

// Type corresponding to derived attribute.
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
  // Format: {0} will be expanded to the attribute. So
  // '{0}.getValue().convertToFloat()' for 'FloatAttr val' will expand to
  // 'getAttrOfType<FloatAttr>("val").getValue().convertToFloat()'.
  code convertFromStorage = "{0}.getValue()";
}

def BoolAttr : Attr<I1> {
  let storageType = [{ BoolAttr }];
  let returnType = [{ bool }];
}
def ElementsAttr : Attr<?> {
  let storageType = [{ ElementsAttr }];
  let returnType = [{ ElementsAttr }];
  code convertFromStorage = "{0}";
}
def F32Attr : Attr<F32> {
  let storageType = [{ FloatAttr }];
  let returnType = [{ float }];
  let convertFromStorage = [{ {0}.getValue().convertToFloat() }];
}
def I32Attr : Attr<I32> {
  let storageType = [{ IntegerAttr }];
  let returnType = [{ int }];
  let convertFromStorage = [{ {0}.getValue().getSExtValue() }];
}
def StrAttr : Attr<String> {
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

// Derived attribute that returns a mlir::Type.
class DerivedTypeAttr<code body> : DerivedAttr<"Type", body>;

// Represents a constant attribute of specific Attr type. The leaf class that
// derives from this should additionally include a `value` member.
class ConstantAttr<Attr attribute> {
  Attr attr = attribute;
}

// The values for const F32 attributes are set as strings as floating point
// values can't be provided directly in TableGen.
class ConstF32Attr<string val> : ConstantAttr<F32Attr> {
  string value = val;
}

//===----------------------------------------------------------------------===//
// Op Properties
//===----------------------------------------------------------------------===//

class OpProperty;

//
// Note: These are hard coded into mlir-tblgen.
//
def Commutative   : OpProperty;   // X op Y == Y op X
def NoSideEffect  : OpProperty;   // op has no side effect

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

// Marker used to identify the argument list for an op.
def ins;

// Base class for all ops.
class Op<string mnemonic, list<OpProperty> props = []> {
  // The mnemonic of the op.
  string opName = mnemonic;

  // One-line human-readable description of what the op does.
  string summary = ?;

  // Additional, longer human-readable description of what the op does.
  string description = ?;

  // Dag containting the arguments of the op. Default 0 arguments.
  dag arguments = (ins);

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
  //                     Value arg0, Value arg1, ...,
  //                     Attribute <attr0-name>, Attribute <attr1-name>, ...);
  //
  //   * where the attributes follow the same declaration order as in the op.
  //
  //   static void build(Builder* builder, OperationState* result,
  //                     ArrayRef<Type> resultTypes,
  //                     ArrayRef<Value> args,
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

  // Whether this op has a constant folder.
  bit hasConstantFolder = 0b0;

  // Op properties.
  list<OpProperty> properties = props;
}

// The arguments of an op.
class Arguments<dag types> {
  dag arguments = types;
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
    Op<mnemonic, props>, Arguments<(ins Tensor:$arg)>, Results<[Tensor]>;

class BinaryOp<string mnemonic, list<OpProperty> props> :
    Op<mnemonic, props>, Arguments<(ins Tensor:$lhs, Tensor:$rhs)>,
                                    Results<[Tensor]>;

class TernaryOp<string mnemonic, list<OpProperty> props> :
    Op<mnemonic, props>, Arguments<(ins Tensor, Tensor, Tensor)>;

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//
// Base class for op+ -> op+ rewrite patterns. These allow declaratively
// specifying rewrite patterns.
// TODO(jpienaar): Add the constraint list along with the Pattern.
class Pattern<dag patternToMatch, list<dag> resultOps> {
  dag PatternToMatch = patternToMatch;
  list<dag> ResultOps = resultOps;
}

// Form of a pattern which produces a single result.
class Pat<dag pattern, dag result> : Pattern<pattern, [result]>;

#endif // OP_BASE
