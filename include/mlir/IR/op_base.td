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

// A logical predicate.
class Pred;

// Logical predicate wrapping a C expression.
class CPred<code pred> : Pred {
  code predCall = "(" # pred # ")";
}

// Kinds of combined logical predicates.  These must closesly match the
// predicates implemented by the C++ backend (tblgen::PredCombinerKind).
class PredCombinerKind;
def PredCombinerAnd : PredCombinerKind;
def PredCombinerOr : PredCombinerKind;
def PredCombinerNot : PredCombinerKind;
def PredCombinerSubstLeaves : PredCombinerKind;

// A predicate that combines other predicates as defined by PredCombinerKind.
// Instantiated below.
class CombinedPred<PredCombinerKind k, list<Pred> c> : Pred {
  PredCombinerKind kind = k;
  list<Pred> children = c;
}

// A predicate that holds if all of its children hold.  Always holds for zero
// children.
class AllOf<list<Pred> children> : CombinedPred<PredCombinerAnd, children>;

// A predicate that holds if any of its children hold.  Never holds for zero
// children.
class AnyOf<list<Pred> children> : CombinedPred<PredCombinerOr, children>;

// A predicate that holds if its child does not.
class Neg<Pred child> : CombinedPred<PredCombinerNot, [child]>;

// A predicate that substitutes "pat" with "repl" in predicate calls of the
// leaves of the predicate tree (i.e., not CombinedPredicates).  This is plain
// string substitution without regular expressions or captures, new predicates
// with more complex logical can be introduced should the need arise.
class SubstLeaves<string pat, string repl, Pred child>
    : CombinedPred<PredCombinerSubstLeaves, [child]> {
  string pattern = pat;
  string replacement = repl;
}

//===----------------------------------------------------------------------===//
// Type predicates.  ({0} is replaced by an instance of mlir::Type)
//===----------------------------------------------------------------------===//

// Whether a type is a VectorType.
def IsVectorTypePred : CPred<"{0}.isa<VectorType>()">;

// Whether a type is a TensorType.
def IsTensorTypePred : CPred<"{0}.isa<TensorType>()">;

// For a TensorType, verify that it is a statically shaped tensor.
def IsStaticShapeTensorTypePred :
  CPred<"{0}.cast<TensorType>().hasStaticShape()">;

//===----------------------------------------------------------------------===//
// Type constraints and types.
//===----------------------------------------------------------------------===//

// A constraint on types.  This can be used to check the validity of
// instruction arguments.
class TypeConstraint<Pred condition, string descr> {
  // The predicates that this type satisfies.
  // Format: {0} will be expanded to the type.
  Pred predicate = condition;
  // User-readable description used, e.g., for error reporting.  If empty,
  // a generic message will be used instead.
  string description = descr;
}

// A type, carries type constraints, but accepts any type by default.
class Type<Pred condition, string descr = "">
    : TypeConstraint<condition, descr>;

// A type that can be constructed using MLIR::Builder.
// Note that this does not "inherit" from Type because it would require
// duplicating Type subclasses for buildable and non-buildable cases to avoid
// diamond "inheritance".
// TODO(zinenko): we may extend this to a more general 'Buildable' trait,
// making some Types and some Attrs buildable.
class BuildableType<code builder> {
  // The builder call to invoke (if specified) to construct the BuildableType.
  // Format: this will be affixed to the builder.
  code builderCall = builder;
}

// Integer types.
class IntegerBase<CPred pred, string descr> : Type<pred, descr>;

// Any integer type irrespective of its width.
def Integer : IntegerBase<CPred<"{0}.isa<IntegerType>()">, "integer">;

// Index type.
def Index : IntegerBase<CPred<"{0}.isa<IndexType>()">, "index">;

// Integer type of a specific width.
class I<int width>
    : IntegerBase<CPred<"{0}.isInteger(" # width # ")">,
                  width # "-bit integer">,
      BuildableType<"getIntegerType(" # width # ")"> {
  int bitwidth = width;
}
def I1  : I<1>;
def I32 : I<32>;

// Floating point types.
class FloatBase<CPred pred, string descr> : Type<pred, descr>;

// Any float type irrespective of its width.
def Float : FloatBase<CPred<"{0}.isa<FloatType>()">, "floating-point">;

// Float type of a specific width.
class F<int width>
    : FloatBase<CPred<"{0}.isF" # width # "()">,
                width # "-bit float">,
      BuildableType<"getF" # width # "Type()"> {
  int bitwidth = width;
}

def F32 : F<32>;

// A container type is a type that has another type embedded within it.
class ContainerType<Type etype, Pred containerPred, code elementTypeCall,
                    string descr> :
    // First, check the container predicate.  Then, substitute the extracted
    // element into the element type checker.
    Type<AllOf<[containerPred,
                SubstLeaves<"{0}", !cast<string>(elementTypeCall),
                etype.predicate>]>,
         descr # " of " # etype.description # " values"> {
  // The type of elements in the container.
  Type elementType = etype;

  // Call to retrieve.
  code getElementTypeCall = elementTypeCall;
}

// Vector types.
class TypedVector<Type t> : ContainerType<t, IsVectorTypePred,
    "{0}.cast<VectorType>().getElementType()", "vector">;

class Vector<Type t, list<int> dims> : ContainerType<t, AllOf<[
    IsVectorTypePred,
    // Match dims. Construct an ArrayRef with the elements of `dims` by folding
    // over the list.
    CPred<"{0}.cast<VectorType>().getShape() == ArrayRef{{" #
      !foldl("", dims, sum, element, sum #
       !if(!empty(sum), "", ",") # !cast<string>(element)) # "}">]>,
    "{0}.cast<VectorType>().getElementType()",
    "vector"> {
  list<int> dimensions = dims;
}

// Tensor type.

// This represents a generic tensor without constraints on elemental type,
// rank, size. As there is no constraint on elemental type, derive from Type
// directly instead of ContainerType.
def Tensor : Type<IsTensorTypePred, "tensor">;

// A tensor with static shape but no other constraints. Note: as
// Tensor is a def this doesn't derive from it, but reuses the predicate
// that must hold for it to be a tensor.
def StaticShapeTensor
    : Type<AllOf<[Tensor.predicate, IsStaticShapeTensorTypePred]>,
           "statically shaped tensor">;

// For typed tensors.
class TypedTensor<Type t>
    : ContainerType<t, Tensor.predicate,
                    "{0}.cast<TensorType>().getElementType()",
                    "tensor">;

def F32Tensor : TypedTensor<F32>;

// Type constraint for integer-like types: integers, indices, vectors of
// integers, tensors of integers.
def IntegerLike : TypeConstraint<AnyOf<[Integer.predicate, Index.predicate,
        TypedVector<Integer>.predicate, TypedTensor<Integer>.predicate]>,
    "integer-like">;

// Type constraint for float-like types: floats, vectors or tensors thereof.
def FloatLike : TypeConstraint<AnyOf<[Float.predicate,
        TypedVector<Float>.predicate, TypedTensor<Float>.predicate]>,
    "floating-point-like">;

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

// A constraint on attributes. This can be used to check the validity of
// instruction attributes.
class AttrConstraint<Pred condition> {
  // The predicates that this attribute satisfies.
  // Format: {0} will be expanded to the attribute.
  Pred predicate = condition;
}

// Base class for all attributes.
class Attr<Pred condition = CPred<"true">> : AttrConstraint<condition> {
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

  // The call expression that builds an attribute from a constant value.
  //
  // Format: {0} will be expanded to an instance of mlir::Builder, {1} will be
  // expanded to the constant value of the attribute.  For example,
  // '{0}.getStringAttr("{1}")' for 'StringAttr:"foo"' will expand to
  // 'builder.getStringAttr("foo")'.
  code constBuilderCall = ?;

  // Default value for attribute.
  // Requires a constBuilderCall defined.
  string defaultValue = ?;
}

// A generic attribute that must be constructed around a specific type.
// Backed by a C++ class "attrName".
class TypeBasedAttr<BuildableType t, string attrName> : Attr {
  let constBuilderCall =
      "{0}.get" # attrName # "({0}." # t.builderCall # ", {1})";
  let storageType = attrName;
}

// An attribute backed by a string type.
class StringBasedAttr : Attr {
  let constBuilderCall = [{ {0}.getStringAttr("{1}") }];
  let storageType = [{ StringAttr }];
}

// Base class for instantiating float attributes of fixed width.
class FloatAttrBase<BuildableType t> : TypeBasedAttr<t, "FloatAttr">;

// Base class for instantiating integer attributes of fixed width.
class IntegerAttrBase<BuildableType t> : TypeBasedAttr<t, "IntegerAttr">;

def BoolAttr : Attr {
  let storageType = [{ BoolAttr }];
  let returnType = [{ bool }];
  let constBuilderCall = [{ {0}.getBoolAttr({1}) }];
}
def ArrayAttr : Attr {
  let storageType = [{ ArrayAttr }];
  let returnType = [{ ArrayAttr }];
  code convertFromStorage = "{0}";
}
def ElementsAttr : Attr {
  let storageType = [{ ElementsAttr }];
  let returnType = [{ ElementsAttr }];
  let convertFromStorage = "{0}";
}
def F32Attr : FloatAttrBase<F32> {
  let returnType = [{ float }];
  let convertFromStorage = [{ {0}.getValue().convertToFloat() }];
}
def I32Attr : IntegerAttrBase<I32> {
  let storageType = [{ IntegerAttr }];
  let returnType = [{ int }];
  let convertFromStorage = [{ {0}.getValue().getSExtValue() }];
}
def StrAttr : StringBasedAttr {
  let storageType = [{ StringAttr }];
  let returnType = [{ StringRef }];
  let constBuilderCall = [{ {0}.getStringAttr("{1}") }];
}

// DerivedAttr are attributes whose value is computed from properties
// of the operation. They do not require additional storage and are
// materialized as needed.
class DerivedAttr<code ReturnType, code Body> : Attr {
  let returnType = ReturnType;
  code body = Body;
}

// Derived attribute that returns a mlir::Type.
class DerivedTypeAttr<code body> : DerivedAttr<"Type", body>;

// Represents a constant attribute of specific Attr type. A constant
// attribute can be specified only of attributes that have a constant
// builder call defined. The constant value is specified as a string.
class ConstantAttr<Attr attribute, string val> {
  Attr attr = attribute;
  string value = val;
}

class ConstF32Attr<string val> : ConstantAttr<F32Attr, val>;

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

// Marker used to identify the result list for an op.
def outs;

// Base class for all ops.
class Op<string mnemonic, list<OpProperty> props = []> {
  // The mnemonic of the op.
  string opName = mnemonic;

  // One-line human-readable description of what the op does.
  string summary = "";

  // Additional, longer human-readable description of what the op does.
  string description = "";

  // Dag containting the arguments of the op. Default to 0 arguments.
  dag arguments = (ins);

  // The list of results of the op. Default to 0 results.
  dag results = (outs);

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
  bit hasCanonicalizer = 0b0;

  // Whether this op has a constant folder.
  bit hasConstantFolder = 0b0;

  // Whether this op has a folder.
  bit hasFolder = 0b0;

  // Op properties.
  list<OpProperty> properties = props;
}

// The arguments of an op.
class Arguments<dag args> {
  dag arguments = args;
}

// The results of an op.
class Results<dag rets> {
  dag results = rets;
}

// The traits of an op.
// Traits are defined in C++ and need to be included for the generated
// op definitions.
class Traits<list<string> Traits> {
  list<string> traits = Traits;
}

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

// Attribute matcher. This is the base class to specify a predicate
// that has to match. Used on the input attributes of a rewrite rule.
class mAttr<Pred pred> : AttrConstraint<pred>;

// Attribute transforms. This is the base class to specify a
// transformation of a matched attribute. Used on the output of a rewrite
// rule.
class tAttr<code transform> {
  // Code to transform the attribute.
  // Format: {0} represents the attribute.
  code attrTransform = transform;
}

// Marker used to indicate that no new result op are generated by applying the
// rewrite pattern, so to replace the matched DAG with an existing SSA value.
def replaceWithValue;

#endif // OP_BASE
