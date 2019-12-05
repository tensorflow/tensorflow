//===-- OpBase.td - Base op definition file ----------------*- tablegen -*-===//
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

#ifndef OP_BASE
#define OP_BASE

//===----------------------------------------------------------------------===//
// Common utilities for defining TableGen mechanisms
//===----------------------------------------------------------------------===//

// A workaround for the inability to define functions in Tablegen.
//
// The template parameter defines a string that can be extracted from an
// instance of this class by accessing the "result" member. Subclasses can take
// their own template parameters as function "arguments" and use them to
// populate result.
// For example, if it didn't already exist, a concat function could be defined
// like:
//
// class StrConcat<list<string> strings> :
//     StrFunc<!foldl("", strings, prev, cur, prev # cur)>
//
// and then called like
//
// StrConcat<["a", "b", "c"]>.result
//
// to get the string "abc"
class StrFunc<string r> {
  string result = r;
}

// Concatenates a list of strings with a separator (default ", ")
class StrJoin<list<string> strings, string sep = ", "> :
    StrFunc<!if(!empty(strings), "",
         !foldl(!head(strings), !tail(strings), prev, cur, prev # sep # cur))>;

// Concatenates a list of integers into a string with a separator (default ", ")
class StrJoinInt<list<int> integers, string sep = ", "> :
    StrJoin<!foreach(i, integers, !cast<string>(i)), sep>;

//===----------------------------------------------------------------------===//
// Predicate definitions
//===----------------------------------------------------------------------===//

// Base class for logical predicates.
//
// Predicates are used to compose constraints (see next section for details).
// There are two categories of predicates:
//
// 1. CPred: the primitive leaf predicate.
// 2. Compound predicate: a predicate composed from child predicates using
//    predicate combiners ("conjunction", "disjunction", "negation" or
//    "substitution").
class Pred;

// A logical predicate wrapping any C expression.
//
// This is the basis for composing more complex predicates. It is the "atom"
// predicate from the perspective of TableGen and the "interface" between
// TableGen and C++. What is inside is already C++ code, which will be treated
// as opaque strings with special placeholders to be substituted.
//
// ## Special placeholders
//
// Special placeholders can be used to refer to entities in the context where
// this predicate is used. They serve as "hooks" to the enclosing environment.
// The following special placeholders are supported in constraints for an op:
//
// * `$_builder` will be replaced by a mlir::Builder instance.
// * `$_op` will be replaced by the current operation.
// * `$_self` will be replaced with the entity this predicate is attached to.
//   E.g., `BoolAttr` is an attribute constraint that wraps a
//   `CPred<"$_self.isa<BoolAttr>()">` (see the following sections for details).
//   Then for `F32:$attr`,`$_self` will be replaced by `$attr`.
//   For type constraints, it's a little bit special since we want the
//   constraints on each type definition reads naturally and we want to attach
//   type constraints directly to an operand/result, $_self will be replaced
//   by the operand/result's type. E.g., for `F32` in `F32:$operand`, its
//   `$_self` will be expanded as `getOperand(...)->getType()`.
class CPred<code pred> : Pred {
  code predExpr = "(" # pred # ")";
}

// Kinds of predicate combiners.  These must closely match the predicates
// implemented by the C++ backend (tblgen::PredCombinerKind).
class PredCombinerKind;
def PredCombinerAnd : PredCombinerKind;
def PredCombinerOr : PredCombinerKind;
def PredCombinerNot : PredCombinerKind;
def PredCombinerSubstLeaves : PredCombinerKind;
def PredCombinerConcat : PredCombinerKind;

// A predicate that combines other predicates as defined by PredCombinerKind.
// Instantiated below.
class CombinedPred<PredCombinerKind k, list<Pred> c> : Pred {
  PredCombinerKind kind = k;
  list<Pred> children = c;
}

// Predicate combiners

// A predicate that holds if all of its children hold.  Always holds for zero
// children.
class And<list<Pred> children> : CombinedPred<PredCombinerAnd, children>;

// A predicate that holds if any of its children hold.  Never holds for zero
// children.
class Or<list<Pred> children> : CombinedPred<PredCombinerOr, children>;

// A predicate that holds if its child does not.
class Neg<Pred child> : CombinedPred<PredCombinerNot, [child]>;

// A predicate that substitutes "pat" with "repl" in predicate calls of the
// leaves of the predicate tree (i.e., not CombinedPred).
//
// This is plain string substitution without regular expressions or captures.
// New predicates with more complex logical can be introduced should the need
// arise.
class SubstLeaves<string pat, string repl, Pred child>
    : CombinedPred<PredCombinerSubstLeaves, [child]> {
  string pattern = pat;
  string replacement = repl;
}

// A predicate that prepends `pre` and appends `suf` to the final predicate
// string composed from `child`. This is plain string concatenation and there
// will be no substitution happening for `pre` and `suf`.
class Concat<string pre, Pred child, string suf> :
    CombinedPred<PredCombinerConcat, [child]> {
  string prefix = pre;
  string suffix = suf;
}

//===----------------------------------------------------------------------===//
// Constraint definitions
//===----------------------------------------------------------------------===//

// TODO(b/130064155): Merge Constraints into Pred.

// Base class for named constraints.
//
// An op's operands/attributes/results can have various requirements, e.g.,
// having certain types, having values inside a certain range, and so on.
// Besides, for a graph rewrite rule, the source pattern used to match against
// the existing graph has conditions, like the op's operand must be of a more
// constrained subtype, the attribute must have a certain value, and so on.
//
// These requirements and conditions are modeled using this class. Records of
// this class are used to generate verification code in op verifier, and
// matching code in pattern matcher.
//
// Constraints are predicates with descriptive names, to facilitate inspection,
// provide nice error messages, etc.
class Constraint<Pred pred, string desc = ""> {
  // The predicates that this constraint requires.
  Pred predicate = pred;
  // User-readable description used in error reporting messages. If empty, a
  // generic message will be used.
  string description = desc;
}

// Subclasses used to differentiate different constraint kinds. These are used
// as markers for the TableGen backend to handle different constraint kinds
// differently if needed. Constraints not deriving from the following subclasses
// are considered as uncategorized constraints.

// Subclass for constraints on a type.
class TypeConstraint<Pred predicate, string description = ""> :
    Constraint<predicate, description>;

// Subclass for constraints on an attribute.
class AttrConstraint<Pred predicate, string description = ""> :
    Constraint<predicate, description>;

// Subclass for constraints on a region.
class RegionConstraint<Pred predicate, string description = ""> :
    Constraint<predicate, description>;

// How to use these constraint categories:
//
// * Use TypeConstraint to specify
//   * Constraints on an op's operand/result definition
//   * Further constraints to match an op's operand/result in source pattern
//
// * Use Attr (a subclass for AttrConstraint) for
//   * Constraints on an op's attribute definition
// * Use AttrConstraint to specify
//   * Further constraints to match an op's attribute in source pattern
//
// * Use uncategorized constraint to specify
//   * Multi-entity constraints in rewrite rules

//===----------------------------------------------------------------------===//
// Common predicates
//===----------------------------------------------------------------------===//

// Whether a type is a VectorType.
def IsVectorTypePred : CPred<"$_self.isa<VectorType>()">;

// Whether a type is a TensorType.
def IsTensorTypePred : CPred<"$_self.isa<TensorType>()">;

// Whether a type is a MemRefType.
def IsMemRefTypePred : CPred<"$_self.isa<MemRefType>()">;

// Whether a type is a ShapedType.
def IsShapedTypePred : CPred<"$_self.isa<ShapedType>()">;

// For a ShapedType, verify that it has a static shape.
def HasStaticShapePred : CPred<"$_self.cast<ShapedType>().hasStaticShape()">;

// Whether a type is a TupleType.
def IsTupleTypePred : CPred<"$_self.isa<TupleType>()">;

//===----------------------------------------------------------------------===//
// Dialect definitions
//===----------------------------------------------------------------------===//

class Dialect {
  // The name of the dialect.
  string name = ?;

  // Short summary of the dialect.
  string summary = ?;

  // The description of the dialect.
  string description = ?;

  // The C++ namespace that ops of this dialect should be placed into.
  //
  // By default, uses the name of the dialect as the only namespace. To avoid
  // placing in any namespace, use "". To specify nested namespaces, use "::"
  // as the delimiter, e.g., given "A::B", ops will be placed in
  // `namespace A { namespace B { <ops> } }`.
  //
  // Note that this works in conjunction with dialect C++ code. Depending on how
  // the generated files are included into the dialect, you may want to specify
  // a full namespace path or a partial one.
  string cppNamespace = name;
}

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//

// A type, carries type constraints.
class Type<Pred condition, string descr = ""> :
    TypeConstraint<condition, descr> {
  string typeDescription = "";
}

// Allows providing an alternative name and description to an existing type def.
class TypeAlias<Type t, string description = t.description> :
    Type<t.predicate, description> {
  let typeDescription = t.typeDescription;
}

// A type of a specific dialect.
class DialectType<Dialect d, Pred condition, string descr = ""> :
    Type<condition, descr> {
  Dialect dialect = d;
}

// A variadic type constraint. It expands to zero or more of the base type. This
// class is used for supporting variadic operands/results. An op can declare no
// more than one variadic operand/result, and that operand/result must be the
// last one in the operand/result list.
class Variadic<Type type> : TypeConstraint<type.predicate, type.description> {
  Type baseType = type;
}

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

// Any type at all.
def AnyType : Type<CPred<"true">, "any type">;

// None type
def NoneType : Type<CPred<"$_self.isa<NoneType>()">, "none type">;

// Any type from the given list
class AnyTypeOf<list<Type> allowedTypes, string description = ""> : Type<
    // Satisfy any of the allowed type's condition
    Or<!foreach(allowedtype, allowedTypes, allowedtype.predicate)>,
    !if(!eq(description, ""),
        StrJoin<!foreach(t, allowedTypes, t.description), " or ">.result,
        description)>;

// Integer types.
// Any integer type irrespective of its width.
def AnyInteger : Type<CPred<"$_self.isa<IntegerType>()">, "integer">;

// Index type.
def Index : Type<CPred<"$_self.isa<IndexType>()">, "index">;

// Integer type of a specific width.
class I<int width>
    : Type<CPred<"$_self.isInteger(" # width # ")">,
                  width # "-bit integer">,
      BuildableType<"getIntegerType(" # width # ")"> {
  int bitwidth = width;
}

class IntOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, I<w>),
              StrJoinInt<widths, "/">.result # "-bit integer">;

def I1  : I<1>;
def I8  : I<8>;
def I16 : I<16>;
def I32 : I<32>;
def I64 : I<64>;

// Floating point types.

// Any float type irrespective of its width.
def AnyFloat : Type<CPred<"$_self.isa<FloatType>()">, "floating-point">;

// Float type of a specific width.
class F<int width>
    : Type<CPred<"$_self.isF" # width # "()">,
                width # "-bit float">,
      BuildableType<"getF" # width # "Type()"> {
  int bitwidth = width;
}

class FloatOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, F<w>),
              StrJoinInt<widths, "/">.result # "-bit float">;

def F16 : F<16>;
def F32 : F<32>;
def F64 : F<64>;

def BF16 : Type<CPred<"$_self.isBF16()">, "bfloat16 type">,
           BuildableType<"getBF16Type()">;

class Complex<Type type>
    : Type<And<[
          CPred<"$_self.isa<ComplexType>()">,
          SubstLeaves<"$_self", "$_self.cast<ComplexType>().getElementType()",
           type.predicate>]>,
           "complex type with " # type.description # " elements"> {
  Type elementType = type;
}

def AnyComplex : Type<CPred<"$_self.isa<ComplexType>()">, "complex-type">;

class OpaqueType<string dialect, string name, string description>
  : Type<CPred<"isOpaqueTypeWithName($_self, \""#dialect#"\", \""#name#"\")">,
         description>;

// Function Type

// Any function type.
def FunctionType : Type<CPred<"$_self.isa<FunctionType>()">, "function type">;

// A container type is a type that has another type embedded within it.
class ContainerType<Type etype, Pred containerPred, code elementTypeCall,
                    string descr> :
    // First, check the container predicate.  Then, substitute the extracted
    // element into the element type checker.
    Type<And<[containerPred,
                SubstLeaves<"$_self", !cast<string>(elementTypeCall),
                etype.predicate>]>,
         descr # " of " # etype.description # " values"> {
  // The type of elements in the container.
  Type elementType = etype;

  // Call to retrieve.
  code getElementTypeCall = elementTypeCall;
}

class ShapedContainerType<list<Type> allowedTypes, Pred containerPred, string descr> :
    ContainerType<AnyTypeOf<allowedTypes>, containerPred,
                  "$_self.cast<ShapedType>().getElementType()", descr>;

// Whether a shaped type is ranked.
def HasRankPred : CPred<"$_self.cast<ShapedType>().hasRank()">;

// Whether a shaped type has one of the specified ranks.
class HasAnyRankOfPred<list<int> ranks> : And<[
    HasRankPred,
    Or<!foreach(rank, ranks,
                CPred<"$_self.cast<ShapedType>().getRank() == " # rank>)>]>;

// Vector types.

class VectorOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes, IsVectorTypePred, "vector">;

// Whether the number of elements of a vector is from the given
// `allowedLengths` list
class IsVectorOfLengthPred<list<int> allowedLengths> :
  And<[IsVectorTypePred,
       Or<!foreach(allowedlength, allowedLengths,
                   CPred<[{$_self.cast<VectorType>().getNumElements()
                           == }]
                         # allowedlength>)>]>;

// Any vector where the number of elements is from the given
// `allowedLengths` list
class VectorOfLength<list<int> allowedLengths> : Type<
  IsVectorOfLengthPred<allowedLengths>,
  " of length " # StrJoinInt<allowedLengths, "/">.result>;


// Any vector where the number of elements is from the given
// `allowedLengths` list and the type is from the given `allowedTypes`
// list
class VectorOfLengthAndType<list<int> allowedLengths,
                            list<Type> allowedTypes> : Type<
  And<[VectorOf<allowedTypes>.predicate,
       VectorOfLength<allowedLengths>.predicate]>,
  VectorOf<allowedTypes>.description #
  VectorOfLength<allowedLengths>.description>;

def AnyVector : VectorOf<[AnyType]>;

// Tensor types.

// Any tensor type whose element type is from the given `allowedTypes` list
class TensorOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes, IsTensorTypePred, "tensor">;

def AnyTensor : TensorOf<[AnyType]>;

def AnyRankedTensor :
  ShapedContainerType<[AnyType], And<[IsTensorTypePred, HasRankPred]>,
  "ranked tensor">;

// TODO(b/130064155) Have an easy way to add another constraint to a type.
class StaticShapeTensorOf<list<Type> allowedTypes>
    : Type<And<[TensorOf<allowedTypes>.predicate, HasStaticShapePred]>,
           "statically shaped " # TensorOf<allowedTypes>.description>;

def AnyStaticShapeTensor : StaticShapeTensorOf<[AnyType]>;

def I1Tensor   : TensorOf<[I1]>;
def I8Tensor   : TensorOf<[I8]>;
def I16Tensor  : TensorOf<[I16]>;
def I32Tensor  : TensorOf<[I32]>;
def I64Tensor  : TensorOf<[I64]>;

def BF16Tensor : TensorOf<[BF16]>;
def F16Tensor  : TensorOf<[F16]>;
def F32Tensor  : TensorOf<[F32]>;
def F64Tensor  : TensorOf<[F64]>;

// Ranked tensor type with one of the specified types and ranks.
class TensorRankOf<list<Type> allowedTypes, list<int> ranks> :
    Type<And<[TensorOf<allowedTypes>.predicate, HasAnyRankOfPred<ranks>]>,
         StrJoin<!foreach(rank, ranks, rank # "D"), "/">.result # " " #
         TensorOf<allowedTypes>.description>;

class 0DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [0]>;
class 1DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [1]>;
class 2DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [2]>;
class 3DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [3]>;
class 4DTensorOf<list<Type> allowedTypes> : TensorRankOf<allowedTypes, [4]>;

// Memref type.

// Memrefs are blocks of data with fixed type and rank.
class MemRefOf<list<Type> allowedTypes> :
    ShapedContainerType<allowedTypes, IsMemRefTypePred, "memref">;

def AnyMemRef : MemRefOf<[AnyType]>;

// Memref declarations handle any memref, independent of rank, size, (static or
// dynamic), layout, or memory space.
def I1MemRef  : MemRefOf<[I1]>;
def I8MemRef  : MemRefOf<[I8]>;
def I16MemRef : MemRefOf<[I16]>;
def I32MemRef : MemRefOf<[I32]>;
def I64MemRef : MemRefOf<[I64]>;

def BF16MemRef : MemRefOf<[BF16]>;
def F16MemRef  : MemRefOf<[F16]>;
def F32MemRef  : MemRefOf<[F32]>;
def F64MemRef  : MemRefOf<[F64]>;

// TODO(b/130064155) Have an easy way to add another constraint to a type.
class MemRefRankOf<list<Type> allowedTypes, list<int> ranks> :
    Type<And<[MemRefOf<allowedTypes>.predicate, HasAnyRankOfPred<ranks>]>,
         StrJoin<!foreach(rank, ranks, rank # "D"), "/">.result # " " #
         MemRefOf<allowedTypes>.description>;

class StaticShapeMemRefOf<list<Type> allowedTypes>
    : Type<And<[MemRefOf<allowedTypes>.predicate, HasStaticShapePred]>,
           "statically shaped " # MemRefOf<allowedTypes>.description>;

def AnyStaticShapeMemRef : StaticShapeMemRefOf<[AnyType]>;

// For a MemRefType, verify that it has strides.
def HasStridesPred : CPred<[{ isStrided($_self.cast<MemRefType>()) }]>;

class StridedMemRefOf<list<Type> allowedTypes>
    : Type<And<[MemRefOf<allowedTypes>.predicate, HasStridesPred]>,
           "strided " # MemRefOf<allowedTypes>.description>;

def AnyStridedMemRef : StridedMemRefOf<[AnyType]>;

class AnyStridedMemRefOfRank<int rank> :
  Type<And<[AnyStridedMemRef.predicate,
            MemRefRankOf<[AnyType], [rank]>.predicate]>,
       AnyStridedMemRef.description # " of rank " # rank>;

// This represents a generic tuple without any constraints on element type.
def AnyTuple : Type<IsTupleTypePred, "tuple">;

// A container type that has other types embedded in it, but (unlike
// ContainerType) can hold elements with a mix of types. Requires a call that
// produces a list of all elements' types.
class MixedContainerType<Type etype, Pred containerPred, code elementTypesCall,
                         string descr> :
    Type<
        And<[
            containerPred,
            Concat<
                "llvm::all_of(" # elementTypesCall # ", [](Type t) { return ",
                SubstLeaves<"$_self", "t", etype.predicate>,
                "; })"
            >
        ]>,
        descr # " with any combination of " # etype.description # " values"> {
  // The type of elements in the container.
  Type elementType = etype;

  // Call to retrieve.
  code getElementTypesCall = elementTypesCall;
}

// A Tuple that holds a mix of elements of the allowed types.
class TupleOf<list<Type> allowedTypes>
    : MixedContainerType<AnyTypeOf<allowedTypes>, IsTupleTypePred,
                         "$_self.cast<TupleType>().getTypes()", "tuple">;

// A Tuple with arbitrary nesting, where all elements are a mix of the allowed
// types.
class NestedTupleOf<list<Type> allowedTypes> :
    MixedContainerType<AnyTypeOf<allowedTypes>, IsTupleTypePred,
                       "getFlattenedTypes($_self.cast<TupleType>())",
                       "nested tuple">;

//===----------------------------------------------------------------------===//
// Common type constraints
//===----------------------------------------------------------------------===//

// Type constraint for bool-like types: bools, vectors of bools, tensors of
// bools.
def BoolLike : TypeConstraint<Or<[I1.predicate, VectorOf<[I1]>.predicate,
                                  TensorOf<[I1]>.predicate]>,
    "bool-like">;

// Type constraint for integer-like types: integers, indices, vectors of
// integers, tensors of integers.
def IntegerLike : TypeConstraint<Or<[AnyInteger.predicate, Index.predicate,
        VectorOf<[AnyInteger]>.predicate, TensorOf<[AnyInteger]>.predicate]>,
    "integer-like">;

// Type constraint for float-like types: floats, vectors or tensors thereof.
def FloatLike : TypeConstraint<Or<[AnyFloat.predicate,
        VectorOf<[AnyFloat]>.predicate, TensorOf<[AnyFloat]>.predicate]>,
    "floating-point-like">;


//===----------------------------------------------------------------------===//
// Attribute definitions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Base attribute definition

// Base class for all attributes.
class Attr<Pred condition, string descr = ""> :
    AttrConstraint<condition, descr> {
  code storageType = ?; // The backing mlir::Attribute type
  code returnType = ?;  // The underlying C++ value type

  // The call expression to convert from the storage type to the return
  // type. For example, an enum can be stored as an int but returned as an
  // enum class.
  //
  // Format: $_self will be expanded to the attribute.
  //
  // For example, `$_self.getValue().getSExtValue()` for `IntegerAttr val` will
  // expand to `getAttrOfType<IntegerAttr>("val").getValue().getSExtValue()`.
  code convertFromStorage = "$_self.getValue()";

  // The call expression to build an attribute from a constant value.
  //
  // Format: $0 will be expanded to the constant value of the attribute.
  //
  // For example, `$_builder.getStringAttr("$0")` for `StringAttr:"foo"` will
  // expand to `builder.getStringAttr("foo")`.
  string constBuilderCall = ?;

  // Default value for attribute.
  // Requires a constBuilderCall defined.
  string defaultValue = ?;

  // Whether the attribute is optional. Typically requires a custom
  // convertFromStorage method to handle the case where the attribute is
  // not present.
  bit isOptional = 0;

  // What is the base-level Attr instantiation that this Attr is built upon.
  // Unset means this is a base-level Attr.
  //
  // This field is used by attribute wrapper classes (DefaultValuedAttr,
  // OptionalAttr, etc.) to retrieve the base-level attribute definition.
  // This can be used for getting its name; otherwise, we will see
  // "anonymous_<number>" as the attribute def name because of template
  // instantiation.
  // TOOD(b/132458159): deduplicate the fields in attribute wrapper classes.
  Attr baseAttr = ?;
}

//===----------------------------------------------------------------------===//
// Attribute modifier definition

// Decorates an attribute to have an (unvalidated) default value if not present.
class DefaultValuedAttr<Attr attr, string val> :
    Attr<attr.predicate, attr.description> {
  // Construct this attribute with the input attribute and change only
  // the default value.
  // Note: this has to be kept up to date with Attr above.
  let storageType = attr.storageType;
  let returnType = attr.returnType;
  let convertFromStorage = attr.convertFromStorage;
  let constBuilderCall = attr.constBuilderCall;
  let defaultValue = val;

  let baseAttr = attr;
}

// Decorates an attribute as optional. The return type of the generated
// attribute accessor method will be Optional<>.
class OptionalAttr<Attr attr> : Attr<attr.predicate, attr.description> {
  // Rewrite the attribute to be optional.
  // Note: this has to be kept up to date with Attr above.
  let storageType = attr.storageType;
  let returnType = "Optional<" # attr.returnType #">";
  let convertFromStorage = "$_self ? " # returnType # "(" #
                           attr.convertFromStorage # ") : (llvm::None)";
  let isOptional = 1;

  let baseAttr = attr;
}

//===----------------------------------------------------------------------===//
// Primitive attribute kinds

// A generic attribute that must be constructed around a specific type
// `attrValType`. Backed by MLIR attribute kind `attrKind`.
class TypedAttrBase<BuildableType attrValType, string attrKind,
                    Pred condition, string descr> :
    Attr<condition, descr> {
  let constBuilderCall = "$_builder.get" # attrKind # "($_builder." #
                         attrValType.builderCall # ", $0)";
  let storageType = attrKind;
}

// Any attribute.
def AnyAttr : Attr<CPred<"true">, "any attribute"> {
  let storageType = "Attribute";
  let returnType = "Attribute";
  let convertFromStorage = "$_self";
  let constBuilderCall = "$0";
}

def BoolAttr : Attr<CPred<"$_self.isa<BoolAttr>()">, "bool attribute"> {
  let storageType = [{ BoolAttr }];
  let returnType = [{ bool }];
  let constBuilderCall = "$_builder.getBoolAttr($0)";
}

// Base class for integer attributes of fixed width.
class IntegerAttrBase<I attrValType, string descr> :
    TypedAttrBase<
      attrValType, "IntegerAttr",
      And<[CPred<"$_self.isa<IntegerAttr>()">,
           CPred<"$_self.cast<IntegerAttr>().getType()."
                 "isInteger(" # attrValType.bitwidth # ")">]>,
      descr> {
  let returnType = [{ APInt }];
}

def APIntAttr : Attr<CPred<"$_self.isa<IntegerAttr>()">,
                     "arbitrary integer attribute"> {
  let storageType = [{ IntegerAttr }];
  let returnType = [{ APInt }];
}

def I1Attr  : IntegerAttrBase<I1,  "1-bit integer attribute">;
def I8Attr  : IntegerAttrBase<I8,  "8-bit integer attribute">;
def I16Attr : IntegerAttrBase<I16, "16-bit integer attribute">;
def I32Attr : IntegerAttrBase<I32, "32-bit integer attribute">;
def I64Attr : IntegerAttrBase<I64, "64-bit integer attribute">;

class NonNegativeIntAttrBase<I attrValType, string descr> :
    TypedAttrBase<
      attrValType, "IntegerAttr",
      And<[IntegerAttrBase<attrValType, "">.predicate,
           CPred<"!$_self.cast<IntegerAttr>().getValue().isNegative()">]>,
      descr> {
  let returnType = [{ APInt }];
}

def NonNegativeI32Attr : NonNegativeIntAttrBase<
    I32, "non-negative 32-bit integer attribute">;
def NonNegativeI64Attr : NonNegativeIntAttrBase<
    I64, "non-negative 64-bit integer attribute">;

class PositiveIntAttrBase<I attrValType, string descr> :
    TypedAttrBase<
      attrValType, "IntegerAttr",
      And<[IntegerAttrBase<attrValType, "">.predicate,
           CPred<"$_self.cast<IntegerAttr>().getValue()"
                 ".isStrictlyPositive()">]>,
      descr> {
  let returnType = [{ APInt }];
}

def PositiveI32Attr : PositiveIntAttrBase<
    I32, "positive 32-bit integer attribute">;
def PositiveI64Attr : PositiveIntAttrBase<
    I64, "positive 64-bit integer attribute">;

// Base class for float attributes of fixed width.
class FloatAttrBase<F attrValType, string descr> :
    TypedAttrBase<attrValType, "FloatAttr",
              And<[CPred<"$_self.isa<FloatAttr>()">,
                     CPred<"$_self.cast<FloatAttr>().getType().isF" #
                           attrValType.bitwidth # "()">]>,
              descr> {
  let returnType = [{ APFloat }];
}

def F32Attr : FloatAttrBase<F32, "32-bit float attribute">;
def F64Attr : FloatAttrBase<F64, "64-bit float attribute">;

// An attribute backed by a string type.
class StringBasedAttr<Pred condition, string descr> : Attr<condition, descr> {
  let constBuilderCall = "$_builder.getStringAttr(\"$0\")";
  let storageType = [{ StringAttr }];
  let returnType = [{ StringRef }];
}

def StrAttr : StringBasedAttr<CPred<"$_self.isa<StringAttr>()">,
                              "string attribute">;

// Base class for attributes containing types. Example:
//   def IntTypeAttr : TypeAttrBase<"IntegerType", "integer type attribute">
// defines a type attribute containing an integer type.
class TypeAttrBase<string retType, string description> :
    Attr<And<[
      CPred<"$_self.isa<TypeAttr>()">,
      CPred<"$_self.cast<TypeAttr>().getValue().isa<" # retType # ">()">]>,
    description> {
  let storageType = [{ TypeAttr }];
  let returnType = retType;
  let convertFromStorage = "$_self.getValue().cast<" # retType # ">()";
}

def TypeAttr : TypeAttrBase<"Type", "any type attribute">;

// The mere presence of unit attributes has a meaning.  Therefore, unit
// attributes are always treated as optional and accessors to them return
// "true" if the attribute is present and "false" otherwise.
def UnitAttr : Attr<CPred<"$_self.isa<UnitAttr>()">, "unit attribute"> {
  let storageType = [{ UnitAttr }];
  let constBuilderCall = "$_builder.getUnitAttr()";
  let convertFromStorage = "$_self != nullptr";
  let returnType = "bool";
  let isOptional = 1;
}

//===----------------------------------------------------------------------===//
// Enum attribute kinds

// Additional information for an enum attribute case.
class EnumAttrCaseInfo<string sym, int val> {
  // The C++ enumerant symbol
  string symbol = sym;

  // The C++ enumerant value
  // If less than zero, there will be no explicit discriminator values assigned
  // to enumerators in the generated enum class.
  int value = val;
}

// An enum attribute case stored with StringAttr.
class StrEnumAttrCase<string sym, int val = -1> :
    EnumAttrCaseInfo<sym, val>,
    StringBasedAttr<
      CPred<"$_self.cast<StringAttr>().getValue() == \"" # sym # "\"">,
      "case " # sym>;

// An enum attribute case stored with IntegerAttr.
class IntEnumAttrCaseBase<I intType, string sym, int val> :
    EnumAttrCaseInfo<sym, val>,
    IntegerAttrBase<intType, "case " # sym> {
  let predicate =
    CPred<"$_self.cast<IntegerAttr>().getInt() == " # val>;
}

class I32EnumAttrCase<string sym, int val> : IntEnumAttrCaseBase<I32, sym, val>;
class I64EnumAttrCase<string sym, int val> : IntEnumAttrCaseBase<I64, sym, val>;

// A bit enum case stored with 32-bit IntegerAttr. `val` here is *not* the
// ordinal number of the bit that is set. It is the 32-bit integer with only
// one bit set.
class BitEnumAttrCase<string sym, int val> :
    EnumAttrCaseInfo<sym, val>,
    IntegerAttrBase<I32, "case " # sym> {
  let predicate = CPred<
    "$_self.cast<IntegerAttr>().getValue().getZExtValue() & " # val # "u">;
}

// Additional information for an enum attribute.
class EnumAttrInfo<string name, list<EnumAttrCaseInfo> cases> {
  // The C++ enum class name
  string className = name;

  // List of all accepted cases
  list<EnumAttrCaseInfo> enumerants = cases;

  // The following fields are only used by the EnumsGen backend to generate
  // an enum class definition and conversion utility functions.

  // The underlying type for the C++ enum class. An empty string mean the
  // underlying type is not explicitly specified.
  string underlyingType = "";

  // The C++ namespaces that the enum class definition and utility functions
  // should be placed into.
  //
  // Normally you want to place the full namespace path here. If it is nested,
  // use "::" as the delimiter, e.g., given "A::B", generated code will be
  // placed in `namespace A { namespace B { ... } }`. To avoid placing in any
  // namespace, use "".
  // TODO(b/134741431): use dialect to provide the namespace.
  string cppNamespace = "";

  // The name of the utility function that converts a value of the underlying
  // type to the corresponding symbol. It will have the following signature:
  //
  // ```c++
  // llvm::Optional<<qualified-enum-class-name>> <fn-name>(<underlying-type>);
  // ```
  string underlyingToSymbolFnName = "symbolize" # name;

  // The name of the utility function that converts a string to the
  // corresponding symbol. It will have the following signature:
  //
  // ```c++
  // llvm::Optional<<qualified-enum-class-name>> <fn-name>(llvm::StringRef);
  // ```
  string stringToSymbolFnName = "symbolize" # name;

  // The name of the utility function that converts a symbol to the
  // corresponding string. It will have the following signature:
  //
  // ```c++
  // <return-type> <fn-name>(<qualified-enum-class-name>);
  // ```
  string symbolToStringFnName = "stringify" # name;
  string symbolToStringFnRetType = "llvm::StringRef";

  // The name of the utility function that returns the max enum value used
  // within the enum class. It will have the following signature:
  //
  // ```c++
  // static constexpr unsigned <fn-name>();
  // ```
  string maxEnumValFnName = "getMaxEnumValFor" # name;
}

// An enum attribute backed by StringAttr.
//
// Op attributes of this kind are stored as StringAttr. Extra verification will
// be generated on the string though: only the symbols of the allowed cases are
// permitted as the string value.
class StrEnumAttr<string name, string description,
                  list<StrEnumAttrCase> cases> :
    EnumAttrInfo<name, cases>,
    StringBasedAttr<
      And<[StrAttr.predicate, Or<!foreach(case, cases, case.predicate)>]>,
      !if(!empty(description), "allowed string cases: " #
          StrJoin<!foreach(case, cases, "'" # case.symbol # "'")>.result,
          description)>;

// An enum attribute backed by IntegerAttr.
//
// Op attributes of this kind are stored as IntegerAttr. Extra verification will
// be generated on the integer though: only the values of the allowed cases are
// permitted as the integer value.
class IntEnumAttr<I intType, string name, string description,
                  list<IntEnumAttrCaseBase> cases> :
    EnumAttrInfo<name, cases>,
    IntegerAttrBase<intType,
      !if(!empty(description), "allowed " # intType.description # " cases: " #
          StrJoinInt<!foreach(case, cases, case.value)>.result, description)> {
  let predicate = And<[
    IntegerAttrBase<intType, "">.predicate,
    Or<!foreach(case, cases, case.predicate)>]>;
}

class I32EnumAttr<string name, string description,
                  list<I32EnumAttrCase> cases> :
    IntEnumAttr<I32, name, description, cases> {
  let returnType = cppNamespace # "::" # name;
  let underlyingType = "uint32_t";
  let convertFromStorage = "static_cast<" # returnType # ">($_self.getInt())";
  let constBuilderCall = "$_builder.getI32IntegerAttr(static_cast<int32_t>($0))";
}
class I64EnumAttr<string name, string description,
                  list<I64EnumAttrCase> cases> :
    IntEnumAttr<I64, name, description, cases> {
  let returnType = cppNamespace # "::" # name;
  let underlyingType = "uint64_t";
  let convertFromStorage = "static_cast<" # returnType # ">($_self.getInt())";
  let constBuilderCall = "$_builder.getI64IntegerAttr(static_cast<int64_t>($0))";
}

// A bit enum stored with 32-bit IntegerAttr.
//
// Op attributes of this kind are stored as IntegerAttr. Extra verification will
// be generated on the integer to make sure only allowed bit are set. Besides,
// helper methods are generated to parse a string separated with a specified
// delimiter to a symbol and vice versa.
class BitEnumAttr<string name, string description,
                  list<BitEnumAttrCase> cases> :
    EnumAttrInfo<name, cases>, IntegerAttrBase<I32, description> {
  let predicate = And<[
    IntegerAttrBase<I32, "">.predicate,
    // Make sure we don't have unknown bit set.
    CPred<"!($_self.cast<IntegerAttr>().getValue().getZExtValue() & (~(" #
          StrJoin<!foreach(case, cases, case.value # "u"), "|">.result #
          ")))">
  ]>;

  let returnType = cppNamespace # "::" # name;
  let underlyingType = "uint32_t";
  let convertFromStorage = "static_cast<" # returnType # ">($_self.getInt())";
  let constBuilderCall = "$_builder.getI32IntegerAttr(static_cast<int32_t>($0))";

  // We need to return a string because we may concatenate symbols for multiple
  // bits together.
  let symbolToStringFnRetType = "std::string";

  // The delimiter used to separate bit enum cases in strings.
  string separator = "|";
}

//===----------------------------------------------------------------------===//
// Composite attribute kinds

class DictionaryAttrBase : Attr<CPred<"$_self.isa<DictionaryAttr>()">,
                          "dictionary of named attribute values"> {
  let storageType = [{ DictionaryAttr }];
  let returnType = [{ DictionaryAttr }];
  let convertFromStorage = "$_self";
}

def DictionaryAttr : DictionaryAttrBase;

class ElementsAttrBase<Pred condition, string description> :
    Attr<condition, description> {
  let storageType = [{ ElementsAttr }];
  let returnType = [{ ElementsAttr }];
  let convertFromStorage = "$_self";
}

def ElementsAttr : ElementsAttrBase<CPred<"$_self.isa<ElementsAttr>()">,
                                   "constant vector/tensor attribute">;

class IntElementsAttr<int width> : ElementsAttrBase<
  CPred<"$_self.isa<DenseIntElementsAttr>() &&"
      "$_self.cast<DenseIntElementsAttr>().getType()."
      "getElementType().isInteger(" # width # ")">,
  width # "-bit integer elements attribute"> {

  let storageType = [{ DenseIntElementsAttr }];
  let returnType = [{ DenseIntElementsAttr }];

  // Note that this is only constructing scalar elements attribute.
  let constBuilderCall = "DenseElementsAttr::get("
    "RankedTensorType::get({}, $_builder.getIntegerType(" # width # ")), "
    "llvm::makeArrayRef($0)).cast<DenseIntElementsAttr>()";
  let convertFromStorage = "$_self";
}

def I32ElementsAttr : IntElementsAttr<32>;
def I64ElementsAttr : IntElementsAttr<64>;

class FloatElementsAttr<int width> : ElementsAttrBase<
  CPred<"$_self.isa<DenseFPElementsAttr>() &&"
      "$_self.cast<DenseElementsAttr>().getType()."
      "getElementType().isF" # width # "()">,
  width # "-bit float elements attribute"> {

  let storageType = [{ DenseElementsAttr }];
  let returnType = [{ DenseElementsAttr }];

  // Note that this is only constructing scalar elements attribute.
  let constBuilderCall = "DenseElementsAttr::get("
    "RankedTensorType::get({}, $_builder.getF" # width # "Type()),"
    "llvm::makeArrayRef($0))";
  let convertFromStorage = "$_self";
}

def F64ElementsAttr : FloatElementsAttr<64>;

// A `width`-bit floating point elements attribute. The attribute should be
// ranked and has a shape as specified in `dims`.
class RankedFloatElementsAttr<int width, list<int> dims> : ElementsAttrBase<
  CPred<"$_self.isa<DenseFPElementsAttr>() &&"
      "$_self.cast<DenseFPElementsAttr>().getType()."
      "getElementType().isF" # width # "() && "
      // Check that this is ranked and has the specified shape.
      "$_self.cast<DenseFPElementsAttr>().getType().hasRank() && "
      "$_self.cast<DenseFPElementsAttr>().getType().getShape() == "
      "llvm::ArrayRef<int64_t>({" # StrJoinInt<dims>.result # "})">,
  width # "-bit float elements attribute of shape [" #
  StrJoinInt<dims>.result # "]"> {

  let storageType = [{ DenseFPElementsAttr }];
  let returnType = [{ DenseFPElementsAttr }];

  let constBuilderCall = "DenseElementsAttr::get("
    "RankedTensorType::get({" # StrJoinInt<dims>.result #
    "}, $_builder.getF" # width # "Type()), "
    "llvm::makeArrayRef($0)).cast<DenseFPElementsAttr>()";
  let convertFromStorage = "$_self";
}

class RankedF32ElementsAttr<list<int> dims> : RankedFloatElementsAttr<32, dims>;
class RankedF64ElementsAttr<list<int> dims> : RankedFloatElementsAttr<64, dims>;

// Base class for array attributes.
class ArrayAttrBase<Pred condition, string description> :
    Attr<condition, description> {
  let storageType = [{ ArrayAttr }];
  let returnType = [{ ArrayAttr }];
  let convertFromStorage = "$_self";
}

def ArrayAttr : ArrayAttrBase<CPred<"$_self.isa<ArrayAttr>()">,
                              "array attribute">;

// Base class for array attributes whose elements are of the same kind.
// `element` specifies the element attribute kind stored in this array.
class TypedArrayAttrBase<Attr element, string description>: ArrayAttrBase<
    And<[
      // Guarantee this is an ArrayAttr first
      CPred<"$_self.isa<ArrayAttr>()">,
      // Guarantee all elements satisfy the constraints from `element`
      Concat<"llvm::all_of($_self.cast<ArrayAttr>(), "
                          "[](Attribute attr) { return ",
                             SubstLeaves<"$_self", "attr", element.predicate>,
                          "; })">]>,
    description> {
  let constBuilderCall = "$_builder.getArrayAttr($0)";
}

def I32ArrayAttr : TypedArrayAttrBase<I32Attr,
                                      "32-bit integer array attribute"> {
  let constBuilderCall = "$_builder.getI32ArrayAttr($0)";
}
def I64ArrayAttr : TypedArrayAttrBase<I64Attr,
                                      "64-bit integer array attribute"> {
  let constBuilderCall = "$_builder.getI64ArrayAttr($0)";
}
def F32ArrayAttr : TypedArrayAttrBase<F32Attr, "32-bit float array attribute"> {
  let constBuilderCall = "$_builder.getF32ArrayAttr($0)";
}
def F64ArrayAttr : TypedArrayAttrBase<F64Attr, "64-bit float array attribute"> {
  let constBuilderCall = "$_builder.getF64ArrayAttr($0)";
}
def StrArrayAttr : TypedArrayAttrBase<StrAttr, "string array attribute"> {
  let constBuilderCall = "$_builder.getStrArrayAttr($0)";
}
def TypeArrayAttr : TypedArrayAttrBase<TypeAttr, "type array attribute"> {
  let constBuilderCall = ?;
}

// Attribute information for an Attribute field within a StructAttr.
class StructFieldAttr<string thisName, Attr thisType> {
  // Name of this field in the StructAttr.
  string name = thisName;

  // Attribute type wrapped by the struct attr.
  Attr type = thisType;
}

// Structured attribute that wraps a DictionaryAttr and provides both a
// validation method and set of accessors for a fixed set of fields. This is
// useful when representing data that would normally be in a structure.
class StructAttr<string name, Dialect dialect,
                 list<StructFieldAttr> attributes> : DictionaryAttrBase {
  // Name for this StructAttr.
  string className = name;

  // Return type should match the name of the structure.
  let returnType = name;

  // Storage type should match the name of the structure.
  let storageType = name;

  // The dialect this StructAttr belongs to.
  Dialect structDialect = dialect;

  // List of fields that the StructAttr contains.
  list<StructFieldAttr> fields = attributes;
}

// Attributes containing symbol references.
def SymbolRefAttr : Attr<CPred<"$_self.isa<SymbolRefAttr>()">,
                        "symbol reference attribute"> {
  let storageType = [{ SymbolRefAttr }];
  let returnType = [{ SymbolRefAttr }];
  let constBuilderCall = "$_builder.getSymbolRefAttr($0)";
  let convertFromStorage = "$_self";
}
def FlatSymbolRefAttr : Attr<CPred<"$_self.isa<FlatSymbolRefAttr>()">,
                                   "flat symbol reference attribute"> {
  let storageType = [{ FlatSymbolRefAttr }];
  let returnType = [{ StringRef }];
  let constBuilderCall = "$_builder.getSymbolRefAttr($0)";
  let convertFromStorage = "$_self.getValue()";
}

def SymbolRefArrayAttr :
  TypedArrayAttrBase<SymbolRefAttr, "symbol ref array attribute"> {
  let constBuilderCall = ?;
}

//===----------------------------------------------------------------------===//
// Derive attribute kinds

// DerivedAttr are attributes whose value is computed from properties
// of the operation. They do not require additional storage and are
// materialized as needed.
class DerivedAttr<code ret, code b> : Attr<CPred<"true">, "derived attribute"> {
  let returnType = ret;
  code body = b;
}

// Derived attribute that returns a mlir::Type.
class DerivedTypeAttr<code body> : DerivedAttr<"Type", body>;

//===----------------------------------------------------------------------===//
// Constant attribute kinds

// Represents a constant attribute of specific Attr type. A constant
// attribute can be specified only of attributes that have a constant
// builder call defined. The constant value is specified as a string.
//
// If used as a constraint, it generates a matcher on a constant attribute by
// using the constant value builder of the attribute and the value.
class ConstantAttr<Attr attribute, string val> : AttrConstraint<
    CPred<"$_self == " # !subst("$0", val, attribute.constBuilderCall)>,
    "constant attribute " # val> {
  Attr attr = attribute;
  string value = val;
}

class ConstF32Attr<string val> : ConstantAttr<F32Attr, val>;
def ConstBoolAttrFalse : ConstantAttr<BoolAttr, "false">;
def ConstBoolAttrTrue : ConstantAttr<BoolAttr, "true">;
def ConstUnitAttr : ConstantAttr<UnitAttr, "unit">;

//===----------------------------------------------------------------------===//
// Common attribute constraints
//===----------------------------------------------------------------------===//

// A general mechanism to further confine the given `attr` with all the
// `constraints`. This allows to compose complex constraints out of a series
// of more primitive ones.
class Confined<Attr attr, list<AttrConstraint> constraints> : Attr<
    And<!listconcat([attr.predicate],
                      !foreach(pred, constraints, pred.predicate))>,
    !foldl(/*init*/attr.description, /*list*/constraints,
           prev, cur, prev # " " # cur.description)> {
  let storageType = attr.storageType;
  let returnType = attr.returnType;
  let convertFromStorage = attr.convertFromStorage;
  let constBuilderCall = attr.constBuilderCall;
  let defaultValue = attr.defaultValue;
  let isOptional = attr.isOptional;

  let baseAttr = attr;
}

// An AttrConstraint that holds if all attr constraints specified in
// 'constraints' hold.
class AllAttrConstraintsOf<list<AttrConstraint> constraints> : AttrConstraint<
    And<!listconcat([!head(constraints).predicate],
                      !foreach(pred, !tail(constraints), pred.predicate))>,
    !foldl(/*init*/!head(constraints).description, /*list*/!tail(constraints),
           prev, cur, prev # " and " # cur.description)> {
}

class IntMinValue<int n> : AttrConstraint<
    CPred<"$_self.cast<IntegerAttr>().getInt() >= " # n>,
    "whose minimal value is " # n>;

class ArrayMinCount<int n> : AttrConstraint<
    CPred<"$_self.cast<ArrayAttr>().size() >= " # n>,
    "with at least " # n # " elements">;

class ArrayCount<int n> : AttrConstraint<
    CPred<"$_self.cast<ArrayAttr>().size() == " #n>,
    "with exactly " # n # " elements">;

class IntArrayNthElemEq<int index, int value> : AttrConstraint<
    And<[
      CPred<"$_self.cast<ArrayAttr>().size() > " # index>,
      CPred<"$_self.cast<ArrayAttr>().getValue()[" # index # "]"
        ".cast<IntegerAttr>().getInt() == " # value>
       ]>,
    "whose " # index # "-th element must be " # value>;

class IntArrayNthElemMinValue<int index, int min> : AttrConstraint<
    And<[
      CPred<"$_self.cast<ArrayAttr>().size() > " # index>,
      CPred<"$_self.cast<ArrayAttr>().getValue()[" # index # "]"
        ".cast<IntegerAttr>().getInt() >= " # min>
        ]>,
    "whose " # index # "-th element must be at least " # min>;

def IsNullAttr : AttrConstraint<
    CPred<"!$_self">, "empty attribute (for optional attributes)">;

// An attribute constraint on FlatSymbolRefAttr that requires that the
// reference point to an op of `opClass` within the closest parent with a symbol
// table.
// TODO(riverriddle) Add support for nested symbol references.
class ReferToOp<string opClass> : AttrConstraint<
    CPred<"isa_and_nonnull<" # opClass # ">("
            "::mlir::SymbolTable::lookupNearestSymbolFrom("
              "&$_op, $_self.cast<FlatSymbolRefAttr>().getValue()))">,
    "referencing to a '" # opClass # "' symbol">;

//===----------------------------------------------------------------------===//
// Region definitions
//===----------------------------------------------------------------------===//

class Region<Pred condition, string descr = ""> :
    RegionConstraint<condition, descr>;

// Any region.
def AnyRegion : Region<CPred<"true">, "any region">;

// A region with the given number of blocks.
class SizedRegion<int numBlocks> : Region<
  CPred<"$_self.getBlocks().size() == " # numBlocks>,
  "region with " # numBlocks # " blocks">;

//===----------------------------------------------------------------------===//
// OpTrait definitions
//===----------------------------------------------------------------------===//

// OpTrait represents a trait regarding an op.
class OpTrait;

// NativeOpTrait corresponds to the MLIR C++ OpTrait mechanism. The
// purpose to wrap around C++ symbol string with this class is to make
// traits specified for ops in TableGen less alien and more integrated.
class NativeOpTrait<string prop> : OpTrait {
  string trait = "OpTrait::" # prop;
}

// ParamNativeOpTrait corresponds to the template-parameterized traits in the
// C++ implementation.  MLIR uses nested class templates to implement such
// traits leading to constructs of the form "TraitName<Parameters>::Impl". Use
// the value in `prop` as the trait name and the value in `params` as
// parameters to construct the native trait class name.
class ParamNativeOpTrait<string prop, string params>
    : NativeOpTrait<prop # "<" # params # ">::Impl">;

// GenInternalOpTrait is an op trait that does not have direct C++ mapping but
// affects op definition generator internals, like how op builders and
// operand/attribute/result getters are generated.
class GenInternalOpTrait<string prop> : OpTrait {
  string trait = "OpTrait::" # prop;
}

// PredOpTrait is an op trait implemented by way of a predicate on the op.
class PredOpTrait<string descr, Pred pred> : OpTrait {
  string description = descr;
  Pred predicate = pred;
}

// Op supports operand broadcast behavior.
def Broadcastable  : NativeOpTrait<"BroadcastableTwoOperandsOneResult">;
// X op Y == Y op X
def Commutative  : NativeOpTrait<"IsCommutative">;
// Op is isolated from above.
def IsolatedFromAbove : NativeOpTrait<"IsIsolatedFromAbove">;
// Op results are float or vectors/tensors thereof.
def ResultsAreFloatLike : NativeOpTrait<"ResultsAreFloatLike">;
// Op has no side effect.
def NoSideEffect : NativeOpTrait<"HasNoSideEffect">;
// Op has the same operand type.
def SameTypeOperands : NativeOpTrait<"SameTypeOperands">;
// Op has same shape for all operands.
def SameOperandsShape : NativeOpTrait<"SameOperandsShape">;
// Op has same operand and result shape.
def SameOperandsAndResultShape : NativeOpTrait<"SameOperandsAndResultShape">;
// Op has the same operand and result type.
def SameOperandsAndResultType : NativeOpTrait<"SameOperandsAndResultType">;
// Op has the same element type (or type itself, if scalar) for all operands.
def SameOperandsElementType : NativeOpTrait<"SameOperandsElementType">;
// Op has the same operand and result element type (or type itself, if scalar).
def SameOperandsAndResultElementType :
  NativeOpTrait<"SameOperandsAndResultElementType">;
// Op is a symbol.
def Symbol : NativeOpTrait<"Symbol">;
// Op defines a symbol table.
def SymbolTable : NativeOpTrait<"SymbolTable">;
// Op is a terminator.
def Terminator : NativeOpTrait<"IsTerminator">;

// Op's regions have a single block with the specified terminator.
class SingleBlockImplicitTerminator<string op>
    : ParamNativeOpTrait<"SingleBlockImplicitTerminator", op>;

// Op's parent operation is the provided one.
class HasParent<string op>
    : ParamNativeOpTrait<"HasParent", op>;

// Op result type is derived from the first attribute. If the attribute is an
// subclass of `TypeAttrBase`, its value is used, otherwise, the type of the
// attribute content is used.
def FirstAttrDerivedResultType :
  GenInternalOpTrait<"FirstAttrDerivedResultType">;

// TODO(antiagainst): Turn the following into normal traits and generate
// verification for them.

// All variadic operands of the op have the same number of values.
// A variadic operand contains an array of values whose array size is only
// known at runtime. This trait requires all variadic operands of an op
// to have the same array size.
def SameVariadicOperandSize : GenInternalOpTrait<"SameVariadicOperandSize">;
// All variadic results of the op have the same number of values.
// A variadic result contains an array of values whose array size is only
// known at runtime. This trait requires all variadic results of an op
// to have the same array size.
def SameVariadicResultSize : GenInternalOpTrait<"SameVariadicResultSize">;

// Uses an attribute named `operand_segment_sizes` to specify how many actual
// operand each ODS-declared operand (variadic or not) corresponds to.
// This trait is used for ops that have multiple variadic operands but do
// not know statically their size relationship. The attribute must be a 1D
// vector that has the same number of elements as the number of ODS declared
// operands. That means even if some operands are non-variadic, the attribute
// still need to have an element for its size, which is always 1.
def AttrSizedOperandSegments : NativeOpTrait<"AttrSizedOperandSegments">;
// Similar to AttrSizedOperandSegments, but used for results. The attribute
// should be named as `result_segment_sizes`.
def AttrSizedResultSegments  : NativeOpTrait<"AttrSizedResultSegments">;

//===----------------------------------------------------------------------===//
// OpInterface definitions
//===----------------------------------------------------------------------===//

// Marker used to identify the argument list for an op or interface method.
def ins;

// OpInterfaceTrait corresponds to a specific 'OpInterface' class defined in
// C++. The purpose to wrap around C++ symbol string with this class is to make
// interfaces specified for ops in TableGen less alien and more integrated.
class OpInterfaceTrait<string name> : NativeOpTrait<""> {
  let trait = name # "::Trait";
}

// This class represents a single, optionally static, interface method.
// Note: non-static interface methods have an implicit 'op' parameter
// corresponding to an instance of the derived operation.
class InterfaceMethod<string desc, string retTy, string methodName,
                      dag args = (ins), code methodBody = [{}]> {
  // A human-readable description of what this method does.
  string description = desc;

  // The name of the interface method.
  string name = methodName;

  // The c++ type-name of the return type.
  string returnType = retTy;

  // A dag of string that correspond to the arguments of the method.
  dag arguments = args;

  // An optional body to the method.
  code body = methodBody;
}

// This class represents a single static interface method.
class StaticInterfaceMethod<string desc, string retTy, string methodName,
                            dag args = (ins), code methodBody = [{}]>
    : InterfaceMethod<desc, retTy, methodName, args, methodBody>;

// OpInterface represents an interface regarding an op.
class OpInterface<string name> : OpInterfaceTrait<name> {
  // A human-readable description of what this interface does.
  string description = "";

  // The name given to the c++ interface class.
  string cppClassName = name;

  // The list of methods defined by this interface.
  list<InterfaceMethod> methods = [];
}

// Whether to declare the op interface methods in the op's header. This class
// simply wraps an OpInterface but is used to indicate that the method
// declarations should be generated.
class DeclareOpInterfaceMethods<OpInterface interface> :
  OpInterface<interface.cppClassName> {
    let description = interface.description;
    let cppClassName = interface.cppClassName;
    let methods = interface.methods;
}

//===----------------------------------------------------------------------===//
// Op definitions
//===----------------------------------------------------------------------===//

// Marker used to identify the result list for an op.
def outs;

// Marker used to identify the region list for an op.
def region;

// Class for defining a custom builder.
//
// TableGen generates several generic builders for each op by default (see
// comment in the `Op` class). If the default generated ones cannot cover
// some use case, custom builders can be defined using instances of this class.
//
// The signature of the builder is always
//
// ```c++
// static void build(Builder *builder, OperationState &state,
//                   <other-parameters>...) {
//   <body>...
// }
// ```
//
// To define a custom builder, the parameter list (*including* the `Builder
// *builder, OperationState &state` part) and body should be passed in
// as separate template arguments to this class. This is because we generate
// op declaration and definition into separate files. If an empty string is
// passed in for `body`, then *only* the builder declaration will be
// generated; this provides a way to define complicated builders entirely
// in C++.
class OpBuilder<string p, code b = ""> {
  string params = p;
  code body = b;
}

// Base class for all ops.
class Op<Dialect dialect, string mnemonic, list<OpTrait> props = []> {
  // The dialect of the op.
  Dialect opDialect = dialect;

  // The mnemonic of the op.
  string opName = mnemonic;

  // One-line human-readable description of what the op does.
  string summary = "";

  // Additional, longer human-readable description of what the op does.
  string description = "";

  // Dag containing the arguments of the op. Default to 0 arguments.
  dag arguments = (ins);

  // The list of results of the op. Default to 0 results.
  dag results = (outs);

  // The list of regions of the op. Default to 0 regions.
  dag regions = (region);

  // Attribute getters can be added to the op by adding an Attr member
  // with the name and type of the attribute. E.g., adding int attribute
  // with name "value" and type "i32":
  //   I32Attr value;

  // Define the hooks used for building, parsing, printing, verification.

  // Custom builder.
  // In addition to the custom builder provided here, and unless
  // skipDefaultBuilders is set, two default builders are generated, with the
  // following signatures:
  //
  // ```c++
  // static void build(Builder *, OperationState &tblgen_state,
  //                   Type <result0-name>, Type <result1-name>, ...,
  //                   Value <arg0-name>, Value <arg1-name>, ...,
  //                   Attribute <attr0-name>, Attribute <attr1-name>, ...);
  // ```
  // * where the attributes follow the same declaration order as in the op.
  //
  // ```c++
  // static void build(Builder *, OperationState &tblgen_state,
  //                   ArrayRef<Type> resultTypes,
  //                   ArrayRef<Value> operands,
  //                   ArrayRef<NamedAttribute> attributes);
  // ```
  list<OpBuilder> builders = ?;

  // Avoid generating default build functions.  Custom builders must be
  // provided.
  bit skipDefaultBuilders = 0;

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
  bit hasCanonicalizer = 0;

  // Whether this op has a folder.
  bit hasFolder = 0;

  // Op traits.
  list<OpTrait> traits = props;

  // Additional code that will be added to the public part of the generated
  // C++ code of the op declaration.
  code extraClassDeclaration = ?;
}

// The arguments of an op.
class Arguments<dag args> {
  dag arguments = args;
}

// The results of an op.
class Results<dag rets> {
  dag results = rets;
}

//===----------------------------------------------------------------------===//
// Common value constraints
//===----------------------------------------------------------------------===//

def HasNoUseOf: Constraint<
    CPred<"$_self->use_begin() == $_self->use_end()">, "has no use">;

//===----------------------------------------------------------------------===//
// Common op type constraints
//===----------------------------------------------------------------------===//

// These traits are for verifying properties of an op that require knowledge of
// multiple arguments or results. For verifying properties of a single argument
// or result, prefer operand type constraints.

// These traits often require including "mlir/IR/TypeUtilities.h".

// TODO(b/135033717): Improve the autogenerated error messages.

class Rank<string name> :
    StrFunc<"$" # name # ".getType().cast<ShapedType>().getRank()">;

class Shape<string name> :
    StrFunc<"$" # name # ".getType().cast<ShapedType>().getShape()">;

class ElementCount<string name> :
  StrFunc<"$" # name # ".getType().cast<ShapedType>().getNumElements()">;

class ElementType<string name> : StrFunc<"getElementTypeOrSelf($" # name # ")">;

class AllMatchPred<list<string> values> :
    CPred<"llvm::is_splat(llvm::makeArrayRef({"# StrJoin<values>.result #"}))">;

class AllMatch<list<string> values, string description> :
    PredOpTrait<description, AllMatchPred<values>>;

// TODO(b/135032064): Only works for non-variadic.
class AllMatchSameOperatorPred<list<string> names, string operator> :
    AllMatchPred<!foreach(n, names, !subst("$_self", "$" # n, operator))>;

class AllMatchSameOperatorTrait<list<string> names, string operator,
                                string description> :
    PredOpTrait<
        "all of {" # StrJoin<names>.result # "} have same " # description,
        AllMatchSameOperatorPred<names, operator>>;

class AllElementCountsMatch<list<string> names> :
    AllMatchSameOperatorTrait<names, ElementCount<"_self">.result,
                              "element count">;

class AllElementTypesMatch<list<string> names> :
    AllMatchSameOperatorTrait<names, ElementType<"_self">.result,
                              "element type">;

class AllRanksMatch<list<string> names> :
    AllMatchSameOperatorTrait<names, Rank<"_self">.result, "rank">;

class AllShapesMatch<list<string> names> :
    AllMatchSameOperatorTrait<names, Shape<"_self">.result, "shape">;

class AllTypesMatch<list<string> names> :
    AllMatchSameOperatorTrait<names, "$_self.getType()", "type">;

// Type Constraint operand `idx`'s Element type is `type`.
class TCopVTEtIs<int idx, Type type> : And<[
   CPred<"$_op.getNumOperands() > " # idx>,
   SubstLeaves<"$_self", "$_op.getOperand(" # idx # ")->getType()",
     IsShapedTypePred>,
   SubstLeaves<"$_self", "getElementTypeOrSelf($_op.getOperand(" # idx # "))",
     type.predicate>]>;

// Predicate to verify that a named argument or result's element type matches a
// given type.
class TypeIsPred<string name, Type type> :
   SubstLeaves<"$_self", "$" # name # ".getType()", type.predicate>;
class TypeIs<string name, Type type> : PredOpTrait<
  "'" # name # "' is " # type.description, TypeIsPred<name, type>>;

// Predicate to verify that a named argument or result's element type matches a
// given type.
class ElementTypeIsPred<string name, Type type> : And<[
   SubstLeaves<"$_self", "$" # name # ".getType()", IsShapedTypePred>,
   SubstLeaves<"$_self", "getElementTypeOrSelf($" # name # ")",
     type.predicate>]>;
class ElementTypeIs<string name, Type type> : PredOpTrait<
  "'" # name # "' is " # type.description, ElementTypeIsPred<name, type>>;

// Predicate to verify that the i'th operand and the j'th operand have the same
// elemental type.
// Type Constraint operand `i`'s Element type is Same As operand `j`'s Element
// type.
class TCopVTEtIsSameAs<int i, int j> : And<[
    CPred<"$_op.getNumOperands() > std::max(" # i # "u," # j # "u)">,
    SubstLeaves<"$_self", "$_op.getOperand(" # i # ")->getType()",
      IsShapedTypePred>,
    SubstLeaves<"$_self", "$_op.getOperand(" # j # ")->getType()",
      IsShapedTypePred>,
    CPred<"mlir::getElementTypeOrSelf($_op.getOperand(" # i # ")) == "
          "mlir::getElementTypeOrSelf($_op.getOperand(" # j # "))">]>;

// Predicate to verify that the i'th result and the j'th operand exist and has
// shaped types.
class TCOpResIsShapedTypePred<int i, int j> : And<[
    CPred<"$_op.getNumResults() > " # i>,
    CPred<"$_op.getNumOperands() > " # j>,
    SubstLeaves<"$_self", "$_op.getResult(" # i # ")->getType()",
      IsShapedTypePred>,
    SubstLeaves<"$_self", "$_op.getOperand(" # j # ")->getType()",
      IsShapedTypePred>]>;

// Predicate to verify that the i'th result and the j'th operand have the same
// type.
class TCresIsSameAsOpBase<int i, int j> :
    CPred<"$_op.getResult(" # i # ")->getType() == "
          "$_op.getOperand(" # j # ")->getType()">;

// Basic Predicate to verify that the i'th result and the j'th operand have the
// same elemental type.
class TCresVTEtIsSameAsOpBase<int i, int j> :
    CPred<"getElementTypeOrSelf($_op.getResult(" # i # ")) == "
          "getElementTypeOrSelf($_op.getOperand(" # j # "))">;

// Predicate to verify that the i'th result and the j'th operand have the same
// elemental type.
// Type Constraint result`i`'s Element type is Same As Operand `j`'s Element
// type.
class TCresVTEtIsSameAsOp<int i, int j> : And<[
    TCOpResIsShapedTypePred<i, j>,
    TCresVTEtIsSameAsOpBase<i, j>]>;

// Predicate to verify that the opId'th operand can be broadcasted to the type
// of the resId'th result.
class TCOpIsBroadcastableToRes<int opId, int resId> : And<[
    TCOpResIsShapedTypePred<opId, resId>,
    CPred<"OpTrait::util::getBroadcastedType("
              "$_op.getOperand(" # opId # ")->getType(), "
              "$_op.getResult(" # resId # ")->getType())">]>;

// Predicate to verify that all the operands at the given `indices`
// have the same element type.
// Type Constraint operands' Element type are all Same At the given `indices`.
// We query the operands' types into a list and check they are all the same.
// Precondition:
// 1) all operands involved are of shaped type and
// 2) the indices are not out of range.
class TCopVTEtAreSameAt<list<int> indices> : CPred<
  "llvm::is_splat(mlir::functional::map("
    "[this](unsigned i) { return getElementTypeOrSelf(this->getOperand(i)); }, "
    "llvm::ArrayRef<unsigned>({" # StrJoinInt<indices>.result # "})))">;

//===----------------------------------------------------------------------===//
// Pattern definitions
//===----------------------------------------------------------------------===//

// Marker used to identify the delta value added to the default benefit value.
def addBenefit;

// Base class for op+ -> op+ rewrite rules. These allow declaratively
// specifying rewrite rules.
//
// A rewrite rule contains two components: a source pattern and one or more
// result patterns. Each pattern is specified as a (recursive) DAG node (tree)
// in the form of `(node arg0, arg1, ...)`.
//
// The `node` are normally MLIR ops, but it can also be one of the directives
// listed later in this section.
//
// ## Symbol binding
//
// In the source pattern, `argN` can be used to specify matchers (e.g., using
// type/attribute type constraints, etc.) and bound to a name for later use.
// We can also bound names to op instances to reference them later in
// multi-entity constraints.
//
// In the result pattern, `argN` can be used to refer to a previously bound
// name, with potential transformations (e.g., using tAttr, etc.). `argN` can
// itself be nested DAG node. We can also bound names to ops to reference
// them later in other result patterns.
//
// For example,
//
// ```
// def : Pattern<(OneResultOp1:$op1 $arg0, $arg1),
//               [(OneResultOp2:$op2 $arg0, $arg1),
//                (OneResultOp3 $op2 (OneResultOp4))],
//               [(HasStaticShapePred $op1)]>;
// ```
//
// `$argN` is bound to the `OneResultOp1`'s N-th argument and used later to
// build `OneResultOp2`. `$op1` is bound to `OneResultOp1` and used to
// check whether the result's shape is static. `$op2` is bound to
// `OneResultOp2` and used to build `OneResultOp3`.
//
// ## Multi-result op
//
// To create multi-result ops in result pattern, you can use a syntax similar
// to uni-result op, and it will act as a value pack for all results:
//
// ```
// def : Pattern<(ThreeResultOp ...),
//               [(TwoResultOp ...), (OneResultOp ...)]>;
// ```
//
// Then `TwoResultOp` will replace the first two values of `ThreeResultOp`.
//
// You can also use `$<name>__N` to explicitly access the N-th result.
// ```
// def : Pattern<(FiveResultOp ...),
//               [(TwoResultOp1:$res1__1 ...), (replaceWithValue $res1__0),
//                (TwoResultOp2:$res2 ...), (replaceWithValue $res2__1)]>;
// ```
//
// Then the values generated by `FiveResultOp` will be replaced by
//
// * `FiveResultOp`#0: `TwoResultOp1`#1
// * `FiveResultOp`#1: `TwoResultOp1`#0
// * `FiveResultOp`#2: `TwoResultOp2`#0
// * `FiveResultOp`#3: `TwoResultOp2`#1
// * `FiveResultOp`#4: `TwoResultOp2`#1
class Pattern<dag source, list<dag> results, list<dag> preds = [],
  dag benefitAdded = (addBenefit 0)> {
  dag sourcePattern = source;
  // Result patterns. Each result pattern is expected to replace one result
  // of the root op in the source pattern. In the case of more result patterns
  // than needed to replace the source op, only the last N results generated
  // by the last N result pattern is used to replace a N-result source op.
  // So that the beginning result patterns can be used to generate additional
  // ops to aid building the results used for replacement.
  list<dag> resultPatterns = results;
  // Multi-entity constraints. Each constraint here involves multiple entities
  // matched in source pattern and places further constraints on them as a
  // whole.
  list<dag> constraints = preds;
  // The delta value added to the default benefit value. The default value is
  // the number of ops in the source pattern. The rule with the highest final
  // benefit value will be applied first if there are multiple rules matches.
  // This delta value can be either positive or negative.
  dag benefitDelta = benefitAdded;
}

// Form of a pattern which produces a single result.
class Pat<dag pattern, dag result, list<dag> preds = [],
  dag benefitAdded = (addBenefit 0)> :
  Pattern<pattern, [result], preds, benefitAdded>;

// Native code call wrapper. This allows invoking an arbitrary C++ expression
// to create an op operand/attribute or replace an op result.
//
// ## Placeholders
//
// If used as a DAG leaf, i.e., `(... NativeCodeCall<"...">:$arg, ...)`,
// the wrapped expression can take special placeholders listed below:
//
// * `$_builder` will be replaced by the current `mlir::PatternRewriter`.
// * `$_self` will be replaced with the entity this transformer is attached to.
//   E.g., with the definition `def transform : NativeCodeCall<"$_self...">`,
//   `$_self` in `transform:$attr` will be replaced by the value for `$attr`.
//
// If used as a DAG node, i.e., `(NativeCodeCall<"..."> <arg0>, ..., <argN>)`,
// then positional placeholders are also supported; placeholder `$N` in the
// wrapped C++ expression will be replaced by `<argN>`.

class NativeCodeCall<string expr> {
  string expression = expr;
}

//===----------------------------------------------------------------------===//
// Common directives
//===----------------------------------------------------------------------===//

// Directive used in result pattern to indicate that no new op are generated,
// so to replace the matched DAG with an existing SSA value.
def replaceWithValue;

#endif // OP_BASE
