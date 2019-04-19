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

#ifdef OP_BASE
#else
#define OP_BASE

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

// Kinds of predicate combiners.  These must closesly match the predicates
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
class AllOf<list<Pred> children> : CombinedPred<PredCombinerAnd, children>;

// A predicate that holds if any of its children hold.  Never holds for zero
// children.
class AnyOf<list<Pred> children> : CombinedPred<PredCombinerOr, children>;

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

// Whether a type is a VectorOrTensorType.
def IsVectorOrTensorTypePred : CPred<"$_self.isa<VectorOrTensorType>()">;

// Whether a type is a TupleType.
def IsTupleTypePred : CPred<"$_self.isa<TupleType>()">;

// Whether a type is a MemRefType.
def IsMemRefTypePred : CPred<"$_self.isa<MemRefType>()">;

// For a TensorType, verify that it is a statically shaped tensor.
def IsStaticShapeTensorTypePred :
  CPred<"$_self.cast<TensorType>().hasStaticShape()">;

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//

// A type, carries type constraints.
class Type<Pred condition, string descr = ""> :
    TypeConstraint<condition, descr>;

// A variadic type constraint. It expands to zero or more of the base type. This
// class is used for supporting variadic operands/results. An op can declare no
// more than one variadic operand/result, and that operand/result must be the
// last one in the operand/result list.
class Variadic<Type type, string descr = "">
    // TODO: support variadic type conditions
    : TypeConstraint<CPred<"true">, descr> {
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

// Any type from the given list
class AnyTypeOf<list<Type> allowedTypes, string description> : Type<
    // Satisfy any of the allowed type's condition
    AnyOf<!foreach(allowedtype, allowedTypes, allowedtype.predicate)>,
    !if(!eq(description, ""),
        // Join all allowed types' descriptions with " or " as the description
        // if not provided during template specialization
        !foldl(/*init*/"", /*list*/allowedTypes, prev, cur,
               prev # " or " # cur.description),
        // Otherwise use the provided one
        description)>;

// Integer types.
class IntegerBase<CPred pred, string descr> : Type<pred, descr>;

// Any integer type irrespective of its width.
def Integer : IntegerBase<CPred<"$_self.isa<IntegerType>()">, "integer">;

// Index type.
def Index : IntegerBase<CPred<"$_self.isa<IndexType>()">, "index">;

// Integer type of a specific width.
class I<int width>
    : IntegerBase<CPred<"$_self.isInteger(" # width # ")">,
                  width # "-bit integer">,
      BuildableType<"getIntegerType(" # width # ")"> {
  int bitwidth = width;
}
def I1  : I<1>;
def I8  : I<8>;
def I16 : I<16>;
def I32 : I<32>;
def I64 : I<64>;

// Floating point types.
class FloatBase<CPred pred, string descr> : Type<pred, descr>;

// Any float type irrespective of its width.
def Float : FloatBase<CPred<"$_self.isa<FloatType>()">, "floating-point">;

// Float type of a specific width.
class F<int width>
    : FloatBase<CPred<"$_self.isF" # width # "()">,
                width # "-bit float">,
      BuildableType<"getF" # width # "Type()"> {
  int bitwidth = width;
}

def F16 : F<16>;
def F32 : F<32>;
def F64 : F<64>;

def BF16 : Type<CPred<"$_self.isBF16()">, "bfloat16 type">,
           BuildableType<"getBF16Type()">;

// A container type is a type that has another type embedded within it.
class ContainerType<Type etype, Pred containerPred, code elementTypeCall,
                    string descr> :
    // First, check the container predicate.  Then, substitute the extracted
    // element into the element type checker.
    Type<AllOf<[containerPred,
                SubstLeaves<"$_self", !cast<string>(elementTypeCall),
                etype.predicate>]>,
         descr # " of " # etype.description # " values"> {
  // The type of elements in the container.
  Type elementType = etype;

  // Call to retrieve.
  code getElementTypeCall = elementTypeCall;
}

// Vector types.
class TypedVector<Type t> : ContainerType<t, IsVectorTypePred,
    "$_self.cast<VectorType>().getElementType()", "vector">;

class Vector<Type t, list<int> dims> : ContainerType<t, AllOf<[
    IsVectorTypePred,
    // Match dims. Construct an ArrayRef with the elements of `dims` by folding
    // over the list.
    CPred<"$_self.cast<VectorType>().getShape() == ArrayRef{{" #
      !foldl("", dims, sum, element, sum #
       !if(!empty(sum), "", ",") # !cast<string>(element)) # "}">]>,
    "$_self.cast<VectorType>().getElementType()",
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
                    "$_self.cast<TensorType>().getElementType()",
                    "tensor">;

class TypedStaticShapeTensor<Type t>
    : Type<AllOf<[ TypedTensor<t>.predicate, IsStaticShapeTensorTypePred ]>,
           "statically shaped tensor">;

def I1Tensor      : TypedTensor<I1>;
def I8Tensor      : TypedTensor<I8>;
def I16Tensor     : TypedTensor<I16>;
def I32Tensor     : TypedTensor<I32>;
def I64Tensor     : TypedTensor<I64>;

def BF16Tensor : TypedTensor<BF16>;
def F16Tensor  : TypedTensor<F16>;
def F32Tensor  : TypedTensor<F32>;
def F64Tensor  : TypedTensor<F64>;

// This represents a generic tuple without any constraints on elemental type,
// ranks, or size. As Tuples can contain tensors, vectors, or scalar values
// there is not only a single elemental type.
def Tuple : Type<IsTupleTypePred, "tuple">;


// A Tuple that only holds elements of a certain type. This cannot inherit from
// ContainerType because tuples do not always have a single element type that
// could be retrieved with elementTypeCall.
class TypedTuple<Type t> :
    Type<AllOf<[
        Tuple.predicate,
        Concat<
            [{
              llvm::all_of(
                $_self.cast<TupleType>().getTypes(),
                [](Type t) {
                  return
            }],
            SubstLeaves<"$_self", "t", t.predicate>,
            "; })">
    ]>, "tuple">;

// Memref type.

// Memrefs are blocks of data with fixed type and rank.
class MemRef<Type t>
    : ContainerType<t, IsMemRefTypePred,
                    "$_self.cast<MemRefType>().getElementType()", "memref">;

// Memref declarations handle any memref, independent of rank, size, (static or
// dynamic), layout, or memory space.
def I1MemRef  : MemRef<I1>;
def I8MemRef  : MemRef<I8>;
def I16MemRef : MemRef<I16>;
def I32MemRef : MemRef<I32>;
def I64MemRef : MemRef<I64>;

def BF16MemRef : MemRef<BF16>;
def F16MemRef  : MemRef<F16>;
def F32MemRef  : MemRef<F32>;
def F64MemRef  : MemRef<F64>;

//===----------------------------------------------------------------------===//
// Common type constraints
//===----------------------------------------------------------------------===//

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
// Attribute definitions
//===----------------------------------------------------------------------===//

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
  bit isOptional = 0b0;
}

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
  let isOptional = 0b1;
}

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
    TypedAttrBase<attrValType, "IntegerAttr",
              AllOf<[CPred<"$_self.isa<IntegerAttr>()">,
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

def I32Attr : IntegerAttrBase<I32, "32-bit integer attribute">;
def I64Attr : IntegerAttrBase<I64, "64-bit integer attribute">;

// Base class for float attributes of fixed width.
class FloatAttrBase<F attrValType, string descr> :
    TypedAttrBase<attrValType, "FloatAttr",
              AllOf<[CPred<"$_self.isa<FloatAttr>()">,
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

// An enum attribute case.
class EnumAttrCase<string sym> : StringBasedAttr<
    CPred<"$_self.cast<StringAttr>().getValue() == \"" # sym # "\"">,
    "case " # sym> {
  // The C++ enumerant symbol
  string symbol = sym;
}

// An enum attribute. Its value can only be one from the given list of `cases`.
// Enum attributes are emulated via mlir::StringAttr, plus extra verification
// on the string: only the symbols of the allowed cases are permitted as the
// string value.
class EnumAttr<string name, string description, list<EnumAttrCase> cases> :
    StringBasedAttr<AllOf<[StrAttr.predicate,
                           AnyOf<!foreach(case, cases, case.predicate)>]>,
                    description> {
  // The C++ enum class name
  string className = name;
  // List of all accepted cases
  list<EnumAttrCase> enumerants = cases;
}

class ElementsAttrBase<Pred condition, string description> :
    Attr<condition, description> {
  let storageType = [{ ElementsAttr }];
  let returnType = [{ ElementsAttr }];
  let convertFromStorage = "$_self";
}

def ElementsAttr: ElementsAttrBase<CPred<"$_self.isa<ElementsAttr>()">,
                                   "constant vector/tensor attribute">;

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
    AllOf<[
      // Guranatee this is an ArrayAttr first
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

// Attributes containing functions.
def FunctionAttr : Attr<CPred<"$_self.isa<FunctionAttr>()">,
                        "function attribute"> {
  let storageType = [{ FunctionAttr }];
  let returnType = [{ Function * }];
  let constBuilderCall = "$_builder.getFunctionAttr($0)";
}

// Base class for attributes containing types. Example:
//   def IntTypeAttr : TypeAttrBase<"IntegerType", "integer type attribute">
// defines a type attribute containing an integer type.
class TypeAttrBase<string retType, string description> :
    Attr<AllOf<[
      CPred<"$_self.isa<TypeAttr>()">,
      CPred<"$_self.cast<TypeAttr>().getValue().isa<" # retType # ">()">]>,
    description> {
  let storageType = [{ TypeAttr }];
  let returnType = retType;
  let convertFromStorage = "$_self.getValue().cast<" # retType # ">()";
}

// DerivedAttr are attributes whose value is computed from properties
// of the operation. They do not require additional storage and are
// materialized as needed.
class DerivedAttr<code ret, code b> : Attr<CPred<"true">, "derived attribute"> {
  let returnType = ret;
  code body = b;
}

// Derived attribute that returns a mlir::Type.
class DerivedTypeAttr<code body> : DerivedAttr<"Type", body>;

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

//===----------------------------------------------------------------------===//
// Common attribute constraints
//===----------------------------------------------------------------------===//

// A general mechanism to further confine the given `attr` with all the
// `constraints`. This allows to compose complex constraints out of a series
// of more primitive ones.
class Confined<Attr attr, list<AttrConstraint> constraints> : Attr<
    AllOf<!listconcat([attr.predicate],
                      !foreach(pred, constraints, pred.predicate))>,
    !foldl(/*init*/attr.description, /*list*/constraints,
           prev, cur, prev # " " # cur.description)> {
  let storageType = attr.storageType;
  let returnType = attr.returnType;
  let convertFromStorage = attr.convertFromStorage;
  let constBuilderCall = attr.constBuilderCall;
  let defaultValue = attr.defaultValue;
  let isOptional = attr.isOptional;
}

// An AttrConstraint that holds if all attr constraints specified in
// 'constraints' hold.
class AllAttrConstraintsOf<list<AttrConstraint> constraints> : AttrConstraint<
    AllOf<!listconcat([!head(constraints).predicate],
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

class IntArrayNthElemEq<int index, int value> : AttrConstraint<
    AllOf<[
      CPred<"$_self.cast<ArrayAttr>().size() > " # index>,
      CPred<"$_self.cast<ArrayAttr>().getValue()[" # index # "]"
        ".cast<IntegerAttr>().getInt() == " # value>
       ]>,
    "whose " # index # "-th element must be " # value>;

class IntArrayNthElemMinValue<int index, int min> : AttrConstraint<
    AllOf<[
      CPred<"$_self.cast<ArrayAttr>().size() > " # index>,
      CPred<"$_self.cast<ArrayAttr>().getValue()[" # index # "]"
        ".cast<IntegerAttr>().getInt() >= " # min>
        ]>,
    "whose " # index # "-th element must be at least " # min>;

//===----------------------------------------------------------------------===//
// OpTrait definitions
//===----------------------------------------------------------------------===//

// OpTrait represents a trait regarding an op.
class OpTrait;

// NativeOpTrait corresponds to the MLIR C++ OpTrait mechanism. The
// purpose to wrap around C++ symbol string with this class is to make
// traits specified for ops in TableGen less alien and more
// integrated.
class NativeOpTrait<string prop> : OpTrait {
  string trait = prop;
}

// Specify a trait to control op definition generator internals.
class OpGenInternalTrait<string prop> : OpTrait {
  string trait = prop;
}

// Specify a trait by way of a predicate on the operation.
class PredOpTrait<string d, Pred p> : OpTrait {
  string desc = d;
  Pred pred = p;
}

// op supports operand broadcast behavior
def Broadcastable    : NativeOpTrait<"BroadcastableTwoOperandsOneResult">;
// X op Y == Y op X
def Commutative      : NativeOpTrait<"IsCommutative">;
// op results are float or vectors/tensors thereof
def FloatLikeResults : NativeOpTrait<"ResultsAreFloatLike">;
// op has no side effect
def NoSideEffect     : NativeOpTrait<"HasNoSideEffect">;
// op has same operand and result shape
def SameValueShape   : NativeOpTrait<"SameOperandsAndResultShape">;
// op has the same operand and result type
def SameValueType    : NativeOpTrait<"SameOperandsAndResultType">;
// op is a terminator
def Terminator       : NativeOpTrait<"IsTerminator">;

// op result type is derived from the first attribute. If the attribute is an
// subclass of `TypeAttrBase`, its value is used, otherwise, the type of the
// attribute content is used.
def FirstAttrDerivedResultType :
  OpGenInternalTrait<"FirstAttrDerivedResultType">;

//===----------------------------------------------------------------------===//
// Op definitions
//===----------------------------------------------------------------------===//

// Marker used to identify the argument list for an op.
def ins;

// Marker used to identify the result list for an op.
def outs;

// Class for defining a custom builder.
//
// TableGen generates several generic builders for each op by default (see
// comment in the `Op` class). If the default generated ones cannot cover
// some use case, custom builders can be defined using instances of this class.
//
// The signature of the builder is always
//
// ```c++
// static void build(Builder *builder, OperationState *result,
//                   <other-parameters>...) {
//   <body>...
// }
// ```
//
// To define a custom builder, the parameter list (*including* the `Builder
// *builder, OperationState *result` part) and body should be passed in
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
class Op<string mnemonic, list<OpTrait> props = []> {
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
  // In addtion to the custom builder provided here, two default builders
  // are generated, with the following signatures:
  //
  // ```c++
  // static void build(Builder *, OperationState *tblgen_state,
  //                   Type <result0-name>, Type <result1-name>, ...,
  //                   Value <arg0-name>, Value <arg1-name>, ...,
  //                   Attribute <attr0-name>, Attribute <attr1-name>, ...);
  // ```
  // * where the attributes follow the same declaration order as in the op.
  //
  // ```c++
  // static void build(Builder *, OperationState *tblgen_state,
  //                   ArrayRef<Type> resultTypes,
  //                   ArrayRef<Value> operands,
  //                   ArrayRef<NamedAttribute> attributes);
  // ```
  list<OpBuilder> builders = ?;

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

  // Op traits.
  list<OpTrait> traits = props;
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
// Common op type constraints
//===----------------------------------------------------------------------===//

// Type Constraint operand `idx`'s Vector or Tensor Element type is `type`.
class TCopVTEtIs<int idx, Type type> : AllOf<[
   CPred<"$_op.getNumOperands() > " # idx>,
   SubstLeaves<"$_self",  "$_op.getOperand(" # idx # ")->getType()",
     IsVectorOrTensorTypePred>,
   SubstLeaves<"$_self",  "$_op.getOperand(" # idx #
     ")->getType().cast<VectorOrTensorType>().getElementType()",
     type.predicate>]>;

// Predicate to verify that the i'th operand and the j'th operand have the same
// elemental type.
// Type Constraint operand `i`'s Vector or Tensor Element type is Same As
// operand `j`'s element type.
class TCopVTEtIsSameAs<int i, int j> : AllOf<[
    CPred<"$_op.getNumOperands() > std::max(" # i # "," # j # ")">,
    SubstLeaves<"$_self",  "$_op.getOperand(" # i # ")->getType()",
      IsVectorOrTensorTypePred>,
    SubstLeaves<"$_self",  "$_op.getOperand(" # j # ")->getType()",
      IsVectorOrTensorTypePred>,
    // TODO: This could be made into C++ function instead.
    CPred<"$_op.getOperand(" # i # ")->getType().cast<VectorOrTensorType>()."
          "getElementType() == $_op.getOperand(" # j # ")->getType()."
          "cast<VectorOrTensorType>().getElementType()">]>;

// Predicate to verify that the i'th result and the j'th operand have the same
// elemental type.
// Type Constraint result`i`'s Vector or Tensor Element type is Same As
// Type Constraint Operand `j`'s Vector or Tensor Element type.
class TCresVTEtIsSameAsOp<int i, int j> : AllOf<[
    CPred<"$_op.getNumResults() > " # i>,
    CPred<"$_op.getNumOperands() > " # j>,
    SubstLeaves<"$_self",  "$_op.getResult(" # i # ")->getType()",
      IsVectorOrTensorTypePred>,
    SubstLeaves<"$_self",  "$_op.getOperand(" # j # ")->getType()",
      IsVectorOrTensorTypePred>,
    // TODO: This could be made into C++ function instead.
    CPred<"$_op.getResult(" # i # ")->getType().cast<VectorOrTensorType>()."
          "getElementType() == $_op.getOperand(" # j # ")->getType()."
          "cast<VectorOrTensorType>().getElementType()">]>;

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
// In the source pattern, `argN` can be used to specify matchers (e.g., using
// type/attribute type constraints, etc.) and bound to a name for later use.
// We can also bound names to op instances to reference them later in
// multi-entity constraints.
//
// In the result pattern, `argN` can be used to refer to a previously bound
// name, with potential transformations (e.g., using tAttr, etc.). `argN` can
// itself be nested DAG node. We can also bound names to op results to reference
// them later in other result patterns.
//
// For example,
//
// ```
// def : Pattern<(OneResultOp1:$res1 $arg0, $arg1),
//               [(OneResultOp2:$res2 $arg0, $arg1),
//                (OneResultOp3 $res2 (OneResultOp4))],
//               [(IsStaticShapeTensorTypePred $res1)]>;
// ```
//
// `$argN` is bound to the `OneResultOp1`'s N-th argument and used later to
// build `OneResultOp2`. `$res1` is bound to `OneResultOp1`'s result and used to
// check whether the result's shape is static. `$res2` is bound to the result of
// `OneResultOp2` and used to build `OneResultOp3`.
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

// Attribute transformation. This is the base class to specify a transformation
// of matched attributes. Used on the output attribute of a rewrite rule.
//
// ## Placeholders
//
// The following special placeholders are supported
//
// * `$_builder` will be replaced by the current `mlir::PatternRewriter`.
// * `$_self` will be replaced with the entity this transformer is attached to.
//   E.g., with the definition `def transform : tAttr<$_self...>`, `$_self` in
//   `transform:$attr` will be replaced by  the value for `$att`.

// Besides, if this is used as a DAG node, i.e., `(tAttr <arg0>, ..., <argN>)`,
// then positional placeholders are supported and placholder `$N` will be
// replaced by `<argN>`.
class tAttr<code transform> {
  code attrTransform = transform;
}

// Native code op creation method. This allows performing an arbitrary op
// creation/replacement by invoking a C++ function with the operands and
// attributes. The function specified needs to have the signature:
//
//   void f(Operation *op, ArrayRef<Value *> operands,
//          ArrayRef<Attribute> attrs, PatternRewriter &rewriter);
//
// The operands and attributes are passed to this function in the order of
// the DAG specified. It is the responsibility of this function to replace the
// matched op(s) using the rewriter. This is intended for the long tail op
// creation and replacement.
// TODO(antiagainst): Unify this and tAttr into a single creation mechanism.
class cOp<string f> {
  // Function to invoke with the given arguments to construct a new op. The
  // operands will be passed to the function first followed by the attributes
  // (as in the function signature above and required by Op arguments).
  string function = f;
}

//===----------------------------------------------------------------------===//
// Common directives
//===----------------------------------------------------------------------===//

// Directive used in result pattern to indicate that no new result op are
// generated, so to replace the matched DAG with an existing SSA value.
def replaceWithValue;

// Directive used in result pattern to indicate that no replacement is generated
// for the current result. Predicates are generated to make sure the
// corresponding result in source pattern is unused.
//   syntax: (verifyUnusedValue)
def verifyUnusedValue;

#endif // OP_BASE
