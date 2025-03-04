/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Utility predicates that are shared by multiple passes.

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mlir/IR/PatternBase.td"

////////////////////////////////////////////////////////////////////////////////
///////////////// TENSOR RANK UTILITIES ////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Checks if the rank of the value is less than or equal to the rank of the
// other value.
def IsRankLessThanEqualTo : Constraint<CPred<
  "$0.getType().cast<ShapedType>().getRank() <= "
  "$1.getType().cast<ShapedType>().getRank()">>;

// Checks if the value has rank at most 'n'.
class HasRankAtMost<int n> : Constraint<
    CPred<"$0.getType().cast<ShapedType>().hasRank() && "
          "$0.getType().cast<ShapedType>().getRank() <= " # n>>;

////////////////////////////////////////////////////////////////////////////////
///////////////// DENSE UTILITIES /////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

def DenseFPElementsAttrPred : CPred<"$_self.isa<DenseFPElementsAttr>()">;
def DenseIntElementsAttrPred : CPred<"$_self.isa<DenseIntElementsAttr>()">;

////////////////////////////////////////////////////////////////////////////////
///////////////// SPLAT CONSTANT UTILITIES /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

def DenseElementsAttrIsSplatPred
    : CPred<"$_self.cast<DenseElementsAttr>().isSplat()">;

class DenseFPElementsAttrSplatValueEqualToPred<string val>
    : CPred<"$_self.cast<DenseFPElementsAttr>().getSplatValue<FloatAttr>()"
            ".getValueAsDouble() == " # val>;

class DenseFPElementsAttrSplatValueEqualToPredWithTolerance<string val, string tolerance>
    : CPred<"std::abs($_self.cast<DenseFPElementsAttr>().getSplatValue<FloatAttr>()"
            ".getValueAsDouble() - " # val # ") <= "#tolerance>;

class DenseIntElementsAttrSplatValueEqualToPred<string val>
    : CPred<"$_self.isa<DenseIntElementsAttr>() && "
            "$_self.cast<DenseIntElementsAttr>().getElementType()"
            "  .isa<IntegerType>() && "
            "$_self.cast<DenseIntElementsAttr>().isSplat() && "
            "$_self.cast<DenseIntElementsAttr>().getSplatValue<IntegerAttr>()"
            "  .getValue().getSExtValue() == " # val>;

// AttrConstraint to match a floating point dense elements attribute with a
// splat value equals to `Value`.
class FPSplatConstAttr<string Value> : AttrConstraint<And<[
    DenseFPElementsAttrPred, DenseElementsAttrIsSplatPred,
    DenseFPElementsAttrSplatValueEqualToPred<Value>]>>;

class FPSplatConstAttrWithTolerance<string Value, string Tolerance> :
    AttrConstraint<And<[
        DenseFPElementsAttrPred, DenseElementsAttrIsSplatPred,
        DenseFPElementsAttrSplatValueEqualToPredWithTolerance<Value, Tolerance>]>>;

// AttrConstraint to match a dense elements attribute with a splat value equal
// to `Value`.
class IntSplatConstAttr<string value> : AttrConstraint<
    DenseIntElementsAttrSplatValueEqualToPred<value>>;

// A constant tensor that has elements with the same value.
def SplatElementsAttr : ElementsAttrBase<DenseElementsAttrIsSplatPred,
  "splat tensor attr">;

// A constant tensor that has elements with the same floating point value.
def SplatFPElementsAttr : ElementsAttrBase<
  And<[DenseFPElementsAttrPred, DenseElementsAttrIsSplatPred]>,
  "f32 splat tensor attr">;

// A constant tensor that has elements with the same integer value.
def SplatIntElementsAttr : ElementsAttrBase<
  And<[DenseIntElementsAttrPred, DenseElementsAttrIsSplatPred]>,
  "integer splat tensor attr">;

// Extracts the scalar constant value.
def GetScalarElementsAttrFromSplat : NativeCodeCall<
    "DenseElementsAttr::get("
    " RankedTensorType::get({},"
        " $0.cast<mlir::DenseElementsAttr>().getType().getElementType()),"
    " $0.cast<mlir::DenseElementsAttr>().getSplatValue<mlir::Attribute>())">;

////////////////////////////////////////////////////////////////////////////////
///////////////// OP BROADCASTING UTILITIES ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

def OperandsBroadcastToOutputType : Constraint<CPred<
  "TFL::OperandsBroadcastToOutputType($0.getType(), $1.getType(), "
                                     "$2.getType())">>;

////////////////////////////////////////////////////////////////////////////////
///////////////// TENSOR SHAPE UTILITIES ///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

def HasSameStaticShapes : Constraint<
    CPred<"$0.getType().cast<ShapedType>().hasStaticShape() && "
          "$1.getType().cast<ShapedType>().hasStaticShape() && "
          "$0.getType().cast<ShapedType>().getShape() =="
          "$1.getType().cast<ShapedType>().getShape()">,
    "have the same static shape">;

def CreateNoneValue : NativeCodeCall<
  "$_builder.create<TFL::NoValueOp>($0.getLoc(), $_builder.getUnitAttr())">;

// Returns shape of a ranked tensor.
// if called without a ranked tensor it will fail.
def GetShape: NativeCodeCall<"GetShape($0)">;

// Return the resultant shape if the shape of the supplied attribute/value is
// expanded by n leading 1s'.
class GetExpandedShapeAttr<int n> : NativeCodeCall<
  "GetExpandedShapeAttr($0, " # n # ")">;

// Return the resultant shape type if the shape of the supplied attribute/value
// is expanded by n leading 1s'.
class GetExpandedShapeType<int n> : NativeCodeCall<
  "GetExpandedShapeType($0, " # n # ")">;

// Constraint that values in list attribute are all ones.
def IsAllOnesConstant : Constraint<CPred<"TFL::IsAllOnesConstant($0)">>;

// Constraint that checks if the transpose op is trivial. Trivial means that
// the permutation is a cyclic permutation of the original shape with only the
// identity dimensions permuted.
def IsTransposeTrivial : Constraint<CPred<
  "TFL::IsTransposeTrivial($0.getType().cast<ShapedType>().getShape(), $1)">>;

// Constraint that checks if the reshape op is equivalent to a transpose op.
// This is true if the reshape op is a trivial reshape op, meaning no change in
// the order of non-identity dimensions.
def IsReshapeEquivalentToTranspose : Constraint<CPred<
  "TFL::IsReshapeEquivalentToTranspose("
    "$0.getType().cast<ShapedType>(),"
    "$1.getType().cast<ShapedType>())">>;

// Returns the permutation of the trivial reshape op, this will be used to
// construct the transpose op.
def GetPermutationFromTrivialReshape : NativeCodeCall<
  "TFL::GetPermutationFromTrivialReshape("
    "$0.getType().cast<ShapedType>(),"
    "$1.getType().cast<ShapedType>())">;

// Constraint that checks if all values in offset between two
// attributes are non-negative.
def HasNonNegativeOffset : Constraint<CPred<"TFL::HasNonNegativeOffset($0, $1)">>;

// Constraint that checks if all values in list attribute are non-negative.
def HasNonNegativeValues : Constraint<CPred<"TFL::HasNonNegativeValues($0)">>;

// Utility function to get the offset between two dense attribute values.
def GetOffSet : NativeCodeCall<"TFL::GetOffSet($0, $1)">;

// Attribute Constraint that checks if the attribute value is zero.
def ZeroIntAttr
  : AttrConstraint<CPred<"$_self.cast<::mlir::IntegerAttr>().getInt() == 0">>;

// Checks if the value has rank at most 'n'.
class HasRankAtLeast<int n> : Constraint<
    CPred<"$0.getType().cast<ShapedType>().hasRank() && "
          "$0.getType().cast<ShapedType>().getRank() >= " # n>>;

// Accepts two inputs and check if both have the same element type.
def SameElementType : Constraint<
  CPred<"getElementTypeOrSelf($0) == getElementTypeOrSelf($1)">>;

// Returns a ShapedType for a permutation and the shape of input after
// applying the permutation to the given shape through a transpose.
class GetTransposedType<string perm> : NativeCodeCall<
  "GetTransposedType($0, " # perm # ")">;

// Function to map final permutation to initial permutation
// initial -> permutation1 -> permutation2 -> final
def RemapPermutation: NativeCodeCall<"RemapPermutation($0, $1)">;

// Checks if all of an ops inputs are the same static shape.
// BUILD NOTE: "OpHasSameStaticShapes" here refers to the C++ function defined 
// in `utils/utils.h`. The `utils.h` header is included in `tfl_ops.h` so all
// of our files will have access to `OpHasSameStaticShapes` when including files
// generated from table-gen.
def OpHasSameStaticShapesPred : CPred<"OpHasSameStaticShapes($0.getDefiningOp())">;
def OpHasSameStaticShapes : Constraint<OpHasSameStaticShapesPred, "op must have static same input shapes">;
def OpHasNotSameStaticShapes : Constraint<Neg<OpHasSameStaticShapesPred>, "op must have not static same input shapes">;

def TransposeFCLastTwoDims:
  NativeCodeCall<"TransposeLastTwoDims($0[0].getType())">;

def AreLastTwoDimsTransposed : Constraint<CPred<
  "TFL::AreLastTwoDimsTransposed($0)">>;

// Checks if the param passed is of NoneType.
def IsNoneType : Constraint<CPred<"$0.getType().isa<NoneType>()">>;

def ConstantLikePred : CPred<"::mlir::matchPattern($0, ::mlir::m_Constant())">;
def IsConstantLike : Constraint<ConstantLikePred>;
def NotConstantLike : Constraint<Neg<ConstantLikePred>>;

// Here, the element type can be any integer or float type. But, note that only
// 32 bit integers are supported for the values.
class GetScalarOfType<int value> : NativeCodeCall<
  "GetScalarOfType(getElementTypeOrSelf($0)," # value # ")">;
