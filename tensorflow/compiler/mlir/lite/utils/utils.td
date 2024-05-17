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

def CreateNoneValue : NativeCodeCall<
  "$_builder.create<TFL::NoValueOp>($0.getLoc(), $_builder.getUnitAttr())">;

// Returns shape of a ranked tensor.
// if called without a ranked tensor it will fail.
def GetShape: NativeCodeCall<"GetShape($0)">;

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

// Checks value is not produced by a TFL_Quant or
// from TFL_Quant Op with same quant type.
def NotFromQuantOpOrSameQuantType : Constraint<
  CPred<"tflite::NotFromQuantOpOrSameQuantType($0,$1)">>;

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