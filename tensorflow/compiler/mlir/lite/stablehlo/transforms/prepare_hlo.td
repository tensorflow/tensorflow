// /* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================*/

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mhlo/IR/hlo_ops.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"

//===------------------------------------------------------------------------===
// mhlo.convolution
//===------------------------------------------------------------------------===

// Re-Layout Patterns
//
// mhlo.convolution can encode arbitrary layouts for all its parameters
// (mhlo::ConvDimensionNumbersAttr). TFL convolutions require parameter layouts
// to be of a specific configuration:
//
// TFL Native Standard Conv Layouts:
// 2D             : [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// 3D             : [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]
// 2D (depthwise) : [b, 0, 1, f]x[i, 0, 1, o]->[b, 0, 1, f]
//
// The following patterns transpose the inputs and output of mhlo.convolution
// ops until they are complicit with the TFL required layout.
// Some of these defs depend on types in `conv_util.h`.

// Given from, to `Layout` get the permutation array for transposing one to
// the other.
def PermForReLayout : NativeCodeCall<
   "DenseIntElementsAttr::get("
      "RankedTensorType::get("
        "{$0.Rank()},"
        "$_builder.getI64Type()),"
      "$0.GetPermForReLayout($1))">;

def AreDnumsFullyDefined : Constraint<CPred<
  "$0.getDefiningOp<mhlo::ConvolutionOp>()"
    ".getLhs().getType().getShape().size()"
  "== $0.getDefiningOp<mhlo::ConvolutionOp>()"
    ".getDimensionNumbers().getInputSpatialDimensions().size() + 2">>;

// Given $0: Layout (from layout), $1: Layout (to layout), $2: Value
// get a new Type with the same element type as value, but with
// shape permuted based on from and to layouts.
def PermuteShape : NativeCodeCall<
  "llvm::cast<RankedTensorType>($2.getType())"
    ".clone($0.PermuteShape("
      "$1,"
      "llvm::cast<RankedTensorType>($2.getType()).getShape()))">;

def IsStandardConv : Constraint<CPred<
  "IsStandardConv($0.getDefiningOp<mhlo::ConvolutionOp>())">>;

def IsDepthwiseConv : Constraint<CPred<
  "IsDepthwiseConv($0.getDefiningOp<mhlo::ConvolutionOp>())">>;

def IsSupportedNonTrivialConv : Constraint<CPred<
  "IsSupportedNonTrivialConv($0.getDefiningOp<mhlo::ConvolutionOp>())">>;

def IsSupportedConv : Constraint<Or
  <[IsStandardConv.predicate, IsDepthwiseConv.predicate, IsSupportedNonTrivialConv.predicate]>>;

def IsSupportedStandardOrNonTrivialConv : Constraint<Or
  <[IsStandardConv.predicate, IsSupportedNonTrivialConv.predicate]>>;

def IsStandardOrDepthwiseConv : Constraint<Or
  <[IsStandardConv.predicate, IsDepthwiseConv.predicate]>>;

//
// Re-layout input (lhs) to [b, spatials..., f]
//===---------------------------------------------------------------------------

def IsInputNotTFLNativeLayout : Constraint<CPred<
    "Layout("
      "$0.getInputBatchDimension(),"
      "$0.getInputFeatureDimension(),"
      "$0.getInputSpatialDimensions())"
    "!= GetTFLNativeInputOrOutputLayout($0)">>;

def InputLayout : NativeCodeCall<
    "Layout($0.getInputBatchDimension(),"
                "$0.getInputFeatureDimension(),"
                "$0.getInputSpatialDimensions())">;

def TFLNativeInputLayout : NativeCodeCall<
    "GetTFLNativeInputOrOutputLayout($0)">;

def InputHasIotaSpatials : Constraint<CPred<
    "Layout($0.getInputBatchDimension(),"
                "$0.getInputFeatureDimension(),"
                "$0.getInputSpatialDimensions()).AreSpatialsIota()">>;

// Copy dnums with the input layout set to [b, spatials..., f].
def CloneDnumsWithTFLNativeInputLayout : NativeCodeCall<
    "CloneDnumsWithInputLayout("
      "$_builder,"
      "$0,"
      "GetTFLNativeInputOrOutputLayout($0))">;

def ReLayoutConvInput : Pat<(MHLO_ConvolutionOp:$conv
                                $input,
                                $kernel,
                                $strides,
                                $padding,
                                $lhs_dilation,
                                $rhs_dilation,
                                $window_reversal,
                                $dnums,
                                $feature_groups,
                                $batch_groups,
                                $precision_config
                            ),
                            (MHLO_ConvolutionOp
                                (MHLO_TransposeOp
                                  $input,
                                    (PermForReLayout
                                      (InputLayout $dnums),
                                      (TFLNativeInputLayout $dnums))
                                ),
                                $kernel,
                                $strides,
                                $padding,
                                $lhs_dilation,
                                $rhs_dilation,
                                $window_reversal,
                                (CloneDnumsWithTFLNativeInputLayout $dnums),
                                $feature_groups,
                                $batch_groups,
                                $precision_config
                            ),
                            [(AreDnumsFullyDefined $conv),
                             (InputHasIotaSpatials $dnums),
                             (IsInputNotTFLNativeLayout $dnums),
                             (IsSupportedConv $conv)],
                             [],
                             (addBenefit 1)>;


//
// Re-layout kernel
//===---------------------------------------------------------------------------

def KernelHasIotaSpatials : Constraint<CPred<
    "Layout($0.getKernelInputFeatureDimension(),"
                "$0.getKernelOutputFeatureDimension(),"
                "$0.getKernelSpatialDimensions()).AreSpatialsIota()">>;

def KernelLayout : NativeCodeCall<
    "Layout($0.getKernelInputFeatureDimension(),"
                "$0.getKernelOutputFeatureDimension(),"
                "$0.getKernelSpatialDimensions())">;

//
// standard conv kernel = [o, spatials..., i].
//=-----

def IsKernelNotTFLNativeStandardConvLayout : Constraint<CPred<
    "Layout("
      "$0.getKernelInputFeatureDimension(),"
      "$0.getKernelOutputFeatureDimension(),"
      "$0.getKernelSpatialDimensions())"
    "!= GetTFLNativeStandardConvKernelLayout($0)">>;

def TFLNativeStandardConvKernelLayout : NativeCodeCall<
    "GetTFLNativeStandardConvKernelLayout($0)">;

// Copy dnums with the kernel layout set to [o, spatials..., i].
def CloneDnumsWithTFLNativeStandardConvKernelLayout : NativeCodeCall<
    "CloneDnumsWithKernelLayout("
      "$_builder,"
      "$0,"
      "GetTFLNativeStandardConvKernelLayout($0))">;


def ReLayoutConvKernel : Pat<(MHLO_ConvolutionOp:$conv
                                $input,
                                $kernel,
                                $strides,
                                $padding,
                                $lhs_dilation,
                                $rhs_dilation,
                                $window_reversal,
                                $dnums,
                                $feature_groups,
                                $batch_groups,
                                $precision_config
                            ),
                            (MHLO_ConvolutionOp
                                $input,
                                (MHLO_TransposeOp
                                  $kernel,
                                    (PermForReLayout
                                      (KernelLayout $dnums),
                                      (TFLNativeStandardConvKernelLayout $dnums))
                                ),
                                $strides,
                                $padding,
                                $lhs_dilation,
                                $rhs_dilation,
                                $window_reversal,
                                (CloneDnumsWithTFLNativeStandardConvKernelLayout $dnums),
                                $feature_groups,
                                $batch_groups,
                                $precision_config
                            ),
                            [(AreDnumsFullyDefined $conv),
                             (KernelHasIotaSpatials $dnums),
                             (IsKernelNotTFLNativeStandardConvLayout $dnums),
                             (IsSupportedStandardOrNonTrivialConv $conv)],
                             [],
                             (addBenefit 1)>;


//
// depthwise conv kernel = [i, spatials..., o].
//=-----

def IsKernelNotTFLNativeDepthwiseLayout : Constraint<CPred<
    "Layout("
      "$0.getKernelInputFeatureDimension(),"
      "$0.getKernelOutputFeatureDimension(),"
      "$0.getKernelSpatialDimensions())"
    "!= GetTFLNativeDepthwiseConvKernelLayout()">>;

def TFLNativeDepthwiseConvKernelLayout : NativeCodeCall<
    "GetTFLNativeDepthwiseConvKernelLayout()">;

def CloneDnumsWithTFLNativeDepthwiseConvKernelLayout : NativeCodeCall<
    "CloneDnumsWithKernelLayout("
      "$_builder,"
      "$0,"
      "GetTFLNativeDepthwiseConvKernelLayout())">;


def ReLayoutConvKernelDepthwise : Pat<(MHLO_ConvolutionOp:$conv
                                $input,
                                $kernel,
                                $strides,
                                $padding,
                                $lhs_dilation,
                                $rhs_dilation,
                                $window_reversal,
                                $dnums,
                                $feature_groups,
                                $batch_groups,
                                $precision_config
                            ),
                            (MHLO_ConvolutionOp
                                $input,
                                (MHLO_TransposeOp
                                  $kernel,
                                    (PermForReLayout
                                      (KernelLayout $dnums),
                                      (TFLNativeDepthwiseConvKernelLayout))
                                ),
                                $strides,
                                $padding,
                                $lhs_dilation,
                                $rhs_dilation,
                                $window_reversal,
                                (CloneDnumsWithTFLNativeDepthwiseConvKernelLayout $dnums),
                                $feature_groups,
                                $batch_groups,
                                $precision_config
                            ),
                            [(AreDnumsFullyDefined $conv),
                             (KernelHasIotaSpatials $dnums),
                             (IsKernelNotTFLNativeDepthwiseLayout $dnums),
                             (IsDepthwiseConv $conv)],
                             [],
                             (addBenefit 1)>;

//
// Re-layout output to [b, spatials..., f]
//===---------------------------------------------------------------------------

def IsOutputNotTFLNativeLayout : Constraint<CPred<
    "Layout("
      "$0.getOutputBatchDimension(),"
      "$0.getOutputFeatureDimension(),"
      "$0.getOutputSpatialDimensions())"
    "!= GetTFLNativeInputOrOutputLayout($0)">>;

def OutputLayout : NativeCodeCall<
    "Layout($0.getOutputBatchDimension(),"
                "$0.getOutputFeatureDimension(),"
                "$0.getOutputSpatialDimensions())">;

def TFLNativeOutputLayout : NativeCodeCall<
    "GetTFLNativeInputOrOutputLayout($0)">;

def OutputHasIotaSpatials : Constraint<CPred<
    "Layout($0.getOutputBatchDimension(),"
                "$0.getOutputFeatureDimension(),"
                "$0.getOutputSpatialDimensions()).AreSpatialsIota()">>;

// Copy dnums with the output layout set to [b, spatials..., f].
def CloneDnumsWithTFLNativeOutputLayout : NativeCodeCall<
    "CloneDnumsWithOutputLayout("
      "$_builder,"
      "$0,"
      "GetTFLNativeInputOrOutputLayout($0))">;

def ReLayoutConvOutput : Pat<(MHLO_ConvolutionOp:$conv
                                $input,
                                $kernel,
                                $strides,
                                $padding,
                                $lhs_dilation,
                                $rhs_dilation,
                                $window_reversal,
                                $dnums,
                                $feature_groups,
                                $batch_groups,
                                $precision_config
                            ),
                            (MHLO_TransposeOp
                              (MHLO_ConvolutionOp
                                  $input,
                                  $kernel,
                                  $strides,
                                  $padding,
                                  $lhs_dilation,
                                  $rhs_dilation,
                                  $window_reversal,
                                  (CloneDnumsWithTFLNativeOutputLayout $dnums),
                                  $feature_groups,
                                  $batch_groups,
                                  $precision_config,
                                  (returnType
                                    (PermuteShape
                                      (OutputLayout $dnums),
                                      (TFLNativeOutputLayout $dnums),
                                      $conv))
                              ),
                              (PermForReLayout
                                (TFLNativeOutputLayout $dnums),
                                (OutputLayout $dnums))
                            ),
                            [(AreDnumsFullyDefined $conv),
                             (KernelHasIotaSpatials $dnums),
                             (IsOutputNotTFLNativeLayout $dnums),
                             (IsSupportedConv $conv)]>;


// Pull out non-trivial padding into separate explicit pad_op.
//
// This has the benifit of allowing for a single point of control
// for turning negative padding into slices. TFL convs can fuse
// "SAME" padding back in post-legalization. Note when lhs dilations
// are non-trivial, the mhlo.convolution has the semantics of a deconvolution.
// In this case padding is interpreted differently and so we leave it in the op.
//===---------------------------------------------------------------------------

// Given DenseElements (i64), check if they are all equal to "val".
class AreI64ElementsAll<int val> : Constraint<CPred<
  "!$0 || llvm::all_of($0.getValues<int64_t>(),"
    "[](auto v) { return v == "# val #"; })">>;

class AreI64ElementsNotAll<int val> :
  Constraint<Neg<AreI64ElementsAll<val>.predicate>>;

// Gets a tuple of DenseElements (i64) given result from mhlo.convolution.
def GetExplicitPaddingArgs : NativeCodeCall<
  "GetExplicitPaddingArgs($_builder,"
    "$0.getDefiningOp<mhlo::ConvolutionOp>())">;

// Gets element type from Value.
def GetElementType : NativeCodeCall<
  "llvm::cast<RankedTensorType>($0.getType()).getElementType()">;

// Given element type, get a DenseElements with scalar shape and 0 value.
def GetZeroScalarAttrFromType : NativeCodeCall<
  "$_builder.getZeroAttr("
    "RankedTensorType::get({}, $0))">;

// Given padding attr, get new padding attr for trivial (no) padding.
def GetTrivialPaddingAttr : NativeCodeCall<
  "$_builder.getZeroAttr($0.getType())">;

// Given mhlo.convolution result, build an explicit mhlo.pad op
// which is semantically equivalant.
def ExplicitlyPadInput : NativeCodeCall<
  "CreatePadOpFromConvPadding($_builder,"
     "$0.getDefiningOp<mhlo::ConvolutionOp>())">;

def UnfuseConvWithExplicitPadding : Pat<(MHLO_ConvolutionOp:$conv
                                $input,
                                $kernel,
                                $strides,
                                $padding,
                                $lhs_dilation,
                                $rhs_dilation,
                                $window_reversal,
                                $dnums,
                                $feature_groups,
                                $batch_groups,
                                $precision_config
                            ),
                            (MHLO_ConvolutionOp
                                (ExplicitlyPadInput $conv),
                                $kernel,
                                $strides,
                                (GetTrivialPaddingAttr $padding),
                                $lhs_dilation,
                                $rhs_dilation,
                                $window_reversal,
                                $dnums,
                                $feature_groups,
                                $batch_groups,
                                $precision_config
                            ),
                            [(AreDnumsFullyDefined $conv),
                             (KernelHasIotaSpatials $dnums),
                             (IsStandardOrDepthwiseConv $conv),
                             (AreI64ElementsNotAll<0> $padding)]>;



//===------------------------------------------------------------------------===
// mhlo.pad
//===------------------------------------------------------------------------===

// Pull negative pads into slice
//===---------------------------

def TrivialStrides : NativeCodeCall<
  "DenseIntElementsAttr::get("
    "RankedTensorType::get({llvm::cast<ShapedType>($0.getType()).getRank()},"
      "$_builder.getI64Type()),"
      "llvm::SmallVector<int64_t>(llvm::cast<ShapedType>($0.getType()).getRank(),"
        "1))">;

def SliceStart : NativeCodeCall<
  "SliceStartFromNegPadLows($0.getDefiningOp<mhlo::PadOp>())">;

def SliceEnd : NativeCodeCall<
  "SliceEndFromNegPadHighs($0.getDefiningOp<mhlo::PadOp>())">;

def AnyNegativePads : Constraint<CPred<
  "AnyNegativePads($0.getDefiningOp<mhlo::PadOp>())">>;

def MakePadPositive : NativeCodeCall<
  "ReplaceNegsWithZero($0)">;

def TrivialInterior : Constraint<CPred<
  "TrivialInterior($0.getDefiningOp<mhlo::PadOp>())">>;

def PullOutNegativePads : Pat<(MHLO_PadOp:$pad
                                $input,
                                $val,
                                $low,
                                $high,
                                $interior
                              ),
                              (MHLO_PadOp
                                (MHLO_SliceOp
                                  $input,
                                  (SliceStart $pad),
                                  (SliceEnd $pad),
                                  (TrivialStrides $pad)),
                                $val,
                                (MakePadPositive $low),
                                (MakePadPositive $high),
                                $interior
                              ),
                              [(AnyNegativePads $pad),
                               (TrivialInterior $pad)]>;
