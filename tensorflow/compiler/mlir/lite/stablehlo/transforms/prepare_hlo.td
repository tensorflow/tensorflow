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
// 2D : [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// 3D : [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]
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
  "$2.getType().cast<RankedTensorType>()"
    ".clone($0.PermuteShape("
      "$1,"
      "$2.getType().cast<RankedTensorType>().getShape()))">;

def IsStandardConv : Constraint<CPred<
  "IsStandardConv($0.getDefiningOp<mhlo::ConvolutionOp>())">>;

// Re-layout input (lhs) to [b, spatials..., f]
//===------------------------------------------

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
                             (IsStandardConv $conv)]>;


// Re-layout kernel to [o, spatials..., i] (2d) or [spatials..., i, o] (3d)
//===----------------------------------------------------------------------

def IsKernelNotTFLNativeLayout : Constraint<CPred<
    "Layout("
      "$0.getKernelInputFeatureDimension(),"
      "$0.getKernelOutputFeatureDimension(),"
      "$0.getKernelSpatialDimensions())"
    "!= GetTFLNativeKernelLayout($0)">>;

def KernelLayout : NativeCodeCall<
    "Layout($0.getKernelInputFeatureDimension(),"
                "$0.getKernelOutputFeatureDimension(),"
                "$0.getKernelSpatialDimensions())">;

def TFLNativeKernelLayout : NativeCodeCall<
    "GetTFLNativeKernelLayout($0)">;

def KernelHasIotaSpatials : Constraint<CPred<
    "Layout($0.getKernelInputFeatureDimension(),"
                "$0.getKernelOutputFeatureDimension(),"
                "$0.getKernelSpatialDimensions()).AreSpatialsIota()">>;

// Copy dnums with the kernel layout set to [o, spatials..., i].
def CloneDnumsWithTFLNativeKernelLayout : NativeCodeCall<
    "CloneDnumsWithKernelLayout("
      "$_builder,"
      "$0,"
      "GetTFLNativeKernelLayout($0))">;

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
                                      (TFLNativeKernelLayout $dnums))
                                ),
                                $strides,
                                $padding,
                                $lhs_dilation,
                                $rhs_dilation,
                                $window_reversal,
                                (CloneDnumsWithTFLNativeKernelLayout $dnums),
                                $feature_groups,
                                $batch_groups,
                                $precision_config
                            ),
                            [(AreDnumsFullyDefined $conv),
                             (KernelHasIotaSpatials $dnums),
                             (IsKernelNotTFLNativeLayout $dnums),
                             (IsStandardConv $conv)]>;

// Re-layout output to [b, spatials..., f]
//===-------------------------------------

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
                             (IsStandardConv $conv)]>;


//===------------------------------------------------------------------------===
// mhlo.pad
//===------------------------------------------------------------------------===

// Pull negative pads into slice
//===---------------------------

def TrivialStrides : NativeCodeCall<
  "DenseIntElementsAttr::get("
    "RankedTensorType::get({$0.getType().cast<ShapedType>().getRank()},"
      "$_builder.getI64Type()),"
      "llvm::SmallVector<int64_t>($0.getType().cast<ShapedType>().getRank(),"
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
