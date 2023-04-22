/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef HLO_OPS_BASE_ENUMS
#define HLO_OPS_BASE_ENUMS

//===----------------------------------------------------------------------===//
// Precision Config enum definitions.
//===----------------------------------------------------------------------===//

// These mirror the XLA PrecisionConfig proto enum.
def HLO_PRECISION_DEFAULT : StrEnumAttrCase<"DEFAULT">;
def HLO_PRECISION_HIGH    : StrEnumAttrCase<"HIGH">;
def HLO_PRECISION_HIGHEST : StrEnumAttrCase<"HIGHEST">;

def HLO_PrecisionAttr : StrEnumAttr<"Precision",
    "XLA precision for an operand. Has backend specific meaning.",
    [HLO_PRECISION_DEFAULT,  HLO_PRECISION_HIGH, HLO_PRECISION_HIGHEST]> {
  let cppNamespace = "::mlir::mhlo";
}

//===----------------------------------------------------------------------===//
// Fast Fourier Transform Type enum definitions.
//===----------------------------------------------------------------------===//

// These mirror the XLA FftType proto enum.
def HLO_FFT_TYPE_FFT : StrEnumAttrCase<"FFT">;
def HLO_FFT_TYPE_IFFT : StrEnumAttrCase<"IFFT">;
def HLO_FFT_TYPE_RFFT : StrEnumAttrCase<"RFFT">;
def HLO_FFT_TYPE_IRFFT : StrEnumAttrCase<"IRFFT">;

def HLO_FftTypeAttr : StrEnumAttr<"FftType",
    "XLA fast fourier transform type.",
    [HLO_FFT_TYPE_FFT, HLO_FFT_TYPE_IFFT,
     HLO_FFT_TYPE_RFFT, HLO_FFT_TYPE_IRFFT]> {
  let cppNamespace = "::mlir::mhlo";
}

//===----------------------------------------------------------------------===//
// Comparison op definitions.
//===----------------------------------------------------------------------===//

// These mirror the XLA ComparisonDirection enum.
def HLO_COMPARISON_DIRECTION_EQ : StrEnumAttrCase<"EQ">;
def HLO_COMPARISON_DIRECTION_NE : StrEnumAttrCase<"NE">;
def HLO_COMPARISON_DIRECTION_GE : StrEnumAttrCase<"GE">;
def HLO_COMPARISON_DIRECTION_GT : StrEnumAttrCase<"GT">;
def HLO_COMPARISON_DIRECTION_LE : StrEnumAttrCase<"LE">;
def HLO_COMPARISON_DIRECTION_LT : StrEnumAttrCase<"LT">;

def HLO_ComparisonDirectionAttr : StrEnumAttr<"ComparisonDirection",
    "Which comparison operation to perform.",
    [
      HLO_COMPARISON_DIRECTION_EQ,
      HLO_COMPARISON_DIRECTION_NE,
      HLO_COMPARISON_DIRECTION_GE,
      HLO_COMPARISON_DIRECTION_GT,
      HLO_COMPARISON_DIRECTION_LE,
      HLO_COMPARISON_DIRECTION_LT
    ]> {
  let cppNamespace = "::mlir::mhlo";
}

def HLO_DEFAULT_COMPARISON_TYPE : NativeCodeCall<"StringAttr()">;
def HLO_COMPARISON_TYPE_FLOAT : StrEnumAttrCase<"FLOAT">;
def HLO_COMPARISON_TYPE_FLOAT_TOTAL_ORDER : StrEnumAttrCase<"TOTALORDER">;
def HLO_COMPARISON_TYPE_SIGNED : StrEnumAttrCase<"SIGNED">;
def HLO_COMPARISON_TYPE_UNSIGNED : StrEnumAttrCase<"UNSIGNED">;

def HLO_ComparisonTypeAttr : StrEnumAttr<"ComparisonType",
    "Which comparison type to use.",
    [
      HLO_COMPARISON_TYPE_FLOAT,
      HLO_COMPARISON_TYPE_FLOAT_TOTAL_ORDER,
      HLO_COMPARISON_TYPE_SIGNED,
      HLO_COMPARISON_TYPE_UNSIGNED
    ]> {
  let cppNamespace = "::mlir::mhlo";
}

// These mirror the XLA Dequantize mode string enum.
def HLO_MIN_COMBINED : StrEnumAttrCase<"MIN_COMBINED">;

def HLO_DequantizeModeAttr : StrEnumAttr<"DequantizeMode",
  "Dequantization mode. Only MIN_COMBINED is supported.",
  [HLO_MIN_COMBINED]> {
  let cppNamespace = "::mlir::mhlo";
}

// These mirror the XLA Transpose enum in Triangular Solve options.
def HLO_TRANSPOSE_INVALID : StrEnumAttrCase<"TRANSPOSE_INVALID">;
def HLO_NO_TRANSPOSE : StrEnumAttrCase<"NO_TRANSPOSE">;
def HLO_TRANSPOSE : StrEnumAttrCase<"TRANSPOSE">;
def HLO_ADJOINT : StrEnumAttrCase<"ADJOINT">;

def HLO_TransposeAttr : StrEnumAttr<"Transpose",
    "Transpose options",
    [
      HLO_TRANSPOSE_INVALID,
      HLO_NO_TRANSPOSE,
      HLO_TRANSPOSE,
      HLO_ADJOINT
    ]> {
  let cppNamespace = "::mlir::mhlo";
}

#endif // HLO_OPS_BASE_ENUMS
