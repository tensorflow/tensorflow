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

include "mlir/IR/EnumAttr.td"
include "mlir/IR/PatternBase.td"

//===----------------------------------------------------------------------===//
// Precision Config enum definitions.
//===----------------------------------------------------------------------===//

// These mirror the XLA PrecisionConfig proto enum.
def HLO_PRECISION_DEFAULT : I32EnumAttrCase<"DEFAULT", 0>;
def HLO_PRECISION_HIGH    : I32EnumAttrCase<"HIGH", 1>;
def HLO_PRECISION_HIGHEST : I32EnumAttrCase<"HIGHEST", 2>;

def HLO_Precision : I32EnumAttr<"Precision",
    "XLA precision for an operand. Has backend specific meaning.",
    [HLO_PRECISION_DEFAULT,  HLO_PRECISION_HIGH, HLO_PRECISION_HIGHEST]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mhlo";
}

def HLO_PrecisionAttr : EnumAttr<HLO_Dialect, HLO_Precision, "precision">;

//===----------------------------------------------------------------------===//
// Domain Metadata Kind  enum definitions.
//===----------------------------------------------------------------------===//

// These mirror the XLA FftType proto enum.
def HLO_DOMAIN_KIND_SHARDING : I32EnumAttrCase<"sharding", 0>;

def HLO_DomainKind : I32EnumAttr<"DomainKind",
    "Kind of domain metatdata attached to an HLO domain.",
    [HLO_DOMAIN_KIND_SHARDING]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mhlo";
}

def HLO_DomainKindAttr : EnumAttr<HLO_Dialect, HLO_DomainKind, "kind">;

//===----------------------------------------------------------------------===//
// Fast Fourier Transform Type enum definitions.
//===----------------------------------------------------------------------===//

// These mirror the XLA FftType proto enum.
def HLO_FFT_TYPE_FFT : I32EnumAttrCase<"FFT", 0>;
def HLO_FFT_TYPE_IFFT : I32EnumAttrCase<"IFFT", 1>;
def HLO_FFT_TYPE_RFFT : I32EnumAttrCase<"RFFT", 2>;
def HLO_FFT_TYPE_IRFFT : I32EnumAttrCase<"IRFFT", 3>;

def HLO_FftType : I32EnumAttr<"FftType",
    "XLA fast fourier transform type.",
    [HLO_FFT_TYPE_FFT, HLO_FFT_TYPE_IFFT,
     HLO_FFT_TYPE_RFFT, HLO_FFT_TYPE_IRFFT]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mhlo";
}

def HLO_FftTypeAttr : EnumAttr<HLO_Dialect, HLO_FftType, "fft_type">;

//===----------------------------------------------------------------------===//
// Custom call enum definitions.
//===----------------------------------------------------------------------===//

// TODO(b/189822916): Remove this enum when all clients are migrated to the
// status-returning API.
def HLO_CUSTOM_CALL_API_VERISON_UNSPECIFIED :
    I32EnumAttrCase<"API_VERSION_UNSPECIFIED", 0>;
def HLO_CUSTOM_CALL_API_VERSION_ORIGINAL :
    I32EnumAttrCase<"API_VERSION_ORIGINAL", 1>;
def HLO_CUSTOM_CALL_API_VERSION_STATUS_RETURNING :
    I32EnumAttrCase<"API_VERSION_STATUS_RETURNING", 2>;
def HLO_CUSTOM_CALL_API_VERSION_STATUS_RETURNING_UNIFIED :
    I32EnumAttrCase<"API_VERSION_STATUS_RETURNING_UNIFIED", 3>;
def HLO_CustomCallApiVersionAttr :
    I32EnumAttr<"CustomCallApiVersion", "Custom call API version", [
        HLO_CUSTOM_CALL_API_VERISON_UNSPECIFIED,
        HLO_CUSTOM_CALL_API_VERSION_ORIGINAL,
        HLO_CUSTOM_CALL_API_VERSION_STATUS_RETURNING,
        HLO_CUSTOM_CALL_API_VERSION_STATUS_RETURNING_UNIFIED
    ]> {
  let cppNamespace = "::mlir::mhlo";
}

//===----------------------------------------------------------------------===//
// Comparison op definitions.
//===----------------------------------------------------------------------===//

// These mirror the XLA ComparisonDirection enum.
def HLO_COMPARISON_DIRECTION_EQ : I32EnumAttrCase<"EQ", 0>;
def HLO_COMPARISON_DIRECTION_NE : I32EnumAttrCase<"NE", 1>;
def HLO_COMPARISON_DIRECTION_GE : I32EnumAttrCase<"GE", 2>;
def HLO_COMPARISON_DIRECTION_GT : I32EnumAttrCase<"GT", 3>;
def HLO_COMPARISON_DIRECTION_LE : I32EnumAttrCase<"LE", 4>;
def HLO_COMPARISON_DIRECTION_LT : I32EnumAttrCase<"LT", 5>;

def HLO_ComparisonDirection : I32EnumAttr<"ComparisonDirection",
    "Which comparison operation to perform.",
    [
      HLO_COMPARISON_DIRECTION_EQ,
      HLO_COMPARISON_DIRECTION_NE,
      HLO_COMPARISON_DIRECTION_GE,
      HLO_COMPARISON_DIRECTION_GT,
      HLO_COMPARISON_DIRECTION_LE,
      HLO_COMPARISON_DIRECTION_LT
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mhlo";
}

def HLO_ComparisonDirectionAttr : EnumAttr<HLO_Dialect, HLO_ComparisonDirection, "comparison_direction">;

def HLO_DEFAULT_COMPARISON_TYPE : NativeCodeCall<"::mlir::mhlo::ComparisonTypeAttr()">;
def HLO_COMPARISON_TYPE_NOTYPE : I32EnumAttrCase<"NOTYPE", 0>;
def HLO_COMPARISON_TYPE_FLOAT : I32EnumAttrCase<"FLOAT", 1>;
def HLO_COMPARISON_TYPE_FLOAT_TOTAL_ORDER : I32EnumAttrCase<"TOTALORDER", 2>;
def HLO_COMPARISON_TYPE_SIGNED : I32EnumAttrCase<"SIGNED", 3>;
def HLO_COMPARISON_TYPE_UNSIGNED : I32EnumAttrCase<"UNSIGNED", 4>;

def HLO_ComparisonType : I32EnumAttr<"ComparisonType",
    "Which comparison type to use.",
    [
      HLO_COMPARISON_TYPE_NOTYPE,
      HLO_COMPARISON_TYPE_FLOAT,
      HLO_COMPARISON_TYPE_FLOAT_TOTAL_ORDER,
      HLO_COMPARISON_TYPE_SIGNED,
      HLO_COMPARISON_TYPE_UNSIGNED
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mhlo";
}

def HLO_ComparisonTypeAttr : EnumAttr<HLO_Dialect, HLO_ComparisonType, "comparison_type">;

// These mirror the XLA Dequantize mode string enum.
def HLO_MIN_COMBINED : I32EnumAttrCase<"MIN_COMBINED", 0>;

def HLO_DequantizeMode : I32EnumAttr<"DequantizeMode",
  "Dequantization mode. Only MIN_COMBINED is supported.",
  [HLO_MIN_COMBINED]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mhlo";
}

def HLO_DequantizeModeAttr : EnumAttr<HLO_Dialect, HLO_DequantizeMode, "dequantize_mode">;

// These mirror the XLA Transpose enum in Triangular Solve options.
def HLO_TRANSPOSE_INVALID : I32EnumAttrCase<"TRANSPOSE_INVALID", 0>;
def HLO_NO_TRANSPOSE : I32EnumAttrCase<"NO_TRANSPOSE", 1>;
def HLO_TRANSPOSE : I32EnumAttrCase<"TRANSPOSE", 2>;
def HLO_ADJOINT : I32EnumAttrCase<"ADJOINT", 3>;

def HLO_Transpose : I32EnumAttr<"Transpose",
    "Transpose options",
    [
      HLO_TRANSPOSE_INVALID,
      HLO_NO_TRANSPOSE,
      HLO_TRANSPOSE,
      HLO_ADJOINT
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mhlo";
}

def HLO_TransposeAttr : EnumAttr<HLO_Dialect, HLO_Transpose, "transpose">;

def HLO_LOOP_FUSION : I32EnumAttrCase<"kLoop", 0>;
def HLO_INPUT_FUSION : I32EnumAttrCase<"kInput", 1>;
def HLO_OUTPUT_FUSION : I32EnumAttrCase<"kOutput", 2>;
def HLO_CUSTOM_FUSION : I32EnumAttrCase<"kCustom", 3>;
def HLO_FusionKind : I32EnumAttr<"FusionKind", "fusion kind", [
    HLO_LOOP_FUSION, HLO_INPUT_FUSION, HLO_OUTPUT_FUSION, HLO_CUSTOM_FUSION
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mhlo";
}

def HLO_RNG_DISTRIBUTION_UNIFORM : I32EnumAttrCase<"UNIFORM", 1>;
def HLO_RNG_DISTRIBUTION_NORMAL : I32EnumAttrCase<"NORMAL", 2>;

def HLO_RNG_DISTRIBUTION : I32EnumAttr<"RngDistribution",
    "XLA PRNG distribution to be used.",
    [
      HLO_RNG_DISTRIBUTION_UNIFORM,
      HLO_RNG_DISTRIBUTION_NORMAL
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mhlo";
}

def HLO_RngDistributionAttr : EnumAttr<HLO_Dialect, HLO_RNG_DISTRIBUTION, "rng_distribution"> {
  let assemblyFormat = "`<` $value `>`";
}

def HLO_FusionKindAttr : EnumAttr<HLO_Dialect, HLO_FusionKind, "fusion_kind">;

def HLO_RNG_ALGORITHM_DEFAULT : I32EnumAttrCase<"DEFAULT", 0>;
def HLO_RNG_ALGORITHM_THREE_FRY : I32EnumAttrCase<"THREE_FRY", 1>;
def HLO_RNG_ALGORITHM_PHILOX : I32EnumAttrCase<"PHILOX", 2>;

def HLO_RNG_ALGORITHM : I32EnumAttr<"RngAlgorithm",
    "XLA PRNG algorithm to be used.",
    [
      HLO_RNG_ALGORITHM_DEFAULT,
      HLO_RNG_ALGORITHM_THREE_FRY,
      HLO_RNG_ALGORITHM_PHILOX
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mhlo";
}

def HLO_RngAlgorithmAttr : EnumAttr<HLO_Dialect, HLO_RNG_ALGORITHM, "rng_algorithm"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // HLO_OPS_BASE_ENUMS
