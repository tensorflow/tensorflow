/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

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

#ifndef STABLEHLO_DIALECT_STABLEHLO_ENUMS
#define STABLEHLO_DIALECT_STABLEHLO_ENUMS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/PatternBase.td"

//===----------------------------------------------------------------------===//
// Precision Config enum definitions.
//===----------------------------------------------------------------------===//

// These mirror the XLA PrecisionConfig proto enum.
def STABLEHLO_PRECISION_DEFAULT : I32EnumAttrCase<"DEFAULT", 0>;
def STABLEHLO_PRECISION_HIGH    : I32EnumAttrCase<"HIGH", 1>;
def STABLEHLO_PRECISION_HIGHEST : I32EnumAttrCase<"HIGHEST", 2>;

def StableHLO_Precision : I32EnumAttr<"Precision",
    "XLA precision for an operand. Has backend specific meaning.",
    [
      STABLEHLO_PRECISION_DEFAULT,
      STABLEHLO_PRECISION_HIGH,
      STABLEHLO_PRECISION_HIGHEST
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::stablehlo";
}

def StableHLO_PrecisionAttr : EnumAttr<StableHLO_Dialect, StableHLO_Precision, "precision">;

// TODO(b/129153247) See if it's possible to also validate the size.
def StableHLO_PrecisionConfigAttr:
    OptionalAttr<
          TypedArrayAttrBase<StableHLO_PrecisionAttr, "Precision Config attribute">>;

//===----------------------------------------------------------------------===//
// Fast Fourier Transform Type enum definitions.
//===----------------------------------------------------------------------===//

// These mirror the XLA FftType proto enum.
def STABLEHLO_FFT_TYPE_FFT : I32EnumAttrCase<"FFT", 0>;
def STABLEHLO_FFT_TYPE_IFFT : I32EnumAttrCase<"IFFT", 1>;
def STABLEHLO_FFT_TYPE_RFFT : I32EnumAttrCase<"RFFT", 2>;
def STABLEHLO_FFT_TYPE_IRFFT : I32EnumAttrCase<"IRFFT", 3>;

def StableHLO_FftType : I32EnumAttr<"FftType",
    "XLA fast fourier transform type.",
    [
      STABLEHLO_FFT_TYPE_FFT,
      STABLEHLO_FFT_TYPE_IFFT,
      STABLEHLO_FFT_TYPE_RFFT,
      STABLEHLO_FFT_TYPE_IRFFT
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::stablehlo";
}

def StableHLO_FftTypeAttr : EnumAttr<StableHLO_Dialect, StableHLO_FftType, "fft_type">;

//===----------------------------------------------------------------------===//
// Custom call enum definitions.
//===----------------------------------------------------------------------===//

// TODO(b/189822916): Remove this enum when all clients are migrated to the
// status-returning API.
def STABLEHLO_CUSTOM_CALL_API_VERISON_UNSPECIFIED :
    I32EnumAttrCase<"API_VERSION_UNSPECIFIED", 0>;
def STABLEHLO_CUSTOM_CALL_API_VERSION_ORIGINAL :
    I32EnumAttrCase<"API_VERSION_ORIGINAL", 1>;
def STABLEHLO_CUSTOM_CALL_API_VERSION_STATUS_RETURNING :
    I32EnumAttrCase<"API_VERSION_STATUS_RETURNING", 2>;
def STABLEHLO_CUSTOM_CALL_API_VERSION_STATUS_RETURNING_UNIFIED :
    I32EnumAttrCase<"API_VERSION_STATUS_RETURNING_UNIFIED", 3>;
def StableHLO_CustomCallApiVersionAttr :
    I32EnumAttr<"CustomCallApiVersion", "Custom call API version", [
        STABLEHLO_CUSTOM_CALL_API_VERISON_UNSPECIFIED,
        STABLEHLO_CUSTOM_CALL_API_VERSION_ORIGINAL,
        STABLEHLO_CUSTOM_CALL_API_VERSION_STATUS_RETURNING,
        STABLEHLO_CUSTOM_CALL_API_VERSION_STATUS_RETURNING_UNIFIED
    ]> {
  let cppNamespace = "::mlir::stablehlo";
}

//===----------------------------------------------------------------------===//
// Comparison op definitions.
//===----------------------------------------------------------------------===//

// These mirror the XLA ComparisonDirection enum.
def STABLEHLO_COMPARISON_DIRECTION_EQ : I32EnumAttrCase<"EQ", 0>;
def STABLEHLO_COMPARISON_DIRECTION_NE : I32EnumAttrCase<"NE", 1>;
def STABLEHLO_COMPARISON_DIRECTION_GE : I32EnumAttrCase<"GE", 2>;
def STABLEHLO_COMPARISON_DIRECTION_GT : I32EnumAttrCase<"GT", 3>;
def STABLEHLO_COMPARISON_DIRECTION_LE : I32EnumAttrCase<"LE", 4>;
def STABLEHLO_COMPARISON_DIRECTION_LT : I32EnumAttrCase<"LT", 5>;

def StableHLO_ComparisonDirection : I32EnumAttr<"ComparisonDirection",
    "Which comparison operation to perform.",
    [
      STABLEHLO_COMPARISON_DIRECTION_EQ,
      STABLEHLO_COMPARISON_DIRECTION_NE,
      STABLEHLO_COMPARISON_DIRECTION_GE,
      STABLEHLO_COMPARISON_DIRECTION_GT,
      STABLEHLO_COMPARISON_DIRECTION_LE,
      STABLEHLO_COMPARISON_DIRECTION_LT
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::stablehlo";
}

def StableHLO_ComparisonDirectionAttr : EnumAttr<StableHLO_Dialect, StableHLO_ComparisonDirection, "comparison_direction">;

def STABLEHLO_DEFAULT_COMPARISON_TYPE : NativeCodeCall<"::mlir::stablehlo::ComparisonTypeAttr()">;
def STABLEHLO_COMPARISON_TYPE_NOTYPE : I32EnumAttrCase<"NOTYPE", 0>;
def STABLEHLO_COMPARISON_TYPE_FLOAT : I32EnumAttrCase<"FLOAT", 1>;
def STABLEHLO_COMPARISON_TYPE_FLOAT_TOTAL_ORDER : I32EnumAttrCase<"TOTALORDER", 2>;
def STABLEHLO_COMPARISON_TYPE_SIGNED : I32EnumAttrCase<"SIGNED", 3>;
def STABLEHLO_COMPARISON_TYPE_UNSIGNED : I32EnumAttrCase<"UNSIGNED", 4>;

def StableHLO_ComparisonType : I32EnumAttr<"ComparisonType",
    "Which comparison type to use.",
    [
      STABLEHLO_COMPARISON_TYPE_NOTYPE,
      STABLEHLO_COMPARISON_TYPE_FLOAT,
      STABLEHLO_COMPARISON_TYPE_FLOAT_TOTAL_ORDER,
      STABLEHLO_COMPARISON_TYPE_SIGNED,
      STABLEHLO_COMPARISON_TYPE_UNSIGNED
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::stablehlo";
}

def StableHLO_ComparisonTypeAttr : EnumAttr<StableHLO_Dialect, StableHLO_ComparisonType, "comparison_type">;

// These mirror the XLA Transpose enum in Triangular Solve options.
def STABLEHLO_TRANSPOSE_INVALID : I32EnumAttrCase<"TRANSPOSE_INVALID", 0>;
def STABLEHLO_NO_TRANSPOSE : I32EnumAttrCase<"NO_TRANSPOSE", 1>;
def STABLEHLO_TRANSPOSE : I32EnumAttrCase<"TRANSPOSE", 2>;
def STABLEHLO_ADJOINT : I32EnumAttrCase<"ADJOINT", 3>;

def StableHLO_Transpose : I32EnumAttr<"Transpose",
    "Transpose options",
    [
      STABLEHLO_TRANSPOSE_INVALID,
      STABLEHLO_NO_TRANSPOSE,
      STABLEHLO_TRANSPOSE,
      STABLEHLO_ADJOINT
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::stablehlo";
}

def StableHLO_TransposeAttr : EnumAttr<StableHLO_Dialect, StableHLO_Transpose, "transpose">;

def STABLEHLO_RNG_DISTRIBUTION_UNIFORM : I32EnumAttrCase<"UNIFORM", 1>;
def STABLEHLO_RNG_DISTRIBUTION_NORMAL : I32EnumAttrCase<"NORMAL", 2>;

def StableHLO_RngDistribution : I32EnumAttr<"RngDistribution",
    "XLA PRNG distribution to be used.",
    [
      STABLEHLO_RNG_DISTRIBUTION_UNIFORM,
      STABLEHLO_RNG_DISTRIBUTION_NORMAL
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::stablehlo";
}

def StableHLO_RngDistributionAttr : EnumAttr<StableHLO_Dialect, StableHLO_RngDistribution, "rng_distribution"> {
  let assemblyFormat = "`<` $value `>`";
}

def STABLEHLO_RNG_ALGORITHM_DEFAULT : I32EnumAttrCase<"DEFAULT", 0>;
def STABLEHLO_RNG_ALGORITHM_THREE_FRY : I32EnumAttrCase<"THREE_FRY", 1>;
def STABLEHLO_RNG_ALGORITHM_PHILOX : I32EnumAttrCase<"PHILOX", 2>;

def StableHLO_RngAlgorithm : I32EnumAttr<"RngAlgorithm",
    "XLA PRNG algorithm to be used.",
    [
      STABLEHLO_RNG_ALGORITHM_DEFAULT,
      STABLEHLO_RNG_ALGORITHM_THREE_FRY,
      STABLEHLO_RNG_ALGORITHM_PHILOX
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::stablehlo";
}

def StableHLO_RngAlgorithmAttr : EnumAttr<StableHLO_Dialect, StableHLO_RngAlgorithm, "rng_algorithm"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // STABLEHLO_DIALECT_STABLEHLO_ENUMS
