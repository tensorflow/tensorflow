/* Copyright 2020 The OpenXLA Authors.

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

#ifndef MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS_ENUMS
#define MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS_ENUMS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/PatternBase.td"

//===----------------------------------------------------------------------===//
// Precision Config enum definitions.
//===----------------------------------------------------------------------===//

// These mirror the XLA PrecisionConfig proto enum.
def MHLO_PRECISION_DEFAULT : I32EnumAttrCase<"DEFAULT", 0>;
def MHLO_PRECISION_HIGH    : I32EnumAttrCase<"HIGH", 1>;
def MHLO_PRECISION_HIGHEST : I32EnumAttrCase<"HIGHEST", 2>;

def MHLO_Precision : I32EnumAttr<"Precision",
    "XLA precision for an operand. Has backend specific meaning.",
    [MHLO_PRECISION_DEFAULT, MHLO_PRECISION_HIGH, MHLO_PRECISION_HIGHEST]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mhlo";
}

def MHLO_PrecisionAttr : EnumAttr<MHLO_Dialect, MHLO_Precision, "precision">;

// TODO(b/129153247) See if it's possible to also validate the size.
def MHLO_PrecisionConfigAttr:
    TypedArrayAttrBase<MHLO_PrecisionAttr, "Precision Config attribute">;

//===----------------------------------------------------------------------===//
// Custom call schedule hints
//===----------------------------------------------------------------------===//

// These mirror the XLA CustomCallSchedule proto enum.
def MHLO_CUSTOM_CALL_SCHEDULE_NONE : I32EnumAttrCase<"NONE", 0>;
def MHLO_CUSTOM_CALL_SCHEDULE_LATEST : I32EnumAttrCase<"LATEST", 1>;
def MHLO_CUSTOM_CALL_SCHEDULE_EARLIEST : I32EnumAttrCase<"EARLIEST", 2>;

// mhlo.custom_call_schedule gives us a scheduling hint for placing calls
// LATEST indicates that the operation should be scheduled just before the first
//   user in the use/def chain.
// EARLIEST indicates that the operation should be scheduled just after the last
//   operation that defines an argument of this operation in the use/def chain.
// NONE indicates no hint for the compiler.
def MHLO_CustomCallSchedule : I32EnumAttr<"CustomCallSchedule",
    "Specifies the desired schedule for the custom-call.",
    [MHLO_CUSTOM_CALL_SCHEDULE_NONE, MHLO_CUSTOM_CALL_SCHEDULE_LATEST, MHLO_CUSTOM_CALL_SCHEDULE_EARLIEST]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mhlo";
}

def MHLO_CustomCallScheduleAttr : EnumAttr<MHLO_Dialect, MHLO_CustomCallSchedule, "custom_call_schedule">;

//===----------------------------------------------------------------------===//
// Domain Metadata Kind  enum definitions.
//===----------------------------------------------------------------------===//

// These mirror the XLA FftType proto enum.
def MHLO_DOMAIN_KIND_SHARDING : I32EnumAttrCase<"sharding", 0>;

def MHLO_DomainKind : I32EnumAttr<"DomainKind",
    "Kind of domain metatdata attached to an HLO domain.",
    [MHLO_DOMAIN_KIND_SHARDING]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mhlo";
}

def MHLO_DomainKindAttr : EnumAttr<MHLO_Dialect, MHLO_DomainKind, "kind">;

//===----------------------------------------------------------------------===//
// Fast Fourier Transform Type enum definitions.
//===----------------------------------------------------------------------===//

// These mirror the XLA FftType proto enum.
def MHLO_FFT_TYPE_FFT : I32EnumAttrCase<"FFT", 0>;
def MHLO_FFT_TYPE_IFFT : I32EnumAttrCase<"IFFT", 1>;
def MHLO_FFT_TYPE_RFFT : I32EnumAttrCase<"RFFT", 2>;
def MHLO_FFT_TYPE_IRFFT : I32EnumAttrCase<"IRFFT", 3>;

def MHLO_FftType : I32EnumAttr<"FftType",
    "XLA fast fourier transform type.",
    [MHLO_FFT_TYPE_FFT, MHLO_FFT_TYPE_IFFT,
     MHLO_FFT_TYPE_RFFT, MHLO_FFT_TYPE_IRFFT]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mhlo";
}

def MHLO_FftTypeAttr : EnumAttr<MHLO_Dialect, MHLO_FftType, "fft_type">;

//===----------------------------------------------------------------------===//
// Custom call enum definitions.
//===----------------------------------------------------------------------===//

// TODO(b/189822916): Remove this enum when all clients are migrated to the
// status-returning API.
def MHLO_CUSTOM_CALL_API_VERISON_UNSPECIFIED :
    I32EnumAttrCase<"API_VERSION_UNSPECIFIED", 0>;
def MHLO_CUSTOM_CALL_API_VERSION_ORIGINAL :
    I32EnumAttrCase<"API_VERSION_ORIGINAL", 1>;
def MHLO_CUSTOM_CALL_API_VERSION_STATUS_RETURNING :
    I32EnumAttrCase<"API_VERSION_STATUS_RETURNING", 2>;
def MHLO_CUSTOM_CALL_API_VERSION_STATUS_RETURNING_UNIFIED :
    I32EnumAttrCase<"API_VERSION_STATUS_RETURNING_UNIFIED", 3>;
def MHLO_CUSTOM_CALL_API_VERSION_TYPED_FFI :
    I32EnumAttrCase<"API_VERSION_TYPED_FFI", 4>;
def MHLO_CustomCallApiVersionAttr :
    I32EnumAttr<"CustomCallApiVersion", "Custom call API version", [
        MHLO_CUSTOM_CALL_API_VERISON_UNSPECIFIED,
        MHLO_CUSTOM_CALL_API_VERSION_ORIGINAL,
        MHLO_CUSTOM_CALL_API_VERSION_STATUS_RETURNING,
        MHLO_CUSTOM_CALL_API_VERSION_STATUS_RETURNING_UNIFIED,
        MHLO_CUSTOM_CALL_API_VERSION_TYPED_FFI
    ]> {
  let cppNamespace = "::mlir::mhlo";
}

//===----------------------------------------------------------------------===//
// Comparison op definitions.
//===----------------------------------------------------------------------===//

// These mirror the XLA ComparisonDirection enum.
def MHLO_COMPARISON_DIRECTION_EQ : I32EnumAttrCase<"EQ", 0>;
def MHLO_COMPARISON_DIRECTION_NE : I32EnumAttrCase<"NE", 1>;
def MHLO_COMPARISON_DIRECTION_GE : I32EnumAttrCase<"GE", 2>;
def MHLO_COMPARISON_DIRECTION_GT : I32EnumAttrCase<"GT", 3>;
def MHLO_COMPARISON_DIRECTION_LE : I32EnumAttrCase<"LE", 4>;
def MHLO_COMPARISON_DIRECTION_LT : I32EnumAttrCase<"LT", 5>;

def MHLO_ComparisonDirection : I32EnumAttr<"ComparisonDirection",
    "Which comparison operation to perform.",
    [
      MHLO_COMPARISON_DIRECTION_EQ,
      MHLO_COMPARISON_DIRECTION_NE,
      MHLO_COMPARISON_DIRECTION_GE,
      MHLO_COMPARISON_DIRECTION_GT,
      MHLO_COMPARISON_DIRECTION_LE,
      MHLO_COMPARISON_DIRECTION_LT
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mhlo";
}

def MHLO_ComparisonDirectionAttr : EnumAttr<MHLO_Dialect, MHLO_ComparisonDirection, "comparison_direction">;

def MHLO_DEFAULT_COMPARISON_TYPE : NativeCodeCall<"::mlir::mhlo::ComparisonTypeAttr()">;
def MHLO_COMPARISON_TYPE_NOTYPE : I32EnumAttrCase<"NOTYPE", 0>;
def MHLO_COMPARISON_TYPE_FLOAT : I32EnumAttrCase<"FLOAT", 1>;
def MHLO_COMPARISON_TYPE_FLOAT_TOTAL_ORDER : I32EnumAttrCase<"TOTALORDER", 2>;
def MHLO_COMPARISON_TYPE_SIGNED : I32EnumAttrCase<"SIGNED", 3>;
def MHLO_COMPARISON_TYPE_UNSIGNED : I32EnumAttrCase<"UNSIGNED", 4>;

def MHLO_ComparisonType : I32EnumAttr<"ComparisonType",
    "Which comparison type to use.",
    [
      MHLO_COMPARISON_TYPE_NOTYPE,
      MHLO_COMPARISON_TYPE_FLOAT,
      MHLO_COMPARISON_TYPE_FLOAT_TOTAL_ORDER,
      MHLO_COMPARISON_TYPE_SIGNED,
      MHLO_COMPARISON_TYPE_UNSIGNED
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mhlo";
}

def MHLO_ComparisonTypeAttr : EnumAttr<MHLO_Dialect, MHLO_ComparisonType, "comparison_type">;

// These mirror the XLA Dequantize mode string enum.
def MHLO_MIN_COMBINED : I32EnumAttrCase<"MIN_COMBINED", 0>;

def MHLO_DequantizeMode : I32EnumAttr<"DequantizeMode",
  "Dequantization mode. Only MIN_COMBINED is supported.",
  [MHLO_MIN_COMBINED]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mhlo";
}

def MHLO_DequantizeModeAttr : EnumAttr<MHLO_Dialect, MHLO_DequantizeMode, "dequantize_mode">;

// These mirror the XLA Transpose enum in Triangular Solve options.
def MHLO_TRANSPOSE_INVALID : I32EnumAttrCase<"TRANSPOSE_INVALID", 0>;
def MHLO_NO_TRANSPOSE : I32EnumAttrCase<"NO_TRANSPOSE", 1>;
def MHLO_TRANSPOSE : I32EnumAttrCase<"TRANSPOSE", 2>;
def MHLO_ADJOINT : I32EnumAttrCase<"ADJOINT", 3>;

def MHLO_Transpose : I32EnumAttr<"Transpose",
    "Transpose options",
    [
      MHLO_TRANSPOSE_INVALID,
      MHLO_NO_TRANSPOSE,
      MHLO_TRANSPOSE,
      MHLO_ADJOINT
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mhlo";
}

def MHLO_TransposeAttr : EnumAttr<MHLO_Dialect, MHLO_Transpose, "transpose">;

def MHLO_LOOP_FUSION : I32EnumAttrCase<"kLoop", 0>;
def MHLO_INPUT_FUSION : I32EnumAttrCase<"kInput", 1>;
def MHLO_OUTPUT_FUSION : I32EnumAttrCase<"kOutput", 2>;
def MHLO_CUSTOM_FUSION : I32EnumAttrCase<"kCustom", 3>;
def MHLO_FusionKind : I32EnumAttr<"FusionKind", "fusion kind", [
    MHLO_LOOP_FUSION, MHLO_INPUT_FUSION, MHLO_OUTPUT_FUSION, MHLO_CUSTOM_FUSION
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mhlo";
}

def MHLO_RNG_DISTRIBUTION_UNIFORM : I32EnumAttrCase<"UNIFORM", 1>;
def MHLO_RNG_DISTRIBUTION_NORMAL : I32EnumAttrCase<"NORMAL", 2>;

def MHLO_RngDistribution : I32EnumAttr<"RngDistribution",
    "XLA PRNG distribution to be used.",
    [
      MHLO_RNG_DISTRIBUTION_UNIFORM,
      MHLO_RNG_DISTRIBUTION_NORMAL
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mhlo";
}

def MHLO_RngDistributionAttr : EnumAttr<MHLO_Dialect, MHLO_RngDistribution, "rng_distribution"> {
  let assemblyFormat = "`<` $value `>`";
}

def MHLO_FusionKindAttr : EnumAttr<MHLO_Dialect, MHLO_FusionKind, "fusion_kind">;

def MHLO_RNG_ALGORITHM_DEFAULT : I32EnumAttrCase<"DEFAULT", 0>;
def MHLO_RNG_ALGORITHM_THREE_FRY : I32EnumAttrCase<"THREE_FRY", 1>;
def MHLO_RNG_ALGORITHM_PHILOX : I32EnumAttrCase<"PHILOX", 2>;

def MHLO_RngAlgorithm : I32EnumAttr<"RngAlgorithm",
    "XLA PRNG algorithm to be used.",
    [
      MHLO_RNG_ALGORITHM_DEFAULT,
      MHLO_RNG_ALGORITHM_THREE_FRY,
      MHLO_RNG_ALGORITHM_PHILOX
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mhlo";
}

def MHLO_RngAlgorithmAttr : EnumAttr<MHLO_Dialect, MHLO_RngAlgorithm, "rng_algorithm"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Result Accuracy enum definitions.
//===----------------------------------------------------------------------===//

def MHLO_RESULT_ACCURACY_DEFAULT : I32EnumAttrCase<"DEFAULT", 0>;
def MHLO_RESULT_ACCURACY_HIGHEST : I32EnumAttrCase<"HIGHEST", 1>;
def MHLO_RESULT_ACCURACY_TOLERANCE: I32EnumAttrCase<"TOLERANCE", 2>;

def MHLO_ResultAccuracyMode : I32EnumAttr<"ResultAccuracyMode",
    "XLA result accuracy mode.",
    [
      MHLO_RESULT_ACCURACY_DEFAULT,
      MHLO_RESULT_ACCURACY_HIGHEST,
      MHLO_RESULT_ACCURACY_TOLERANCE
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mhlo";
}

def MHLO_ResultAccuracyModeAttr : EnumAttr<MHLO_Dialect, MHLO_ResultAccuracyMode, "result_accuracy_mode"> {
  let assemblyFormat = "`<` $value `>`";
}


#endif // MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS_ENUMS
