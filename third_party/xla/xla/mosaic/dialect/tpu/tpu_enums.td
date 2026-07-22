/* Copyright 2026 The OpenXLA Authors.

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

#ifndef TPU_ENUMS
#define TPU_ENUMS

include "mlir/IR/EnumAttr.td"
include "xla/mosaic/dialect/tpu/tpu.td"

def TPU_CoreType : I32EnumAttr<"CoreType", "Core type", [
  I32EnumAttrCase<"kTc", 0, "tc">,
  I32EnumAttrCase<"kScScalarSubcore", 1, "sc_scalar_subcore">,
  I32EnumAttrCase<"kScVectorSubcore", 2, "sc_vector_subcore">
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_CoreTypeEnum : EnumAttr<TPU_Dialect, TPU_CoreType, "core_type"> {
    let assemblyFormat = "`<` $value `>`";
}

def TPU_PipelineMode : I32EnumAttr<"PipelineMode", "Pipeline mode", [
  I32EnumAttrCase<"kSynchronous", 1, "synchronous">,
  I32EnumAttrCase<"kDoubleBuffered", 2, "double_buffered">
  ]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_PipelineModeEnum : EnumAttr<TPU_Dialect, TPU_PipelineMode, "pipeline_mode"> {
    let assemblyFormat = "`<` $value `>`";
}

def TPU_RevisitMode : I32EnumAttr<"RevisitMode", "Revisit mode", [
  I32EnumAttrCase<"kImmediate", 0, "immediate">,
  I32EnumAttrCase<"kAny", 1, "any">
  ]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_RevisitModeEnum : EnumAttr<TPU_Dialect, TPU_RevisitMode, "revisit_mode"> {
    let assemblyFormat = "`<` $value `>`";
}

def TPU_DimensionSemantics : I32EnumAttr<"DimensionSemantics", "Dimension semantics", [
  I32EnumAttrCase<"parallel", 0>,
  I32EnumAttrCase<"arbitrary", 1>,
  I32EnumAttrCase<"core_parallel", 2>,
  I32EnumAttrCase<"subcore_parallel", 3>
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_DimensionSemanticsEnum
    : EnumAttr<TPU_Dialect, TPU_DimensionSemantics, "dimension_semantics"> {
    let assemblyFormat = "`<` $value `>`";
}


def TPU_ContractPrecision : I32EnumAttr<"ContractPrecision", "Contraction precision", [
  I32EnumAttrCase<"kBF16", 0, "bf16">,
  I32EnumAttrCase<"kFP32", 1, "fp32">
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_ContractPrecisionEnum
    : EnumAttr<TPU_Dialect, TPU_ContractPrecision, "contract_precision"> {
    let assemblyFormat = "`<` $value `>`";
}

def TPU_PackFormat : I32EnumAttr<"PackFormat", "Pack format", [
  I32EnumAttrCase<"kCompressed", 0, "compressed">,
  I32EnumAttrCase<"kInterleaved", 1, "interleaved">
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_PackFormatEnum : EnumAttr<TPU_Dialect, TPU_PackFormat, "pack_format"> {
    let assemblyFormat = "`<` $value `>`";
}

def TPU_TiledCase   : I32EnumAttrCase<"tiled", 0>;
def TPU_LaneCase    : I32EnumAttrCase<"lanes", 1>;
def TPU_SublaneCase : I32EnumAttrCase<"sublanes", 2>;
def TPU_VectorLayoutDim : I32EnumAttr<
  "VectorLayoutDim", "", [TPU_TiledCase, TPU_LaneCase, TPU_SublaneCase]>;


def TPU_MemorySpace : I32EnumAttr<"MemorySpace", "Memory space", [
  I32EnumAttrCase<"kAny", 4294967295, "any">,
  I32EnumAttrCase<"kVmem", 0, "vmem">,
  I32EnumAttrCase<"kSmem", 1, "smem">,
  I32EnumAttrCase<"kHbm", 2, "hbm">,
  I32EnumAttrCase<"kCmem", 3, "cmem">,
  I32EnumAttrCase<"kSemaphoreMem", 4, "semaphore_mem">,
  I32EnumAttrCase<"kVmemShared", 5, "vmem_shared">,
  I32EnumAttrCase<"kHost", 6, "host">
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_ReductionKind : I32EnumAttr<"ReductionKind", "Reduction kind", [
  I32EnumAttrCase<"kSum", 0, "sum">,
  I32EnumAttrCase<"kMax", 1, "max">,
  I32EnumAttrCase<"kMin", 2, "min">,
  I32EnumAttrCase<"kArgMax", 3, "arg_max">,
  I32EnumAttrCase<"kArgMin", 4, "arg_min">,
  I32EnumAttrCase<"kFindFirstSet", 5, "find_first_set">
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_ReductionKindAttr
    : EnumAttr<TPU_Dialect, TPU_ReductionKind, "reduction_kind"> {
    let assemblyFormat = "`<` $value `>`";
}

def TPU_RoundingMode : I32EnumAttr<"RoundingMode", "Rounding mode", [
  I32EnumAttrCase<"kTowardsZero", 0, "towards_zero">,
  I32EnumAttrCase<"kToNearestEven", 1, "to_nearest_even">,
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_RoundingModeEnum : EnumAttr<TPU_Dialect, TPU_RoundingMode, "rounding_mode"> {
    let assemblyFormat = "`<` $value `>`";
}

#endif  // TPU_ENUMS
