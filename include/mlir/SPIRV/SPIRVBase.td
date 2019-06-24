//===-- SPIRVOps.td - MLIR SPIR-V Op Definitions Spec ------*- tablegen -*-===//
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
// This is the base file for SPIR-V operation definition specification.
// This file defines the SPIR-V dialect, common SPIR-V types, and utilities
// for facilitating defining SPIR-V ops.
//
//===----------------------------------------------------------------------===//

#ifdef SPIRV_BASE
#else
#define SPIRV_BASE

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

//===----------------------------------------------------------------------===//
// SPIR-V dialect definitions
//===----------------------------------------------------------------------===//

def SPV_Dialect : Dialect {
  let name = "spv";

  let description = [{
    The SPIR-V dialect in MLIR.

    SPIR-V is the Khronos Group's binary intermediate language for representing
    graphical-shader stages and compute kernels for multiple Khronos APIs,
    including OpenCL, OpenGL, and Vulkan.
    See https://www.khronos.org/registry/spir-v for more details.

    This dialect aims to be a simple proxy for the SPIR-V binary format to
    enable straightforward and lightweight conversion from/to the binary
    format. Ops in this dialect should stay at the same semantic level and
    try to be a mechanical mapping to the corresponding SPIR-V instructions;
    but they may deviate representationally to allow using MLIR mechanisms.
    As a convention, if such deviation happens, the op name follows "snake_case"
    style; otherwise, the op name just follows the SPIR-V mnemonic (by removing
    the leading `Op` prefix) to use "CamelCase" style.
  }];

  let cppNamespace = "spirv";
}

//===----------------------------------------------------------------------===//
// SPIR-V type definitions
//===----------------------------------------------------------------------===//

def SPV_IsPtrType : CPred<"$_self.isa<::mlir::spirv::PointerType>()">;
def SPV_IsArrayType : CPred<"$_self.isa<::mlir::spirv::ArrayType>()">;
def SPV_IsRTArrayType : CPred<"$_self.isa<::mlir::spirv::RuntimeArrayType>()">;

// See https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_types
// for the definition of the following types and type categories.

def SPV_Void : TypeAlias<NoneType, "void type">;
def SPV_Bool : IntOfWidths<[1]>;
def SPV_Integer : IntOfWidths<[8, 16, 32, 64]>;
def SPV_Float : FloatOfWidths<[16, 32, 64]>;
def SPV_Vector : VectorOf<[SPV_Bool, SPV_Integer, SPV_Float]>;
// Component type check is done in the type parser for the following SPIR-V
// dialect-specific types so we use "Any" here.
def SPV_AnyPtr : Type<SPV_IsPtrType, "any SPIR-V pointer type">;
def SPV_AnyArray : Type<SPV_IsArrayType, "any SPIR-V array type">;
def SPV_AnyRTArray : Type<SPV_IsRTArrayType, "any SPIR-V runtime array type">;

def SPV_Numerical : AnyTypeOf<[SPV_Integer, SPV_Float]>;
def SPV_Scalar : AnyTypeOf<[SPV_Numerical, SPV_Bool]>;
def SPV_Aggregrate : AnyTypeOf<[SPV_AnyArray]>;
def SPV_Composite: AnyTypeOf<[SPV_Vector, SPV_AnyArray]>;
def SPV_Type : AnyTypeOf<[
    SPV_Void, SPV_Bool, SPV_Integer, SPV_Float, SPV_Vector,
    SPV_AnyPtr, SPV_AnyArray, SPV_AnyRTArray
  ]>;

class SPV_ScalarOrVectorOf<Type type> :
    Type<Or<[type.predicate, VectorOf<[type]>.predicate]>,
         "scalar/vector of " # type.description>;

// TODO(antiagainst): Use a more appropriate way to model optional operands
class SPV_Optional<Type type> : Variadic<type>;

//===----------------------------------------------------------------------===//
// SPIR-V enum definitions
//===----------------------------------------------------------------------===//

// Begin enum section. Generated from SPIR-V spec; DO NOT MODIFY!

def SPV_AM_Logical                    : EnumAttrCase<"Logical", 0>;
def SPV_AM_Physical32                 : EnumAttrCase<"Physical32", 1>;
def SPV_AM_Physical64                 : EnumAttrCase<"Physical64", 2>;
def SPV_AM_PhysicalStorageBuffer64EXT : EnumAttrCase<"PhysicalStorageBuffer64EXT", 5348>;

def SPV_AddressingModelAttr :
    EnumAttr<"AddressingModel", "valid SPIR-V AddressingModel", [
      SPV_AM_Logical, SPV_AM_Physical32, SPV_AM_Physical64,
      SPV_AM_PhysicalStorageBuffer64EXT
    ]> {
  let cppNamespace = "::mlir::spirv";
  let underlyingType = "uint32_t";
}

def SPV_D_1D          : EnumAttrCase<"1D", 0>;
def SPV_D_2D          : EnumAttrCase<"2D", 1>;
def SPV_D_3D          : EnumAttrCase<"3D", 2>;
def SPV_D_Cube        : EnumAttrCase<"Cube", 3>;
def SPV_D_Rect        : EnumAttrCase<"Rect", 4>;
def SPV_D_Buffer      : EnumAttrCase<"Buffer", 5>;
def SPV_D_SubpassData : EnumAttrCase<"SubpassData", 6>;

def SPV_DimAttr :
    EnumAttr<"Dim", "valid SPIR-V Dim", [
      SPV_D_1D, SPV_D_2D, SPV_D_3D, SPV_D_Cube, SPV_D_Rect, SPV_D_Buffer,
      SPV_D_SubpassData
    ]> {
  let cppNamespace = "::mlir::spirv";
  let underlyingType = "uint32_t";
}

def SPV_IF_Unknown      : EnumAttrCase<"Unknown", 0>;
def SPV_IF_Rgba32f      : EnumAttrCase<"Rgba32f", 1>;
def SPV_IF_Rgba16f      : EnumAttrCase<"Rgba16f", 2>;
def SPV_IF_R32f         : EnumAttrCase<"R32f", 3>;
def SPV_IF_Rgba8        : EnumAttrCase<"Rgba8", 4>;
def SPV_IF_Rgba8Snorm   : EnumAttrCase<"Rgba8Snorm", 5>;
def SPV_IF_Rg32f        : EnumAttrCase<"Rg32f", 6>;
def SPV_IF_Rg16f        : EnumAttrCase<"Rg16f", 7>;
def SPV_IF_R11fG11fB10f : EnumAttrCase<"R11fG11fB10f", 8>;
def SPV_IF_R16f         : EnumAttrCase<"R16f", 9>;
def SPV_IF_Rgba16       : EnumAttrCase<"Rgba16", 10>;
def SPV_IF_Rgb10A2      : EnumAttrCase<"Rgb10A2", 11>;
def SPV_IF_Rg16         : EnumAttrCase<"Rg16", 12>;
def SPV_IF_Rg8          : EnumAttrCase<"Rg8", 13>;
def SPV_IF_R16          : EnumAttrCase<"R16", 14>;
def SPV_IF_R8           : EnumAttrCase<"R8", 15>;
def SPV_IF_Rgba16Snorm  : EnumAttrCase<"Rgba16Snorm", 16>;
def SPV_IF_Rg16Snorm    : EnumAttrCase<"Rg16Snorm", 17>;
def SPV_IF_Rg8Snorm     : EnumAttrCase<"Rg8Snorm", 18>;
def SPV_IF_R16Snorm     : EnumAttrCase<"R16Snorm", 19>;
def SPV_IF_R8Snorm      : EnumAttrCase<"R8Snorm", 20>;
def SPV_IF_Rgba32i      : EnumAttrCase<"Rgba32i", 21>;
def SPV_IF_Rgba16i      : EnumAttrCase<"Rgba16i", 22>;
def SPV_IF_Rgba8i       : EnumAttrCase<"Rgba8i", 23>;
def SPV_IF_R32i         : EnumAttrCase<"R32i", 24>;
def SPV_IF_Rg32i        : EnumAttrCase<"Rg32i", 25>;
def SPV_IF_Rg16i        : EnumAttrCase<"Rg16i", 26>;
def SPV_IF_Rg8i         : EnumAttrCase<"Rg8i", 27>;
def SPV_IF_R16i         : EnumAttrCase<"R16i", 28>;
def SPV_IF_R8i          : EnumAttrCase<"R8i", 29>;
def SPV_IF_Rgba32ui     : EnumAttrCase<"Rgba32ui", 30>;
def SPV_IF_Rgba16ui     : EnumAttrCase<"Rgba16ui", 31>;
def SPV_IF_Rgba8ui      : EnumAttrCase<"Rgba8ui", 32>;
def SPV_IF_R32ui        : EnumAttrCase<"R32ui", 33>;
def SPV_IF_Rgb10a2ui    : EnumAttrCase<"Rgb10a2ui", 34>;
def SPV_IF_Rg32ui       : EnumAttrCase<"Rg32ui", 35>;
def SPV_IF_Rg16ui       : EnumAttrCase<"Rg16ui", 36>;
def SPV_IF_Rg8ui        : EnumAttrCase<"Rg8ui", 37>;
def SPV_IF_R16ui        : EnumAttrCase<"R16ui", 38>;
def SPV_IF_R8ui         : EnumAttrCase<"R8ui", 39>;

def SPV_ImageFormatAttr :
    EnumAttr<"ImageFormat", "valid SPIR-V ImageFormat", [
      SPV_IF_Unknown, SPV_IF_Rgba32f, SPV_IF_Rgba16f, SPV_IF_R32f, SPV_IF_Rgba8,
      SPV_IF_Rgba8Snorm, SPV_IF_Rg32f, SPV_IF_Rg16f, SPV_IF_R11fG11fB10f,
      SPV_IF_R16f, SPV_IF_Rgba16, SPV_IF_Rgb10A2, SPV_IF_Rg16, SPV_IF_Rg8,
      SPV_IF_R16, SPV_IF_R8, SPV_IF_Rgba16Snorm, SPV_IF_Rg16Snorm, SPV_IF_Rg8Snorm,
      SPV_IF_R16Snorm, SPV_IF_R8Snorm, SPV_IF_Rgba32i, SPV_IF_Rgba16i, SPV_IF_Rgba8i,
      SPV_IF_R32i, SPV_IF_Rg32i, SPV_IF_Rg16i, SPV_IF_Rg8i, SPV_IF_R16i, SPV_IF_R8i,
      SPV_IF_Rgba32ui, SPV_IF_Rgba16ui, SPV_IF_Rgba8ui, SPV_IF_R32ui,
      SPV_IF_Rgb10a2ui, SPV_IF_Rg32ui, SPV_IF_Rg16ui, SPV_IF_Rg8ui, SPV_IF_R16ui,
      SPV_IF_R8ui
    ]> {
  let cppNamespace = "::mlir::spirv";
  let underlyingType = "uint32_t";
}

def SPV_MA_None                    : EnumAttrCase<"None", 0x0000>;
def SPV_MA_Volatile                : EnumAttrCase<"Volatile", 0x0001>;
def SPV_MA_Aligned                 : EnumAttrCase<"Aligned", 0x0002>;
def SPV_MA_Nontemporal             : EnumAttrCase<"Nontemporal", 0x0004>;
def SPV_MA_MakePointerAvailableKHR : EnumAttrCase<"MakePointerAvailableKHR", 0x0008>;
def SPV_MA_MakePointerVisibleKHR   : EnumAttrCase<"MakePointerVisibleKHR", 0x0010>;
def SPV_MA_NonPrivatePointerKHR    : EnumAttrCase<"NonPrivatePointerKHR", 0x0020>;

def SPV_MemoryAccessAttr :
    EnumAttr<"MemoryAccess", "valid SPIR-V MemoryAccess", [
      SPV_MA_None, SPV_MA_Volatile, SPV_MA_Aligned, SPV_MA_Nontemporal,
      SPV_MA_MakePointerAvailableKHR, SPV_MA_MakePointerVisibleKHR,
      SPV_MA_NonPrivatePointerKHR
    ]> {
  let cppNamespace = "::mlir::spirv";
  let underlyingType = "uint32_t";
}

def SPV_MM_Simple    : EnumAttrCase<"Simple", 0>;
def SPV_MM_GLSL450   : EnumAttrCase<"GLSL450", 1>;
def SPV_MM_OpenCL    : EnumAttrCase<"OpenCL", 2>;
def SPV_MM_VulkanKHR : EnumAttrCase<"VulkanKHR", 3>;

def SPV_MemoryModelAttr :
    EnumAttr<"MemoryModel", "valid SPIR-V MemoryModel", [
      SPV_MM_Simple, SPV_MM_GLSL450, SPV_MM_OpenCL, SPV_MM_VulkanKHR
    ]> {
  let cppNamespace = "::mlir::spirv";
  let underlyingType = "uint32_t";
}

def SPV_SC_UniformConstant          : EnumAttrCase<"UniformConstant", 0>;
def SPV_SC_Input                    : EnumAttrCase<"Input", 1>;
def SPV_SC_Uniform                  : EnumAttrCase<"Uniform", 2>;
def SPV_SC_Output                   : EnumAttrCase<"Output", 3>;
def SPV_SC_Workgroup                : EnumAttrCase<"Workgroup", 4>;
def SPV_SC_CrossWorkgroup           : EnumAttrCase<"CrossWorkgroup", 5>;
def SPV_SC_Private                  : EnumAttrCase<"Private", 6>;
def SPV_SC_Function                 : EnumAttrCase<"Function", 7>;
def SPV_SC_Generic                  : EnumAttrCase<"Generic", 8>;
def SPV_SC_PushConstant             : EnumAttrCase<"PushConstant", 9>;
def SPV_SC_AtomicCounter            : EnumAttrCase<"AtomicCounter", 10>;
def SPV_SC_Image                    : EnumAttrCase<"Image", 11>;
def SPV_SC_StorageBuffer            : EnumAttrCase<"StorageBuffer", 12>;
def SPV_SC_CallableDataNV           : EnumAttrCase<"CallableDataNV", 5328>;
def SPV_SC_IncomingCallableDataNV   : EnumAttrCase<"IncomingCallableDataNV", 5329>;
def SPV_SC_RayPayloadNV             : EnumAttrCase<"RayPayloadNV", 5338>;
def SPV_SC_HitAttributeNV           : EnumAttrCase<"HitAttributeNV", 5339>;
def SPV_SC_IncomingRayPayloadNV     : EnumAttrCase<"IncomingRayPayloadNV", 5342>;
def SPV_SC_ShaderRecordBufferNV     : EnumAttrCase<"ShaderRecordBufferNV", 5343>;
def SPV_SC_PhysicalStorageBufferEXT : EnumAttrCase<"PhysicalStorageBufferEXT", 5349>;

def SPV_StorageClassAttr :
    EnumAttr<"StorageClass", "valid SPIR-V StorageClass", [
      SPV_SC_UniformConstant, SPV_SC_Input, SPV_SC_Uniform, SPV_SC_Output,
      SPV_SC_Workgroup, SPV_SC_CrossWorkgroup, SPV_SC_Private, SPV_SC_Function,
      SPV_SC_Generic, SPV_SC_PushConstant, SPV_SC_AtomicCounter, SPV_SC_Image,
      SPV_SC_StorageBuffer, SPV_SC_CallableDataNV, SPV_SC_IncomingCallableDataNV,
      SPV_SC_RayPayloadNV, SPV_SC_HitAttributeNV, SPV_SC_IncomingRayPayloadNV,
      SPV_SC_ShaderRecordBufferNV, SPV_SC_PhysicalStorageBufferEXT
    ]> {
  let cppNamespace = "::mlir::spirv";
  let underlyingType = "uint32_t";
}

// End enum section. Generated from SPIR-V spec; DO NOT MODIFY!

// Enums added manually that are not part of SPIRV spec

def SPV_IDI_NoDepth    : EnumAttrCase<"NoDepth", 0>;
def SPV_IDI_IsDepth   : EnumAttrCase<"IsDepth", 1>;
def SPV_IDI_DepthUnknown : EnumAttrCase<"DepthUnknown", 2>;

def SPV_DepthAttr :
    EnumAttr<"ImageDepthInfo", "valid SPIR-V Image Depth specification",[
    SPV_IDI_NoDepth, SPV_IDI_IsDepth, SPV_IDI_DepthUnknown]> {
  let cppNamespace = "::mlir::spirv";
  let underlyingType = "uint32_t";
}

def SPV_IAI_NonArrayed : EnumAttrCase<"NonArrayed", 0>;
def SPV_IAI_Arrayed     : EnumAttrCase<"Arrayed", 1>;

def SPV_ArrayedAttr :
    EnumAttr<"ImageArrayedInfo", "valid SPIR-V Image Arrayed specification", [
    SPV_IAI_NonArrayed, SPV_IAI_Arrayed]> {
  let cppNamespace = "::mlir::spirv";
  let underlyingType = "uint32_t";
}

def SPV_ISI_SingleSampled : EnumAttrCase<"SingleSampled", 0>;
def SPV_ISI_MultiSampled  : EnumAttrCase<"MultiSampled", 1>;

def SPV_SamplingAttr:
    EnumAttr<"ImageSamplingInfo", "valid SPIR-V Image Sampling specification", [
    SPV_ISI_SingleSampled, SPV_ISI_MultiSampled]> {
  let cppNamespace = "::mlir::spirv";
  let underlyingType = "uint32_t";
}

def SPV_ISUI_SamplerUnknown : EnumAttrCase<"SamplerUnknown", 0>;
def SPV_ISUI_NeedSampler    : EnumAttrCase<"NeedSampler", 1>;
def SPV_ISUI_NoSampler      : EnumAttrCase<"NoSampler", 2>;

def SPV_SamplerUseAttr:
    EnumAttr<"ImageSamplerUseInfo", "valid SPIR-V Sampler Use specification", [
    SPV_ISUI_SamplerUnknown, SPV_ISUI_NeedSampler, SPV_ISUI_NoSampler]> {
  let cppNamespace = "::mlir::spirv";
  let underlyingType = "uint32_t";
}

//===----------------------------------------------------------------------===//
// SPIR-V op definitions
//===----------------------------------------------------------------------===//

// Base class for all SPIR-V ops.
class SPV_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<SPV_Dialect, mnemonic, traits> {
  // Opcode for the binary format. Ops cannot be directly serialized will
  // leave this field as unset.
  int opcode = ?;

  // For each SPIR-V op, the following static functions need to be defined
  // in SPVOps.cpp:
  //
  // * static ParseResult parse<op-c++-class-name>(OpAsmParser *parser,
  //                                               OperationState *result)
  // * static void print(OpAsmPrinter *p, <op-c++-class-name> op)
  // * static LogicalResult verify(<op-c++-class-name> op)
  let parser = [{ return ::parse$cppClass(parser, result); }];
  let printer = [{ return ::print(*this, p); }];
  let verifier = [{ return ::verify(*this); }];
}

#endif // SPIRV_BASE
