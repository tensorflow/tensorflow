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

def SPV_AM_Logical                    : I32EnumAttrCase<"Logical", 0>;
def SPV_AM_Physical32                 : I32EnumAttrCase<"Physical32", 1>;
def SPV_AM_Physical64                 : I32EnumAttrCase<"Physical64", 2>;
def SPV_AM_PhysicalStorageBuffer64EXT : I32EnumAttrCase<"PhysicalStorageBuffer64EXT", 5348>;

def SPV_AddressingModelAttr :
    I32EnumAttr<"AddressingModel", "valid SPIR-V AddressingModel", [
      SPV_AM_Logical, SPV_AM_Physical32, SPV_AM_Physical64,
      SPV_AM_PhysicalStorageBuffer64EXT
    ]> {
  let returnType = "::mlir::spirv::AddressingModel";
  let convertFromStorage = "static_cast<::mlir::spirv::AddressingModel>($_self.getInt())";
  let cppNamespace = "::mlir::spirv";
}

def SPV_D_1D          : I32EnumAttrCase<"1D", 0>;
def SPV_D_2D          : I32EnumAttrCase<"2D", 1>;
def SPV_D_3D          : I32EnumAttrCase<"3D", 2>;
def SPV_D_Cube        : I32EnumAttrCase<"Cube", 3>;
def SPV_D_Rect        : I32EnumAttrCase<"Rect", 4>;
def SPV_D_Buffer      : I32EnumAttrCase<"Buffer", 5>;
def SPV_D_SubpassData : I32EnumAttrCase<"SubpassData", 6>;

def SPV_DimAttr :
    I32EnumAttr<"Dim", "valid SPIR-V Dim", [
      SPV_D_1D, SPV_D_2D, SPV_D_3D, SPV_D_Cube, SPV_D_Rect, SPV_D_Buffer,
      SPV_D_SubpassData
    ]> {
  let returnType = "::mlir::spirv::Dim";
  let convertFromStorage = "static_cast<::mlir::spirv::Dim>($_self.getInt())";
  let cppNamespace = "::mlir::spirv";
}

def SPV_IF_Unknown      : I32EnumAttrCase<"Unknown", 0>;
def SPV_IF_Rgba32f      : I32EnumAttrCase<"Rgba32f", 1>;
def SPV_IF_Rgba16f      : I32EnumAttrCase<"Rgba16f", 2>;
def SPV_IF_R32f         : I32EnumAttrCase<"R32f", 3>;
def SPV_IF_Rgba8        : I32EnumAttrCase<"Rgba8", 4>;
def SPV_IF_Rgba8Snorm   : I32EnumAttrCase<"Rgba8Snorm", 5>;
def SPV_IF_Rg32f        : I32EnumAttrCase<"Rg32f", 6>;
def SPV_IF_Rg16f        : I32EnumAttrCase<"Rg16f", 7>;
def SPV_IF_R11fG11fB10f : I32EnumAttrCase<"R11fG11fB10f", 8>;
def SPV_IF_R16f         : I32EnumAttrCase<"R16f", 9>;
def SPV_IF_Rgba16       : I32EnumAttrCase<"Rgba16", 10>;
def SPV_IF_Rgb10A2      : I32EnumAttrCase<"Rgb10A2", 11>;
def SPV_IF_Rg16         : I32EnumAttrCase<"Rg16", 12>;
def SPV_IF_Rg8          : I32EnumAttrCase<"Rg8", 13>;
def SPV_IF_R16          : I32EnumAttrCase<"R16", 14>;
def SPV_IF_R8           : I32EnumAttrCase<"R8", 15>;
def SPV_IF_Rgba16Snorm  : I32EnumAttrCase<"Rgba16Snorm", 16>;
def SPV_IF_Rg16Snorm    : I32EnumAttrCase<"Rg16Snorm", 17>;
def SPV_IF_Rg8Snorm     : I32EnumAttrCase<"Rg8Snorm", 18>;
def SPV_IF_R16Snorm     : I32EnumAttrCase<"R16Snorm", 19>;
def SPV_IF_R8Snorm      : I32EnumAttrCase<"R8Snorm", 20>;
def SPV_IF_Rgba32i      : I32EnumAttrCase<"Rgba32i", 21>;
def SPV_IF_Rgba16i      : I32EnumAttrCase<"Rgba16i", 22>;
def SPV_IF_Rgba8i       : I32EnumAttrCase<"Rgba8i", 23>;
def SPV_IF_R32i         : I32EnumAttrCase<"R32i", 24>;
def SPV_IF_Rg32i        : I32EnumAttrCase<"Rg32i", 25>;
def SPV_IF_Rg16i        : I32EnumAttrCase<"Rg16i", 26>;
def SPV_IF_Rg8i         : I32EnumAttrCase<"Rg8i", 27>;
def SPV_IF_R16i         : I32EnumAttrCase<"R16i", 28>;
def SPV_IF_R8i          : I32EnumAttrCase<"R8i", 29>;
def SPV_IF_Rgba32ui     : I32EnumAttrCase<"Rgba32ui", 30>;
def SPV_IF_Rgba16ui     : I32EnumAttrCase<"Rgba16ui", 31>;
def SPV_IF_Rgba8ui      : I32EnumAttrCase<"Rgba8ui", 32>;
def SPV_IF_R32ui        : I32EnumAttrCase<"R32ui", 33>;
def SPV_IF_Rgb10a2ui    : I32EnumAttrCase<"Rgb10a2ui", 34>;
def SPV_IF_Rg32ui       : I32EnumAttrCase<"Rg32ui", 35>;
def SPV_IF_Rg16ui       : I32EnumAttrCase<"Rg16ui", 36>;
def SPV_IF_Rg8ui        : I32EnumAttrCase<"Rg8ui", 37>;
def SPV_IF_R16ui        : I32EnumAttrCase<"R16ui", 38>;
def SPV_IF_R8ui         : I32EnumAttrCase<"R8ui", 39>;

def SPV_ImageFormatAttr :
    I32EnumAttr<"ImageFormat", "valid SPIR-V ImageFormat", [
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
  let returnType = "::mlir::spirv::ImageFormat";
  let convertFromStorage = "static_cast<::mlir::spirv::ImageFormat>($_self.getInt())";
  let cppNamespace = "::mlir::spirv";
}

def SPV_MA_None                    : I32EnumAttrCase<"None", 0x0000>;
def SPV_MA_Volatile                : I32EnumAttrCase<"Volatile", 0x0001>;
def SPV_MA_Aligned                 : I32EnumAttrCase<"Aligned", 0x0002>;
def SPV_MA_Nontemporal             : I32EnumAttrCase<"Nontemporal", 0x0004>;
def SPV_MA_MakePointerAvailableKHR : I32EnumAttrCase<"MakePointerAvailableKHR", 0x0008>;
def SPV_MA_MakePointerVisibleKHR   : I32EnumAttrCase<"MakePointerVisibleKHR", 0x0010>;
def SPV_MA_NonPrivatePointerKHR    : I32EnumAttrCase<"NonPrivatePointerKHR", 0x0020>;

def SPV_MemoryAccessAttr :
    I32EnumAttr<"MemoryAccess", "valid SPIR-V MemoryAccess", [
      SPV_MA_None, SPV_MA_Volatile, SPV_MA_Aligned, SPV_MA_Nontemporal,
      SPV_MA_MakePointerAvailableKHR, SPV_MA_MakePointerVisibleKHR,
      SPV_MA_NonPrivatePointerKHR
    ]> {
  let returnType = "::mlir::spirv::MemoryAccess";
  let convertFromStorage = "static_cast<::mlir::spirv::MemoryAccess>($_self.getInt())";
  let cppNamespace = "::mlir::spirv";
}

def SPV_MM_Simple    : I32EnumAttrCase<"Simple", 0>;
def SPV_MM_GLSL450   : I32EnumAttrCase<"GLSL450", 1>;
def SPV_MM_OpenCL    : I32EnumAttrCase<"OpenCL", 2>;
def SPV_MM_VulkanKHR : I32EnumAttrCase<"VulkanKHR", 3>;

def SPV_MemoryModelAttr :
    I32EnumAttr<"MemoryModel", "valid SPIR-V MemoryModel", [
      SPV_MM_Simple, SPV_MM_GLSL450, SPV_MM_OpenCL, SPV_MM_VulkanKHR
    ]> {
  let returnType = "::mlir::spirv::MemoryModel";
  let convertFromStorage = "static_cast<::mlir::spirv::MemoryModel>($_self.getInt())";
  let cppNamespace = "::mlir::spirv";
}

def SPV_SC_UniformConstant          : I32EnumAttrCase<"UniformConstant", 0>;
def SPV_SC_Input                    : I32EnumAttrCase<"Input", 1>;
def SPV_SC_Uniform                  : I32EnumAttrCase<"Uniform", 2>;
def SPV_SC_Output                   : I32EnumAttrCase<"Output", 3>;
def SPV_SC_Workgroup                : I32EnumAttrCase<"Workgroup", 4>;
def SPV_SC_CrossWorkgroup           : I32EnumAttrCase<"CrossWorkgroup", 5>;
def SPV_SC_Private                  : I32EnumAttrCase<"Private", 6>;
def SPV_SC_Function                 : I32EnumAttrCase<"Function", 7>;
def SPV_SC_Generic                  : I32EnumAttrCase<"Generic", 8>;
def SPV_SC_PushConstant             : I32EnumAttrCase<"PushConstant", 9>;
def SPV_SC_AtomicCounter            : I32EnumAttrCase<"AtomicCounter", 10>;
def SPV_SC_Image                    : I32EnumAttrCase<"Image", 11>;
def SPV_SC_StorageBuffer            : I32EnumAttrCase<"StorageBuffer", 12>;
def SPV_SC_CallableDataNV           : I32EnumAttrCase<"CallableDataNV", 5328>;
def SPV_SC_IncomingCallableDataNV   : I32EnumAttrCase<"IncomingCallableDataNV", 5329>;
def SPV_SC_RayPayloadNV             : I32EnumAttrCase<"RayPayloadNV", 5338>;
def SPV_SC_HitAttributeNV           : I32EnumAttrCase<"HitAttributeNV", 5339>;
def SPV_SC_IncomingRayPayloadNV     : I32EnumAttrCase<"IncomingRayPayloadNV", 5342>;
def SPV_SC_ShaderRecordBufferNV     : I32EnumAttrCase<"ShaderRecordBufferNV", 5343>;
def SPV_SC_PhysicalStorageBufferEXT : I32EnumAttrCase<"PhysicalStorageBufferEXT", 5349>;

def SPV_StorageClassAttr :
    I32EnumAttr<"StorageClass", "valid SPIR-V StorageClass", [
      SPV_SC_UniformConstant, SPV_SC_Input, SPV_SC_Uniform, SPV_SC_Output,
      SPV_SC_Workgroup, SPV_SC_CrossWorkgroup, SPV_SC_Private, SPV_SC_Function,
      SPV_SC_Generic, SPV_SC_PushConstant, SPV_SC_AtomicCounter, SPV_SC_Image,
      SPV_SC_StorageBuffer, SPV_SC_CallableDataNV, SPV_SC_IncomingCallableDataNV,
      SPV_SC_RayPayloadNV, SPV_SC_HitAttributeNV, SPV_SC_IncomingRayPayloadNV,
      SPV_SC_ShaderRecordBufferNV, SPV_SC_PhysicalStorageBufferEXT
    ]> {
  let returnType = "::mlir::spirv::StorageClass";
  let convertFromStorage = "static_cast<::mlir::spirv::StorageClass>($_self.getInt())";
  let cppNamespace = "::mlir::spirv";
}

// End enum section. Generated from SPIR-V spec; DO NOT MODIFY!

// Enums added manually that are not part of SPIRV spec

def SPV_IDI_NoDepth      : I32EnumAttrCase<"NoDepth", 0>;
def SPV_IDI_IsDepth      : I32EnumAttrCase<"IsDepth", 1>;
def SPV_IDI_DepthUnknown : I32EnumAttrCase<"DepthUnknown", 2>;

def SPV_DepthAttr :
    I32EnumAttr<"ImageDepthInfo", "valid SPIR-V Image Depth specification",
      [SPV_IDI_NoDepth, SPV_IDI_IsDepth, SPV_IDI_DepthUnknown]> {
  let cppNamespace = "::mlir::spirv";
}

def SPV_IAI_NonArrayed : I32EnumAttrCase<"NonArrayed", 0>;
def SPV_IAI_Arrayed    : I32EnumAttrCase<"Arrayed", 1>;

def SPV_ArrayedAttr :
    I32EnumAttr<"ImageArrayedInfo", "valid SPIR-V Image Arrayed specification",
      [SPV_IAI_NonArrayed, SPV_IAI_Arrayed]> {
  let cppNamespace = "::mlir::spirv";
}

def SPV_ISI_SingleSampled : I32EnumAttrCase<"SingleSampled", 0>;
def SPV_ISI_MultiSampled  : I32EnumAttrCase<"MultiSampled", 1>;

def SPV_SamplingAttr:
    I32EnumAttr<"ImageSamplingInfo", "valid SPIR-V Image Sampling specification",
      [SPV_ISI_SingleSampled, SPV_ISI_MultiSampled]> {
  let cppNamespace = "::mlir::spirv";
}

def SPV_ISUI_SamplerUnknown : I32EnumAttrCase<"SamplerUnknown", 0>;
def SPV_ISUI_NeedSampler    : I32EnumAttrCase<"NeedSampler", 1>;
def SPV_ISUI_NoSampler      : I32EnumAttrCase<"NoSampler", 2>;

def SPV_SamplerUseAttr:
    I32EnumAttr<"ImageSamplerUseInfo", "valid SPIR-V Sampler Use specification",
      [SPV_ISUI_SamplerUnknown, SPV_ISUI_NeedSampler, SPV_ISUI_NoSampler]> {
  let cppNamespace = "::mlir::spirv";
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
