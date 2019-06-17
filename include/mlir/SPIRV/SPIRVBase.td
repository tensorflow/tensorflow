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

//===----------------------------------------------------------------------===//
// SPIR-V op definitions
//===----------------------------------------------------------------------===//

// Base class for all SPIR-V ops.
class SPV_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<SPV_Dialect, mnemonic, traits>;

#endif // SPIRV_BASE
