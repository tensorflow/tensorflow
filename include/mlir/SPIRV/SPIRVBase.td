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

class SPV_ScalarOrVectorOf<Type type> :
    Type<Or<[type.predicate, VectorOf<[type]>.predicate]>,
         "scalar/vector of " # type.description>;

// Begin enum section. Generated from SPIR-V spec; DO NOT MODIFY!

def SPV_AM_Logical                    : EnumAttrCase<"Logical">;
def SPV_AM_Physical32                 : EnumAttrCase<"Physical32">;
def SPV_AM_Physical64                 : EnumAttrCase<"Physical64">;
def SPV_AM_PhysicalStorageBuffer64EXT : EnumAttrCase<"PhysicalStorageBuffer64EXT">;
def SPV_AddressingModelAttr :
    EnumAttr<"AddressingModel", "valid SPIR-V AddressingModel", [
      SPV_AM_Logical, SPV_AM_Physical32, SPV_AM_Physical64,
      SPV_AM_PhysicalStorageBuffer64EXT
    ]>;

def SPV_MM_Simple    : EnumAttrCase<"Simple">;
def SPV_MM_GLSL450   : EnumAttrCase<"GLSL450">;
def SPV_MM_OpenCL    : EnumAttrCase<"OpenCL">;
def SPV_MM_VulkanKHR : EnumAttrCase<"VulkanKHR">;
def SPV_MemoryModelAttr :
    EnumAttr<"MemoryModel", "valid SPIR-V MemoryModel", [
      SPV_MM_Simple, SPV_MM_GLSL450, SPV_MM_OpenCL, SPV_MM_VulkanKHR
    ]>;

// End enum section. Generated from SPIR-V spec; DO NOT MODIFY!

//===----------------------------------------------------------------------===//
// SPIR-V op definitions
//===----------------------------------------------------------------------===//

// Base class for all SPIR-V ops.
class SPV_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<SPV_Dialect, mnemonic, traits>;

#endif // SPIRV_BASE
