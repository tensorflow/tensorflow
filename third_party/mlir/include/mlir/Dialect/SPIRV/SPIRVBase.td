//===- SPIRVBase.td - MLIR SPIR-V Op Definitions Base file -*- tablegen -*-===//
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
// SPIR-V opcode specification
//===----------------------------------------------------------------------===//

class SPV_OpCode<string name, int val> {
  // Name used as reference to retrieve the opcode
  string opname = name;

  // Opcode associated with the name
  int opcode = val;
}

// Begin opcode section. Generated from SPIR-V spec; DO NOT MODIFY!

def SPV_OC_OpNop                   : I32EnumAttrCase<"OpNop", 0>;
def SPV_OC_OpName                  : I32EnumAttrCase<"OpName", 5>;
def SPV_OC_OpMemoryModel           : I32EnumAttrCase<"OpMemoryModel", 14>;
def SPV_OC_OpEntryPoint            : I32EnumAttrCase<"OpEntryPoint", 15>;
def SPV_OC_OpExecutionMode         : I32EnumAttrCase<"OpExecutionMode", 16>;
def SPV_OC_OpTypeVoid              : I32EnumAttrCase<"OpTypeVoid", 19>;
def SPV_OC_OpTypeBool              : I32EnumAttrCase<"OpTypeBool", 20>;
def SPV_OC_OpTypeInt               : I32EnumAttrCase<"OpTypeInt", 21>;
def SPV_OC_OpTypeFloat             : I32EnumAttrCase<"OpTypeFloat", 22>;
def SPV_OC_OpTypeVector            : I32EnumAttrCase<"OpTypeVector", 23>;
def SPV_OC_OpTypeArray             : I32EnumAttrCase<"OpTypeArray", 28>;
def SPV_OC_OpTypePointer           : I32EnumAttrCase<"OpTypePointer", 32>;
def SPV_OC_OpTypeFunction          : I32EnumAttrCase<"OpTypeFunction", 33>;
def SPV_OC_OpConstantTrue          : I32EnumAttrCase<"OpConstantTrue", 41>;
def SPV_OC_OpConstantFalse         : I32EnumAttrCase<"OpConstantFalse", 42>;
def SPV_OC_OpConstant              : I32EnumAttrCase<"OpConstant", 43>;
def SPV_OC_OpConstantComposite     : I32EnumAttrCase<"OpConstantComposite", 44>;
def SPV_OC_OpConstantNull          : I32EnumAttrCase<"OpConstantNull", 46>;
def SPV_OC_OpSpecConstantTrue      : I32EnumAttrCase<"OpSpecConstantTrue", 48>;
def SPV_OC_OpSpecConstantFalse     : I32EnumAttrCase<"OpSpecConstantFalse", 49>;
def SPV_OC_OpSpecConstant          : I32EnumAttrCase<"OpSpecConstant", 50>;
def SPV_OC_OpSpecConstantComposite : I32EnumAttrCase<"OpSpecConstantComposite", 51>;
def SPV_OC_OpFunction              : I32EnumAttrCase<"OpFunction", 54>;
def SPV_OC_OpFunctionParameter     : I32EnumAttrCase<"OpFunctionParameter", 55>;
def SPV_OC_OpFunctionEnd           : I32EnumAttrCase<"OpFunctionEnd", 56>;
def SPV_OC_OpVariable              : I32EnumAttrCase<"OpVariable", 59>;
def SPV_OC_OpLoad                  : I32EnumAttrCase<"OpLoad", 61>;
def SPV_OC_OpStore                 : I32EnumAttrCase<"OpStore", 62>;
def SPV_OC_OpAccessChain           : I32EnumAttrCase<"OpAccessChain", 65>;
def SPV_OC_OpDecorate              : I32EnumAttrCase<"OpDecorate", 71>;
def SPV_OC_OpCompositeExtract      : I32EnumAttrCase<"OpCompositeExtract", 81>;
def SPV_OC_OpIAdd                  : I32EnumAttrCase<"OpIAdd", 128>;
def SPV_OC_OpFAdd                  : I32EnumAttrCase<"OpFAdd", 129>;
def SPV_OC_OpISub                  : I32EnumAttrCase<"OpISub", 130>;
def SPV_OC_OpFSub                  : I32EnumAttrCase<"OpFSub", 131>;
def SPV_OC_OpIMul                  : I32EnumAttrCase<"OpIMul", 132>;
def SPV_OC_OpFMul                  : I32EnumAttrCase<"OpFMul", 133>;
def SPV_OC_OpUDiv                  : I32EnumAttrCase<"OpUDiv", 134>;
def SPV_OC_OpSDiv                  : I32EnumAttrCase<"OpSDiv", 135>;
def SPV_OC_OpFDiv                  : I32EnumAttrCase<"OpFDiv", 136>;
def SPV_OC_OpUMod                  : I32EnumAttrCase<"OpUMod", 137>;
def SPV_OC_OpSRem                  : I32EnumAttrCase<"OpSRem", 138>;
def SPV_OC_OpSMod                  : I32EnumAttrCase<"OpSMod", 139>;
def SPV_OC_OpFRem                  : I32EnumAttrCase<"OpFRem", 140>;
def SPV_OC_OpFMod                  : I32EnumAttrCase<"OpFMod", 141>;
def SPV_OC_OpIEqual                : I32EnumAttrCase<"OpIEqual", 170>;
def SPV_OC_OpINotEqual             : I32EnumAttrCase<"OpINotEqual", 171>;
def SPV_OC_OpUGreaterThan          : I32EnumAttrCase<"OpUGreaterThan", 172>;
def SPV_OC_OpSGreaterThan          : I32EnumAttrCase<"OpSGreaterThan", 173>;
def SPV_OC_OpUGreaterThanEqual     : I32EnumAttrCase<"OpUGreaterThanEqual", 174>;
def SPV_OC_OpSGreaterThanEqual     : I32EnumAttrCase<"OpSGreaterThanEqual", 175>;
def SPV_OC_OpULessThan             : I32EnumAttrCase<"OpULessThan", 176>;
def SPV_OC_OpSLessThan             : I32EnumAttrCase<"OpSLessThan", 177>;
def SPV_OC_OpULessThanEqual        : I32EnumAttrCase<"OpULessThanEqual", 178>;
def SPV_OC_OpSLessThanEqual        : I32EnumAttrCase<"OpSLessThanEqual", 179>;
def SPV_OC_OpReturn                : I32EnumAttrCase<"OpReturn", 253>;

def SPV_OpcodeAttr :
    I32EnumAttr<"Opcode", "valid SPIR-V instructions", [
      SPV_OC_OpNop, SPV_OC_OpName, SPV_OC_OpMemoryModel, SPV_OC_OpEntryPoint,
      SPV_OC_OpExecutionMode, SPV_OC_OpTypeVoid, SPV_OC_OpTypeBool, SPV_OC_OpTypeInt,
      SPV_OC_OpTypeFloat, SPV_OC_OpTypeVector, SPV_OC_OpTypeArray,
      SPV_OC_OpTypePointer, SPV_OC_OpTypeFunction, SPV_OC_OpConstantTrue,
      SPV_OC_OpConstantFalse, SPV_OC_OpConstant, SPV_OC_OpConstantComposite,
      SPV_OC_OpConstantNull, SPV_OC_OpSpecConstantTrue, SPV_OC_OpSpecConstantFalse,
      SPV_OC_OpSpecConstant, SPV_OC_OpSpecConstantComposite, SPV_OC_OpFunction,
      SPV_OC_OpFunctionParameter, SPV_OC_OpFunctionEnd, SPV_OC_OpVariable,
      SPV_OC_OpLoad, SPV_OC_OpStore, SPV_OC_OpAccessChain, SPV_OC_OpDecorate,
      SPV_OC_OpCompositeExtract, SPV_OC_OpIAdd, SPV_OC_OpFAdd, SPV_OC_OpISub,
      SPV_OC_OpFSub, SPV_OC_OpIMul, SPV_OC_OpFMul, SPV_OC_OpUDiv, SPV_OC_OpSDiv,
      SPV_OC_OpFDiv, SPV_OC_OpUMod, SPV_OC_OpSRem, SPV_OC_OpSMod, SPV_OC_OpFRem,
      SPV_OC_OpFMod, SPV_OC_OpIEqual, SPV_OC_OpINotEqual, SPV_OC_OpUGreaterThan,
      SPV_OC_OpSGreaterThan, SPV_OC_OpUGreaterThanEqual, SPV_OC_OpSGreaterThanEqual,
      SPV_OC_OpULessThan, SPV_OC_OpSLessThan, SPV_OC_OpULessThanEqual,
      SPV_OC_OpSLessThanEqual, SPV_OC_OpReturn
      ]> {
    let returnType = "::mlir::spirv::Opcode";
    let convertFromStorage = "static_cast<::mlir::spirv::Opcode>($_self.getInt())";
    let cppNamespace = "::mlir::spirv";
}

// End opcode section. Generated from SPIR-V spec; DO NOT MODIFY!


//===----------------------------------------------------------------------===//
// SPIR-V type definitions
//===----------------------------------------------------------------------===//

def SPV_IsPtrType : CPred<"$_self.isa<::mlir::spirv::PointerType>()">;
def SPV_IsArrayType : CPred<"$_self.isa<::mlir::spirv::ArrayType>()">;
def SPV_IsRTArrayType : CPred<"$_self.isa<::mlir::spirv::RuntimeArrayType>()">;
def SPV_IsStructType : CPred<"$_self.isa<::mlir::spirv::StructType>()">;

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
def SPV_AnyStruct : Type<SPV_IsStructType, "any SPIR-V struct type">;

def SPV_Numerical : AnyTypeOf<[SPV_Integer, SPV_Float]>;
def SPV_Scalar : AnyTypeOf<[SPV_Numerical, SPV_Bool]>;
def SPV_Aggregrate : AnyTypeOf<[SPV_AnyArray, SPV_AnyStruct]>;
def SPV_Composite: AnyTypeOf<[SPV_Vector, SPV_AnyArray, SPV_AnyStruct]>;
def SPV_Type : AnyTypeOf<[
    SPV_Void, SPV_Bool, SPV_Integer, SPV_Float, SPV_Vector,
    SPV_AnyPtr, SPV_AnyArray, SPV_AnyRTArray, SPV_AnyStruct
  ]>;

class SPV_ScalarOrVectorOf<Type type> :
    Type<Or<[type.predicate, VectorOf<[type]>.predicate]>,
         "scalar/vector of " # type.description>;

// TODO(antiagainst): Use a more appropriate way to model optional operands
class SPV_Optional<Type type> : Variadic<type>;

def SPV_IsEntryPointType :
    CPred<"$_self.isa<::mlir::spirv::EntryPointType>()">;
def SPV_EntryPoint : Type<SPV_IsEntryPointType, "SPIR-V entry point type">;

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

def SPV_BI_Position                    : I32EnumAttrCase<"Position", 0>;
def SPV_BI_PointSize                   : I32EnumAttrCase<"PointSize", 1>;
def SPV_BI_ClipDistance                : I32EnumAttrCase<"ClipDistance", 3>;
def SPV_BI_CullDistance                : I32EnumAttrCase<"CullDistance", 4>;
def SPV_BI_VertexId                    : I32EnumAttrCase<"VertexId", 5>;
def SPV_BI_InstanceId                  : I32EnumAttrCase<"InstanceId", 6>;
def SPV_BI_PrimitiveId                 : I32EnumAttrCase<"PrimitiveId", 7>;
def SPV_BI_InvocationId                : I32EnumAttrCase<"InvocationId", 8>;
def SPV_BI_Layer                       : I32EnumAttrCase<"Layer", 9>;
def SPV_BI_ViewportIndex               : I32EnumAttrCase<"ViewportIndex", 10>;
def SPV_BI_TessLevelOuter              : I32EnumAttrCase<"TessLevelOuter", 11>;
def SPV_BI_TessLevelInner              : I32EnumAttrCase<"TessLevelInner", 12>;
def SPV_BI_TessCoord                   : I32EnumAttrCase<"TessCoord", 13>;
def SPV_BI_PatchVertices               : I32EnumAttrCase<"PatchVertices", 14>;
def SPV_BI_FragCoord                   : I32EnumAttrCase<"FragCoord", 15>;
def SPV_BI_PointCoord                  : I32EnumAttrCase<"PointCoord", 16>;
def SPV_BI_FrontFacing                 : I32EnumAttrCase<"FrontFacing", 17>;
def SPV_BI_SampleId                    : I32EnumAttrCase<"SampleId", 18>;
def SPV_BI_SamplePosition              : I32EnumAttrCase<"SamplePosition", 19>;
def SPV_BI_SampleMask                  : I32EnumAttrCase<"SampleMask", 20>;
def SPV_BI_FragDepth                   : I32EnumAttrCase<"FragDepth", 22>;
def SPV_BI_HelperInvocation            : I32EnumAttrCase<"HelperInvocation", 23>;
def SPV_BI_NumWorkgroups               : I32EnumAttrCase<"NumWorkgroups", 24>;
def SPV_BI_WorkgroupSize               : I32EnumAttrCase<"WorkgroupSize", 25>;
def SPV_BI_WorkgroupId                 : I32EnumAttrCase<"WorkgroupId", 26>;
def SPV_BI_LocalInvocationId           : I32EnumAttrCase<"LocalInvocationId", 27>;
def SPV_BI_GlobalInvocationId          : I32EnumAttrCase<"GlobalInvocationId", 28>;
def SPV_BI_LocalInvocationIndex        : I32EnumAttrCase<"LocalInvocationIndex", 29>;
def SPV_BI_WorkDim                     : I32EnumAttrCase<"WorkDim", 30>;
def SPV_BI_GlobalSize                  : I32EnumAttrCase<"GlobalSize", 31>;
def SPV_BI_EnqueuedWorkgroupSize       : I32EnumAttrCase<"EnqueuedWorkgroupSize", 32>;
def SPV_BI_GlobalOffset                : I32EnumAttrCase<"GlobalOffset", 33>;
def SPV_BI_GlobalLinearId              : I32EnumAttrCase<"GlobalLinearId", 34>;
def SPV_BI_SubgroupSize                : I32EnumAttrCase<"SubgroupSize", 36>;
def SPV_BI_SubgroupMaxSize             : I32EnumAttrCase<"SubgroupMaxSize", 37>;
def SPV_BI_NumSubgroups                : I32EnumAttrCase<"NumSubgroups", 38>;
def SPV_BI_NumEnqueuedSubgroups        : I32EnumAttrCase<"NumEnqueuedSubgroups", 39>;
def SPV_BI_SubgroupId                  : I32EnumAttrCase<"SubgroupId", 40>;
def SPV_BI_SubgroupLocalInvocationId   : I32EnumAttrCase<"SubgroupLocalInvocationId", 41>;
def SPV_BI_VertexIndex                 : I32EnumAttrCase<"VertexIndex", 42>;
def SPV_BI_InstanceIndex               : I32EnumAttrCase<"InstanceIndex", 43>;
def SPV_BI_SubgroupEqMask              : I32EnumAttrCase<"SubgroupEqMask", 4416>;
def SPV_BI_SubgroupGeMask              : I32EnumAttrCase<"SubgroupGeMask", 4417>;
def SPV_BI_SubgroupGtMask              : I32EnumAttrCase<"SubgroupGtMask", 4418>;
def SPV_BI_SubgroupLeMask              : I32EnumAttrCase<"SubgroupLeMask", 4419>;
def SPV_BI_SubgroupLtMask              : I32EnumAttrCase<"SubgroupLtMask", 4420>;
def SPV_BI_BaseVertex                  : I32EnumAttrCase<"BaseVertex", 4424>;
def SPV_BI_BaseInstance                : I32EnumAttrCase<"BaseInstance", 4425>;
def SPV_BI_DrawIndex                   : I32EnumAttrCase<"DrawIndex", 4426>;
def SPV_BI_DeviceIndex                 : I32EnumAttrCase<"DeviceIndex", 4438>;
def SPV_BI_ViewIndex                   : I32EnumAttrCase<"ViewIndex", 4440>;
def SPV_BI_BaryCoordNoPerspAMD         : I32EnumAttrCase<"BaryCoordNoPerspAMD", 4992>;
def SPV_BI_BaryCoordNoPerspCentroidAMD : I32EnumAttrCase<"BaryCoordNoPerspCentroidAMD", 4993>;
def SPV_BI_BaryCoordNoPerspSampleAMD   : I32EnumAttrCase<"BaryCoordNoPerspSampleAMD", 4994>;
def SPV_BI_BaryCoordSmoothAMD          : I32EnumAttrCase<"BaryCoordSmoothAMD", 4995>;
def SPV_BI_BaryCoordSmoothCentroidAMD  : I32EnumAttrCase<"BaryCoordSmoothCentroidAMD", 4996>;
def SPV_BI_BaryCoordSmoothSampleAMD    : I32EnumAttrCase<"BaryCoordSmoothSampleAMD", 4997>;
def SPV_BI_BaryCoordPullModelAMD       : I32EnumAttrCase<"BaryCoordPullModelAMD", 4998>;
def SPV_BI_FragStencilRefEXT           : I32EnumAttrCase<"FragStencilRefEXT", 5014>;
def SPV_BI_ViewportMaskNV              : I32EnumAttrCase<"ViewportMaskNV", 5253>;
def SPV_BI_SecondaryPositionNV         : I32EnumAttrCase<"SecondaryPositionNV", 5257>;
def SPV_BI_SecondaryViewportMaskNV     : I32EnumAttrCase<"SecondaryViewportMaskNV", 5258>;
def SPV_BI_PositionPerViewNV           : I32EnumAttrCase<"PositionPerViewNV", 5261>;
def SPV_BI_ViewportMaskPerViewNV       : I32EnumAttrCase<"ViewportMaskPerViewNV", 5262>;
def SPV_BI_FullyCoveredEXT             : I32EnumAttrCase<"FullyCoveredEXT", 5264>;
def SPV_BI_TaskCountNV                 : I32EnumAttrCase<"TaskCountNV", 5274>;
def SPV_BI_PrimitiveCountNV            : I32EnumAttrCase<"PrimitiveCountNV", 5275>;
def SPV_BI_PrimitiveIndicesNV          : I32EnumAttrCase<"PrimitiveIndicesNV", 5276>;
def SPV_BI_ClipDistancePerViewNV       : I32EnumAttrCase<"ClipDistancePerViewNV", 5277>;
def SPV_BI_CullDistancePerViewNV       : I32EnumAttrCase<"CullDistancePerViewNV", 5278>;
def SPV_BI_LayerPerViewNV              : I32EnumAttrCase<"LayerPerViewNV", 5279>;
def SPV_BI_MeshViewCountNV             : I32EnumAttrCase<"MeshViewCountNV", 5280>;
def SPV_BI_MeshViewIndicesNV           : I32EnumAttrCase<"MeshViewIndicesNV", 5281>;
def SPV_BI_BaryCoordNV                 : I32EnumAttrCase<"BaryCoordNV", 5286>;
def SPV_BI_BaryCoordNoPerspNV          : I32EnumAttrCase<"BaryCoordNoPerspNV", 5287>;
def SPV_BI_FragSizeEXT                 : I32EnumAttrCase<"FragSizeEXT", 5292>;
def SPV_BI_FragInvocationCountEXT      : I32EnumAttrCase<"FragInvocationCountEXT", 5293>;
def SPV_BI_LaunchIdNV                  : I32EnumAttrCase<"LaunchIdNV", 5319>;
def SPV_BI_LaunchSizeNV                : I32EnumAttrCase<"LaunchSizeNV", 5320>;
def SPV_BI_WorldRayOriginNV            : I32EnumAttrCase<"WorldRayOriginNV", 5321>;
def SPV_BI_WorldRayDirectionNV         : I32EnumAttrCase<"WorldRayDirectionNV", 5322>;
def SPV_BI_ObjectRayOriginNV           : I32EnumAttrCase<"ObjectRayOriginNV", 5323>;
def SPV_BI_ObjectRayDirectionNV        : I32EnumAttrCase<"ObjectRayDirectionNV", 5324>;
def SPV_BI_RayTminNV                   : I32EnumAttrCase<"RayTminNV", 5325>;
def SPV_BI_RayTmaxNV                   : I32EnumAttrCase<"RayTmaxNV", 5326>;
def SPV_BI_InstanceCustomIndexNV       : I32EnumAttrCase<"InstanceCustomIndexNV", 5327>;
def SPV_BI_ObjectToWorldNV             : I32EnumAttrCase<"ObjectToWorldNV", 5330>;
def SPV_BI_WorldToObjectNV             : I32EnumAttrCase<"WorldToObjectNV", 5331>;
def SPV_BI_HitTNV                      : I32EnumAttrCase<"HitTNV", 5332>;
def SPV_BI_HitKindNV                   : I32EnumAttrCase<"HitKindNV", 5333>;
def SPV_BI_IncomingRayFlagsNV          : I32EnumAttrCase<"IncomingRayFlagsNV", 5351>;
def SPV_BI_WarpsPerSMNV                : I32EnumAttrCase<"WarpsPerSMNV", 5374>;
def SPV_BI_SMCountNV                   : I32EnumAttrCase<"SMCountNV", 5375>;
def SPV_BI_WarpIDNV                    : I32EnumAttrCase<"WarpIDNV", 5376>;
def SPV_BI_SMIDNV                      : I32EnumAttrCase<"SMIDNV", 5377>;

def SPV_BuiltInAttr :
    I32EnumAttr<"BuiltIn", "valid SPIR-V BuiltIn", [
      SPV_BI_Position, SPV_BI_PointSize, SPV_BI_ClipDistance, SPV_BI_CullDistance,
      SPV_BI_VertexId, SPV_BI_InstanceId, SPV_BI_PrimitiveId, SPV_BI_InvocationId,
      SPV_BI_Layer, SPV_BI_ViewportIndex, SPV_BI_TessLevelOuter,
      SPV_BI_TessLevelInner, SPV_BI_TessCoord, SPV_BI_PatchVertices,
      SPV_BI_FragCoord, SPV_BI_PointCoord, SPV_BI_FrontFacing, SPV_BI_SampleId,
      SPV_BI_SamplePosition, SPV_BI_SampleMask, SPV_BI_FragDepth,
      SPV_BI_HelperInvocation, SPV_BI_NumWorkgroups, SPV_BI_WorkgroupSize,
      SPV_BI_WorkgroupId, SPV_BI_LocalInvocationId, SPV_BI_GlobalInvocationId,
      SPV_BI_LocalInvocationIndex, SPV_BI_WorkDim, SPV_BI_GlobalSize,
      SPV_BI_EnqueuedWorkgroupSize, SPV_BI_GlobalOffset, SPV_BI_GlobalLinearId,
      SPV_BI_SubgroupSize, SPV_BI_SubgroupMaxSize, SPV_BI_NumSubgroups,
      SPV_BI_NumEnqueuedSubgroups, SPV_BI_SubgroupId,
      SPV_BI_SubgroupLocalInvocationId, SPV_BI_VertexIndex, SPV_BI_InstanceIndex,
      SPV_BI_SubgroupEqMask, SPV_BI_SubgroupGeMask, SPV_BI_SubgroupGtMask,
      SPV_BI_SubgroupLeMask, SPV_BI_SubgroupLtMask, SPV_BI_BaseVertex,
      SPV_BI_BaseInstance, SPV_BI_DrawIndex, SPV_BI_DeviceIndex, SPV_BI_ViewIndex,
      SPV_BI_BaryCoordNoPerspAMD, SPV_BI_BaryCoordNoPerspCentroidAMD,
      SPV_BI_BaryCoordNoPerspSampleAMD, SPV_BI_BaryCoordSmoothAMD,
      SPV_BI_BaryCoordSmoothCentroidAMD, SPV_BI_BaryCoordSmoothSampleAMD,
      SPV_BI_BaryCoordPullModelAMD, SPV_BI_FragStencilRefEXT, SPV_BI_ViewportMaskNV,
      SPV_BI_SecondaryPositionNV, SPV_BI_SecondaryViewportMaskNV,
      SPV_BI_PositionPerViewNV, SPV_BI_ViewportMaskPerViewNV, SPV_BI_FullyCoveredEXT,
      SPV_BI_TaskCountNV, SPV_BI_PrimitiveCountNV, SPV_BI_PrimitiveIndicesNV,
      SPV_BI_ClipDistancePerViewNV, SPV_BI_CullDistancePerViewNV,
      SPV_BI_LayerPerViewNV, SPV_BI_MeshViewCountNV, SPV_BI_MeshViewIndicesNV,
      SPV_BI_BaryCoordNV, SPV_BI_BaryCoordNoPerspNV, SPV_BI_FragSizeEXT,
      SPV_BI_FragInvocationCountEXT, SPV_BI_LaunchIdNV, SPV_BI_LaunchSizeNV,
      SPV_BI_WorldRayOriginNV, SPV_BI_WorldRayDirectionNV, SPV_BI_ObjectRayOriginNV,
      SPV_BI_ObjectRayDirectionNV, SPV_BI_RayTminNV, SPV_BI_RayTmaxNV,
      SPV_BI_InstanceCustomIndexNV, SPV_BI_ObjectToWorldNV, SPV_BI_WorldToObjectNV,
      SPV_BI_HitTNV, SPV_BI_HitKindNV, SPV_BI_IncomingRayFlagsNV,
      SPV_BI_WarpsPerSMNV, SPV_BI_SMCountNV, SPV_BI_WarpIDNV, SPV_BI_SMIDNV
    ]> {
  let returnType = "::mlir::spirv::BuiltIn";
  let convertFromStorage = "static_cast<::mlir::spirv::BuiltIn>($_self.getInt())";
  let cppNamespace = "::mlir::spirv";
}

def SPV_D_RelaxedPrecision            : I32EnumAttrCase<"RelaxedPrecision", 0>;
def SPV_D_SpecId                      : I32EnumAttrCase<"SpecId", 1>;
def SPV_D_Block                       : I32EnumAttrCase<"Block", 2>;
def SPV_D_BufferBlock                 : I32EnumAttrCase<"BufferBlock", 3>;
def SPV_D_RowMajor                    : I32EnumAttrCase<"RowMajor", 4>;
def SPV_D_ColMajor                    : I32EnumAttrCase<"ColMajor", 5>;
def SPV_D_ArrayStride                 : I32EnumAttrCase<"ArrayStride", 6>;
def SPV_D_MatrixStride                : I32EnumAttrCase<"MatrixStride", 7>;
def SPV_D_GLSLShared                  : I32EnumAttrCase<"GLSLShared", 8>;
def SPV_D_GLSLPacked                  : I32EnumAttrCase<"GLSLPacked", 9>;
def SPV_D_CPacked                     : I32EnumAttrCase<"CPacked", 10>;
def SPV_D_BuiltIn                     : I32EnumAttrCase<"BuiltIn", 11>;
def SPV_D_NoPerspective               : I32EnumAttrCase<"NoPerspective", 13>;
def SPV_D_Flat                        : I32EnumAttrCase<"Flat", 14>;
def SPV_D_Patch                       : I32EnumAttrCase<"Patch", 15>;
def SPV_D_Centroid                    : I32EnumAttrCase<"Centroid", 16>;
def SPV_D_Sample                      : I32EnumAttrCase<"Sample", 17>;
def SPV_D_Invariant                   : I32EnumAttrCase<"Invariant", 18>;
def SPV_D_Restrict                    : I32EnumAttrCase<"Restrict", 19>;
def SPV_D_Aliased                     : I32EnumAttrCase<"Aliased", 20>;
def SPV_D_Volatile                    : I32EnumAttrCase<"Volatile", 21>;
def SPV_D_Constant                    : I32EnumAttrCase<"Constant", 22>;
def SPV_D_Coherent                    : I32EnumAttrCase<"Coherent", 23>;
def SPV_D_NonWritable                 : I32EnumAttrCase<"NonWritable", 24>;
def SPV_D_NonReadable                 : I32EnumAttrCase<"NonReadable", 25>;
def SPV_D_Uniform                     : I32EnumAttrCase<"Uniform", 26>;
def SPV_D_UniformId                   : I32EnumAttrCase<"UniformId", 27>;
def SPV_D_SaturatedConversion         : I32EnumAttrCase<"SaturatedConversion", 28>;
def SPV_D_Stream                      : I32EnumAttrCase<"Stream", 29>;
def SPV_D_Location                    : I32EnumAttrCase<"Location", 30>;
def SPV_D_Component                   : I32EnumAttrCase<"Component", 31>;
def SPV_D_Index                       : I32EnumAttrCase<"Index", 32>;
def SPV_D_Binding                     : I32EnumAttrCase<"Binding", 33>;
def SPV_D_DescriptorSet               : I32EnumAttrCase<"DescriptorSet", 34>;
def SPV_D_Offset                      : I32EnumAttrCase<"Offset", 35>;
def SPV_D_XfbBuffer                   : I32EnumAttrCase<"XfbBuffer", 36>;
def SPV_D_XfbStride                   : I32EnumAttrCase<"XfbStride", 37>;
def SPV_D_FuncParamAttr               : I32EnumAttrCase<"FuncParamAttr", 38>;
def SPV_D_FPRoundingMode              : I32EnumAttrCase<"FPRoundingMode", 39>;
def SPV_D_FPFastMathMode              : I32EnumAttrCase<"FPFastMathMode", 40>;
def SPV_D_LinkageAttributes           : I32EnumAttrCase<"LinkageAttributes", 41>;
def SPV_D_NoContraction               : I32EnumAttrCase<"NoContraction", 42>;
def SPV_D_InputAttachmentIndex        : I32EnumAttrCase<"InputAttachmentIndex", 43>;
def SPV_D_Alignment                   : I32EnumAttrCase<"Alignment", 44>;
def SPV_D_MaxByteOffset               : I32EnumAttrCase<"MaxByteOffset", 45>;
def SPV_D_AlignmentId                 : I32EnumAttrCase<"AlignmentId", 46>;
def SPV_D_MaxByteOffsetId             : I32EnumAttrCase<"MaxByteOffsetId", 47>;
def SPV_D_NoSignedWrap                : I32EnumAttrCase<"NoSignedWrap", 4469>;
def SPV_D_NoUnsignedWrap              : I32EnumAttrCase<"NoUnsignedWrap", 4470>;
def SPV_D_ExplicitInterpAMD           : I32EnumAttrCase<"ExplicitInterpAMD", 4999>;
def SPV_D_OverrideCoverageNV          : I32EnumAttrCase<"OverrideCoverageNV", 5248>;
def SPV_D_PassthroughNV               : I32EnumAttrCase<"PassthroughNV", 5250>;
def SPV_D_ViewportRelativeNV          : I32EnumAttrCase<"ViewportRelativeNV", 5252>;
def SPV_D_SecondaryViewportRelativeNV : I32EnumAttrCase<"SecondaryViewportRelativeNV", 5256>;
def SPV_D_PerPrimitiveNV              : I32EnumAttrCase<"PerPrimitiveNV", 5271>;
def SPV_D_PerViewNV                   : I32EnumAttrCase<"PerViewNV", 5272>;
def SPV_D_PerTaskNV                   : I32EnumAttrCase<"PerTaskNV", 5273>;
def SPV_D_PerVertexNV                 : I32EnumAttrCase<"PerVertexNV", 5285>;
def SPV_D_NonUniformEXT               : I32EnumAttrCase<"NonUniformEXT", 5300>;
def SPV_D_RestrictPointerEXT          : I32EnumAttrCase<"RestrictPointerEXT", 5355>;
def SPV_D_AliasedPointerEXT           : I32EnumAttrCase<"AliasedPointerEXT", 5356>;
def SPV_D_CounterBuffer               : I32EnumAttrCase<"CounterBuffer", 5634>;
def SPV_D_UserSemantic                : I32EnumAttrCase<"UserSemantic", 5635>;
def SPV_D_UserTypeGOOGLE              : I32EnumAttrCase<"UserTypeGOOGLE", 5636>;

def SPV_DecorationAttr :
    I32EnumAttr<"Decoration", "valid SPIR-V Decoration", [
      SPV_D_RelaxedPrecision, SPV_D_SpecId, SPV_D_Block, SPV_D_BufferBlock,
      SPV_D_RowMajor, SPV_D_ColMajor, SPV_D_ArrayStride, SPV_D_MatrixStride,
      SPV_D_GLSLShared, SPV_D_GLSLPacked, SPV_D_CPacked, SPV_D_BuiltIn,
      SPV_D_NoPerspective, SPV_D_Flat, SPV_D_Patch, SPV_D_Centroid, SPV_D_Sample,
      SPV_D_Invariant, SPV_D_Restrict, SPV_D_Aliased, SPV_D_Volatile, SPV_D_Constant,
      SPV_D_Coherent, SPV_D_NonWritable, SPV_D_NonReadable, SPV_D_Uniform,
      SPV_D_UniformId, SPV_D_SaturatedConversion, SPV_D_Stream, SPV_D_Location,
      SPV_D_Component, SPV_D_Index, SPV_D_Binding, SPV_D_DescriptorSet, SPV_D_Offset,
      SPV_D_XfbBuffer, SPV_D_XfbStride, SPV_D_FuncParamAttr, SPV_D_FPRoundingMode,
      SPV_D_FPFastMathMode, SPV_D_LinkageAttributes, SPV_D_NoContraction,
      SPV_D_InputAttachmentIndex, SPV_D_Alignment, SPV_D_MaxByteOffset,
      SPV_D_AlignmentId, SPV_D_MaxByteOffsetId, SPV_D_NoSignedWrap,
      SPV_D_NoUnsignedWrap, SPV_D_ExplicitInterpAMD, SPV_D_OverrideCoverageNV,
      SPV_D_PassthroughNV, SPV_D_ViewportRelativeNV,
      SPV_D_SecondaryViewportRelativeNV, SPV_D_PerPrimitiveNV, SPV_D_PerViewNV,
      SPV_D_PerTaskNV, SPV_D_PerVertexNV, SPV_D_NonUniformEXT,
      SPV_D_RestrictPointerEXT, SPV_D_AliasedPointerEXT, SPV_D_CounterBuffer,
      SPV_D_UserSemantic, SPV_D_UserTypeGOOGLE
    ]> {
  let returnType = "::mlir::spirv::Decoration";
  let convertFromStorage = "static_cast<::mlir::spirv::Decoration>($_self.getInt())";
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

def SPV_EM_Invocations                      : I32EnumAttrCase<"Invocations", 0>;
def SPV_EM_SpacingEqual                     : I32EnumAttrCase<"SpacingEqual", 1>;
def SPV_EM_SpacingFractionalEven            : I32EnumAttrCase<"SpacingFractionalEven", 2>;
def SPV_EM_SpacingFractionalOdd             : I32EnumAttrCase<"SpacingFractionalOdd", 3>;
def SPV_EM_VertexOrderCw                    : I32EnumAttrCase<"VertexOrderCw", 4>;
def SPV_EM_VertexOrderCcw                   : I32EnumAttrCase<"VertexOrderCcw", 5>;
def SPV_EM_PixelCenterInteger               : I32EnumAttrCase<"PixelCenterInteger", 6>;
def SPV_EM_OriginUpperLeft                  : I32EnumAttrCase<"OriginUpperLeft", 7>;
def SPV_EM_OriginLowerLeft                  : I32EnumAttrCase<"OriginLowerLeft", 8>;
def SPV_EM_EarlyFragmentTests               : I32EnumAttrCase<"EarlyFragmentTests", 9>;
def SPV_EM_PointMode                        : I32EnumAttrCase<"PointMode", 10>;
def SPV_EM_Xfb                              : I32EnumAttrCase<"Xfb", 11>;
def SPV_EM_DepthReplacing                   : I32EnumAttrCase<"DepthReplacing", 12>;
def SPV_EM_DepthGreater                     : I32EnumAttrCase<"DepthGreater", 14>;
def SPV_EM_DepthLess                        : I32EnumAttrCase<"DepthLess", 15>;
def SPV_EM_DepthUnchanged                   : I32EnumAttrCase<"DepthUnchanged", 16>;
def SPV_EM_LocalSize                        : I32EnumAttrCase<"LocalSize", 17>;
def SPV_EM_LocalSizeHint                    : I32EnumAttrCase<"LocalSizeHint", 18>;
def SPV_EM_InputPoints                      : I32EnumAttrCase<"InputPoints", 19>;
def SPV_EM_InputLines                       : I32EnumAttrCase<"InputLines", 20>;
def SPV_EM_InputLinesAdjacency              : I32EnumAttrCase<"InputLinesAdjacency", 21>;
def SPV_EM_Triangles                        : I32EnumAttrCase<"Triangles", 22>;
def SPV_EM_InputTrianglesAdjacency          : I32EnumAttrCase<"InputTrianglesAdjacency", 23>;
def SPV_EM_Quads                            : I32EnumAttrCase<"Quads", 24>;
def SPV_EM_Isolines                         : I32EnumAttrCase<"Isolines", 25>;
def SPV_EM_OutputVertices                   : I32EnumAttrCase<"OutputVertices", 26>;
def SPV_EM_OutputPoints                     : I32EnumAttrCase<"OutputPoints", 27>;
def SPV_EM_OutputLineStrip                  : I32EnumAttrCase<"OutputLineStrip", 28>;
def SPV_EM_OutputTriangleStrip              : I32EnumAttrCase<"OutputTriangleStrip", 29>;
def SPV_EM_VecTypeHint                      : I32EnumAttrCase<"VecTypeHint", 30>;
def SPV_EM_ContractionOff                   : I32EnumAttrCase<"ContractionOff", 31>;
def SPV_EM_Initializer                      : I32EnumAttrCase<"Initializer", 33>;
def SPV_EM_Finalizer                        : I32EnumAttrCase<"Finalizer", 34>;
def SPV_EM_SubgroupSize                     : I32EnumAttrCase<"SubgroupSize", 35>;
def SPV_EM_SubgroupsPerWorkgroup            : I32EnumAttrCase<"SubgroupsPerWorkgroup", 36>;
def SPV_EM_SubgroupsPerWorkgroupId          : I32EnumAttrCase<"SubgroupsPerWorkgroupId", 37>;
def SPV_EM_LocalSizeId                      : I32EnumAttrCase<"LocalSizeId", 38>;
def SPV_EM_LocalSizeHintId                  : I32EnumAttrCase<"LocalSizeHintId", 39>;
def SPV_EM_PostDepthCoverage                : I32EnumAttrCase<"PostDepthCoverage", 4446>;
def SPV_EM_DenormPreserve                   : I32EnumAttrCase<"DenormPreserve", 4459>;
def SPV_EM_DenormFlushToZero                : I32EnumAttrCase<"DenormFlushToZero", 4460>;
def SPV_EM_SignedZeroInfNanPreserve         : I32EnumAttrCase<"SignedZeroInfNanPreserve", 4461>;
def SPV_EM_RoundingModeRTE                  : I32EnumAttrCase<"RoundingModeRTE", 4462>;
def SPV_EM_RoundingModeRTZ                  : I32EnumAttrCase<"RoundingModeRTZ", 4463>;
def SPV_EM_StencilRefReplacingEXT           : I32EnumAttrCase<"StencilRefReplacingEXT", 5027>;
def SPV_EM_OutputLinesNV                    : I32EnumAttrCase<"OutputLinesNV", 5269>;
def SPV_EM_OutputPrimitivesNV               : I32EnumAttrCase<"OutputPrimitivesNV", 5270>;
def SPV_EM_DerivativeGroupQuadsNV           : I32EnumAttrCase<"DerivativeGroupQuadsNV", 5289>;
def SPV_EM_DerivativeGroupLinearNV          : I32EnumAttrCase<"DerivativeGroupLinearNV", 5290>;
def SPV_EM_OutputTrianglesNV                : I32EnumAttrCase<"OutputTrianglesNV", 5298>;
def SPV_EM_PixelInterlockOrderedEXT         : I32EnumAttrCase<"PixelInterlockOrderedEXT", 5366>;
def SPV_EM_PixelInterlockUnorderedEXT       : I32EnumAttrCase<"PixelInterlockUnorderedEXT", 5367>;
def SPV_EM_SampleInterlockOrderedEXT        : I32EnumAttrCase<"SampleInterlockOrderedEXT", 5368>;
def SPV_EM_SampleInterlockUnorderedEXT      : I32EnumAttrCase<"SampleInterlockUnorderedEXT", 5369>;
def SPV_EM_ShadingRateInterlockOrderedEXT   : I32EnumAttrCase<"ShadingRateInterlockOrderedEXT", 5370>;
def SPV_EM_ShadingRateInterlockUnorderedEXT : I32EnumAttrCase<"ShadingRateInterlockUnorderedEXT", 5371>;

def SPV_ExecutionModeAttr :
    I32EnumAttr<"ExecutionMode", "valid SPIR-V ExecutionMode", [
      SPV_EM_Invocations, SPV_EM_SpacingEqual, SPV_EM_SpacingFractionalEven,
      SPV_EM_SpacingFractionalOdd, SPV_EM_VertexOrderCw, SPV_EM_VertexOrderCcw,
      SPV_EM_PixelCenterInteger, SPV_EM_OriginUpperLeft, SPV_EM_OriginLowerLeft,
      SPV_EM_EarlyFragmentTests, SPV_EM_PointMode, SPV_EM_Xfb, SPV_EM_DepthReplacing,
      SPV_EM_DepthGreater, SPV_EM_DepthLess, SPV_EM_DepthUnchanged, SPV_EM_LocalSize,
      SPV_EM_LocalSizeHint, SPV_EM_InputPoints, SPV_EM_InputLines,
      SPV_EM_InputLinesAdjacency, SPV_EM_Triangles, SPV_EM_InputTrianglesAdjacency,
      SPV_EM_Quads, SPV_EM_Isolines, SPV_EM_OutputVertices, SPV_EM_OutputPoints,
      SPV_EM_OutputLineStrip, SPV_EM_OutputTriangleStrip, SPV_EM_VecTypeHint,
      SPV_EM_ContractionOff, SPV_EM_Initializer, SPV_EM_Finalizer,
      SPV_EM_SubgroupSize, SPV_EM_SubgroupsPerWorkgroup,
      SPV_EM_SubgroupsPerWorkgroupId, SPV_EM_LocalSizeId, SPV_EM_LocalSizeHintId,
      SPV_EM_PostDepthCoverage, SPV_EM_DenormPreserve, SPV_EM_DenormFlushToZero,
      SPV_EM_SignedZeroInfNanPreserve, SPV_EM_RoundingModeRTE,
      SPV_EM_RoundingModeRTZ, SPV_EM_StencilRefReplacingEXT, SPV_EM_OutputLinesNV,
      SPV_EM_OutputPrimitivesNV, SPV_EM_DerivativeGroupQuadsNV,
      SPV_EM_DerivativeGroupLinearNV, SPV_EM_OutputTrianglesNV,
      SPV_EM_PixelInterlockOrderedEXT, SPV_EM_PixelInterlockUnorderedEXT,
      SPV_EM_SampleInterlockOrderedEXT, SPV_EM_SampleInterlockUnorderedEXT,
      SPV_EM_ShadingRateInterlockOrderedEXT, SPV_EM_ShadingRateInterlockUnorderedEXT
    ]> {
  let returnType = "::mlir::spirv::ExecutionMode";
  let convertFromStorage = "static_cast<::mlir::spirv::ExecutionMode>($_self.getInt())";
  let cppNamespace = "::mlir::spirv";
}

def SPV_EM_Vertex                 : I32EnumAttrCase<"Vertex", 0>;
def SPV_EM_TessellationControl    : I32EnumAttrCase<"TessellationControl", 1>;
def SPV_EM_TessellationEvaluation : I32EnumAttrCase<"TessellationEvaluation", 2>;
def SPV_EM_Geometry               : I32EnumAttrCase<"Geometry", 3>;
def SPV_EM_Fragment               : I32EnumAttrCase<"Fragment", 4>;
def SPV_EM_GLCompute              : I32EnumAttrCase<"GLCompute", 5>;
def SPV_EM_Kernel                 : I32EnumAttrCase<"Kernel", 6>;
def SPV_EM_TaskNV                 : I32EnumAttrCase<"TaskNV", 5267>;
def SPV_EM_MeshNV                 : I32EnumAttrCase<"MeshNV", 5268>;
def SPV_EM_RayGenerationNV        : I32EnumAttrCase<"RayGenerationNV", 5313>;
def SPV_EM_IntersectionNV         : I32EnumAttrCase<"IntersectionNV", 5314>;
def SPV_EM_AnyHitNV               : I32EnumAttrCase<"AnyHitNV", 5315>;
def SPV_EM_ClosestHitNV           : I32EnumAttrCase<"ClosestHitNV", 5316>;
def SPV_EM_MissNV                 : I32EnumAttrCase<"MissNV", 5317>;
def SPV_EM_CallableNV             : I32EnumAttrCase<"CallableNV", 5318>;

def SPV_ExecutionModelAttr :
    I32EnumAttr<"ExecutionModel", "valid SPIR-V ExecutionModel", [
      SPV_EM_Vertex, SPV_EM_TessellationControl, SPV_EM_TessellationEvaluation,
      SPV_EM_Geometry, SPV_EM_Fragment, SPV_EM_GLCompute, SPV_EM_Kernel,
      SPV_EM_TaskNV, SPV_EM_MeshNV, SPV_EM_RayGenerationNV, SPV_EM_IntersectionNV,
      SPV_EM_AnyHitNV, SPV_EM_ClosestHitNV, SPV_EM_MissNV, SPV_EM_CallableNV
    ]> {
  let returnType = "::mlir::spirv::ExecutionModel";
  let convertFromStorage = "static_cast<::mlir::spirv::ExecutionModel>($_self.getInt())";
  let cppNamespace = "::mlir::spirv";
}

def SPV_FC_None       : I32EnumAttrCase<"None", 0x0000>;
def SPV_FC_Inline     : I32EnumAttrCase<"Inline", 0x0001>;
def SPV_FC_DontInline : I32EnumAttrCase<"DontInline", 0x0002>;
def SPV_FC_Pure       : I32EnumAttrCase<"Pure", 0x0004>;
def SPV_FC_Const      : I32EnumAttrCase<"Const", 0x0008>;

def SPV_FunctionControlAttr :
    I32EnumAttr<"FunctionControl", "valid SPIR-V FunctionControl", [
      SPV_FC_None, SPV_FC_Inline, SPV_FC_DontInline, SPV_FC_Pure, SPV_FC_Const
    ]> {
  let returnType = "::mlir::spirv::FunctionControl";
  let convertFromStorage = "static_cast<::mlir::spirv::FunctionControl>($_self.getInt())";
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

def SPV_LT_Export : I32EnumAttrCase<"Export", 0>;
def SPV_LT_Import : I32EnumAttrCase<"Import", 1>;

def SPV_LinkageTypeAttr :
    I32EnumAttr<"LinkageType", "valid SPIR-V LinkageType", [
      SPV_LT_Export, SPV_LT_Import
    ]> {
  let returnType = "::mlir::spirv::LinkageType";
  let convertFromStorage = "static_cast<::mlir::spirv::LinkageType>($_self.getInt())";
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
// SPIR-V OpTrait definitions
//===----------------------------------------------------------------------===//

// Check that an op can only be used with SPIR-V ModuleOp
def IsModuleOnlyPred :
  CPred<"llvm::isa_and_nonnull<spirv::ModuleOp>($_op.getParentOp())">;

def ModuleOnly :
  PredOpTrait<"op can only be used in a 'spv.module' block", IsModuleOnlyPred>;

//===----------------------------------------------------------------------===//
// SPIR-V op definitions
//===----------------------------------------------------------------------===//

// Base class for all SPIR-V ops.
class SPV_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<SPV_Dialect, mnemonic, traits> {

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

  // Specifies whether this op has a direct corresponding SPIR-V binary
  // instruction opcode. The (de)serializer use this field to determine whether
  // to auto-generate an entry in the (de)serialization dispatch table for this
  // op. If set, this field also futher enables `autogenSerialization` (see
  // below for details).
  bit hasOpcode = 1;

  // Name of the corresponding SPIR-V op. Only valid to use when hasOpcode is 1.
  string spirvOpName = "Op" # mnemonic;

  // Controls whether to auto-generate this op's (de)serialization method.
  // If set, it results in generation of the following methods:
  //
  // ```c++
  // template<typename OpTy> Serializer::processOp(OpTy op);
  // template<typename OpTy> Deserializer::processOp(ArrayRef<uint32_t>);
  // ```
  //
  // If this field is not set, then manual implementation of a specialization of
  // these methods is required.
  //
  // Note:
  //
  // 1) If hasOpcode is set but autogenSerialization is not set, the
  //    (de)serializer dispatch method still calls the above method for
  //    (de)serializing this op.
  //
  // 2) If hasOpcode is not set, then this field is not interpreted; this op's
  //    (de)serialization method will not be auto-generated regardless. Neither
  //    does the handling in the (de)serialization dispatch table. Both
  //    (de)serializing this op and its dispatch should be handled manually.
  bit autogenSerialization = 1;
}

class SPV_BinaryOp<string mnemonic, Type resultType, Type operandsType,
                   list<OpTrait> traits = []> :
      SPV_Op<mnemonic, traits> {
  let arguments = (ins
    SPV_ScalarOrVectorOf<operandsType>:$operand1,
    SPV_ScalarOrVectorOf<operandsType>:$operand2
  );
  let results = (outs
    SPV_ScalarOrVectorOf<resultType>:$result
  );
  let parser = [{ return impl::parseBinaryOp(parser, result); }];
  let printer = [{ return impl::printBinaryOp(getOperation(), p); }];
  // No additional verification needed in addition to the ODS-generated ones.
  let verifier = [{ return success(); }];
}

class SPV_ArithmeticOp<string mnemonic, Type type,
                       list<OpTrait> traits = []> :
      // Operands type same as result type.
      SPV_BinaryOp<mnemonic, type, type,
                   !listconcat(traits,
                               [NoSideEffect, SameOperandsAndResultType])> {
}

class SPV_LogicalOp<string mnemonic, Type operandsType,
                    list<OpTrait> traits = []> :
      // Result type is SPV_Bool.
      SPV_BinaryOp<mnemonic, SPV_Bool, operandsType,
                   !listconcat(traits,
                               [NoSideEffect, SameTypeOperands,
                                SameOperandsAndResultShape])> {
  let parser = [{ return ::parseBinaryLogicalOp(parser, result); }];
  let printer = [{ return ::printBinaryLogicalOp(getOperation(), p); }];
}


#endif // SPIRV_BASE
