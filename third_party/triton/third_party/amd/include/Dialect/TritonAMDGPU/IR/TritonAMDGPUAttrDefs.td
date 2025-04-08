/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITON_AMDGPU_ATTRDEFS
#define TRITON_AMDGPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "TritonAMDGPUDialect.td"
include "mlir/IR/EnumAttr.td"

class TritonAMDGPU_Attr<string name, list<Trait> traits = [],
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<TritonAMDGPU_Dialect, name, traits, baseCppClass> {
}

def TritonAMDGPU_OpIdxAttr : TritonAMDGPU_Attr<"OpIdx"> {
  let cppNamespace = "::mlir::triton::amdgpu";
  let mnemonic = "OpIdx";
  let summary = "An operand index attribute.";
  let description = [{
    The attribute is a way to describe which input argument of the target
    operation (e.g., `tt.dot`) the result of a given operation belongs to.
  }];

  let parameters = (ins "uint32_t":$value);
  let assemblyFormat = "`<` $value `>`";
}

def TritonAMDGPU_InstCounter : TritonAMDGPU_Attr<"InstCounter"> {
  let cppNamespace = "::mlir::triton::amdgpu";
  let mnemonic = "InstCounter";
  let summary = "An instruction counter attribute.";
  let description = [{
    The attribute holds the number of issued LLVM instructions of a specific kind as well as
    the data type.
  }];

  let parameters = (ins "uint32_t":$value, "Type":$type);
  let assemblyFormat = "`<` params `>`";
}

class TritonAMDGPU_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::amdgpu";
}

class TritonAMDGPU_I32EnumAttr<string mnemonic, TritonAMDGPU_I32Enum enumInfo> :
    EnumAttr<TritonAMDGPU_Dialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
  let cppNamespace = "::mlir::triton::amdgpu";
}

def SchedHintCaseNone : I32EnumAttrCase<"none", 0>;
def SchedHintCaseLLVMIglp0 : I32EnumAttrCase<"llvm_iglp_0", 1>;
def SchedHintCaseLLVMIglp1 : I32EnumAttrCase<"llvm_iglp_1", 2>;
def SchedHintCaseLocalPrefetch : I32EnumAttrCase<"local_prefetch", 3>;

def TritonAMDGPU_SchedHintsEnum : TritonAMDGPU_I32Enum<
  "SchedHint", "Instruction Scheduling Hints for AMD GPUs", [
    SchedHintCaseNone,
    SchedHintCaseLLVMIglp0,
    SchedHintCaseLLVMIglp1,
    SchedHintCaseLocalPrefetch,
  ]>;

def TritonAMDGPU_SchedHintVariantAttr :
  TritonAMDGPU_I32EnumAttr<"SchedHintVariant", TritonAMDGPU_SchedHintsEnum>;

#endif
