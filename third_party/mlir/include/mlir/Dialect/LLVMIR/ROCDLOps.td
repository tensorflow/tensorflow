//===-- ROCDLOps.td - ROCDL IR dialect op definition file --*- tablegen -*-===//
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
// This is the ROCDL IR operation definition file.
//
//===----------------------------------------------------------------------===//

#ifndef ROCDLIR_OPS
#define ROCDLIR_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

//===----------------------------------------------------------------------===//
// ROCDL dialect definitions
//===----------------------------------------------------------------------===//

def ROCDL_Dialect : Dialect {
  let name = "rocdl";
  let cppNamespace = "ROCDL";
}

//===----------------------------------------------------------------------===//
// ROCDL op definitions
//===----------------------------------------------------------------------===//

class ROCDL_Op<string mnemonic, list<OpTrait> traits = []> :
  LLVM_OpBase<ROCDL_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// ROCDL special register op definitions
//===----------------------------------------------------------------------===//

class ROCDL_SpecialRegisterOp<string mnemonic,
    list<OpTrait> traits = []> :
  ROCDL_Op<mnemonic, !listconcat(traits, [NoSideEffect])>,
  Results<(outs LLVM_Type:$res)>, Arguments<(ins)> {
  string llvmBuilder = "$res = createIntrinsicCall(builder,"
    # "llvm::Intrinsic::amdgcn_" # !subst(".","_", mnemonic) # ");";
  let parser = [{ return parseROCDLOp(parser, result); }];
  let printer = [{ printROCDLOp(p, this->getOperation()); }];
}

class ROCDL_DeviceFunctionOp<string mnemonic, string device_function,
                             int parameter, list<OpTrait> traits = []> :
  ROCDL_Op<mnemonic, !listconcat(traits, [NoSideEffect])>,
  Results<(outs LLVM_Type:$res)>, Arguments<(ins)> {
  string llvmBuilder = "$res = createDeviceFunctionCall(builder, \""
  # device_function # "\", " # parameter # ");";
  let parser = [{ return parseROCDLOp(parser, result); }];
  let printer = [{ printROCDLOp(p, this->getOperation()); }];
}

//===----------------------------------------------------------------------===//
// Thread index and Block index

def ROCDL_ThreadIdXOp : ROCDL_SpecialRegisterOp<"workitem.id.x">;
def ROCDL_ThreadIdYOp : ROCDL_SpecialRegisterOp<"workitem.id.y">;
def ROCDL_ThreadIdZOp : ROCDL_SpecialRegisterOp<"workitem.id.z">;

def ROCDL_BlockIdXOp : ROCDL_SpecialRegisterOp<"workgroup.id.x">;
def ROCDL_BlockIdYOp : ROCDL_SpecialRegisterOp<"workgroup.id.y">;
def ROCDL_BlockIdZOp : ROCDL_SpecialRegisterOp<"workgroup.id.z">;

//===----------------------------------------------------------------------===//
// Thread range and Block range

def ROCDL_BlockDimXOp : ROCDL_DeviceFunctionOp<"workgroup.dim.x",
                                               "__ockl_get_local_size", 0>;

def ROCDL_BlockDimYOp : ROCDL_DeviceFunctionOp<"workgroup.dim.y",
                                               "__ockl_get_local_size", 1>;

def ROCDL_BlockDimZOp : ROCDL_DeviceFunctionOp<"workgroup.dim.z",
                                               "__ockl_get_local_size", 2>;

def ROCDL_GridDimXOp : ROCDL_DeviceFunctionOp<"grid.dim.x",
                                               "__ockl_get_global_size", 0>;

def ROCDL_GridDimYOp : ROCDL_DeviceFunctionOp<"grid.dim.y",
                                               "__ockl_get_global_size", 1>;

def ROCDL_GridDimZOp : ROCDL_DeviceFunctionOp<"grid.dim.z",
                                               "__ockl_get_global_size", 2>;


#endif // ROCDLIR_OPS
