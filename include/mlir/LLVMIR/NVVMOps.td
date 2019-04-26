//===-- NVVMOps.td - NVVM IR dialect op definition file ----*- tablegen -*-===//
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
// This is the NVVM IR operation definition file.
//
//===----------------------------------------------------------------------===//

#ifdef NVVMIR_OPS
#else
#define NVVMIR_OPS

include "mlir/LLVMIR/LLVMOpBase.td"

class NVVM_Op<string mnemonic, list<OpTrait> traits = []> :
  LLVM_OpBase<!strconcat("nvvm.", mnemonic), traits> {
}

class NVVM_SpecialRegisterOp<string mnemonic,
    list<OpTrait> traits = []> :
  NVVM_Op<mnemonic, !listconcat(traits, [NoSideEffect])>,
  Results<(outs LLVM_Type:$res)>, Arguments<(ins)> {
  string llvmBuilder = "createIntrinsicCall(builder, llvm::Intrinsic::nvvm_"
    # !subst(".","_", mnemonic) # ");";
  let parser = [{ return parseNVVMSpecialRegisterOp(parser, result); }];
  let printer = [{ printNVVMSpecialRegisterOp(p, this->getOperation()); }];
}

def NVVM_ThreadIdXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.tid.x">;
def NVVM_ThreadIdYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.tid.y">;
def NVVM_ThreadIdZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.tid.z">;
def NVVM_ThreadDimXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ntid.x">;
def NVVM_ThreadDimYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ntid.y">;
def NVVM_ThreadDimZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ntid.z">;
def NVVM_BlockIdXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ctaid.x">;
def NVVM_BlockIdYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ctaid.y">;
def NVVM_BlockIdZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ctaid.z">;
def NVVM_BlockDimXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.nctaid.x">;
def NVVM_BlockDimYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.nctaid.y">;
def NVVM_BlockDimZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.nctaid.z">;

#endif // NVVMIR_OPS