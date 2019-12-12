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

#ifndef NVVMIR_OPS
#define NVVMIR_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

//===----------------------------------------------------------------------===//
// NVVM dialect definitions
//===----------------------------------------------------------------------===//

def NVVM_Dialect : Dialect {
  let name = "nvvm";
  let cppNamespace = "NVVM";
}

//===----------------------------------------------------------------------===//
// NVVM op definitions
//===----------------------------------------------------------------------===//

class NVVM_Op<string mnemonic, list<OpTrait> traits = []> :
  LLVM_OpBase<NVVM_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// NVVM special register op definitions
//===----------------------------------------------------------------------===//

class NVVM_SpecialRegisterOp<string mnemonic,
    list<OpTrait> traits = []> :
  NVVM_Op<mnemonic, !listconcat(traits, [NoSideEffect])>,
  Results<(outs LLVM_Type:$res)>, Arguments<(ins)> {
  string llvmBuilder = "$res = createIntrinsicCall(builder,"
    # "llvm::Intrinsic::nvvm_" # !subst(".","_", mnemonic) # ");";
  let parser = [{ return parseNVVMSpecialRegisterOp(parser, result); }];
  let printer = [{ printNVVMIntrinsicOp(p, this->getOperation()); }];
}

//===----------------------------------------------------------------------===//
// Lane index and range
def NVVM_LaneIdOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.laneid">;
def NVVM_WarpSizeOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.warpsize">;

//===----------------------------------------------------------------------===//
// Thread index and range
def NVVM_ThreadIdXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.tid.x">;
def NVVM_ThreadIdYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.tid.y">;
def NVVM_ThreadIdZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.tid.z">;
def NVVM_BlockDimXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ntid.x">;
def NVVM_BlockDimYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ntid.y">;
def NVVM_BlockDimZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ntid.z">;

//===----------------------------------------------------------------------===//
// Block index and range
def NVVM_BlockIdXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ctaid.x">;
def NVVM_BlockIdYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ctaid.y">;
def NVVM_BlockIdZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ctaid.z">;
def NVVM_GridDimXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.nctaid.x">;
def NVVM_GridDimYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.nctaid.y">;
def NVVM_GridDimZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.nctaid.z">;

//===----------------------------------------------------------------------===//
// NVVM synchronization op definitions
//===----------------------------------------------------------------------===//

def NVVM_Barrier0Op : NVVM_Op<"barrier0"> {
  string llvmBuilder = [{
      createIntrinsicCall(builder, llvm::Intrinsic::nvvm_barrier0);
  }];
  let parser = [{ return success(); }];
  let printer = [{ printNVVMIntrinsicOp(p, this->getOperation()); }];
}

def NVVM_ShflDownOp :
  NVVM_Op<"shfl.sync.down">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_Type:$dst,
                 LLVM_Type:$val,
                 LLVM_Type:$offset,
                 LLVM_Type:$mask_and_clamp,
                 OptionalAttr<UnitAttr>:$return_value_and_is_valid)> {
  string llvmBuilder = [{
      auto intId = getShflDownIntrinsicId(
          $_resultType, static_cast<bool>($return_value_and_is_valid));
      $res = createIntrinsicCall(builder,
          intId, {$dst, $val, $offset, $mask_and_clamp});
  }];
  let parser = [{ return parseNVVMShflSyncDownOp(parser, result); }];
  let printer = [{ printNVVMIntrinsicOp(p, this->getOperation()); }];
  let verifier = [{
    if (!getAttrOfType<UnitAttr>("return_value_and_is_valid"))
      return success();
    auto type = getType().cast<LLVM::LLVMType>();
    if (!type.isStructTy() || type.getStructNumElements() != 2 ||
        !type.getStructElementType(1).isIntegerTy(
            /*Bitwidth=*/1))
      return emitError("expected return type !llvm<\"{ ?, i1 }\">");
    return success();
  }];
}

def NVVM_VoteBallotOp :
  NVVM_Op<"vote.ballot.sync">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_Type:$mask, LLVM_Type:$pred)> {
  string llvmBuilder = [{
      $res = createIntrinsicCall(builder,
            llvm::Intrinsic::nvvm_vote_ballot_sync, {$mask, $pred});
  }];
  let parser = [{ return parseNVVMVoteBallotOp(parser, result); }];
  let printer = [{ printNVVMIntrinsicOp(p, this->getOperation()); }];
}

def NVVM_MmaOp :
  NVVM_Op<"mma.sync">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins Variadic<LLVM_Type>:$args)> {
  string llvmBuilder = [{
    $res = createIntrinsicCall(
        builder, llvm::Intrinsic::nvvm_mma_m8n8k4_row_row_f32_f32, $args);
  }];
  let parser = [{ return parseNVVMMmaOp(parser, result); }];
  let printer = [{ printNVVMMmaOp(p, *this); }];
  let verifier = [{ return mlir::NVVM::verify(*this); }];
}

#endif // NVVMIR_OPS
