//===-- GPUOps.td - GPU dialect operation definitions ------*- tablegen -*-===//
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
// Defines some operations of the GPU dialect.
//
//===----------------------------------------------------------------------===//

#ifdef GPU_OPS
#else
#define GPU_OPS

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def GPU_Dialect : Dialect {
  let name = "gpu";
}

class GPU_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<GPU_Dialect, mnemonic, traits>;

class GPU_IndexOp<string mnemonic, list<OpTrait> traits = []> :
    GPU_Op<mnemonic, !listconcat(traits, [NoSideEffect])>,
    Arguments<(ins StrAttr:$dimension)>, Results<(outs Index)> {
  let verifier = [{ return ::verifyIndexOp(*this); }];
}

def gpu_BlockDim : GPU_IndexOp<"block_dim">;
def gpu_BlockId : GPU_IndexOp<"block_id">;
def gpu_GridDim : GPU_IndexOp<"grid_dim">;
def gpu_ThreadId : GPU_IndexOp<"thread_id">;

def gpu_Return : GPU_Op<"return", [Terminator]>, Arguments<(ins)>,
    Results<(outs)> {
  let summary = "Terminator for GPU launch regions.";
  let description = [{
    A terminator operation for regions that appear in the body of `gpu.launch`
    operation.  These regions are not expected to return any value so the
    terminator takes no operands.
  }];

  let parser = [{ return success(); }];
  let printer = [{ p << getOperationName(); }];
}

def gpu_AllReduce : GPU_Op<"all_reduce", [SameOperandsAndResultType]>,
    Arguments<(ins AnyType)>, Results<(outs AnyType)> {
  let summary = "Reduce values among workgroup.";
  let description = [{
    The "all_reduce" op reduces the value of every invocation across a local
    workgroup.

    For example,
    ```
      %1 = gpu.all_reduce %0 : f32
    ```
    computes the sum of each invocation's %0 value. The value of %1 is always
    equal for all invocations of a local workgroup.

    Either none or all invocations of a local workgroup need to execute this op
    in convergence.
  }];
}

#endif // GPU_OPS
