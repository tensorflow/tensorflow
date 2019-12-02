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

#ifndef GPU_OPS
#define GPU_OPS

include "mlir/IR/OpBase.td"

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

def GPU_BlockDimOp : GPU_IndexOp<"block_dim">;
def GPU_BlockIdOp : GPU_IndexOp<"block_id">;
def GPU_GridDimOp : GPU_IndexOp<"grid_dim">;
def GPU_ThreadIdOp : GPU_IndexOp<"thread_id">;

def GPU_ReturnOp : GPU_Op<"return", [Terminator]>, Arguments<(ins)>,
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

def GPU_YieldOp : GPU_Op<"yield", [Terminator]>,
    Arguments<(ins Variadic<AnyType>:$values)> {
  let summary = "GPU yield operation";
  let description = [{
    "gpu.yield" is a special terminator operation for blocks inside regions
    in gpu ops. It returns values to the immediately enclosing gpu op.

    Example:

       ```gpu.yield %f0, %f1 : f32, f32
       ```
  }];
}

// These mirror the XLA ComparisonDirection enum.
def GPU_AllReduceOpAdd : StrEnumAttrCase<"add">;
def GPU_AllReduceOpMul : StrEnumAttrCase<"mul">;

def GPU_AllReduceOperationAttr : StrEnumAttr<"AllReduceOperationAttr",
    "built-in reduction operations supported by gpu.allreduce.",
    [
      GPU_AllReduceOpAdd,
      GPU_AllReduceOpMul,
    ]>;

def GPU_AllReduceOp : GPU_Op<"all_reduce",
    [SameOperandsAndResultType, IsolatedFromAbove]>,
    Arguments<(ins AnyType:$value,
               OptionalAttr<GPU_AllReduceOperationAttr>:$op)>,
    Results<(outs AnyType)> {
  let summary = "Reduce values among workgroup.";
  let description = [{
    The "all_reduce" op reduces the value of every work item across a local
    workgroup. The result is equal for all work items of a workgroup.

    For example, both
    ```
      %1 = "gpu.all_reduce"(%0) ({}) { op = "add" } : (f32) -> (f32)
      %2 = "gpu.all_reduce"(%0) ({
      ^bb(%lhs : f32, %rhs : f32):
        %sum = addf %lhs, %rhs : f32
        "gpu.yield"(%sum) : (f32) -> ()
      }) : (f32) -> (f32)
    ```
    compute the sum of each work item's %0 value. The first version specifies
    the accumulation as operation, whereas the second version specifies the
    accumulation as code region. The accumulation operation must either be
    `add` or `mul`.

    Either none or all work items of a workgroup need to execute this op
    in convergence.
  }];
  let regions = (region AnyRegion:$body);
  let verifier = [{ return ::verifyAllReduce(*this); }];
}

def GPU_BarrierOp : GPU_Op<"barrier"> {
  let summary = "Synchronizes all work items of a workgroup.";
  let description = [{
    The "barrier" op synchronizes all work items of a workgroup. It is used
    to coordinate communication between the work items of the workgroup.

    ```
      gpu.barrier
    ```
    waits until all work items in the workgroup have reached this point
    and all memory accesses made by these work items prior to the op are
    visible to all work items in the workgroup. Data hazards between work items
    accessing the same memory can be avoided by synchronizing work items
    in-between these accesses.

    Either none or all work items of a workgroup need to execute this op
    in convergence.
  }];
  let parser = [{ return success(); }];
  let printer = [{ p << getOperationName(); }];
}

#endif // GPU_OPS
