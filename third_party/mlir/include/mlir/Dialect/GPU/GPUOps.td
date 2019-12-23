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
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

// Type constraint accepting standard integers, indices and wrapped LLVM integer
// types.
def IntLikeOrLLVMInt : TypeConstraint<
  Or<[AnyInteger.predicate, Index.predicate, LLVMInt.predicate]>,
  "integer, index or LLVM dialect equivalent">;

//===----------------------------------------------------------------------===//
// GPU Dialect operations.
//===----------------------------------------------------------------------===//

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

def GPU_GPUFuncOp : GPU_Op<"func", [FunctionLike, IsolatedFromAbove, Symbol]> {
  let summary = "Function executable on a GPU";

  let description = [{
    Defines a function that can be executed on a GPU. This supports memory
    attribution and its body has a particular execution model.

    GPU functions are either kernels (as indicated by the `kernel` attribute) or
    regular functions. The former can be launched from the host side, while the
    latter are device side only.

    The memory attribution defines SSA values that correspond to memory buffers
    allocated in the memory hierarchy of the GPU (see below).

    The operation has one attached region that corresponds to the body of the
    function. The region arguments consist of the function arguments without
    modification, followed by buffers defined in memory annotations. The body of
    a GPU function, when launched, is executed by multiple work items. There are
    no guarantees on the order in which work items execute, or on the connection
    between them. In particular, work items are not necessarily executed in
    lock-step. Synchronization ops such as "gpu.barrier" should be used to
    coordinate work items. Declarations of GPU functions, i.e. not having the
    body region, are not supported.

    Syntax:

    ```
    op ::= `gpu.func` symbol-ref-id `(` argument-list `)` (`->`
    function-result-list)?
           memory-attribution `kernel`? function-attributes? region

    memory-attribution ::= (`workgroup` `(` ssa-id-and-type-list `)`)?
                           (`private` `(` ssa-id-and-type-list `)`)?
    ```

    Example:

    ```mlir
    gpu.func @foo(%arg0: index)
        workgroup(%workgroup: memref<32xf32, 3>)
        private(%private: memref<1xf32, 5>)
        kernel
        attributes {qux: "quux"} {
      gpu.return
    }
    ```

    The generic form illustrates the concept

    ```mlir
    "gpu.func"(%arg: index) {sym_name: "foo", kernel, qux: "quux"} ({
    ^bb0(%arg0: index, %workgroup: memref<32xf32, 3>,
         %private: memref<1xf32, 5>):
      "gpu.return"() : () -> ()
    }) : (index) -> ()
    ```

    Note the non-default memory spaces used in memref types in memory
    attribution.
  }];

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, StringRef name, "
              "FunctionType type, ArrayRef<Type> workgroupAttributions = {}, "
              "ArrayRef<Type> privateAttributions = {}, "
              "ArrayRef<NamedAttribute> attrs = {}">
  ];

  let extraClassDeclaration = [{
    /// Returns `true` if the GPU function defined by this Op is a kernel, i.e.
    /// it is intended to be launched from host.
    bool isKernel() {
      return getAttrOfType<UnitAttr>(GPUDialect::getKernelFuncAttrName()) !=
             nullptr;
    }

    /// Returns the type of the function this Op defines.
    FunctionType getType() {
      return getTypeAttr().getValue().cast<FunctionType>();
    }

    /// Change the type of this function in place. This is an extremely
    /// dangerous operation and it is up to the caller to ensure that this is
    /// legal for this function, and to restore invariants:
    ///  - the entry block args must be updated to match the function params.
    ///  - the argument/result attributes may need an update: if the new type
    ///  has less parameters we drop the extra attributes, if there are more
    ///  parameters they won't have any attributes.
    // TODO(b/146349912): consider removing this function thanks to rewrite
    // patterns.
    void setType(FunctionType newType);

    /// Returns the number of buffers located in the workgroup memory.
    unsigned getNumWorkgroupAttributions() {
      return getAttrOfType<IntegerAttr>(getNumWorkgroupAttributionsAttrName())
          .getInt();
    }

    /// Returns a list of block arguments that correspond to buffers located in
    /// the workgroup memory
    ArrayRef<BlockArgumentPtr> getWorkgroupAttributions() {
      auto begin =
          std::next(getBody().front().args_begin(), getType().getNumInputs());
      auto end = std::next(begin, getNumWorkgroupAttributions());
      return {begin, end};
    }

    /// Returns a list of block arguments that correspond to buffers located in
    /// the private memory.
    ArrayRef<BlockArgumentPtr> getPrivateAttributions() {
      auto begin =
          std::next(getBody().front().args_begin(),
                    getType().getNumInputs() + getNumWorkgroupAttributions());
      return {begin, getBody().front().args_end()};
    }

    /// Returns the name of the attribute containing the number of buffers
    /// located in the workgroup memory.
    static StringRef getNumWorkgroupAttributionsAttrName() {
      return "workgroup_attributions";
    }

    // FunctionLike trait needs access to the functions below.
    friend class OpTrait::FunctionLike<GPUFuncOp>;

    /// Hooks for the input/output type enumeration in FunctionLike .
    unsigned getNumFuncArguments() { return getType().getNumInputs(); }
    unsigned getNumFuncResults() { return getType().getNumResults(); }

    /// Returns the keywords used in the custom syntax for this Op.
    static StringRef getWorkgroupKeyword() { return "workgroup"; }
    static StringRef getPrivateKeyword() { return "private"; }
    static StringRef getKernelKeyword() { return "kernel"; }

    /// Hook for FunctionLike verifier.
    LogicalResult verifyType();

    /// Verifies the body of the function.
    LogicalResult verifyBody();
  }];

  // let verifier = [{ return ::verifFuncOpy(*this); }];
  let printer = [{ printGPUFuncOp(p, *this); }];
  let parser = [{ return parseGPUFuncOp(parser, result); }];
}

def GPU_LaunchFuncOp : GPU_Op<"launch_func">,
    Arguments<(ins IntLikeOrLLVMInt:$gridSizeX, IntLikeOrLLVMInt:$gridSizeY,
               IntLikeOrLLVMInt:$gridSizeZ, IntLikeOrLLVMInt:$blockSizeX,
               IntLikeOrLLVMInt:$blockSizeY, IntLikeOrLLVMInt:$blockSizeZ,
               Variadic<AnyType>:$operands)>,
    Results<(outs)> {
  let summary = "Launches a function as a GPU kerneel";

  let description = [{
    Launch a kernel function on the specified grid of thread blocks.
    `gpu.launch` operations are lowered to `gpu.launch_func` operations by
    outlining the kernel body into a function in a dedicated module, which
    reflects the separate compilation process. The kernel function is required
    to have the `gpu.kernel` attribute. The module containing the kernel
    function is required to have the `gpu.kernel_module` attribute and must be
    named. And finally, the module containing the kernel module (which thus
    cannot be the top-level module) is required to have the
    `gpu.container_module` attribute. The `gpu.launch_func` operation has a
    string attribute named `kernel` to specify the name of the kernel function
    to launch and an attribute named `kernel_module` to specify the name of the
    module containing that kernel function.

    The operation takes at least six operands, with the first three operands
    being grid sizes along x,y,z dimensions and the following three being block
    sizes along x,y,z dimensions. When a lower-dimensional kernel is required,
    unused sizes must be explicitly set to `1`. The remaining operands are
    passed as arguments to the kernel function.

    A custom syntax for this operation is currently not available.

    Example:

    ```mlir
    module attributes {gpu.container_module} {

      // This module creates a separate compilation unit for the GPU compiler.
      module @kernels attributes {gpu.kernel_module} {
        func @kernel_1(%arg0 : f32, %arg1 : !llvm<"float*">)
            attributes { nvvm.kernel = true } {

          // Operations that produce block/thread IDs and dimensions are
          // injected when outlining the `gpu.launch` body to a function called
          // by `gpu.launch_func`.
          %tIdX = "gpu.thread_id"() {dimension = "x"} : () -> (index)
          %tIdY = "gpu.thread_id"() {dimension = "y"} : () -> (index)
          %tIdZ = "gpu.thread_id"() {dimension = "z"} : () -> (index)

          %bDimX = "gpu.block_dim"() {dimension = "x"} : () -> (index)
          %bDimY = "gpu.block_dim"() {dimension = "y"} : () -> (index)
          %bDimZ = "gpu.block_dim"() {dimension = "z"} : () -> (index)

          %bIdX = "gpu.block_id"() {dimension = "x"} : () -> (index)
          %bIdY = "gpu.block_id"() {dimension = "y"} : () -> (index)
          %bIdZ = "gpu.block_id"() {dimension = "z"} : () -> (index)

          %gDimX = "gpu.grid_dim"() {dimension = "x"} : () -> (index)
          %gDimY = "gpu.grid_dim"() {dimension = "y"} : () -> (index)
          %gDimZ = "gpu.grid_dim"() {dimension = "z"} : () -> (index)

          "some_op"(%bx, %tx) : (index, index) -> ()
          %42 = load %arg1[%bx] : memref<?xf32, 1>
        }
      }

      "gpu.launch_func"(%cst, %cst, %cst,  // Grid sizes.
                        %cst, %cst, %cst,  // Block sizes.
                        %arg0, %arg1)      // Arguments passed to the kernel.
            { kernel_module = @kernels,    // Module containing the kernel.
              kernel = "kernel_1" }        // Kernel function.
            : (index, index, index, index, index, index, f32, !llvm<"float*">)
              -> ()
    }
    ```
  }];

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, GPUFuncOp kernelFunc, "
              "ValuePtr gridSizeX, ValuePtr gridSizeY, ValuePtr gridSizeZ, "
              "ValuePtr blockSizeX, ValuePtr blockSizeY, ValuePtr blockSizeZ, "
              "ValueRange kernelOperands">,
    OpBuilder<"Builder *builder, OperationState &result, GPUFuncOp kernelFunc, "
              "KernelDim3 gridSize, KernelDim3 blockSize, "
              "ValueRange kernelOperands">
  ];

  let extraClassDeclaration = [{
    /// The kernel function specified by the operation's `kernel` attribute.
    StringRef kernel();

    /// The number of operands passed to the kernel function.
    unsigned getNumKernelOperands();

    /// The name of the kernel module specified by the operation's
    /// `kernel_module` attribute.
    StringRef getKernelModuleName();

    /// The i-th operand passed to the kernel function.
    ValuePtr getKernelOperand(unsigned i);

    /// Get the SSA values passed as operands to specify the grid size.
    KernelDim3 getGridSizeOperandValues();

    /// Get the SSA values passed as operands to specify the block size.
    KernelDim3 getBlockSizeOperandValues();

    /// The number of launch configuration operands, placed at the leading
    /// positions of the operand list.
    static constexpr unsigned kNumConfigOperands = 6;

    // This needs to quietly verify if attributes with names defined below are
    // present since it is run before the verifier of this op.
    friend LogicalResult GPUDialect::verifyOperationAttribute(Operation *,
                                                              NamedAttribute);

    /// The name of the symbolRef attribute specifying the kernel to launch.
    static StringRef getKernelAttrName() { return "kernel"; }

    /// The name of the symbolRef attribute specifying the name of the module
    /// containing the kernel to launch.
    static StringRef getKernelModuleAttrName() { return "kernel_module"; }
  }];

  let verifier = [{ return ::verify(*this); }];
}

def GPU_LaunchOp : GPU_Op<"launch", [IsolatedFromAbove]>,
    Arguments<(ins Index:$gridSizeX, Index:$gridSizeY, Index:$gridSizeZ,
               Index:$blockSizeX, Index:$blockSizeY, Index:$blockSizeZ,
               Variadic<AnyType>:$operands)>,
    Results<(outs)> {
  let summary = "GPU kernel launch operation";

  let description = [{
    Launch a kernel on the specified grid of thread blocks. The body of the
    kernel is defined by the single region that this operation contains. The
    operation takes at least six operands, with first three operands being grid
    sizes along x,y,z dimensions, the following three arguments being block
    sizes along x,y,z dimension, and the remaining operands are arguments of the
    kernel. When a lower-dimensional kernel is required, unused sizes must be
    explicitly set to `1`.

    The body region has at least _twelve_ arguments, grouped as follows:

    -   three arguments that contain block identifiers along x,y,z dimensions;
    -   three arguments that contain thread identifiers along x,y,z dimensions;
    -   operands of the `gpu.launch` operation as is, including six leading
        operands for grid and block sizes.

    Operations inside the body region, and any operations in the nested regions,
    are _not_ allowed to use values defined outside the _body_ region, as if
    this region was a function. If necessary, values must be passed as kernel
    arguments into the body region. Nested regions inside the kernel body are
    allowed to use values defined in their ancestor regions as long as they
    don't cross the kernel body region boundary.

    Syntax:

    ```
    operation ::= `gpu.launch` `block` `(` ssa-id-list `)` `in` ssa-reassignment
                             `threads` `(` ssa-id-list `)` `in` ssa-reassignment
                               (`args` ssa-reassignment `:` type-list)?
                               region attr-dict?
    ssa-reassignment ::= `(` ssa-id `=` ssa-use (`,` ssa-id `=` ssa-use)* `)`
    ```

    Example:

    ```mlir
    gpu.launch blocks(%bx, %by, %bz) in (%sz_bx = %0, %sz_by = %1, %sz_bz = %2)
               threads(%tx, %ty, %tz) in (%sz_tx = %3, %sz_ty = %4, %sz_tz = %5)
               args(%arg0 = %6, %arg1 = 7) : f32, memref<?xf32, 1> {
      // Block and thread identifiers, as well as block/grid sizes are
      // immediately usable inside body region.
      "some_op"(%bx, %tx) : (index, index) -> ()
      %42 = load %arg1[%bx] : memref<?xf32, 1>
    }

    // Generic syntax explains how the pretty syntax maps to the IR structure.
    "gpu.launch"(%cst, %cst, %c1,  // Grid sizes.
                        %cst, %c1, %c1,   // Block sizes.
                        %arg0, %arg1)     // Actual arguments.
        {/*attributes*/}
        // All sizes and identifiers have "index" size.
        : (index, index, index, index, index, index, f32, memref<?xf32, 1>)
            -> () {
    // The operation passes block and thread identifiers, followed by grid and
    // block sizes, followed by actual arguments to the entry block of the
    // region.
    ^bb0(%bx : index, %by : index, %bz : index,
         %tx : index, %ty : index, %tz : index,
         %num_bx : index, %num_by : index, %num_bz : index,
         %num_tx : index, %num_ty : index, %num_tz : index,
         %arg0 : f32, %arg1 : memref<?xf32, 1>):
      "some_op"(%bx, %tx) : (index, index) -> ()
      %3 = "std.load"(%arg1, %bx) : (memref<?xf32, 1>, index) -> f32
    }
    ```

    Rationale: using operation/block arguments gives analyses a clear way of
    understanding that a value has additional semantics (e.g., we will need to
    know what value corresponds to threadIdx.x for coalescing). We can recover
    these properties by analyzing the operations producing values, but it is
    easier just to have that information by construction.
  }];

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, ValuePtr gridSizeX,"
              "ValuePtr gridSizeY, ValuePtr gridSizeZ, ValuePtr blockSizeX,"
              "ValuePtr blockSizeY, ValuePtr blockSizeZ,"
              "ValueRange operands">
  ];

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    /// Get the SSA values corresponding to kernel block identifiers.
    KernelDim3 getBlockIds();
    /// Get the SSA values corresponding to kernel thread identifiers.
    KernelDim3 getThreadIds();
    /// Get the SSA values corresponding to kernel grid size.
    KernelDim3 getGridSize();
    /// Get the SSA values corresponding to kernel block size.
    KernelDim3 getBlockSize();
    /// Get the operand values passed as kernel arguments.
    operand_range getKernelOperandValues();
    /// Get the operand types passed as kernel arguments.
    operand_type_range getKernelOperandTypes();

    /// Get the SSA values passed as operands to specify the grid size.
    KernelDim3 getGridSizeOperandValues();
    /// Get the SSA values passed as operands to specify the block size.
    KernelDim3 getBlockSizeOperandValues();

    /// Get the SSA values of the kernel arguments.
    iterator_range<Block::args_iterator> getKernelArguments();

    /// Erase the `index`-th kernel argument.  Both the entry block argument and
    /// the operand will be dropped.  The block argument must not have any uses.
    void eraseKernelArgument(unsigned index);

    static StringRef getBlocksKeyword() { return "blocks"; }
    static StringRef getThreadsKeyword() { return "threads"; }
    static StringRef getArgsKeyword() { return "args"; }

    /// The number of launch configuration operands, placed at the leading
    /// positions of the operand list.
    static constexpr unsigned kNumConfigOperands = 6;

    /// The number of region attributes containing the launch configuration,
    /// placed in the leading positions of the argument list.
    static constexpr unsigned kNumConfigRegionAttributes = 12;
  }];

  let parser = [{ return parseLaunchOp(parser, result); }];
  let printer = [{ printLaunchOp(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
}

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

def GPU_ShuffleOpXor : StrEnumAttrCase<"xor">;

def GPU_ShuffleModeAttr : StrEnumAttr<"ShuffleModeAttr",
    "Indexing modes supported by gpu.shuffle.",
    [
      GPU_ShuffleOpXor,
    ]>;

def GPU_ShuffleOp : GPU_Op<"shuffle", [NoSideEffect]>,
    Arguments<(ins AnyType:$value, I32:$offset, I32:$width,
               GPU_ShuffleModeAttr:$mode)>,
    Results<(outs AnyType:$result, I1:$valid)> {
  let summary = "Shuffles values within a subgroup.";
  let description = [{
    The "shuffle" op moves values to a different invocation within the same
    subgroup.

    For example
    ```
      %1, %2 = gpu.shuffle %0, %offset, %width xor : f32
    ```
    for lane k returns the value from lane `k ^ offset` and `true` if that lane
    is smaller than %width. Otherwise it returns an unspecified value and
    `false`. A lane is the index of an invocation relative to its subgroup.

    The width specifies the number of invocations that participate in the
    shuffle. The width needs to be the same for all invocations that participate
    in the shuffle. Exactly the first `width` invocations of a subgroup need to
    execute this op in convergence.
  }];
  let verifier = [{ return ::verifyShuffleOp(*this); }];
  let printer = [{ printShuffleOp(p, *this); }];
  let parser = [{ return parseShuffleOp(parser, result); }];
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
