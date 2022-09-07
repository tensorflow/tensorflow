/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This is the definition file for the TensorFlow Device Dialect.

#ifndef TF_DEVICE_DIALECT
#define TF_DEVICE_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// TensorFlow Device Dialect definitions
//===----------------------------------------------------------------------===//

def TfDevice_Dialect : Dialect {
  let name = "tf_device";

  let description = [{
    The TensorFlow Device dialect.

    This dialect contains operations to describe/launch computations on devices.
    These operations do not map 1-1 to TensorFlow ops and requires a lowering
    pass later to transform them into Compile/Run op pairs, like XlaCompile and
    XlaRun.
}];

  let emitAccessorPrefix = kEmitAccessorPrefix_Raw;
  let cppNamespace = "::mlir::tf_device";
}

//===----------------------------------------------------------------------===//
// TensorFlow Device Dialect Ops definitions
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect.
class TfDevice_Op<string mnemonic, list<Trait> traits = []> :
    Op<TfDevice_Dialect, mnemonic, traits> { }

def TfDevice_LaunchOp : TfDevice_Op<"launch",
    [SingleBlockImplicitTerminator<"ReturnOp">]> {
  let summary = [{
The `tf_device.launch` op launches containing operations on target device.
  }];

  let description = [{
This op captures all needed live-in values.
  }];

  let arguments = (ins
    StrAttr:$device
  );

  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    Block &GetBody() { return getOperation()->getRegion(0).front(); }
    StringRef getDevice() { return device(); }
    bool WrapsSingleOp();
  }];

  let builders = [
    OpBuilder<(ins "StringAttr":$device, "TypeRange":$result_types),
    [{
      $_state.addAttribute("device", device);
      $_state.addTypes(result_types);
      $_state.addRegion();
    }]>
  ];

  let hasCanonicalizer = 1;
}

def TfDevice_ReturnOp : TfDevice_Op<"return", [NoSideEffect, ReturnLike, Terminator]> {
  let summary = [{
The `tf_device.return` operation terminates and returns values from a
`tf_device` dialect operation.
  }];

  let arguments = (ins
    Variadic<AnyType>:$results
  );

  let builders = [
    OpBuilder<(ins),
    [{
      build($_builder, $_state, {});
    }]>
   ];

  let assemblyFormat = "attr-dict ($results^ `:` type($results))?";
}

def TfDevice_LaunchFuncOp : TfDevice_Op<"launch_func", []> {
  let summary = [{
    The `tf_device.launch_func` launches a function on target device.
  }];

  let arguments = (ins
    StrAttr:$device,
    FlatSymbolRefAttr:$func,
    Variadic<AnyType>:$operands);

  let results = (outs
    Variadic<AnyType>:$results
  );

  let extraClassDeclaration = [{
    StringRef getFunc() { return func(); }
    StringRef getDevice() { return device(); }
  }];
}

def TfDevice_ParallelExecuteOp : TfDevice_Op<"parallel_execute",
    [SingleBlockImplicitTerminator<"ReturnOp">]> {
  let description = [{
    ParallelExecute op concurrently executes variadic number of regions. Regions
    must represent separate sets of instructions to execute concurrently. In
    order to represent concurrently executed regions with dependencies, multiple
    ParallelExecute ops can be used instead. As so, regions within
    ParallelExecute op must not have control/data dependencies.

    While explicit dependencies between regions are disallowed, ParallelExecute
    op does not prevent implicit communication between regions (e.g.
    communication via send/recvs). In this case, users of ParallelExecute op
    must provide correct control dependencies between regions to guarantee
    correctness. Regions in ParallelExecute may include Resource ops.

    In the case where different regions include ops access the same resource,
    the users of the ParallelExecute op must provide mechanism (via send/recvs
    or via control dependencies) to guarantee correct ordering. Sequential
    ordering of ops within a region is guaranteed. Also, sequential ordering of
    ops before/after ParallelExecute ops are guaranteed. That is, execution of
    regions inside ParallelExecute op is blocked until all inputs to all regions
    are materialized and ops following ParallelExecute op are blocked until all
    regions are executed.
  }];

  let results = (outs
    Variadic<AnyType>:$execute_outputs
  );

  let regions = (region VariadicRegion<SizedRegion<1>>:$regions);

  let extraClassDeclaration = [{
    Block& GetRegionBlockWithIndex(unsigned index);
    Operation::result_range GetRegionOutputs(unsigned region_index);

    // Checks if a tf_device.parallel_execute index'th region block wraps a
    // single operation and the single operation results are perfectly forwarded
    // to the region block's return.
    bool RegionWrapsSingleOp(unsigned index);
  }];

  let builders = [
    OpBuilder<(ins "int":$num_regions, "TypeRange":$output_types)>,
  ];

  let hasVerifier = 1;
}

def TfDevice_ReplicateOp : TfDevice_Op<"replicate",
    [SingleBlockImplicitTerminator<"ReturnOp">, AttrSizedOperandSegments]> {
  let summary = "Wraps an N-way replicated computation.";

  let description = [{
The region held by this operation represents a computation that is replicated
across multiple devices. The number of replications is based on the `n`
attribute. Explicit devices can be populated in the `devices` attribute, and it
must be a mapping of device alias to list of explicit or aliased device names
from the outer scope. The device name map specifies devices on which replicated
ops inside tf_device.replicate will be executed.

A tf_device.parallel_execute inside the tf_device.replicate op region may be
used to represent computations across a larger set of devices. In that case, the
device alias can be used to specify device assignment and replication of each
concurrent execution (i.e. region) defined by tf_device.parallel_execute op.
The size of each value list in the device name map must match `n`. Within a
replica, the execution semantics follow standard sequential behavior. Ops in the
tf_device.replicate wrapped with a tf_device.launch will have its device set to
the associated replicated device from `devices` if the tf_device.launch refers
to an aliased device name. Otherwise the device already set in tf_device.launch
is used instead.

Operands are replicated inputs and packed inputs.

replicated_inputs: each group of `n` inputs corresponds to an input for a single
individual replica and is mapped to a single region argument. Inside one group
the operands are matching in order the `devices` attribute. Each replicated
input must have compatible shapes and types.
packed_inputs: each input corresponds to an input broadcasted across all
replicas and is mapped to a single region argument.

Operands not replicated can be implicitly captured by ops in the region. Results
are replicated each from the regions terminator.

For example:
```
%0 = "tf.opA"() : () -> tensor<i32>
%1 = "tf.opB"() : () -> tensor<i32>
%2 = "tf.opC"() : () -> tensor<f32>
%3 = "tf.opD"() : () -> tensor<f32>
%4 = "tf.opE"() : () -> tensor<!tf_type.resource>
%5 = "tf.opF"() : () -> tensor<!tf_type.resource>
%6 = "tf.opG"() : () -> tensor<!tf_type.string>
%7 = "tf.opH"() : () -> tensor<!tf_type.string>
%8 = "tf.opI"() : () -> tensor<!tf_type.variant>
%9 = "tf.opJ"() : () -> tensor<i1>
%output:8 = tf_device.replicate([%0, %1] as %input_0: tensor<i32>,
                                [%2, %3] as %input_1: tensor<f32>,
                                [%4, %5] as %input_2: tensor<!tf_type.resource>,
                                [%6, %7] as %input_3: tensor<!tf_type.string>,
                                %8 as %input_4: tensor<!tf_type.variant>)
                {n = 2 : i32,
                 devices = {DEVICE_ALIAS_0 = ["/DEVICE:0", "/DEVICE:1"],
                            DEVICE_ALIAS_1 = ["/DEVICE:2", "/DEVICE:3"]}} {
  // Inside the region, %0, %2, %4, and %6 corresponds to
  // "/DEVICE:0"/"/DEVICE:2" and %1, %3, %5, and %7 corresponds to
  // "/DEVICE:1"/"/DEVICE:3", depending on which device alias is used.
  %k = "tf_device.launch"() ( {
    %9 = "tf.opK"(%input_0, %input_4, %9) :
      (tensor<i32>, tensor<!tf_type.variant>, tensor<i1>) -> tensor<i32>
    tf_device.return %9 : tensor<i32>
  }) {device = "DEVICE_ALIAS_0"} : () -> tensor<i32>
  %l = "tf_device.launch"() ( {
    %10 = "tf.opL"(%input_1, %input_4, %9) :
      (tensor<f32>, tensor<!tf_type.variant>, tensor<i1>) -> tensor<f32>
    tf_device.return %10 : tensor<f32>
  }) {device = "DEVICE_ALIAS_1"} : () -> tensor<f32>
  %m = "tf_device.launch"() ( {
    %11 = "tf.opM"(%input_2, %input_4, %9) :
      (tensor<!tf_type.resource>, tensor<!tf_type.variant>, tensor<i1>)
        -> tensor<!tf_type.resource>
    tf_device.return %11 : tensor<!tf_type.resource>
  }) {device = "/DEVICE:4"} : () -> tensor<f32>
  %n = "tf.opN"(%input_3, %input_4, %9) :
    (tensor<!tf_type.string>, tensor<!tf_type.variant>, tensor<i1>)
      -> tensor<!tf_type.string>
  tf_device.return %k, %l, %m, %n :
    tensor<i32>, tensor<f32>, tensor<!tf_type.resource>, tensor<!tf_type.string>
}
// %output#0 corresponds to %k returned from "/DEVICE:0"
// %output#1 corresponds to %k returned from "/DEVICE:1"
// %output#2 corresponds to %l returned from "/DEVICE:2"
// %output#3 corresponds to %l returned from "/DEVICE:3"
// %output#4, %output#5 corresponds to %m and will be returned from "/DEVICE:4"
// %output#6, %output#7 corresponds to %n and will have no device set
```
  }];

  let arguments = (ins
    Variadic<AnyType>:$replicated_inputs,
    Variadic<AnyType>:$packed_inputs,

    DenseI32ArrayAttr:$operand_segment_sizes,
    ConfinedAttr<I32Attr, [IntMinValue<2>]>:$n,
    OptionalAttr<DictionaryAttr>:$devices
  );

  let results = (outs
    Variadic<AnyType>:$replicated_outputs
  );

  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    Block &GetBody() { return getOperation()->getRegion(0).front(); }
    unsigned GetNumReplicatedBlockArguments();
    unsigned GetNumPackedBlockArguments();
    llvm::ArrayRef<BlockArgument> GetPackedBlockArguments();
    llvm::ArrayRef<BlockArgument> GetReplicatedBlockArguments();
    bool IsReplicatedBlockArgument(BlockArgument block_arg);
    bool IsPackedBlockArgument(BlockArgument block_arg);
    unsigned GetReplicaOperandIndexForBlockArgument(BlockArgument block_arg, unsigned replica);
    Value GetReplicaOperandForBlockArgument(BlockArgument block_arg, unsigned replica);
    MutableArrayRef<OpOperand> GetOperandsForBlockArgument(BlockArgument block_arg);
    bool WrapsSingleOp();
  }];

  let builders = [
    OpBuilder<(ins "int":$n,
      "const llvm::SmallDenseMap<StringRef, llvm::SmallVector<StringRef, 4>>&":$devices,
      "llvm::ArrayRef<std::pair<ValueRange, Type>>":$replicated_inputs,
      "ValueRange":$packed_inputs, "TypeRange":$replica_output_types)>,
    OpBuilder<(ins "int":$n, "llvm::Optional<DictionaryAttr>":$devices,
      "llvm::ArrayRef<std::pair<ValueRange, Type>>":$replicated_inputs,
      "ValueRange":$packed_inputs, "TypeRange":$replica_output_types)>,
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def TfDevice_ClusterOp : TfDevice_Op<"cluster",
    [SingleBlockImplicitTerminator<"ReturnOp">]> {
  let summary = [{
The `tf_device.cluster` op wraps containing operations in a region.
  }];

  let description = [{
This op can be used to group operations, and captures all needed live-in values.

Optional policy attribute allows to tag clusters with a policy name that was
used to form the cluster.
  }];

  let arguments = (ins OptionalAttr<StrAttr>:$policy);

  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region SizedRegion<1>:$body);

  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes),
    [{
      build($_builder, $_state, resultTypes, mlir::StringAttr {});
    }]>
  ];

  let extraClassDeclaration = [{
    Block &GetBody() { return getOperation()->getRegion(0).front(); }
  }];

  let hasCanonicalizer = 1;
}

def TfDevice_ClusterFuncOp : TfDevice_Op<"cluster_func", []> {
  let summary = [{
The `tf_device.cluster_func` launches a function containing the body of a
cluster.
  }];

  let description = [{
This op is used for outlining a cluster.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$func,
    Variadic<AnyType>:$operands
  );

  let results = (outs
    Variadic<AnyType>:$results
  );

  let extraClassDeclaration = [{
    // returns the function that this operation will launch.
    func::FuncOp getFunc() {
      return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, funcAttr());
    }
  }];
}

def TfDevice_RemoteRunOp : TfDevice_Op<"remote_run",
    [SingleBlockImplicitTerminator<"ReturnOp">]> {
  let summary = [{
The `tf_device.remote_run` op launches the containing operations on a specific
host.
  }];

  let description = [{
This op captures all needed live-in values.
  }];

  let arguments = (ins
    StrAttr:$host,
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$callee_args
  );

  let results = (outs
    Variadic<AnyType>:$results
  );

  let assemblyFormat = [{
      $host $callee `(` $callee_args `)` attr-dict `:` functional-type ( $callee_args , $results )
  }];
}

def TfDevice_SendOp : TfDevice_Op<"send", []> {
  let summary = "Send a value to a host.";

  let description = [{
    Send the value to the given host with the given rendezvous key.
  }];

  let arguments = (ins
    AnyType:$value,
    StrAttr:$key,
    StrAttr:$dst_host
  );

  let results = (outs);

  let assemblyFormat = [{$value $key $dst_host attr-dict `:` type($value)}];
}

def TfDevice_ReceiveOp : TfDevice_Op<"receive", []> {
  let summary = "Rceive a value from a host.";

  let description = [{
    Receive a value from the given host with the given rendezvous key.
  }];

  let arguments = (ins
    StrAttr:$key,
    StrAttr:$src_host
  );

  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{$key $src_host attr-dict `:` type($result)}];
}

#endif // TF_DEVICE_DIALECT
