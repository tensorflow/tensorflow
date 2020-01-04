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

  let cppNamespace = "tf_device";
}

//===----------------------------------------------------------------------===//
// TensorFlow Device Dialect Ops definitions
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect.
class TfDevice_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TfDevice_Dialect, mnemonic, traits> { }

def TfDevice_LaunchOp : TfDevice_Op<"launch",
                                    [SingleBlockImplicitTerminator<"ReturnOp">]>
{
  let summary = [{The `tf_device.launch` op captures all needed live-in values
                  and launches containing operations on target device.}];

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
  }];

  let builders = [
    OpBuilder<[{Builder *builder, OperationState &result,
                StringAttr device, ArrayRef<Type> result_types}],
      [{
        result.addAttribute("device", device);
        result.addTypes(result_types);
        result.addRegion();
      }]
    >
  ];

  let hasCanonicalizer = 1;
}

def TfDevice_ReturnOp : TfDevice_Op<"return", [Terminator]> {
  let summary = [{
    The `tf_device.return` operation terminates and returns values from
    `tf_device.launch` operation;
  }];

  let arguments = (ins
    Variadic<AnyType>:$results
  );

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result",
    [{
      build(builder, result, {});
    }]>
   ];

  let parser = [{ return Parse$cppClass(&parser, &result); }];
  let printer = [{ return Print(*this, &p); }];
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
    FunctionType getFuncType();
  }];
}

def TfDevice_ReplicateOp :
    TfDevice_Op<"replicate", [SingleBlockImplicitTerminator<"ReturnOp">]> {
  let summary = "Wraps an N-way replicated computation.";

  let description = [{
The region held by this operation represents a computation that is replicated
across multiple devices. The number of replications is based on the `n`
attribute. Explicit devices can be populated in the `devices` attribute, and its
length must match `n`. Within a replica, the execution semantics follow standard
sequential behavior. Ops in the replicate with no device assigned will have its
device set to the associated replicate device from `devices`. Operands are
replicated inputs: each group of `n` inputs corresponds to an input for a single
individual replica and is mapped to a single region argument. Inside one group
the operands are matching in order the `devices` attribute. Each replicated
input must have compatible shapes and types. Operands not replicated can be
implicitly captured by ops in the region. Results are replicated each from the
regions terminator.

For example:
```
%0 = "tf.opA"() : () -> tensor<i32>
%1 = "tf.opB"() : () -> tensor<i32>
%2 = "tf.opC"() : () -> tensor<f32>
%3 = "tf.opD"() : () -> tensor<f32>
%4 = "tf.opE"() : () -> tensor<i1>
%output:4 = tf_device.replicate([%0, %1] as %input_0:tensor<i32>,
                                [%2, %3] as %input_1:tensor<f32>)
                {n = 2 : i32, devices = ["/DEVICE:0", "/DEVICE:1"]} {
  // Inside the region, %input_0 corresponds to %0 for "/DEVICE:0" and %1 for
  // "/DEVICE:1", and %input_1 corresponds to %2 for "/DEVICE:0" and %3 for
  // "/DEVICE:1".
  %f = "tf.opF"(%input_0, %4) : (tensor<i32>, tensor<i1>) -> tensor<i32>
  %g = "tf.opG"(%input_1, %4) : (tensor<f32>, tensor<i1>) -> tensor<f32>
  tf_device.return %f, %g : tensor<i32>, tensor<f32>
}
// %output#0 corresponds to %f returned from "/DEVICE:0"
// %output#1 corresponds to %f returned from "/DEVICE:1"
// %output#2 corresponds to %g returned from "/DEVICE:0"
// %output#3 corresponds to %g returned from "/DEVICE:1"
```
  }];

  let arguments = (ins
    Variadic<AnyType>:$replicated_inputs,
    Confined<I32Attr, [IntMinValue<2>]>:$n,
    OptionalAttr<StrArrayAttr>:$devices
  );

  let results = (outs
    Variadic<AnyType>:$replicated_outputs
  );

  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    Block &GetBody() { return getOperation()->getRegion(0).front(); }
  }];

  let builders = [
    OpBuilder<"Builder* builder, OperationState& state, int n, "
              "llvm::ArrayRef<llvm::StringRef> devices, "
              "llvm::ArrayRef<std::pair<llvm::ArrayRef<Value>, Type>>"
              "    replicated_inputs, "
              "llvm::ArrayRef<Type> replica_output_types">,
    OpBuilder<"Builder* builder, OperationState& state, int n, "
              "llvm::ArrayRef<llvm::StringRef> devices, "
              "llvm::ArrayRef<std::pair<Operation::operand_range, Type>>"
              "    replicated_inputs, "
              "Operation::result_type_range replica_output_types">
  ];

  let parser = [{ return Parse$cppClass(&parser, &result); }];
  let printer = [{ return Print(*this, &p); }];
  let verifier = [{ return Verify(*this); }];
}

#endif // TF_DEVICE_DIALECT
