/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This is the definition file for the Runtime Fallback Dialect.

#ifdef TFRT_DELEGATE_DIALECT
#else
#define TFRT_DELEGATE_DIALECT

include "tfrt/tfrt_op_base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//
def TfTensorType : OpaqueType<"tfd", "tf_tensor", "!tfd.tf_tensor type">;

//===----------------------------------------------------------------------===//
// Runtime Fallback Dialect definitions
//===----------------------------------------------------------------------===//

def RuntimeFallback_Dialect : Dialect {
  let name = "tfd";

  let description = [{
    The Runtime Fallback dialect.

    This dialect contains operations to run existing TF kernels on TFRT by
    invoking TF Eager API.
  }];

  let cppNamespace = "::mlir::tfd";

  let emitAccessorPrefix = kEmitAccessorPrefix_Raw;
}

//===----------------------------------------------------------------------===//
// Runtime Fallback Dialect Ops definitions
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect.
class RuntimeFallbackDialect_Op<string mnemonic, list<Trait> traits = []> :
    Op<RuntimeFallback_Dialect, mnemonic, traits> { }

def InitEagerContextOp : RuntimeFallbackDialect_Op<"init_eager_context"> {
  let summary = "eager context initialization operation";
  let description = [{
    The "tfd.init_eager_context" operation takes an input chain, creates and
    initializes the TF EagerContext and returns an output chain.

    Example:
      %c1 = "tfd.init_eager_context"(%c0): (!tfrt.chain) -> !tfrt.chain
  }];

  let arguments = (ins TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def DelegateKernelOp : RuntimeFallbackDialect_Op<"delegate_kernel"> {
  let summary = "delegate kernel operation";
  let description = [{
    The "tfd.delegate_kernel" operation takes an input chain, and arbitrary
    number of input arguments, and runs a specified TF op via TFE C API. It
    returns an output chain and variable number of outputs from the TF op.

    The input arguments and attributes are passed to the TF op. The outputs are
    outputs of the TF op.

    Note that `_name` is a required attribute specifying the TF op to run.
    TFRT attributes are sorted alphabetically, passed in as positional
    attributes to the TFRT kernel, rather than as named attributes.

    Example:
      To run "tf.MatMul" op, which has two boolean attributes,
        1. Set _name = "MatMul"
        2. For each TF attribute, split it into two attributes, one for name of
           the TF attribute, and the other for the type and value of the
           attribute value. Attribute value is a string with the format of
           "type$val", where type can be "bool", "string", "tfdtype", "tfshape",
           "tftensor".
           The value serialization format can be found in attr_util.h.

      %out_c, %out_tensor = "tfd.delegate_kernel"(
        %in_c, %in1_tensor, %in2_tensor) {
        _name = "MatMul",
        attr1_name = "transpose_a", attr1_value = "bool$false",
        attr2_name = "transpose_b", attr2_value = "bool$false"
      } : (!tfrt.chain, !tfd.tf_tensor, !tfd.tf_tensor) -> (
        !tfrt.chain, !tfd.tf_tensor)
  }];

  let arguments = (ins TFRT_ChainType, Variadic<AnyType>);
  let results = (outs TFRT_ChainType, Variadic<AnyType>);
}

def PrintTftOp : RuntimeFallbackDialect_Op<"print_tft"> {
  let summary = "print TF tensor operation";
  let description = [{
    The "tfd.print_tft" operation prints the input TF tensor. It takes an input
    TF tensor to be printed and an input chain, and returns an output chain.

    Example:
      %c1 = "tfd.print_tft"(%t, %c) : (!tfd.tf_tensor, !tfrt.chain) -> !tfrt.chain

  }];

  let arguments = (ins TfTensorType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
}

def ConvertTftToDhtOp : RuntimeFallbackDialect_Op<"convert_tft_to_dht", [NoSideEffect]> {
  let summary = "convert TF tensor to TFRT DHT tensor operation";
  let description = [{
    The "tfd.convert_tft_to_dht" operation converts a TF tensor to a TFRT
    DenseHostTensor.

    It takes as input a TF Tensor and an input chain, and returns a converted
    TFRT DHT tensor and an output chain.

    Example:
      %dht, %c0 = "tfd.convert_tft_to_dht"(%tft, %c)
      : (!tfd.tf_tensor, !tfrt.chain) -> (!dht.host_tensor, !tfrt.chain)
  }];

  let arguments = (ins TfTensorType, TFRT_ChainType);
  // Enable registering canonicalization patterns with this operation.
  let hasCanonicalizer = 1;
  let results = (outs TensorType, TFRT_ChainType);
}

def MoveDhtToTftOp : RuntimeFallbackDialect_Op<"move_dht_to_tft", [NoSideEffect]> {
  let summary = "convert TFRT DHT tensor to DHT tensor operation";
  let description = [{
    The "tfd.move_dht_to_tft" operation moves a TFRT tensor into a TF Tensor.

    It takes as input a TFRT Tensor and an input chain, and returns a TF tensor
    with the same underlying buffer and an output chain.

    Example:
      %dht, %c0 = "tfd.convert_tft_to_dht"(%tft, %c)
        : (!tfd.tf_tensor, !tfrt.chain) -> (!dht.host_tensor, !tfrt.chain)
  }];

  let arguments = (ins TensorType, TFRT_ChainType);
  let results = (outs TfTensorType, TFRT_ChainType);
}

#endif // TFRT_DELEGATE_DIALECT
