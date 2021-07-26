/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifdef TF_CPURT_OPS
#else
#define TF_CPURT_OPS

include "mlir/IR/OpBase.td"
include "tfrt/tfrt_op_base.td"
include "tfrt/compiler/opdefs/tfrt_op_interfaces.td"
include "tfrt/compiler/opdefs/tfrt_traits.td"
include "tfrt/cpu/jit/opdefs/cpurt_base.td"

include "tfrt_fallback.td"

//===----------------------------------------------------------------------===//
// TFRT CPU Runtime dialect definitions
//===----------------------------------------------------------------------===//

def TF_CPURT_Dialect : Dialect {
  let name = "tf_cpurt";

  let description = [{
    The Tensorflow JIT CPU runtime dialect.

    This dialect contains Tensorflow specific operations to support JIT
    compilation and exection built on top of TFRT CPURT.
  }];

  let cppNamespace = "::mlir::tf_cpurt";
}

//===----------------------------------------------------------------------===//
// TF CPU Runtime op definitions
//===----------------------------------------------------------------------===//

class TF_CPURT_Op<string mnemonic, list<OpTrait> traits = []> :
      Op<TF_CPURT_Dialect, mnemonic, traits> {
}

def FallbackCompileOp : TF_CPURT_Op<"fallback.compile",
    [TFRT_CostFunctionInterface, TFRT_FixedCost<1>]> {
  let summary = "compiles kernel at runtime using LLVM JIT compiler";
  let description = [{
    `tf_cpurt.fallback.compile` compiles a Tensorflow program defined by the
    kernel function in the nested module to the CPURT JIT executable using
    LLVM JIT APIs and caches it in the JitExecutableCache owned by the
    resource context.

    This kernel can be used in the init function to make sure that when execute
    kernel (see definition below) called at runtime, the compiled kernel will be
    already pre-compiled and ready to run.
    ```
  }];

  let arguments = (ins
    SymbolRefAttr:$kernel,
    StrAttr:$device
  );

  let results = (outs TFRT_ChainType:$compiled);

  let assemblyFormat = [{
    $kernel `device` `(` $device `)` attr-dict
  }];
}

def FallbackExecuteOp : TF_CPURT_Op<"fallback.execute",
    [TFRT_CostFunctionInterface, TFRT_FixedCost<1>]> {
  let summary = "cpurt execute operation with a fallback runtime interop";
  let description = [{
    `tf_cpurt.fallback.execute` compiles a Tensorflow program defined by the
    kernel function in the nested module to the CPURT JIT executable using
    LLVM JIT APIs and calls it with provided fallback tensors operands, and
    returns back a fallback tensor for each result.

    Internally this operation caches compilation results in the request context,
    and the actual compilation happens only once (or a few times if kernel
    specialization is required).

    Example: adding two f32 tensors

    ```mlir
      module @kernel attributes { tfrt.compiled } {
        func @main(%lhs: tensor<?xf32>, %rhs: tensor<?xf32>) -> tensor<?xf32> {
          %0 = "tf.Add"(%lhs, %rhs)
                 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
          return %0: tensor<?xf32>
        }
      }

      %operand = ... : !tfrt_fallback.tf_tensor

      %result = tf_cpurt.fallback.execute @kernel::@main (%operand)
                                           device("/CPU:0")
                  : (!tfrt_fallback.tf_tensor) -> !tfrt_fallback.tf_tensor
    ```
  }];

  let arguments = (ins
    SymbolRefAttr:$kernel,
    Variadic<TFTensorType>:$operands,
    StrAttr:$device
  );

  let results = (outs
    Variadic<TFTensorType>:$results
  );

  let assemblyFormat = [{
    $kernel `(` $operands `)` `device` `(` $device `)` attr-dict `:`
    functional-type($operands, $results)
  }];
}

def FallbackDebugExecuteOp : TF_CPURT_Op<"fallback.debug.execute"> {
  let summary = "cpurt debug-execute operation with a fallback runtime interop";
  let description = [{
    Same as `tf_cpurt.fallback.execute`, except that this prints debug
    information to standard output.
  }];

  let arguments = (ins
    SymbolRefAttr:$kernel,
    Variadic<TFTensorType>:$operands,
    StrAttr:$device
  );

  let results = (outs
    Variadic<TFTensorType>:$results
  );

  let assemblyFormat = [{
    $kernel `(` $operands `)` `device` `(` $device `)` attr-dict `:`
    functional-type($operands, $results)
  }];
}

#endif // TF_CPURT_OPS
