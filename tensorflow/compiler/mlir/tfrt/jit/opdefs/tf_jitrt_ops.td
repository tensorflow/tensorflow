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

#ifdef TF_JITRT_OPS
#else
#define TF_JITRT_OPS

include "mlir/IR/OpBase.td"
include "tfrt/tfrt_op_base.td"
include "tfrt/compiler/opdefs/tfrt_op_interfaces.td"
include "tfrt/compiler/opdefs/tfrt_traits.td"
include "tfrt/jitrt/opdefs/jitrt_base.td"

include "tfrt_fallback.td"

//===----------------------------------------------------------------------===//
// Tensorflow/JitRT Runtime dialect definitions
//===----------------------------------------------------------------------===//

def TF_JITRT_Dialect : Dialect {
  let name = "tf_jitrt";

  let description = [{
    The Tensorflow JIT CPU runtime dialect.

    This dialect contains Tensorflow specific operations to support JIT
    compilation and exection built on top of TFRT JitRt.
  }];

  let cppNamespace = "::mlir::tf_jitrt";
  let useFoldAPI = kEmitFoldAdaptorFolder;
}

//===----------------------------------------------------------------------===//
// TF JitRt op definitions.
//===----------------------------------------------------------------------===//

class TF_JITRT_Op<string mnemonic, list<Trait> traits = []> :
      Op<TF_JITRT_Dialect, mnemonic, traits> {
}

def FallbackCompileOp : TF_JITRT_Op<"fallback.compile",
    [TFRT_CostFunctionInterface, TFRT_FixedCost<1>]> {
  let summary = "compiles kernel at runtime using LLVM JIT compiler";
  let description = [{
    `tf_jitrt.fallback.compile` schedules the compilation of a Tensorflow
    program defined by the kernel function in the nested module to the JitRt JIT
    executable using LLVM JIT APIs and caches it in the JitExecutableCache owned
    by the resource context.

    Compilation happens asynchronously by launching compilation task into the
    dedicated thread pool, and the kernel returns an available chain once the
    task is scheduled.

    This kernel can be used in the init function to make sure that when execute
    kernel (see definition below) called at runtime, the default compiled kernel
    will be already pre-compiled and ready to run.
    ```
  }];

  let arguments = (ins
    SymbolRefAttr:$kernel,
    StrAttr:$device
  );

  let results = (outs TFRT_ChainType:$scheduled);

  let assemblyFormat = [{
    $kernel `device` `(` $device `)` attr-dict
  }];
}

def FallbackExecuteOp : TF_JITRT_Op<"fallback.execute",
    [Pure, DeclareOpInterfaceMethods<TFRT_CostFunctionInterface>]> {
  let summary = "jitrt execute operation with a fallback runtime interop";
  let description = [{
    `tf_jitrt.fallback.execute` compiles a Tensorflow program defined by the
    kernel function in the nested module to the JitRt JIT executable using
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

      %result = tf_jitrt.fallback.execute @kernel::@main (%operand)
                                           device("/CPU:0")
                  : (!tfrt_fallback.tf_tensor) -> !tfrt_fallback.tf_tensor
    ```
  }];

  let arguments = (ins
    SymbolRefAttr:$kernel,
    Variadic<TFTensorType>,
    StrAttr:$device
  );

  let results = (outs Variadic<TFTensorType>);

  let assemblyFormat = [{
    $kernel `(` operands `)` `device` `(` $device `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

def FallbackDebugExecuteOp : TF_JITRT_Op<"fallback.debug.execute"> {
  let summary = "jitrt debug-execute operation with a fallback runtime interop";
  let description = [{
    Same as `tf_jitrt.fallback.execute`, except that this kernel allows to pass
    compilation options to the tf-jitrt-pipeline and optionally print
    specialization-related debug information to standard output.
  }];

  let arguments = (ins
    SymbolRefAttr:$kernel,
    Variadic<TFTensorType>,
    StrAttr:$device,
    // Print to standard output whenever compiled kernel specialized for the
    // operands shapes or values.
    BoolAttr:$debug_specializations,
    // Compiler options for lowering from Tensorflow dialect to the dialects
    // supported by the JitRt (see the `tf-jitrt-pipeline` options).
    BoolAttr:$vectorize,
    // Compiler options for lowering from Tensorflow dialect to the dialects
    // supported by the JitRt (see the `tf-jitrt-pipeline` options).
    BoolAttr:$legalize_i1_tensors
  );

  let results = (outs Variadic<TFTensorType>);

  let assemblyFormat = [{
    $kernel `(` operands `)` `device` `(` $device `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// TF JitRt op definitions for use in tests.
//===----------------------------------------------------------------------===//

def FallbackWaitForCompilationOp : TF_JITRT_Op<"test.wait_for_compilation",
    [TFRT_CostFunctionInterface, TFRT_FixedCost<1>]> {
  let summary = "waits for the completion of all compilation tasks";
  let description = [{
    `tf_jitrt.test.wait_for_compilation` waits for the completion of all
    specialization compilation tasks related to the given kernel.

    This kernel is intended for use in tests to deterministically call
    specialized kernels when specialization is done lazily.
  }];

  let arguments = (ins
    SymbolRefAttr:$kernel,
    TFRT_ChainType:$chain
  );

  let results = (outs TFRT_ChainType:$scheduled);

  let assemblyFormat = [{
    $kernel `(` $chain `)` attr-dict
  }];
}

def FallbackResetCompilationThreadPoolOp :
    TF_JITRT_Op<"test.reset_compilation_thread_pool",
    [TFRT_CostFunctionInterface, TFRT_FixedCost<1>]> {
  let summary = "resets the thread pool running compilation tasks";
  let description = [{
    `tf_jitrt.test.reset_compilation_thread_pool` resets the compilation thread
    pool, i.e., it waits for the completion of all compilation tasks and
    destroys all captured async value references.

    This kernel is intended for use in tests to make sure that we do not trigger
    async value leak checks because of racy execution.
  }];

  let arguments = (ins
    TFRT_ChainType:$chain
  );

  let results = (outs TFRT_ChainType:$scheduled);

  let assemblyFormat = [{
    $chain attr-dict
  }];
}

#endif // TF_JITRT_OPS
