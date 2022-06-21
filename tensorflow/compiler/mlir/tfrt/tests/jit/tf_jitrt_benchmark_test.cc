/*
 * Copyright 2022 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <utility>

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tfrt/jit/tf_jitrt_pipeline.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tfrt/jitrt/jitrt.h"  // from @tf_runtime
#include "tfrt/jitrt/jitrt_compiler.h"  // from @tf_runtime

namespace tensorflow {

static const char* mlir_module = R"(
  func.func @compute(
      %arg0: tensor<?x?x1xf32>,
      %arg1: tensor<?x?x20xf32>,
      %arg2: tensor<?x?x40xf32>,
      %arg3: tensor<?x?x40xf32>,
      %arg4: tensor<?x?x33xf32>,
      %arg5: tensor<?x?x13xf32>
  ) -> (tensor<?x?x20xf32>, tensor<?x?x40xf32>, tensor<?x?x40xf32>,
        tensor<?x?x33xf32>, tensor<?x?x13xf32>)
  {
    %c = "tf.Const"() {value = dense<9.99999997E-7> : tensor<f32>}
         : () -> tensor<f32>
    %0 = "tf.AddV2"(%arg0, %c)
         : (tensor<?x?x1xf32>, tensor<f32>) -> tensor<?x?x1xf32>
    %1 = "tf.RealDiv"(%arg1, %0)
         : (tensor<?x?x20xf32>, tensor<?x?x1xf32>) -> tensor<?x?x20xf32>
    %2 = "tf.RealDiv"(%arg2, %0)
         : (tensor<?x?x40xf32>, tensor<?x?x1xf32>) -> tensor<?x?x40xf32>
    %3 = "tf.RealDiv"(%arg3, %0)
         : (tensor<?x?x40xf32>, tensor<?x?x1xf32>) -> tensor<?x?x40xf32>
    %4 = "tf.RealDiv"(%arg4, %0)
         : (tensor<?x?x33xf32>, tensor<?x?x1xf32>) -> tensor<?x?x33xf32>
    %5 = "tf.RealDiv"(%arg5, %0)
         : (tensor<?x?x13xf32>, tensor<?x?x1xf32>) -> tensor<?x?x13xf32>
    return %1, %2, %3, %4, %5 : tensor<?x?x20xf32>, tensor<?x?x40xf32>,
                                tensor<?x?x40xf32>, tensor<?x?x33xf32>,
                                tensor<?x?x13xf32>
  })";

static const char* entrypoint = "compute";

using ::tfrt::jitrt::CompilationOptions;
using ::tfrt::jitrt::CompilationPipelineOptions;
using ::tfrt::jitrt::CreateDefaultJitRtCompilationPipeline;
using ::tfrt::jitrt::JitExecutable;
using ::tfrt::jitrt::JitExecutableCache;
using ::tfrt::jitrt::RegisterDefaultJitRtDialects;

static void BM_InstantiateExecutable(::testing::benchmark::State& state) {
  // Options for the default JitRt compilation pipeline (lowering to LLVM).
  CompilationPipelineOptions copts;
  copts.alignment = EIGEN_MAX_ALIGN_BYTES;
  copts.num_worker_threads = 16;
  copts.cost_driven_async_parallel_for = false;

  // Options for the JitRt JitExecutable compilation.
  CompilationOptions opts;
  opts.specialization = CompilationOptions::Specialization::kEnabled;

  // Register dialects and interfaces required for the compilation pipeline.
  opts.register_dialects = [](mlir::DialectRegistry& registry) {
    mlir::RegisterAllTensorFlowDialects(registry);
    RegisterDefaultJitRtDialects(registry);
  };

  // Register a custom pipeline for lowering from Tensorflow dialect to LLVM.
  opts.create_compilation_pipeline = [&](mlir::PassManager& pm) {
    TfJitRtPipelineOptions opts;

    // Lower from Tensorflow to Linalg on buffers.
    CreateTfJitRtPipeline(pm, opts);

    // Use default JitRt compilation pipeline to lower to LLVM.
    CreateDefaultJitRtCompilationPipeline(pm, copts);
  };

  // Register a custom pipeline to propagate specialization information.
  opts.create_specialization_pipeline = [&](mlir::PassManager& pm) {
    CreateJitRtSpecializationPipeline(pm);
  };

  // When lowering Tensorflow functions to JitRt we convert all input and
  // result tensors to memrefs, and add a kernel context input.
  opts.calling_convention = CompilationOptions::DefaultCallingConvention(
      mlir::bufferization::BufferizeTypeConverter());

  for (auto _ : state) {
    llvm::Expected<JitExecutable> jit_executable =
        JitExecutable::Instantiate(mlir_module, entrypoint, opts, "benchmark");
    if (auto err = jit_executable.takeError())
      LOG(FATAL) << "Failed to compile the kernel: " << tfrt::StrCat(err);
  }
}

BENCHMARK(BM_InstantiateExecutable);

}  // namespace tensorflow
