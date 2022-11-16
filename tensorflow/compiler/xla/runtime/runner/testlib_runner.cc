/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/mlir/runtime/transforms/tests/testlib_pipeline.h"
#include "tensorflow/compiler/xla/runtime/runner/runner.h"

using namespace xla::runtime;  // NOLINT

static JitExecutable::Options CompileOpts() {
  JitExecutable::Options opts;
  opts.specialization = JitExecutable::Specialization::kDisabled;
  opts.compiler.register_dialects = RegisterXlaRuntimeTestlibDialects;
  opts.compiler.create_compilation_pipeline = CreateXlaRuntimeTestlibPipeline;
  return opts;
}

static Executable::ExecuteOpts ExecuteOpts() {
  Executable::ExecuteOpts opts;
  opts.async_task_runner = xla::runtime::NoAsyncTaskRunner();
  return opts;
}

int main(int argc, char** argv) {
  return xla::runtime::Main(argc, argv, CompileOpts(), ExecuteOpts());
}
