/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/testlib/kernel_runner.h"

#include <random>
#include <string_view>
#include <utility>
#include <vector>

#include "xla/backends/cpu/testlib/llvm_ir_kernel_emitter.h"
#include "xla/codegen/testlib/kernel_runner.h"
#include "xla/literal_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::cpu {

TEST(KernelRunnerTest, Add) {
  // TODO(b/370122948): Add an actual LLVM IR for simple addition kernel.
  static constexpr std::string_view kNoOp = R"(
    define ptr @noop(ptr noundef %0) {
      ret ptr null
    }
  )";

  LlvmIrKernelEmitter::KernelArg arg{1024, BufferUse::kWrite};
  LlvmIrKernelEmitter emitter(kNoOp, "noop", se::ThreadDim(), {arg});

  TF_ASSERT_OK_AND_ASSIGN(auto kernel, emitter.EmitKernelSpec());

  KernelRunner runner(std::move(kernel));

  std::minstd_rand0 engine;

  Shape shape = ShapeUtil::MakeShape(F32, {8});
  auto a = LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);
  auto b = LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);
  auto c = LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);

  std::vector<KernelRunner::Argument> args = {
      KernelRunnerUtil::CreateArgument(*a),
      KernelRunnerUtil::CreateArgument(*b),
      KernelRunnerUtil::CreateArgument(*c),
  };

  auto status = runner.Call(args);
  ASSERT_FALSE(status.ok());
}

}  // namespace xla::cpu
