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

#include "xla/service/cpu/host_kernel_emitter.h"

#include <memory>
#include <vector>

#include "llvm/IR/LLVMContext.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/filecheck.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/test.h"

namespace xla::cpu {
namespace {

TEST(HostKernelEmitterTest, BuildKernelPrototype) {
  llvm::LLVMContext context;
  auto module = std::make_unique<llvm::Module>("test", context);

  auto shape = ShapeUtil::MakeShape(PrimitiveType::F32, {4, 2});
  std::vector<Shape> parameters = {shape};
  std::vector<Shape> results = {shape};

  HostKernelEmitter emitter(module.get());
  HostKernelEmitter::KernelPrototype prototype =
      emitter.BuildKernelPrototype("test", parameters, results);

  ASSERT_TRUE(*RunFileCheck(llvm_ir::DumpToString(module.get()), R"(
    CHECK: define ptr @test(ptr %0) {

    CHECK:   getelementptr %SE_HOST_KernelCallFrame, {{.*}} i64 0
    CHECK:   getelementptr %SE_HOST_KernelThreadDim
    CHECK:   getelementptr %SE_HOST_KernelThreadDim
    CHECK:   getelementptr %SE_HOST_KernelThreadDim
    CHECK:   load i64
    CHECK:   load i64
    CHECK:   load i64

    CHECK:   getelementptr %SE_HOST_KernelCallFrame, {{.*}} i64 1
    CHECK:   getelementptr %SE_HOST_KernelThread
    CHECK:   getelementptr %SE_HOST_KernelThread
    CHECK:   getelementptr %SE_HOST_KernelThread
    CHECK:   load i64
    CHECK:   load i64
    CHECK:   load i64

    CHECK:   getelementptr %SE_HOST_KernelCallFrame, {{.*}} i64 3
    CHECK:   getelementptr %SE_HOST_KernelArg
    CHECK:   getelementptr %SE_HOST_KernelArg

    CHECK:   getelementptr %SE_HOST_KernelCallFrame, {{.*}} i64 3
    CHECK:   getelementptr %SE_HOST_KernelArg
    CHECK:   getelementptr %SE_HOST_KernelArg

    CHECK:   ret ptr null
    CHECK: }
  )"));
}

}  // namespace
}  // namespace xla::cpu
