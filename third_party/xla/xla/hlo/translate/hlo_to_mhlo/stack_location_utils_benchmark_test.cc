/* Copyright 2026 The OpenXLA Authors.

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

// Benchmarks the HLO to MLIR import of a module whose instructions share a few
// deep call stacks, the shape of a traced JAX program: every instruction
// carries a stack frame id, the frames form long chains, and many instructions
// end in the same frame. Importing such a module used to rebuild the whole
// location chain of every instruction, O(instructions x stack depth) attribute
// lookups; with the per frame memo each frame is built once.
//
// Run the benchmarks with --benchmark_filter=all (as a test argument under the
// test runner, with the test output shown).

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "benchmark/benchmark.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_metadata.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/stack_frames.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

// A chain of num_instructions unary ops. The frame table holds num_stacks
// call stacks of depth `depth` that share their outer half; instruction i ends
// in stack i % num_stacks, so every stack is referenced by many instructions.
std::unique_ptr<HloModule> MakeModuleWithSharedStacks(int num_instructions,
                                                      int num_stacks,
                                                      int depth) {
  auto module = std::make_unique<HloModule>("shared_stacks", HloModuleConfig());
  StackFrames& frames = module->mutable_stack_frames();

  StackFrameId shared_prefix;
  for (int d = 0; d < depth / 2; ++d) {
    std::string function_name = absl::StrCat("outer", d);
    shared_prefix = frames.AddStackFrame(
        {"common.py", function_name, d + 1, 1, d + 1, 1, shared_prefix});
  }
  std::vector<StackFrameId> leaves;
  leaves.reserve(num_stacks);
  for (int s = 0; s < num_stacks; ++s) {
    std::string file_name = absl::StrCat("stack", s, ".py");
    StackFrameId id = shared_prefix;
    for (int d = depth / 2; d < depth; ++d) {
      std::string function_name = absl::StrCat("inner", d);
      id = frames.AddStackFrame(
          {file_name, function_name, d + 1, s + 1, d + 1, s + 1, id});
    }
    leaves.push_back(id);
  }

  HloComputation::Builder builder("entry");
  Shape shape = ShapeUtil::MakeShape(F32, {8});
  HloInstruction* value =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "x"));
  for (int i = 0; i < num_instructions; ++i) {
    value = builder.AddInstruction(
        HloInstruction::CreateUnary(shape, HloOpcode::kNegate, value));
    OpMetadata metadata;
    metadata.set_op_name(absl::StrCat("jit(fn)/negate", i));
    metadata.set_stack_frame_id(leaves[i % num_stacks].value);
    value->set_metadata(metadata);
  }
  module->AddEntryComputation(builder.Build());
  return module;
}

int CallStackDepth(mlir::Location location) {
  int depth = 0;
  while (true) {
    if (auto name_loc = mlir::dyn_cast<mlir::NameLoc>(location)) {
      if (mlir::isa<mlir::FileLineColLoc>(name_loc.getChildLoc())) {
        return depth + 1;
      }
      location = name_loc.getChildLoc();
    } else if (auto call_site = mlir::dyn_cast<mlir::CallSiteLoc>(location)) {
      ++depth;
      location = call_site.getCaller();
    } else {
      return depth;
    }
  }
}

TEST(StackLocationBenchmarkTest, ImportsSharedStacks) {
  constexpr int kDepth = 6;
  std::unique_ptr<HloModule> module = MakeModuleWithSharedStacks(
      /*num_instructions=*/10, /*num_stacks=*/3, kDepth);
  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
      ConvertHloToMlirHlo(context, module.get(),
                          /*import_all_computations=*/true,
                          /*flatten_computation_args_result=*/true));
  int negates = 0;
  mlir_module->walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() == "mhlo.negate") {
      ++negates;
      EXPECT_EQ(CallStackDepth(op->getLoc()), kDepth);
    }
  });
  EXPECT_EQ(negates, 10);
}

void BM_ImportWithSharedStacks(benchmark::State& state) {
  const int num_instructions = state.range(0);
  const int depth = state.range(1);
  std::unique_ptr<HloModule> module =
      MakeModuleWithSharedStacks(num_instructions, /*num_stacks=*/256, depth);
  for (auto s : state) {
    // A fresh context per import, as in a compile: nothing is uniqued yet.
    mlir::MLIRContext context;
    auto mlir_module =
        ConvertHloToMlirHlo(context, module.get(),
                            /*import_all_computations=*/true,
                            /*flatten_computation_args_result=*/true);
    CHECK_OK(mlir_module.status());
    benchmark::DoNotOptimize(mlir_module);
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          num_instructions);
}

BENCHMARK(BM_ImportWithSharedStacks)
    ->ArgPair(10000, 32)
    ->ArgPair(50000, 32)
    ->Unit(benchmark::kMillisecond);

}  // namespace
}  // namespace xla
