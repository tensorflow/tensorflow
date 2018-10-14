/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.h"

#include <algorithm>
#include <unordered_set>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/gpu/stream_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_verified_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace gpu {

class GpuHloScheduleTest : public HloVerifiedTestBase {
 protected:
  using HloVec = std::vector<const HloInstruction*>;

  // Pre-canned shapes.
  Shape f32_2x2_ = ShapeUtil::MakeShape(F32, {2, 2});

  static std::unique_ptr<GpuHloSchedule> BuildGpuHloSchedule(
      const HloModule& module, const StreamAssignment& streams) {
    return GpuHloSchedule::Build(module, streams, /*pointer_size=*/8)
        .ConsumeValueOrDie();
  }

  std::unique_ptr<HloModule> CreateNewModule() {
    HloModuleConfig config;
    auto debug_options = GetDebugOptionsForTest();
    debug_options.set_xla_gpu_disable_multi_streaming(false);
    config.set_debug_options(debug_options);
    return absl::make_unique<HloModule>("test_module", config);
  }

  HloVec RemoveHlo(const HloVec& input,
                   const std::unordered_set<const HloInstruction*>& remove) {
    HloVec result(input);
    result.erase(std::remove_if(result.begin(), result.end(),
                                [&remove](const HloInstruction* x) {
                                  return remove.count(x) > 0;
                                }),
                 result.end());
    return result;
  }
};

// Test of a single stream, where data dependencies fully determine the
// execution order.
TEST_F(GpuHloScheduleTest, SequentialMatMul) {
  HloComputation::Builder builder("entry_computation");
  HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, f32_2x2_, /*name=*/"x"));
  HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, f32_2x2_, /*name=*/"y"));
  HloInstruction* z = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, f32_2x2_, /*name=*/"z"));
  HloInstruction* dot1 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, x, y));
  HloInstruction* dot2 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, dot1, z));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build(dot2));

  std::unique_ptr<StreamAssignment> streams = AssignStreams(*module);
  EXPECT_EQ(streams->StreamNumberForHlo(*dot1),
            streams->StreamNumberForHlo(*dot2));

  auto schedule = BuildGpuHloSchedule(*module, *streams);
  // Remove parameters, which are unordered.
  EXPECT_EQ(RemoveHlo(schedule->ThunkLaunchOrder(), {x, y, z}),
            HloVec({dot1, dot2}));

  // Parameters x,y,z are mutually unordered, while dot1 and dot2 are
  // transitively ordered by operands.
  auto order = schedule->ConsumeHloOrdering();
  EXPECT_TRUE(order->ExecutesBefore(x, dot1));
  EXPECT_TRUE(order->ExecutesBefore(x, dot2));
  EXPECT_TRUE(order->ExecutesBefore(y, dot1));
  EXPECT_TRUE(order->ExecutesBefore(y, dot2));
  EXPECT_TRUE(order->ExecutesBefore(z, dot2));
  EXPECT_TRUE(order->ExecutesBefore(dot1, dot2));

  EXPECT_FALSE(order->ExecutesBefore(x, x));
  EXPECT_FALSE(order->ExecutesBefore(x, y));
  EXPECT_FALSE(order->ExecutesBefore(x, z));
  EXPECT_FALSE(order->ExecutesBefore(y, x));
  EXPECT_FALSE(order->ExecutesBefore(y, y));
  EXPECT_FALSE(order->ExecutesBefore(y, z));
  EXPECT_FALSE(order->ExecutesBefore(z, x));
  EXPECT_FALSE(order->ExecutesBefore(z, y));
  EXPECT_FALSE(order->ExecutesBefore(z, z));
  EXPECT_FALSE(order->ExecutesBefore(z, dot1));
  EXPECT_FALSE(order->ExecutesBefore(dot1, x));
  EXPECT_FALSE(order->ExecutesBefore(dot1, y));
  EXPECT_FALSE(order->ExecutesBefore(dot1, z));
  EXPECT_FALSE(order->ExecutesBefore(dot1, dot1));
  EXPECT_FALSE(order->ExecutesBefore(dot2, x));
  EXPECT_FALSE(order->ExecutesBefore(dot2, y));
  EXPECT_FALSE(order->ExecutesBefore(dot2, z));
  EXPECT_FALSE(order->ExecutesBefore(dot2, dot1));
  EXPECT_FALSE(order->ExecutesBefore(dot2, dot2));
}

// Test of a single stream, where data dependencies do not fully determine the
// execution order, but the stream assignment does.
TEST_F(GpuHloScheduleTest, SequentialAdd) {
  HloComputation::Builder builder("entry_computation");
  HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, f32_2x2_, /*name=*/"x"));
  HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, f32_2x2_, /*name=*/"y"));
  HloInstruction* z = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, f32_2x2_, /*name=*/"z"));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_2x2_, HloOpcode::kAdd, x, y));
  HloInstruction* add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_2x2_, HloOpcode::kAdd, y, z));
  HloInstruction* add3 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_2x2_, HloOpcode::kAdd, add1, add2));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build(add3));

  std::unique_ptr<StreamAssignment> streams = AssignStreams(*module);
  EXPECT_EQ(streams->StreamNumberForHlo(*add1),
            streams->StreamNumberForHlo(*add2));
  EXPECT_EQ(streams->StreamNumberForHlo(*add1),
            streams->StreamNumberForHlo(*add3));

  auto schedule = BuildGpuHloSchedule(*module, *streams);
  // Remove parameters, which are unordered.
  EXPECT_EQ(RemoveHlo(schedule->ThunkLaunchOrder(), {x, y, z}),
            HloVec({add1, add2, add3}));

  // Parameters x,y,z are mutually unordered, while add1, add2 and add3 are
  // transitively ordered by operands.
  auto order = schedule->ConsumeHloOrdering();
  EXPECT_TRUE(order->ExecutesBefore(x, add1));
  EXPECT_TRUE(order->ExecutesBefore(x, add2));
  EXPECT_TRUE(order->ExecutesBefore(x, add3));
  EXPECT_TRUE(order->ExecutesBefore(y, add1));
  EXPECT_TRUE(order->ExecutesBefore(y, add2));
  EXPECT_TRUE(order->ExecutesBefore(y, add3));
  EXPECT_TRUE(order->ExecutesBefore(z, add2));
  EXPECT_TRUE(order->ExecutesBefore(z, add3));
  EXPECT_TRUE(order->ExecutesBefore(add1, add3));
  EXPECT_TRUE(order->ExecutesBefore(add2, add3));
  // The HLO graph does not define an ordering for add1 and add2, but their
  // assignment onto the same stream does define an ordering.
  if (order->ExecutesBefore(add1, add2)) {
    EXPECT_FALSE(order->ExecutesBefore(add2, add1));
  } else {
    EXPECT_TRUE(order->ExecutesBefore(add2, add1));
    EXPECT_FALSE(order->ExecutesBefore(add1, add2));
  }

  EXPECT_FALSE(order->ExecutesBefore(x, x));
  EXPECT_FALSE(order->ExecutesBefore(x, y));
  EXPECT_FALSE(order->ExecutesBefore(x, z));
  EXPECT_FALSE(order->ExecutesBefore(y, x));
  EXPECT_FALSE(order->ExecutesBefore(y, y));
  EXPECT_FALSE(order->ExecutesBefore(y, z));
  EXPECT_FALSE(order->ExecutesBefore(z, x));
  EXPECT_FALSE(order->ExecutesBefore(z, y));
  EXPECT_FALSE(order->ExecutesBefore(z, z));
  EXPECT_FALSE(order->ExecutesBefore(z, add1));
  EXPECT_FALSE(order->ExecutesBefore(add1, x));
  EXPECT_FALSE(order->ExecutesBefore(add1, y));
  EXPECT_FALSE(order->ExecutesBefore(add1, z));
  EXPECT_FALSE(order->ExecutesBefore(add1, add1));
  EXPECT_FALSE(order->ExecutesBefore(add2, x));
  EXPECT_FALSE(order->ExecutesBefore(add2, y));
  EXPECT_FALSE(order->ExecutesBefore(add2, z));
  EXPECT_FALSE(order->ExecutesBefore(add2, add2));
}

// Test of two streams.
TEST_F(GpuHloScheduleTest, ConcurrentMatMul) {
  HloComputation::Builder builder("entry_computation");
  HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, f32_2x2_, /*name=*/"x"));
  HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, f32_2x2_, /*name=*/"y"));
  HloInstruction* dot1 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, x, y));
  HloInstruction* dot2 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, y, x));
  HloInstruction* add =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, dot1, dot2));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build(add));

  std::unique_ptr<StreamAssignment> streams = AssignStreams(*module);
  EXPECT_NE(streams->StreamNumberForHlo(*dot1),
            streams->StreamNumberForHlo(*dot2));

  auto schedule = BuildGpuHloSchedule(*module, *streams);
  // Remove parameters, which are unordered.
  HloVec thunk_launch_order = RemoveHlo(schedule->ThunkLaunchOrder(), {x, y});
  EXPECT_TRUE(thunk_launch_order == HloVec({dot1, dot2, add}) ||
              thunk_launch_order == HloVec({dot2, dot1, add}));

  // Parameters x,y are mutually unordered, while dot1, dot2 and add are
  // transitively ordered by operands.
  auto order = schedule->ConsumeHloOrdering();
  EXPECT_TRUE(order->ExecutesBefore(x, dot1));
  EXPECT_TRUE(order->ExecutesBefore(x, dot2));
  EXPECT_TRUE(order->ExecutesBefore(y, dot1));
  EXPECT_TRUE(order->ExecutesBefore(y, dot2));
  EXPECT_TRUE(order->ExecutesBefore(dot1, add));
  EXPECT_TRUE(order->ExecutesBefore(dot2, add));

  EXPECT_FALSE(order->ExecutesBefore(x, x));
  EXPECT_FALSE(order->ExecutesBefore(x, y));
  EXPECT_FALSE(order->ExecutesBefore(y, x));
  EXPECT_FALSE(order->ExecutesBefore(y, y));
  EXPECT_FALSE(order->ExecutesBefore(dot1, x));
  EXPECT_FALSE(order->ExecutesBefore(dot1, y));
  EXPECT_FALSE(order->ExecutesBefore(dot1, dot1));
  EXPECT_FALSE(order->ExecutesBefore(dot1, dot2));
  EXPECT_FALSE(order->ExecutesBefore(dot2, x));
  EXPECT_FALSE(order->ExecutesBefore(dot2, y));
  EXPECT_FALSE(order->ExecutesBefore(dot2, dot1));
  EXPECT_FALSE(order->ExecutesBefore(dot2, dot2));
  EXPECT_FALSE(order->ExecutesBefore(add, x));
  EXPECT_FALSE(order->ExecutesBefore(add, y));
  EXPECT_FALSE(order->ExecutesBefore(add, dot1));
  EXPECT_FALSE(order->ExecutesBefore(add, dot2));
  EXPECT_FALSE(order->ExecutesBefore(add, add));
}

// Test of multiple streams.
TEST_F(GpuHloScheduleTest, LatticeMatMul) {
  //      d00      -- layer 0
  //     /   \
  //   d10   d11   -- layer 1
  //  /   \ /   \
  // d20  d21  d22 -- layer 2
  //  \   / \   /
  //   d30   d31   -- layer 3
  //     \   /
  //      d40      -- layer 4
  HloComputation::Builder builder("entry_computation");
  std::vector<HloInstruction*> params;
  params.reserve(6);
  for (int i = 0; i < 6; ++i) {
    params.push_back(builder.AddInstruction(HloInstruction::CreateParameter(
        i, f32_2x2_, /*name=*/absl::StrFormat("param%d", i))));
  }
  HloInstruction* d00 = builder.AddInstruction(
      CreateCanonicalDot(f32_2x2_, params[2], params[3]));
  HloInstruction* d10 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, params[1], d00));
  HloInstruction* d11 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, d00, params[4]));
  HloInstruction* d20 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, params[0], d10));
  HloInstruction* d21 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, d10, d11));
  HloInstruction* d22 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, d11, params[5]));
  HloInstruction* d30 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, d20, d21));
  HloInstruction* d31 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, d21, d22));
  HloInstruction* d40 =
      builder.AddInstruction(CreateCanonicalDot(f32_2x2_, d30, d31));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build(d40));

  std::unique_ptr<StreamAssignment> streams = AssignStreams(*module);
  // The two dots on layer 1 are concurrent.
  EXPECT_NE(streams->StreamNumberForHlo(*d10),
            streams->StreamNumberForHlo(*d11));
  // The three dots on layer 2 are concurrent.
  EXPECT_NE(streams->StreamNumberForHlo(*d20),
            streams->StreamNumberForHlo(*d21));
  EXPECT_NE(streams->StreamNumberForHlo(*d20),
            streams->StreamNumberForHlo(*d22));
  EXPECT_NE(streams->StreamNumberForHlo(*d21),
            streams->StreamNumberForHlo(*d22));
  // The two dots on layer 3 are concurrent.
  EXPECT_NE(streams->StreamNumberForHlo(*d30),
            streams->StreamNumberForHlo(*d31));

  // We don't check the thunk launch order, since there are many valid total
  // orders, and it's annoying to express.
  auto schedule = BuildGpuHloSchedule(*module, *streams);

  auto order = schedule->ConsumeHloOrdering();
  const HloVec all_params(
      {params[0], params[1], params[2], params[3], params[4], params[5]});
  const HloVec all_ops({d00, d10, d11, d20, d21, d22, d30, d31, d40});

  // Parameters are mutually unordered, and never execute before ops.
  for (const HloInstruction* param : all_params) {
    for (const HloInstruction* param2 : all_params) {
      EXPECT_FALSE(order->ExecutesBefore(param, param2));
    }
    for (const HloInstruction* op : all_ops) {
      EXPECT_FALSE(order->ExecutesBefore(op, param));
    }
  }

  // Check ordering of params before ops.
  for (const HloInstruction* op : all_ops) {
    if (op == d20 || op == d30 || op == d40) {
      EXPECT_TRUE(order->ExecutesBefore(params[0], op));
    } else {
      EXPECT_FALSE(order->ExecutesBefore(params[0], op));
    }
    if (op != d00 && op != d11 && op != d22) {
      EXPECT_TRUE(order->ExecutesBefore(params[1], op));
    } else {
      EXPECT_FALSE(order->ExecutesBefore(params[1], op));
    }
    EXPECT_TRUE(order->ExecutesBefore(params[2], op));
    EXPECT_TRUE(order->ExecutesBefore(params[3], op));
    if (op != d00 && op != d10 && op != d20) {
      EXPECT_TRUE(order->ExecutesBefore(params[4], op));
    } else {
      EXPECT_FALSE(order->ExecutesBefore(params[4], op));
    }
    if (op == d22 || op == d31 || op == d40) {
      EXPECT_TRUE(order->ExecutesBefore(params[5], op));
    } else {
      EXPECT_FALSE(order->ExecutesBefore(params[5], op));
    }
  }

  // Check ordering of ops before ops.
  for (const HloInstruction* op : all_ops) {
    if (op != d00) {
      EXPECT_TRUE(order->ExecutesBefore(d00, op));
    } else {
      EXPECT_FALSE(order->ExecutesBefore(d00, op));
    }

    if (op == d20 || op == d21 || op == d30 || op == d31 || op == d40) {
      EXPECT_TRUE(order->ExecutesBefore(d10, op));
    } else {
      EXPECT_FALSE(order->ExecutesBefore(d10, op));
    }

    if (op == d21 || op == d22 || op == d30 || op == d31 || op == d40) {
      EXPECT_TRUE(order->ExecutesBefore(d11, op));
    } else {
      EXPECT_FALSE(order->ExecutesBefore(d11, op));
    }

    if (op == d30 || op == d40) {
      EXPECT_TRUE(order->ExecutesBefore(d20, op));
    } else {
      EXPECT_FALSE(order->ExecutesBefore(d20, op));
    }

    if (op == d30 || op == d31 || op == d40) {
      EXPECT_TRUE(order->ExecutesBefore(d21, op));
    } else {
      EXPECT_FALSE(order->ExecutesBefore(d21, op));
    }

    if (op == d31 || op == d40) {
      EXPECT_TRUE(order->ExecutesBefore(d22, op));
    } else {
      EXPECT_FALSE(order->ExecutesBefore(d22, op));
    }

    if (op == d40) {
      EXPECT_TRUE(order->ExecutesBefore(d30, op));
      EXPECT_TRUE(order->ExecutesBefore(d31, op));
    } else {
      EXPECT_FALSE(order->ExecutesBefore(d30, op));
      EXPECT_FALSE(order->ExecutesBefore(d31, op));
    }

    EXPECT_FALSE(order->ExecutesBefore(d40, op));
  }
}

}  // namespace gpu
}  // namespace xla
