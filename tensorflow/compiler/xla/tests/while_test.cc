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

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace se = ::perftools::gputools;

namespace xla {
namespace {

class WhileTest : public ClientLibraryTestBase {};

// Tests a while node when the result type T is S32.
//
// int32 result = 0;
// while (result < 5) {
//   result = result + 1;
// }
TEST_F(WhileTest, WhileWithScalarResult) {
  auto result_shape = ShapeUtil::MakeShape(S32, {});

  // Create a computation for the condition: repeat for 5 iterations.
  Computation condition;
  {
    ComputationBuilder builder(client_, "condition");
    auto prev = builder.Parameter(0, result_shape, "prev");
    builder.Gt(builder.ConstantR0<int32>(5), prev);
    condition = builder.Build().ConsumeValueOrDie();
  }

  // Create a computation for the body: add 1 to the result variable.
  Computation body;
  {
    ComputationBuilder builder(client_, "body");
    auto prev = builder.Parameter(0, result_shape, "prev");
    auto input = builder.ConstantR0<int32>(1);
    auto result = builder.Add(input, prev);
    body = builder.Build().ConsumeValueOrDie();
  }

  // Create a While node with computations for the condition and the body.
  ComputationBuilder builder(client_, TestName());
  auto init = builder.ConstantR0<int32>(0);
  auto result = builder.While(condition, body, init);
  auto shape = builder.GetShape(result).ConsumeValueOrDie();

  ComputeAndCompareR0<int32>(&builder, 5, {});
}

// Tests a while node when the result type T is a vector.
//
// All constants are chosen to produce exact results.
// vector<float> result(0);
// while (result.sum() < 15.5f) {
//   result = result + vector<float>(0);
// }
// TODO(b/29185393): does not terminate on CPU.
TEST_F(WhileTest, DISABLED_WhileWithEmptyVectorResult) {
  Shape result_shape = ShapeUtil::MakeShape(F32, {0});

  // Create a computation for the reduction.
  Computation add;
  {
    ComputationBuilder builder(client_, "add");
    auto x = builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "x");
    auto y = builder.Parameter(1, ShapeUtil::MakeShape(F32, {}), "y");
    builder.Add(x, y);
    add = builder.Build().ConsumeValueOrDie();
  }

  // Create a computation for the condition.
  // Repeat until the sum of the result vector is less than 15.5f.
  Computation condition;
  {
    ComputationBuilder builder(client_, "condition");
    auto prev = builder.Parameter(0, result_shape, "prev");
    auto sum = builder.Reduce(prev, builder.ConstantR0<float>(0.0f), add,
                              /*dimensions_to_reduce=*/{0});
    auto test = builder.Gt(builder.ConstantR0<float>(15.5f), sum);
    condition = builder.Build().ConsumeValueOrDie();
  }

  // Create a computation for the body.
  // Add a constant vector of 1.f to the result vector.
  Computation body;
  {
    ComputationBuilder builder(client_, "body");
    auto prev = builder.Parameter(0, result_shape, "prev");
    auto input = builder.ConstantR1<float>({});
    auto result = builder.Add(input, prev);
    body = builder.Build().ConsumeValueOrDie();
  }

  // Create a While node with computations for the condition and the body.
  ComputationBuilder builder(client_, "while");
  auto init = builder.ConstantR1<float>({});
  auto result = builder.While(condition, body, init);
  VLOG(2) << "while = " << ShapeUtil::HumanString(
                               *builder.GetShape(result).ConsumeValueOrDie());

  ComputeAndCompareR1<float>(&builder, {}, {}, ErrorSpec(0.0001));
}

// Tests a while node when the result type T is a vector.
//
// All constants are chosen to produce exact results.
// vector<float> result(8, 0.0f);
// while (result.sum() < 15.5f) {
//   result = result + vector<float>(8, 0.125f);
// }
TEST_F(WhileTest, WhileWithVectorResult) {
  Shape result_shape = ShapeUtil::MakeShape(F32, {8});

  // Create a computation for the reduction.
  Computation add;
  {
    ComputationBuilder builder(client_, "add");
    auto x = builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "x");
    auto y = builder.Parameter(1, ShapeUtil::MakeShape(F32, {}), "y");
    builder.Add(x, y);
    add = builder.Build().ConsumeValueOrDie();
  }

  // Create a computation for the condition.
  // Repeat until the sum of the result vector is less than 5.5f.
  Computation condition;
  {
    ComputationBuilder builder(client_, "condition");
    auto prev = builder.Parameter(0, result_shape, "prev");
    auto sum = builder.Reduce(prev, builder.ConstantR0<float>(0.0f), add,
                              /*dimensions_to_reduce=*/{0});
    auto test = builder.Gt(builder.ConstantR0<float>(15.5f), sum);
    condition = builder.Build().ConsumeValueOrDie();
  }

  // Create a computation for the body.
  // Add a constant vector of 1.f to the result vector.
  Computation body;
  {
    ComputationBuilder builder(client_, "body");
    auto prev = builder.Parameter(0, result_shape, "prev");
    auto input = builder.ConstantR1<float>(8, 0.125f);
    auto result = builder.Add(input, prev);
    body = builder.Build().ConsumeValueOrDie();
  }

  // Create a While node with computations for the condition and the body.
  ComputationBuilder builder(client_, "while");
  auto init = builder.ConstantR1<float>(8, 0.f);
  auto result = builder.While(condition, body, init);
  VLOG(2) << "while = " << ShapeUtil::HumanString(
                               *builder.GetShape(result).ConsumeValueOrDie());

  // Individual elements with increase by 1/8 each time through the loop, so
  // the sum will increase by 1.0.  It will first be >15.5 when the elements
  // have all reached 2.0.
  std::vector<float> expected = {2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

// Tests a while node when the result type T is a Tuple.
//
// tuple<int32, vector<float>> result(0, vector<float>(10, 0.0f));
// while (get<0>(result) < 5) {
//   get<0>(result) = get<0>(result) + 1;
//   get<1>(result) = get<1>(result) + vector<float>(10, 1.0f);
// }
TEST_F(WhileTest, WhileWithTupleResult) {
  std::vector<Shape> shape_elements = {ShapeUtil::MakeShape(S32, {}),
                                       ShapeUtil::MakeShape(F32, {10})};
  Shape result_shape = ShapeUtil::MakeTupleShape(shape_elements);

  // Create a computation for the condition.
  // Repeat for 5 iterations.
  Computation condition;
  {
    ComputationBuilder builder(client_, "condition");
    auto prev = builder.Parameter(0, result_shape, "prev");
    auto iteration = builder.GetTupleElement(prev, 0);
    builder.Gt(builder.ConstantR0<int32>(5), iteration);
    condition = builder.Build().ConsumeValueOrDie();
  }

  // Create a computation for the body.
  // Add 1 to the iteration variable and add a constant vector of 1.0f to
  // the weight variable, both of which are tuple elements.
  Computation body;
  {
    ComputationBuilder builder(client_, "body");
    auto prev = builder.Parameter(0, result_shape, "prev");
    auto iteration = builder.GetTupleElement(prev, 0);
    auto weights = builder.GetTupleElement(prev, 1);
    auto input = builder.ConstantR1<float>(10, 1.f);
    auto new_weights = builder.Add(weights, input);
    auto result = builder.Tuple(
        {builder.Add(iteration, builder.ConstantR0<int32>(1)), new_weights});
    body = builder.Build().ConsumeValueOrDie();
  }

  // Create a While node with computations for the condition and the body.
  ComputationBuilder builder(client_, "while");
  auto init = builder.Tuple(
      {builder.ConstantR0<int32>(0), builder.ConstantR1<float>(10, 0.f)});
  auto result = builder.While(condition, body, init);
  VLOG(2) << "while = " << ShapeUtil::HumanString(
                               *builder.GetShape(result).ConsumeValueOrDie());

  auto expected_counter = LiteralUtil::CreateR0<int32>(5);
  auto expected_data = LiteralUtil::CreateR1<float>(
      {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f});
  auto expected =
      LiteralUtil::MakeTuple({expected_counter.get(), expected_data.get()});
  VLOG(2) << "expected = " << ShapeUtil::HumanString(expected->shape());
  ComputeAndCompareTuple(&builder, *expected, {}, ErrorSpec(0.0001));
}

// WhileTest that uses DynamicUpdateSlice instruction in body computation.
// Loop state tuple element 1 has as its single user operand(0) of
// DynamicUpdateSlice, which will trigger in-place dynamic slice update on GPU.
XLA_TEST_F(WhileTest, WhileWithDynamicUpdateSlice) {
  std::vector<Shape> shape_elements = {ShapeUtil::MakeShape(S32, {}),
                                       ShapeUtil::MakeShape(F32, {10})};
  Shape result_shape = ShapeUtil::MakeTupleShape(shape_elements);

  // Create a computation for the condition.
  // Repeat for 5 iterations.
  Computation condition;
  {
    ComputationBuilder builder(client_, "condition");
    auto prev = builder.Parameter(0, result_shape, "prev");
    auto iteration = builder.GetTupleElement(prev, 0);
    builder.Gt(builder.ConstantR0<int32>(5), iteration);
    condition = builder.Build().ConsumeValueOrDie();
  }

  // Create a computation for the body.
  // Add 1 to the iteration variable and add a constant vector of 1.0f to
  // the weight variable, both of which are tuple elements.
  Computation body;
  {
    ComputationBuilder builder(client_, "body");
    auto prev = builder.Parameter(0, result_shape, "prev");
    // TupleElement 0
    auto iteration = builder.GetTupleElement(prev, 0);
    auto out0 = builder.Add(iteration, builder.ConstantR0<int32>(1));
    // TupleElement 1
    auto input = builder.GetTupleElement(prev, 1);
    // Update.
    auto update = builder.ConvertElementType(builder.Broadcast(out0, {2}), F32);
    // Starts = iteration * 2;
    auto starts = builder.Reshape(
        builder.Mul(iteration, builder.ConstantR0<int32>(2)), {1});
    // UpdateSlice.
    auto out1 = builder.DynamicUpdateSlice(input, update, starts);

    auto result = builder.Tuple({out0, out1});
    body = builder.Build().ConsumeValueOrDie();
  }

  // Create a While node with computations for the condition and the body.
  ComputationBuilder builder(client_, "while");
  auto init = builder.Tuple(
      {builder.ConstantR0<int32>(0), builder.ConstantR1<float>(10, 0.f)});
  auto result = builder.While(condition, body, init);
  VLOG(2) << "while = "
          << ShapeUtil::HumanString(
                 *builder.GetShape(result).ConsumeValueOrDie());

  auto expected_counter = LiteralUtil::CreateR0<int32>(5);
  auto expected_data = LiteralUtil::CreateR1<float>(
      {1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f, 4.0f, 4.0f, 5.0f, 5.0f});
  auto expected =
      LiteralUtil::MakeTuple({expected_counter.get(), expected_data.get()});
  VLOG(2) << "expected = " << ShapeUtil::HumanString(expected->shape());
  ComputeAndCompareTuple(&builder, *expected, {}, ErrorSpec(0.0001));
}

// Tests a while node when the result type T is a vector of S32.
//
// int32 result = (0, 0, 0, 0, 0, 0);
// while (result[0] < count) {
//   result += (1, U[0, 100], U[0, 100], U[0, 100], U[0, 100], U[0, 100]);
// }
//
// This test misuses a vector to represent a pair:
//   ((iteration, (random vector))).
//
// Note: this test currently only tests generating random values within a loop.
// Per backend the values generated can be different as the different backends
// use different random number generators.
// TODO(b/32240857): Extend test to verify outputs.
TEST_F(WhileTest, WhileWithPrngScalarResult) {
  auto v6s32 = ShapeUtil::MakeShape(S32, {6});

  // Create a computation for the condition: repeat for count iterations.
  auto build_condition = [this, v6s32](int count) {
    ComputationBuilder builder(client_, TestName());
    auto prev = builder.Reshape(
        builder.Slice(builder.Parameter(0, v6s32, "prev"), {0}, {1}), {0}, {});
    builder.Gt(builder.ConstantR0<int32>(count), prev);
    return builder.Build().ConsumeValueOrDie();
  };

  // Create a computation for the body: add 1 to the result variable.
  Computation body;
  {
    ComputationBuilder builder(client_, "body");
    auto prev = builder.Parameter(0, v6s32, "prev");
    auto inc = builder.ConcatInDim(
        {builder.ConstantR1<int32>({1}),
         builder.RngUniform(builder.ConstantR0<int32>(0),
                            builder.ConstantR0<int32>(100),
                            ShapeUtil::MakeShape(S32, {5}))},
        0);
    auto result = builder.Add(inc, prev);
    body = builder.Build().ConsumeValueOrDie();
  }

  // Create a While node with computations for the condition and the body.
  auto while_loop = [this, &body, build_condition](int count) {
    ComputationBuilder builder(client_, TestName());
    auto init = builder.ConstantR1<int32>({0, 0, 0, 0, 0, 0});
    auto result = builder.While(build_condition(count), body, init);
    auto shape = builder.GetShape(result).ConsumeValueOrDie();
    return builder.Build();
  };

  for (int i = 1; i < 4; ++i) {
    TF_ASSIGN_OR_ASSERT_OK(auto computation, while_loop(i));

    ExecutionOptions execution_options;
    execution_options.set_seed(65);
    TF_ASSIGN_OR_ASSERT_OK(
        auto result,
        client_->ExecuteAndTransfer(computation, {}, &execution_options));
  }
}

void BM_WhileLoop(int num_iters) {
  // Benchmark a simple kernel to measure while loop overheads.
  tensorflow::testing::StopTiming();

  se::Platform* platform = PlatformUtil::GetDefaultPlatform().ValueOrDie();
  auto executors = PlatformUtil::GetStreamExecutors(platform).ValueOrDie();
  StreamExecutorMemoryAllocator allocator(platform, executors);
  LocalClient* client =
      ClientLibrary::GetOrCreateLocalClient(platform).ValueOrDie();

  Shape loop_state_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(S32, {}), ShapeUtil::MakeShape(F32, {10})});

  // Create while condition computation with 'loop_limit'.
  const int32 loop_limit = 100;
  Computation condition;
  {
    ComputationBuilder builder(client, "condition");
    auto prev = builder.Parameter(0, loop_state_shape, "prev");
    auto iteration = builder.GetTupleElement(prev, 0);
    builder.Lt(iteration, builder.ConstantR0<int32>(loop_limit));
    condition = builder.Build().ConsumeValueOrDie();
  }

  // Create while body computation with unit loop increment.
  Computation body;
  {
    ComputationBuilder builder(client, "body");
    auto prev = builder.Parameter(0, loop_state_shape, "prev");
    auto iteration = builder.GetTupleElement(prev, 0);
    auto weights = builder.GetTupleElement(prev, 1);
    auto one = builder.ConstantR0<int32>(1);
    auto next_iteration = builder.Add(iteration, one);
    auto one_vec = builder.ConstantR1<float>(10, 1.f);
    auto new_weights = builder.Add(weights, one_vec);
    auto result = builder.Tuple({next_iteration, new_weights});
    body = builder.Build().ConsumeValueOrDie();
  }

  // Create a While instruction.
  ComputationBuilder builder(client, "while");
  auto init = builder.Tuple(
      {builder.ConstantR0<int32>(0), builder.ConstantR1<float>(10, 0.f)});
  builder.While(condition, body, init);
  auto computation = builder.Build().ConsumeValueOrDie();

  // Run some warm-up executions.
  LocalExecuteOptions options;
  options.set_allocator(&allocator);
  const int kWarmups = 2;
  for (int i = 0; i < kWarmups; ++i) {
    auto result = client->ExecuteLocally(computation, {}, options);
    ASSERT_TRUE(result.ok());
  }

  // Run benchmark.
  tensorflow::testing::StartTiming();
  for (int i = 0; i < num_iters; ++i) {
    auto result = client->ExecuteLocally(computation, {}, options);
    ASSERT_TRUE(result.ok());
  }
}

// TODO(b/32470510): Benchmark fails on parallel CPU backend.
#ifndef XLA_TEST_BACKEND_CPU_PARALLEL
BENCHMARK(BM_WhileLoop);
#endif

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list;
  xla::legacy_flags::AppendCpuCompilerFlags(&flag_list);
  xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << "\n" << usage;
    return 2;
  }
  testing::InitGoogleTest(&argc, argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return 2;
  }
  tensorflow::testing::RunBenchmarks();
  return RUN_ALL_TESTS();
}
