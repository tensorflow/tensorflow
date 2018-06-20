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

#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/aot/tests/test_graph_tfadd.h"
#include "tensorflow/compiler/aot/tests/test_graph_tfadd_with_ckpt.h"
#include "tensorflow/compiler/aot/tests/test_graph_tfadd_with_ckpt_saver.h"
#include "tensorflow/compiler/aot/tests/test_graph_tfassert_eq.h"
#include "tensorflow/compiler/aot/tests/test_graph_tfcond.h"
#include "tensorflow/compiler/aot/tests/test_graph_tffunction.h"
#include "tensorflow/compiler/aot/tests/test_graph_tfgather.h"
#include "tensorflow/compiler/aot/tests/test_graph_tfmatmul.h"
#include "tensorflow/compiler/aot/tests/test_graph_tfmatmulandadd.h"
#include "tensorflow/compiler/aot/tests/test_graph_tfmatmulandadd_with_profiling.h"
#include "tensorflow/compiler/aot/tests/test_graph_tfsplits.h"
#include "tensorflow/compiler/xla/service/hlo_profile_printer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace tfcompile {
namespace {

using ::testing::HasSubstr;
using ::testing::IsSupersetOf;

TEST(TFCompileTest, Add) {
  AddComp add;
  EXPECT_EQ(add.arg0_data(), add.args()[0]);
  EXPECT_EQ(add.arg1_data(), add.args()[1]);

  add.arg0() = 1;
  add.arg1() = 2;
  EXPECT_TRUE(add.Run());
  EXPECT_EQ(add.error_msg(), "");
  EXPECT_EQ(add.result0(), 3);
  EXPECT_EQ(add.result0_data()[0], 3);
  EXPECT_EQ(add.result0_data(), add.results()[0]);

  add.arg0_data()[0] = 123;
  add.arg1_data()[0] = 456;
  EXPECT_TRUE(add.Run());
  EXPECT_EQ(add.error_msg(), "");
  EXPECT_EQ(add.result0(), 579);
  EXPECT_EQ(add.result0_data()[0], 579);
  EXPECT_EQ(add.result0_data(), add.results()[0]);

  const AddComp& add_const = add;
  EXPECT_EQ(add_const.error_msg(), "");
  EXPECT_EQ(add_const.arg0(), 123);
  EXPECT_EQ(add_const.arg0_data()[0], 123);
  EXPECT_EQ(add_const.arg0_data(), add.args()[0]);
  EXPECT_EQ(add_const.arg1(), 456);
  EXPECT_EQ(add_const.arg1_data()[0], 456);
  EXPECT_EQ(add_const.arg1_data(), add.args()[1]);
  EXPECT_EQ(add_const.result0(), 579);
  EXPECT_EQ(add_const.result0_data()[0], 579);
  EXPECT_EQ(add_const.result0_data(), add_const.results()[0]);
}

// Run tests that use set_argN_data separately, to avoid accidentally re-using
// non-existent buffers.
TEST(TFCompileTest, Add_SetArg) {
  AddComp add(AddComp::AllocMode::RESULTS_PROFILES_AND_TEMPS_ONLY);

  int32 arg_x = 10;
  int32 arg_y = 32;
  add.set_arg0_data(&arg_x);
  add.set_arg1_data(&arg_y);
  EXPECT_EQ(add.arg0_data(), add.args()[0]);
  EXPECT_EQ(add.arg1_data(), add.args()[1]);

  EXPECT_TRUE(add.Run());
  EXPECT_EQ(add.error_msg(), "");
  EXPECT_EQ(add.result0(), 42);
  EXPECT_EQ(add.result0_data()[0], 42);
  EXPECT_EQ(add.result0_data(), add.results()[0]);
}

TEST(TFCompileTest, AddWithCkpt) {
  AddWithCkptComp add;
  EXPECT_EQ(add.arg0_data(), add.args()[0]);

  add.arg0() = 1;
  EXPECT_TRUE(add.Run());
  EXPECT_EQ(add.error_msg(), "");
  EXPECT_EQ(add.result0(), 43);
  EXPECT_EQ(add.result0_data()[0], 43);
  EXPECT_EQ(add.result0_data(), add.results()[0]);

  add.arg0_data()[0] = 111;
  EXPECT_TRUE(add.Run());
  EXPECT_EQ(add.error_msg(), "");
  EXPECT_EQ(add.result0(), 153);
  EXPECT_EQ(add.result0_data()[0], 153);
  EXPECT_EQ(add.result0_data(), add.results()[0]);

  const AddWithCkptComp& add_const = add;
  EXPECT_EQ(add_const.error_msg(), "");
  EXPECT_EQ(add_const.arg0(), 111);
  EXPECT_EQ(add_const.arg0_data()[0], 111);
  EXPECT_EQ(add_const.arg0_data(), add_const.args()[0]);
  EXPECT_EQ(add_const.result0(), 153);
  EXPECT_EQ(add_const.result0_data()[0], 153);
  EXPECT_EQ(add_const.result0_data(), add_const.results()[0]);
}

TEST(TFCompileTest, AddWithCkptSaver) {
  AddWithCkptSaverComp add;
  EXPECT_EQ(add.arg0_data(), add.args()[0]);

  add.arg0() = 1;
  EXPECT_TRUE(add.Run());
  EXPECT_EQ(add.error_msg(), "");
  EXPECT_EQ(add.result0(), 43);
  EXPECT_EQ(add.result0_data()[0], 43);
  EXPECT_EQ(add.result0_data(), add.results()[0]);

  add.arg0_data()[0] = 111;
  EXPECT_TRUE(add.Run());
  EXPECT_EQ(add.error_msg(), "");
  EXPECT_EQ(add.result0(), 153);
  EXPECT_EQ(add.result0_data()[0], 153);
  EXPECT_EQ(add.result0_data(), add.results()[0]);

  const AddWithCkptSaverComp& add_const = add;
  EXPECT_EQ(add_const.error_msg(), "");
  EXPECT_EQ(add_const.arg0(), 111);
  EXPECT_EQ(add_const.arg0_data()[0], 111);
  EXPECT_EQ(add_const.arg0_data(), add_const.args()[0]);
  EXPECT_EQ(add_const.result0(), 153);
  EXPECT_EQ(add_const.result0_data()[0], 153);
  EXPECT_EQ(add_const.result0_data(), add_const.results()[0]);
}

TEST(TFCompileTest, Cond) {
  CondComp cond;
  EXPECT_EQ(cond.arg0_data(), cond.args()[0]);
  EXPECT_EQ(cond.arg1_data(), cond.args()[1]);
  EXPECT_EQ(cond.arg2_data(), cond.args()[2]);
  cond.arg1() = 10;
  cond.arg2() = 20;
  {
    cond.arg0() = true;
    const int32 expected_result = cond.arg1();
    EXPECT_TRUE(cond.Run());
    EXPECT_EQ(cond.result0(), expected_result);
    EXPECT_EQ(cond.result0_data()[0], expected_result);
    EXPECT_EQ(cond.result0_data(), cond.results()[0]);
  }
  {
    cond.arg0() = false;
    const int32 expected_result = cond.arg2();
    EXPECT_TRUE(cond.Run());
    EXPECT_EQ(cond.result0(), expected_result);
    EXPECT_EQ(cond.result0_data()[0], expected_result);
    EXPECT_EQ(cond.result0_data(), cond.results()[0]);
  }
}

TEST(TFCompileTest, Gather) {
  GatherComp gather;
  EXPECT_EQ(gather.arg0_data(), gather.args()[0]);
  EXPECT_EQ(gather.arg1_data(), gather.args()[1]);

  // Successful gather.
  {
    const float params[4] = {1, 2, 3, 4};
    std::copy(params + 0, params + 4, gather.arg0_data());
    const int32 indices[2] = {1, 3};
    std::copy(indices + 0, indices + 2, gather.arg1_data());
    EXPECT_TRUE(gather.Run());
    EXPECT_EQ(gather.error_msg(), "");
    const float results[2] = {2, 4};
    for (int i = 0; i < 2; ++i) {
      EXPECT_EQ(gather.result0(i), results[i]);
      EXPECT_EQ(gather.result0_data()[i], results[i]);
    }
    EXPECT_EQ(gather.result0_data(), gather.results()[0]);

    const GatherComp& gather_const = gather;
    EXPECT_EQ(gather_const.error_msg(), "");
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(gather_const.arg0(i), params[i]);
      EXPECT_EQ(gather_const.arg0_data()[i], params[i]);
    }
    EXPECT_EQ(gather_const.arg0_data(), gather_const.args()[0]);
    for (int i = 0; i < 2; ++i) {
      EXPECT_EQ(gather_const.arg1(i), indices[i]);
      EXPECT_EQ(gather_const.arg1_data()[i], indices[i]);
    }
    EXPECT_EQ(gather_const.arg1_data(), gather_const.args()[1]);
    for (int i = 0; i < 2; ++i) {
      EXPECT_EQ(gather_const.result0(i), results[i]);
      EXPECT_EQ(gather_const.result0_data()[i], results[i]);
    }
    EXPECT_EQ(gather_const.result0_data(), gather.results()[0]);
  }
}

TEST(TFCompileTest, MatMul2) {
  Eigen::ThreadPool tp(2);
  Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());

  foo::bar::MatMulComp matmul;
  matmul.set_thread_pool(&device);
  EXPECT_EQ(matmul.arg0_data(), matmul.args()[0]);
  EXPECT_EQ(matmul.arg1_data(), matmul.args()[1]);

  // Test using the argN() methods.
  {
    matmul.arg0(0, 0) = 1;
    matmul.arg0(0, 1) = 2;
    matmul.arg0(0, 2) = 3;
    matmul.arg0(1, 0) = 4;
    matmul.arg0(1, 1) = 5;
    matmul.arg0(1, 2) = 6;

    matmul.arg1(0, 0) = 7;
    matmul.arg1(0, 1) = 8;
    matmul.arg1(1, 0) = 9;
    matmul.arg1(1, 1) = 10;
    matmul.arg1(2, 0) = 11;
    matmul.arg1(2, 1) = 12;

    EXPECT_TRUE(matmul.Run());
    EXPECT_EQ(matmul.error_msg(), "");
    const float results[4] = {58, 64, 139, 154};
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(matmul.result0(i / 2, i % 2), results[i]);
      EXPECT_EQ(matmul.result0_data()[i], results[i]);
    }
    EXPECT_EQ(matmul.result0_data(), matmul.results()[0]);
  }

  // Test using the argN_data() methods.
  {
    const float args[12] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
    std::copy(args + 0, args + 6, matmul.arg0_data());
    std::copy(args + 6, args + 12, matmul.arg1_data());
    EXPECT_TRUE(matmul.Run());
    EXPECT_EQ(matmul.error_msg(), "");
    const float results[4] = {5800, 6400, 13900, 15400};
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(matmul.result0(i / 2, i % 2), results[i]);
      EXPECT_EQ(matmul.result0_data()[i], results[i]);
    }
    EXPECT_EQ(matmul.result0_data(), matmul.results()[0]);

    const foo::bar::MatMulComp& matmul_const = matmul;
    EXPECT_EQ(matmul_const.error_msg(), "");
    for (int i = 0; i < 6; ++i) {
      EXPECT_EQ(matmul_const.arg0(i / 3, i % 3), args[i]);
      EXPECT_EQ(matmul_const.arg0_data()[i], args[i]);
    }
    EXPECT_EQ(matmul_const.arg0_data(), matmul.args()[0]);
    for (int i = 0; i < 6; ++i) {
      EXPECT_EQ(matmul_const.arg1(i / 2, i % 2), args[i + 6]);
      EXPECT_EQ(matmul_const.arg1_data()[i], args[i + 6]);
    }
    EXPECT_EQ(matmul_const.arg1_data(), matmul.args()[1]);
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(matmul_const.result0(i / 2, i % 2), results[i]);
      EXPECT_EQ(matmul_const.result0_data()[i], results[i]);
    }
    EXPECT_EQ(matmul_const.result0_data(), matmul.results()[0]);
  }
}

// Run tests that use set_argN_data separately, to avoid accidentally re-using
// non-existent buffers.
TEST(TFCompileTest, MatMul2_SetArg) {
  Eigen::ThreadPool tp(2);
  Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());

  foo::bar::MatMulComp matmul(
      foo::bar::MatMulComp::AllocMode::RESULTS_PROFILES_AND_TEMPS_ONLY);
  matmul.set_thread_pool(&device);

  // Test using the set_argN_data() methods.
  float arg0[2][3] = {{1, 2, 3}, {4, 5, 6}};
  float arg1[3][2] = {{7, 8}, {9, 10}, {11, 12}};
  matmul.set_arg0_data(&arg0);
  matmul.set_arg1_data(&arg1);
  EXPECT_EQ(matmul.arg0_data(), matmul.args()[0]);
  EXPECT_EQ(matmul.arg1_data(), matmul.args()[1]);

  EXPECT_TRUE(matmul.Run());
  EXPECT_EQ(matmul.error_msg(), "");
  const float results[4] = {58, 64, 139, 154};
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(matmul.result0(i / 2, i % 2), results[i]);
    EXPECT_EQ(matmul.result0_data()[i], results[i]);
  }
  EXPECT_EQ(matmul.result0_data(), matmul.results()[0]);
}

TEST(TFCompileTest, MatMulAndAdd1) {
  Eigen::ThreadPool tp(1);
  Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());

  MatMulAndAddComp muladd;
  muladd.set_thread_pool(&device);
  EXPECT_EQ(muladd.arg0_data(), muladd.args()[0]);
  EXPECT_EQ(muladd.arg1_data(), muladd.args()[1]);

  // Test methods with positional args and results.
  {
    const float args[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::copy(args + 0, args + 4, muladd.arg0_data());
    std::copy(args + 4, args + 8, muladd.arg1_data());
    EXPECT_TRUE(muladd.Run());
    EXPECT_EQ(muladd.error_msg(), "");
    const float results0[4] = {19, 22, 43, 50};
    const float results1[4] = {6, 8, 10, 12};
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(muladd.result0(i / 2, i % 2), results0[i]);
      EXPECT_EQ(muladd.result0_data()[i], results0[i]);
      EXPECT_EQ(muladd.result1(i / 2, i % 2), results1[i]);
      EXPECT_EQ(muladd.result1_data()[i], results1[i]);
    }
    EXPECT_EQ(muladd.result0_data(), muladd.results()[0]);
    EXPECT_EQ(muladd.result1_data(), muladd.results()[1]);

    const MatMulAndAddComp& muladd_const = muladd;
    EXPECT_EQ(muladd_const.error_msg(), "");
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(muladd_const.arg0(i / 2, i % 2), args[i]);
      EXPECT_EQ(muladd_const.arg0_data()[i], args[i]);
    }
    EXPECT_EQ(muladd_const.arg0_data(), muladd.args()[0]);
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(muladd_const.arg1(i / 2, i % 2), args[i + 4]);
      EXPECT_EQ(muladd_const.arg1_data()[i], args[i + 4]);
    }
    EXPECT_EQ(muladd_const.arg1_data(), muladd.args()[1]);
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(muladd_const.result0(i / 2, i % 2), results0[i]);
      EXPECT_EQ(muladd_const.result0_data()[i], results0[i]);
      EXPECT_EQ(muladd_const.result1(i / 2, i % 2), results1[i]);
      EXPECT_EQ(muladd_const.result1_data()[i], results1[i]);
    }
    EXPECT_EQ(muladd_const.result0_data(), muladd.results()[0]);
    EXPECT_EQ(muladd_const.result1_data(), muladd.results()[1]);
  }

  // Test methods with named args and results.
  {
    const float args[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    std::copy(args + 0, args + 4, muladd.arg_x_data());
    std::copy(args + 4, args + 8, muladd.arg_y_data());
    EXPECT_TRUE(muladd.Run());
    EXPECT_EQ(muladd.error_msg(), "");
    const float results0[4] = {1900, 2200, 4300, 5000};
    const float results1[4] = {60, 80, 100, 120};
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(muladd.result_x_y_prod(i / 2, i % 2), results0[i]);
      EXPECT_EQ(muladd.result_x_y_prod_data()[i], results0[i]);
      EXPECT_EQ(muladd.result_x_y_sum(i / 2, i % 2), results1[i]);
      EXPECT_EQ(muladd.result_x_y_sum_data()[i], results1[i]);
    }
    EXPECT_EQ(muladd.result_x_y_prod_data(), muladd.results()[0]);
    EXPECT_EQ(muladd.result_x_y_sum_data(), muladd.results()[1]);

    // Test const methods.
    const MatMulAndAddComp& muladd_const = muladd;
    EXPECT_EQ(muladd_const.error_msg(), "");
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(muladd_const.arg_x(i / 2, i % 2), args[i]);
      EXPECT_EQ(muladd_const.arg_x_data()[i], args[i]);
    }
    EXPECT_EQ(muladd_const.arg_x_data(), muladd.args()[0]);
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(muladd_const.arg_y(i / 2, i % 2), args[i + 4]);
      EXPECT_EQ(muladd_const.arg_y_data()[i], args[i + 4]);
    }
    EXPECT_EQ(muladd_const.arg_y_data(), muladd.args()[1]);
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(muladd_const.result_x_y_prod(i / 2, i % 2), results0[i]);
      EXPECT_EQ(muladd_const.result_x_y_prod_data()[i], results0[i]);
      EXPECT_EQ(muladd_const.result_x_y_sum(i / 2, i % 2), results1[i]);
      EXPECT_EQ(muladd_const.result_x_y_sum_data()[i], results1[i]);
    }
    EXPECT_EQ(muladd_const.result_x_y_prod_data(), muladd.results()[0]);
    EXPECT_EQ(muladd_const.result_x_y_sum_data(), muladd.results()[1]);
  }
}

TEST(TFCompileTest, Function) {
  // The function is equivalent to an addition
  FunctionComp add_fn;
  EXPECT_EQ(add_fn.arg0_data(), add_fn.args()[0]);
  EXPECT_EQ(add_fn.arg1_data(), add_fn.args()[1]);

  add_fn.arg0() = 1;
  add_fn.arg1() = 2;
  EXPECT_TRUE(add_fn.Run());
  EXPECT_EQ(add_fn.error_msg(), "");
  EXPECT_EQ(add_fn.result0(), 3);
  EXPECT_EQ(add_fn.result0_data()[0], 3);
  EXPECT_EQ(add_fn.result0_data(), add_fn.results()[0]);
}

TEST(TFCompileTest, Splits) {
  Eigen::ThreadPool tp(1);
  Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());

  SplitsComp fn;

  fn.set_thread_pool(&device);
  // x = [[1, 2], [3, 4]]
  fn.arg0(0, 0) = 1;
  fn.arg0(0, 1) = 2;
  fn.arg0(1, 0) = 3;
  fn.arg0(1, 1) = 4;

  // y = [[10, 20], [30, 40]]
  fn.arg1(0, 0) = 10;
  fn.arg1(0, 1) = 20;
  fn.arg1(1, 0) = 30;
  fn.arg1(1, 1) = 40;
  EXPECT_TRUE(fn.Run());
  EXPECT_EQ(fn.error_msg(), "");
  const float expected[] = {7.86375557e+10, 1.34274679e+11, 1.92741717e+12,
                            3.29964742e+12};
  EXPECT_NEAR(expected[0], fn.result0(0, 0), 1e4);
  EXPECT_NEAR(expected[1], fn.result0(0, 1), 1e4);
  EXPECT_NEAR(expected[2], fn.result0(1, 0), 1e4);
  EXPECT_NEAR(expected[3], fn.result0(1, 1), 1e4);
}

TEST(TFCompileTest, AssertEqAndReturnDiff) {
  // Assert is converted into a no-op in XLA, so there is no failure even if the
  // two args are different.
  AssertComp assert;
  EXPECT_EQ(assert.arg0_data(), assert.args()[0]);
  EXPECT_EQ(assert.arg1_data(), assert.args()[1]);

  assert.arg0() = 2;
  assert.arg1() = 1;
  const int32 expected_result = assert.arg0() - assert.arg1();
  EXPECT_TRUE(assert.Run());
  EXPECT_EQ(assert.error_msg(), "");
  EXPECT_EQ(assert.result0(), expected_result);
  EXPECT_EQ(assert.result0_data()[0], expected_result);
  EXPECT_EQ(assert.result0_data(), assert.results()[0]);
}

TEST(TFCompileTest, LookupNameIndex) {
  // add doesn't have any names defined in its config.
  AddComp add;
  EXPECT_FALSE(add.HasNameIndices());

  // muladd has names defined for all feeds and fetches.
  MatMulAndAddComp muladd;
  EXPECT_TRUE(muladd.HasNameIndices());

  EXPECT_EQ(muladd.LookupArgIndex("x"), 0);
  EXPECT_EQ(muladd.LookupArgIndex("y"), 1);
  EXPECT_EQ(muladd.LookupArgIndex(""), -1);
  EXPECT_EQ(muladd.LookupArgIndex("x_hold"), -1);
  EXPECT_EQ(muladd.LookupArgIndex("y_hold"), -1);
  EXPECT_EQ(muladd.LookupArgIndex("x_y_prod"), -1);
  EXPECT_EQ(muladd.LookupArgIndex("x_y_sum"), -1);

  EXPECT_EQ(muladd.LookupResultIndex("x_y_prod"), 0);
  EXPECT_EQ(muladd.LookupResultIndex("x_y_sum"), 1);
  EXPECT_EQ(muladd.LookupResultIndex(""), -1);
  EXPECT_EQ(muladd.LookupResultIndex("x"), -1);
  EXPECT_EQ(muladd.LookupResultIndex("y"), -1);
  EXPECT_EQ(muladd.LookupResultIndex("x_hold"), -1);
  EXPECT_EQ(muladd.LookupResultIndex("y_hold"), -1);
}

TEST(TFCompileTest, ProgramShape) {
  using xla::ShapeUtil;
  const xla::Shape f32_2x2 = ShapeUtil::MakeShape(xla::F32, {2, 2});

  // add doesn't have the program shape defined.
  AddComp add;
  ASSERT_TRUE(add.ProgramShape() == nullptr);

  // muladd has the program shape defined.
  MatMulAndAddComp muladd;
  const xla::ProgramShape* muladd_shape = muladd.ProgramShape();
  ASSERT_TRUE(muladd_shape != nullptr);
  ASSERT_EQ(muladd_shape->parameters_size(), 2);
  EXPECT_TRUE(ShapeUtil::Compatible(muladd_shape->parameters(0), f32_2x2));
  EXPECT_TRUE(ShapeUtil::Compatible(muladd_shape->parameters(1), f32_2x2));

  const xla::Shape& muladd_result = muladd_shape->result();
  ASSERT_EQ(muladd_result.element_type(), xla::TUPLE);
  ASSERT_EQ(ShapeUtil::TupleElementCount(muladd_result), 2);
  const xla::Shape& muladd_result0 =
      ShapeUtil::GetTupleElementShape(muladd_result, 0);
  EXPECT_TRUE(ShapeUtil::Compatible(muladd_result0, f32_2x2));
  const xla::Shape& muladd_result1 =
      ShapeUtil::GetTupleElementShape(muladd_result, 1);
  EXPECT_TRUE(ShapeUtil::Compatible(muladd_result1, f32_2x2));
}

TEST(TFCompileTest, HloProfiling) {
  Eigen::ThreadPool tp(1);
  Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());

  MatMulAndAddCompWithProfiling fn;
  ASSERT_TRUE(fn.hlo_profiling_enabled());

  fn.set_thread_pool(&device);

  // x = [[1, 2], [3, 4]]
  fn.arg0(0, 0) = 1;
  fn.arg0(0, 1) = 2;
  fn.arg0(1, 0) = 3;
  fn.arg0(1, 1) = 4;

  // y = [[10, 20], [30, 40]]
  fn.arg1(0, 0) = 10;
  fn.arg1(0, 1) = 20;
  fn.arg1(1, 0) = 30;
  fn.arg1(1, 1) = 40;

  EXPECT_TRUE(fn.Run());

  string hlo_profile_as_string =
      xla::PrintHloProfile(fn.hlo_profile_printer_data(), fn.profile_counters(),
                           /*clock_rate_ghz=*/1.0);
  VLOG(1) << "HLO profile string:\n" << hlo_profile_as_string;

  std::vector<string> hlo_profile_lines =
      tensorflow::str_util::Split(hlo_profile_as_string, '\n');

  auto header = HasSubstr("Execution profile for");
  auto total_cycles_profile_line = HasSubstr("[total]");
  auto dot_profile_line = HasSubstr(
      "%dot.0.4 = f32[2,2]{1,0} dot(f32[2,2]{1,0} %arg0.0.0, f32[2,2]{1,0} "
      "%arg1.0.1)");
  auto add_profile_line = HasSubstr(
      "%add.0.6 = f32[2,2]{1,0} add(f32[2,2]{1,0} %arg0.0.0, f32[2,2]{1,0} "
      "%arg1.0.1)");
  auto tuple_profile_line = HasSubstr(
      "%tuple.0.8 = (f32[2,2]{1,0}, f32[2,2]{1,0}) tuple(f32[2,2]{1,0} "
      "%dot.0.4, f32[2,2]{1,0} %add.0.6)");
  auto arg0_profile_line = HasSubstr("%arg0.0.0 = f32[2,2]{1,0} parameter(0)");
  auto arg1_profile_line = HasSubstr("%arg1.0.1 = f32[2,2]{1,0} parameter(1)");

  EXPECT_THAT(hlo_profile_lines,
              IsSupersetOf({header, total_cycles_profile_line, dot_profile_line,
                            add_profile_line, tuple_profile_line}));
}

}  // namespace
}  // namespace tfcompile
}  // namespace tensorflow
