/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/tests/local_client_test_base.h"
#include "xla/tests/test_macros.h"
#include "tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

// Tests that ensure outfeed instructions that are contained in nested
// computations in non-root positions are executed.

class OutfeedInNestedComputationTest : public LocalClientTestBase {};

XLA_TEST_F(OutfeedInNestedComputationTest, OutfeedInWhile) {
  XlaBuilder b(TestName());

  Shape state_tuple_array_shape = ShapeUtil::MakeShape(xla::S32, {10, 5});
  Shape int_shape = ShapeUtil::MakeShape(xla::S32, {});
  Shape state_tuple_shape =
      ShapeUtil::MakeTupleShape({int_shape, state_tuple_array_shape});
  Shape xfeed_shape = ShapeUtil::MakeShape(xla::S32, {2});

  XlaOp some_buffer = Broadcast(ConstantR0<int32_t>(&b, 0), {10, 5});
  XlaOp num_iter = Infeed(&b, int_shape);
  XlaOp init_tuple = Tuple(&b, {num_iter, some_buffer});

  TF_ASSERT_OK_AND_ASSIGN(XlaComputation loop_cond, [&] {
    // Condition: iteration variable > 0
    XlaBuilder cond_builder("loop_condition");
    XlaOp state_tuple = Parameter(&cond_builder, 0, state_tuple_shape, "state");
    XlaOp loop_counter = GetTupleElement(state_tuple, 0);
    Outfeed(loop_counter, int_shape, "");
    Gt(loop_counter, ConstantR0<int32_t>(&cond_builder, 0));
    return cond_builder.Build();
  }());

  TF_ASSERT_OK_AND_ASSIGN(XlaComputation loop_body, [&] {
    XlaBuilder body_builder("loop_body");
    XlaOp state_tuple = Parameter(&body_builder, 0, state_tuple_shape, "state");
    XlaOp loop_counter = GetTupleElement(state_tuple, 0);
    XlaOp buffer_inside = GetTupleElement(state_tuple, 1);

    // Read some stuff from Infeed.
    XlaOp some_input = Infeed(&body_builder, xfeed_shape);
    XlaOp sum = Add(some_input, Broadcast(loop_counter, {2}));
    Outfeed(sum, xfeed_shape, "");

    XlaOp iter_left = Sub(loop_counter, ConstantR0<int32_t>(&body_builder, 1));

    Tuple(&body_builder, {iter_left, buffer_inside});
    return body_builder.Build();
  }());

  // Build loop.
  XlaOp result_tuple = While(loop_cond, loop_body, init_tuple);
  GetTupleElement(result_tuple, 0);
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, b.Build());

  Literal comp_result;
  std::unique_ptr<tsl::Thread> thread(tsl::Env::Default()->StartThread(
      tsl::ThreadOptions(), "execute_thread", [&] {
        comp_result =
            local_client_->ExecuteAndTransfer(computation, {}).value();
      }));

  VLOG(1) << "Transferring trip count to computation";
  // Transfer number of iterations to Infeed.
  TF_ASSERT_OK(
      local_client_->TransferToInfeed(LiteralUtil::CreateR0<int32_t>(1)));

  // Pick up value from outfeed
  {
    VLOG(1) << "Reading from condition outfeed";
    TF_ASSERT_OK_AND_ASSIGN(Literal r,
                            local_client_->TransferFromOutfeed(&int_shape));
    EXPECT_EQ(r.Get<int32_t>({}), 1);
  }

  VLOG(1) << "Writing data to infeed";
  // Transfer some stuff to Infeed for use inside of loop.
  TF_ASSERT_OK(local_client_->TransferToInfeed(
      LiteralUtil::CreateR1<int32_t>({10, 20})));

  // Pick up value from outfeed
  {
    VLOG(1) << "Reading from body outfeed";
    TF_ASSERT_OK_AND_ASSIGN(Literal r,
                            local_client_->TransferFromOutfeed(&xfeed_shape));
    EXPECT_EQ(r.Get<int32_t>({0}), 11);
    EXPECT_EQ(r.Get<int32_t>({1}), 21);
  }

  {
    VLOG(1) << "Reading from condition outfeed";
    TF_ASSERT_OK_AND_ASSIGN(Literal r,
                            local_client_->TransferFromOutfeed(&int_shape));
    EXPECT_EQ(r.Get<int32_t>({}), 0);
  }

  // Joins the thread
  thread.reset();

  EXPECT_EQ(comp_result.Get<int32_t>({}), 0);
}

XLA_TEST_F(OutfeedInNestedComputationTest, OutfeedInConditional) {
  XlaBuilder b(TestName());

  Shape condition_shape = ShapeUtil::MakeShape(xla::PRED, {});
  Shape result_shape = ShapeUtil::MakeShape(xla::PRED, {});

  TF_ASSERT_OK_AND_ASSIGN(XlaComputation true_computation, [&] {
    XlaBuilder inner_builder("true_computation");
    XlaOp param = Parameter(&inner_builder, 0, result_shape, "param");
    Outfeed(param, result_shape, "");
    Or(param, param);
    return inner_builder.Build();
  }());

  TF_ASSERT_OK_AND_ASSIGN(XlaComputation false_computation, [&] {
    XlaBuilder inner_builder("false_computation");
    Parameter(&inner_builder, 0, result_shape, "param");
    return inner_builder.Build();
  }());

  XlaOp pred = Infeed(&b, condition_shape);
  Conditional(/*predicate=*/pred, /*true_operand=*/pred,
              /*true_computation=*/true_computation, /*false_operand=*/pred,
              /*false_computation=*/false_computation);

  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, b.Build());

  Literal comp_result;
  std::unique_ptr<tsl::Thread> thread(tsl::Env::Default()->StartThread(
      tsl::ThreadOptions(), "execute_thread", [&] {
        comp_result =
            local_client_->ExecuteAndTransfer(computation, {}).value();
      }));

  TF_ASSERT_OK(
      local_client_->TransferToInfeed(LiteralUtil::CreateR0<bool>(true)));

  TF_ASSERT_OK_AND_ASSIGN(Literal r,
                          local_client_->TransferFromOutfeed(&result_shape));

  EXPECT_EQ(r.Get<bool>({}), true);

  // Join the thread
  thread.reset();
}

}  // namespace
}  // namespace xla
