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

#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_computation.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class ReplayTest : public ClientLibraryTestBase {};

TEST_F(ReplayTest, TwoPlusTwoReplay) {
  // Make 2+2 computation.
  XlaBuilder builder(TestName());
  auto two = ConstantR0<int32>(&builder, 2);
  Add(two, two);
  XlaComputation computation = builder.Build().ConsumeValueOrDie();

  // Serialize it out.
  std::unique_ptr<HloSnapshot> module =
      computation.Snapshot().ConsumeValueOrDie();

  // Replay it.
  XlaComputation replayed = client_->LoadSnapshot(*module).ConsumeValueOrDie();

  // Check signature is the same.
  std::unique_ptr<ProgramShape> original_shape =
      client_->GetComputationShape(computation).ConsumeValueOrDie();
  std::unique_ptr<ProgramShape> replayed_shape =
      client_->GetComputationShape(replayed).ConsumeValueOrDie();
  ASSERT_TRUE(protobuf_util::ProtobufEquals(*original_shape, *replayed_shape));

  // Run it.
  std::unique_ptr<Literal> literal =
      client_
          ->ExecuteAndTransfer(replayed, /*arguments=*/{}, &execution_options_)
          .ConsumeValueOrDie();

  // Expect 4.
  LiteralTestUtil::ExpectR0Equal<int32>(4, *literal);
}

XLA_TEST_F(ReplayTest, XPlusYReplayWithParameters) {
  // Make computation.
  XlaBuilder builder(TestName());
  auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(S32, {}), "x");
  auto y = Parameter(&builder, 1, ShapeUtil::MakeShape(S32, {}), "y");
  Add(x, y);
  XlaComputation computation = builder.Build().ConsumeValueOrDie();

  // Serialize it out.
  std::unique_ptr<HloSnapshot> module =
      computation.Snapshot().ConsumeValueOrDie();

  // Replay it.
  XlaComputation replayed = client_->LoadSnapshot(*module).ConsumeValueOrDie();

  // Check signature is the same.
  std::unique_ptr<ProgramShape> original_shape =
      client_->GetComputationShape(computation).ConsumeValueOrDie();
  std::unique_ptr<ProgramShape> replayed_shape =
      client_->GetComputationShape(replayed).ConsumeValueOrDie();
  ASSERT_TRUE(protobuf_util::ProtobufEquals(*original_shape, *replayed_shape));

  // Run it.
  std::unique_ptr<GlobalData> x_data =
      client_->TransferToServer(*Literal::CreateR0<int32>(2))
          .ConsumeValueOrDie();
  std::unique_ptr<GlobalData> y_data =
      client_->TransferToServer(*Literal::CreateR0<int32>(3))
          .ConsumeValueOrDie();
  std::unique_ptr<Literal> literal =
      client_
          ->ExecuteAndTransfer(replayed,
                               /*arguments=*/{x_data.get(), y_data.get()},
                               &execution_options_)
          .ConsumeValueOrDie();

  // Expect 5.
  LiteralTestUtil::ExpectR0Equal<int32>(5, *literal);
}

TEST_F(ReplayTest, MapPlusTwoOverR1) {
  // As above, but with map(+2) over some constant array.
  XlaBuilder plus_two_builder("plus two");
  auto input =
      Parameter(&plus_two_builder, 0, ShapeUtil::MakeShape(S32, {}), "input");
  Add(input, ConstantR0<int32>(&plus_two_builder, 2));
  XlaComputation plus_two = plus_two_builder.Build().ConsumeValueOrDie();

  XlaBuilder mapper_builder(TestName());
  auto original = ConstantR1<int32>(&mapper_builder, {1, 2, 3});
  Map(&mapper_builder, {original}, plus_two, {0});

  XlaComputation computation = mapper_builder.Build().ConsumeValueOrDie();

  // Serialize it out.
  std::unique_ptr<HloSnapshot> module =
      computation.Snapshot().ConsumeValueOrDie();

  // Replay it.
  XlaComputation replayed = client_->LoadSnapshot(*module).ConsumeValueOrDie();

  // Check signature is the same.
  std::unique_ptr<ProgramShape> original_shape =
      client_->GetComputationShape(computation).ConsumeValueOrDie();
  std::unique_ptr<ProgramShape> replayed_shape =
      client_->GetComputationShape(replayed).ConsumeValueOrDie();
  ASSERT_TRUE(protobuf_util::ProtobufEquals(*original_shape, *replayed_shape));

  // Run it.
  std::unique_ptr<Literal> literal =
      client_
          ->ExecuteAndTransfer(replayed, /*arguments=*/{}, &execution_options_)
          .ConsumeValueOrDie();

  // Expect result.
  LiteralTestUtil::ExpectR1Equal<int32>({3, 4, 5}, *literal);
}

}  // namespace
}  // namespace xla
