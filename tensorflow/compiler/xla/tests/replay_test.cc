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

#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/service/session.pb.h"
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
  ComputationBuilder builder(client_, TestName());
  auto two = builder.ConstantR0<int32>(2);
  builder.Add(two, two);
  Computation computation = builder.Build().ConsumeValueOrDie();

  // Serialize it out.
  std::unique_ptr<SessionModule> module =
      computation.Snapshot().ConsumeValueOrDie();

  // Replay it.
  Computation replayed = client_->LoadSnapshot(*module).ConsumeValueOrDie();

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
  ComputationBuilder builder(client_, TestName());
  auto x = builder.Parameter(0, ShapeUtil::MakeShape(S32, {}), "x");
  auto y = builder.Parameter(1, ShapeUtil::MakeShape(S32, {}), "y");
  builder.Add(x, y);
  Computation computation = builder.Build().ConsumeValueOrDie();

  // Serialize it out.
  std::unique_ptr<SessionModule> module =
      computation.Snapshot().ConsumeValueOrDie();

  // Replay it.
  Computation replayed = client_->LoadSnapshot(*module).ConsumeValueOrDie();

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
  ComputationBuilder plus_two_builder(client_, "plus two");
  auto input =
      plus_two_builder.Parameter(0, ShapeUtil::MakeShape(S32, {}), "input");
  plus_two_builder.Add(input, plus_two_builder.ConstantR0<int32>(2));
  Computation plus_two = plus_two_builder.Build().ConsumeValueOrDie();

  ComputationBuilder mapper_builder(client_, TestName());
  auto original = mapper_builder.ConstantR1<int32>({1, 2, 3});
  mapper_builder.Map({original}, plus_two);

  Computation computation = mapper_builder.Build().ConsumeValueOrDie();

  // Serialize it out.
  std::unique_ptr<SessionModule> module =
      computation.Snapshot().ConsumeValueOrDie();

  // Replay it.
  Computation replayed = client_->LoadSnapshot(*module).ConsumeValueOrDie();

  // Check signature is the same.
  std::unique_ptr<ProgramShape> original_shape =
      client_->GetComputationShape(computation).ConsumeValueOrDie();
  std::unique_ptr<ProgramShape> replayed_shape =
      client_->GetComputationShape(replayed).ConsumeValueOrDie();
  ASSERT_TRUE(protobuf_util::ProtobufEquals(*original_shape, *replayed_shape));

  // Destroy the originals.
  computation.Reset();
  plus_two.Reset();

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
