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
#include <vector>

#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class ClientTest : public ClientLibraryTestBase {};

TEST_F(ClientTest, ExecuteWithLayout) {
  ComputationBuilder b(client_, TestName());

  std::vector<std::vector<int64>> layouts = {{0, 1}, {1, 0}};
  for (const std::vector<int64>& execute_layout : layouts) {
    for (const std::vector<int64>& transfer_layout : layouts) {
      b.Add(b.ConstantR2<int32>({{1, 2}, {3, 4}}),
            b.ConstantR2<int32>({{10, 20}, {30, 40}}));
      auto computation = b.Build();
      ASSERT_TRUE(computation.ok()) << computation.status();

      ExecutionOptions execution_options;
      *execution_options.mutable_shape_with_output_layout() =
          ShapeUtil::MakeShapeWithLayout(S32, /*dimensions=*/{2, 2},
                                         execute_layout);
      std::unique_ptr<GlobalData> data =
          client_->Execute(computation.ValueOrDie(), {}, &execution_options)
              .ConsumeValueOrDie();

      std::unique_ptr<Literal> expected_literal =
          test_utils::CreateR2LiteralWithLayout<int32>({{11, 22}, {33, 44}},
                                                       transfer_layout);

      auto computed = client_->Transfer(*data, &expected_literal->shape());

      LiteralTestUtil::AssertEqualShapesAndLayouts(
          expected_literal->shape(), computed.ValueOrDie()->shape());
      LiteralTestUtil::ExpectEqual(*expected_literal, *computed.ValueOrDie());
    }
  }
}

TEST_F(ClientTest, ExecuteWithTupleLayout) {
  ComputationBuilder b(client_, TestName());

  b.Tuple({b.ConstantR2<int32>({{1, 2}, {3, 4}}),
           b.ConstantR2<int32>({{10, 20}, {30, 40}})});

  auto computation = b.Build();
  ASSERT_TRUE(computation.ok()) << computation.status();

  ExecutionOptions execution_options;
  // Create a result shape with one element column major and the other row
  // major.
  *execution_options.mutable_shape_with_output_layout() =
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeShapeWithLayout(S32, /*dimensions=*/{2, 2},
                                          /*minor_to_major=*/{0, 1}),
           ShapeUtil::MakeShapeWithLayout(S32, /*dimensions=*/{2, 2},
                                          /*minor_to_major=*/{1, 0})});

  auto result =
      client_
          ->ExecuteAndTransfer(computation.ValueOrDie(), {}, &execution_options)
          .ConsumeValueOrDie();
  LiteralTestUtil::ExpectR2Equal<int32>({{1, 2}, {3, 4}},
                                        result->tuple_literals(0));
  LiteralTestUtil::ExpectR2Equal<int32>({{10, 20}, {30, 40}},
                                        result->tuple_literals(1));

  EXPECT_TRUE(ShapeUtil::IsTuple(result->shape()));
  EXPECT_EQ(2, ShapeUtil::TupleElementCount(result->shape()));

  EXPECT_TRUE(ShapeUtil::Equal(
      ShapeUtil::GetTupleElementShape(result->shape(), 0),
      ShapeUtil::MakeShapeWithLayout(S32, /*dimensions=*/{2, 2},
                                     /*minor_to_major=*/{0, 1})));
  EXPECT_TRUE(ShapeUtil::Equal(
      ShapeUtil::GetTupleElementShape(result->shape(), 1),
      ShapeUtil::MakeShapeWithLayout(S32, /*dimensions=*/{2, 2},
                                     /*minor_to_major=*/{1, 0})));
}

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
  return RUN_ALL_TESTS();
}
