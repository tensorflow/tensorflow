// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_text/core/kernels/text_kernels_test_util.h"

namespace tensorflow {

using tensorflow::DT_INT32;
using tensorflow::FakeInput;
using tensorflow::NodeDefBuilder;
using tensorflow::Status;
using tensorflow::TensorShape;
using tensorflow::text_kernels_test_util::MatrixEq;
using tensorflow::text_kernels_test_util::VectorEq;

class ConstrainedSequenceInputValidationTest : public tensorflow::OpsTestBase {
 public:
  void SetUpOpWithDefaults(bool use_start_end,
                           tensorflow::DataType input_datatype) {
    // Prepare graph.
    TF_ASSERT_OK(NodeDefBuilder("tested_op", "ConstrainedSequence")
                     .Attr("Tin", input_datatype)
                     .Attr("use_viterbi", true)
                     .Attr("use_log_space", true)
                     .Attr("use_start_and_end_states", use_start_end)
                     .Input(FakeInput())
                     .Input(FakeInput())
                     .Input(FakeInput())
                     .Input(FakeInput())
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

  void SetUpOpWithStartEnd() { SetUpOpWithDefaults(true, DT_INT32); }

  void SetUpOpWithNoStartEnd() { SetUpOpWithDefaults(false, DT_INT32); }
};
// TODO(b/122968457): There are a bunch of tests that only validate !ok instead
// of looking for specific error messages; fix that.

// This test examines evaluations with only a permissions matrix.
TEST_F(ConstrainedSequenceInputValidationTest, WorksWithInt64InputLengths) {
  // Prepare graph.
  SetUpOpWithDefaults(true, DT_INT64);
  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 12.0, 13.0, 4.0,  //
                               1.0, 12.0, 13.0, 14.0,  //
                               15.0, 2.0, 3.0, 14.0,   //
                           }});

  // Add the sequence_lengths input.
  std::vector<int64> input_lengths({1, 1, 1});
  AddInputFromArray<int64>(TensorShape({3}), input_lengths);

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({5, 5}),
                          {
                              // TO 0 TO 1  TO 2  TO 3  TO OUT
                              true, true, true,  true, true,   // FROM 0
                              true, true, true,  true, true,   // FROM 1
                              true, true, true,  true, true,   // FROM 2
                              true, true, true,  true, true,   // FROM 3
                              true, true, false, true, false,  // FROM 'OUTSIDE'
                          });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  TF_ASSERT_OK(RunOpKernel());

  // The first sequence's highest score is 2, but OUT->2 is not ok, so it's 1.
  // The second sequence's highest score is 3, which is ok.
  // The third sequence's highest score is 0, which is ok.

  // Validate the output.
  std::vector<int32> expected_transitions({1, 3, 0});
  std::vector<int64> expected_offsets({0, 1, 2, 3});

  // Validate the output.
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_transitions));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_offsets));
}

TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnOuterWrongSizePermissionMatrix) {
  // Prepare graph.
  SetUpOpWithStartEnd();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({4, 5}),
                          {
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                          });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  auto result = RunOpKernel();
  EXPECT_FALSE(result.ok());
}
TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnInnerWrongSizePermissionMatrix) {
  // Prepare graph.
  SetUpOpWithStartEnd();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({5, 4}),
                          {
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                          });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  auto result = RunOpKernel();
  EXPECT_FALSE(result.ok());
}
TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnWrongRankPermissionMatrix) {
  // Prepare graph.
  SetUpOpWithStartEnd();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({25}),
                          {
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                          });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  auto result = RunOpKernel();
  EXPECT_FALSE(result.ok());
}

TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnOuterWrongSizeWeightMatrix) {
  // Prepare graph.
  SetUpOpWithStartEnd();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({0, 0}), {});

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({4, 5}), {0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.1, 0.5, 0.5, 1.0, 1.0});
  auto result = RunOpKernel();
  EXPECT_FALSE(result.ok());
}
TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnInnerWrongSizeWeightMatrix) {
  // Prepare graph.
  SetUpOpWithStartEnd();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({0, 0}), {});

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({5, 4}), {0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.1, 0.5, 0.5, 1.0, 1.0});

  auto result = RunOpKernel();
  EXPECT_FALSE(result.ok());
}
TEST_F(ConstrainedSequenceInputValidationTest, FailsOnWrongRankWeightMatrix) {
  // Prepare graph.
  SetUpOpWithStartEnd();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({0, 0}), {});

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({25}), {0.5, 0.5, 0.5, 0.5, 1.0,  //
                                               0.5, 0.5, 0.5, 0.5, 1.0,  //
                                               0.5, 0.5, 0.5, 0.5, 1.0,  //
                                               0.5, 0.5, 0.5, 0.5, 1.0,  //
                                               0.1, 0.5, 0.5, 1.0, 1.0});
  auto result = RunOpKernel();
  EXPECT_FALSE(result.ok());
}

TEST_F(ConstrainedSequenceInputValidationTest,
       PassesWithCorrectSizedWeightAndPermissionsMatrix) {
  // Prepare graph.
  SetUpOpWithNoStartEnd();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({4, 4}), {
                                                   true, true, true, true,  //
                                                   true, true, true, true,  //
                                                   true, true, true, true,  //
                                                   true, true, true, true,  //
                                               });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({4, 4}), {0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 1.0, 1.0});
  auto result = RunOpKernel();
  EXPECT_TRUE(result.ok());
}

TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnOuterWrongSizePermissionMatrixWithNoStartEnd) {
  // Prepare graph.
  SetUpOpWithNoStartEnd();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({4, 5}),
                          {
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                          });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  auto result = RunOpKernel();
  EXPECT_FALSE(result.ok());
}
TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnInnerWrongSizePermissionMatrixWithNoStartEnd) {
  // Prepare graph.
  SetUpOpWithNoStartEnd();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({5, 4}),
                          {
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                              true, true, true, true, true,  //
                          });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  auto result = RunOpKernel();
  EXPECT_FALSE(result.ok());
}
TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnWrongRankPermissionMatrixWithNoStartEnd) {
  // Prepare graph.
  SetUpOpWithNoStartEnd();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({16}), {
                                                 true, true, true, true,  //
                                                 true, true, true, true,  //
                                                 true, true, true, true,  //
                                                 true, true, true, true,  //
                                             });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  auto result = RunOpKernel();
  EXPECT_FALSE(result.ok());
}

TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnOuterWrongSizeWeightMatrixWithNoStartEnd) {
  // Prepare graph.
  SetUpOpWithNoStartEnd();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({0, 0}), {});

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({4, 5}), {0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.1, 0.5, 0.5, 1.0, 1.0});
  auto result = RunOpKernel();
  EXPECT_FALSE(result.ok());
}
TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnInnerWrongSizeWeightMatrixWithNoStartEnd) {
  // Prepare graph.
  SetUpOpWithNoStartEnd();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({0, 0}), {});

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({5, 4}), {0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.1, 0.5, 0.5, 1.0, 1.0});

  auto result = RunOpKernel();
  EXPECT_FALSE(result.ok());
}
TEST_F(ConstrainedSequenceInputValidationTest,
       FailsOnWrongRankWeightMatrixWithNoStartEnd) {
  // Prepare graph.
  SetUpOpWithNoStartEnd();
  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 3.0, 4.0,  //
                               1.0, 12.0, 3.0, 4.0,  //
                               1.0, 2.0, 3.0, 14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({0, 0}), {});

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({16}), {0.5, 0.5, 0.5, 1.0,  //
                                               0.5, 0.5, 0.5, 1.0,  //
                                               0.5, 0.5, 0.5, 1.0,  //
                                               0.5, 0.5, 0.5, 1.0});
  auto result = RunOpKernel();
  EXPECT_FALSE(result.ok());
}

}  // namespace tensorflow
