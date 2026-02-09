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

class ExpViterbiConstrainedSequenceTest : public tensorflow::OpsTestBase {
 public:
  void SetUpOpWithDefaults() {
    // Prepare graph.
    TF_ASSERT_OK(NodeDefBuilder("tested_op", "ConstrainedSequence")
                     .Attr("Tin", DT_INT32)
                     .Attr("use_viterbi", true)
                     .Attr("use_log_space", false)
                     .Attr("use_start_and_end_states", true)
                     .Input(FakeInput())
                     .Input(FakeInput())
                     .Input(FakeInput())
                     .Input(FakeInput())
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

// TODO(b/122968457): There are a bunch of tests that only validate !ok instead
// of looking for specific error messages; fix that.

// This test examines evaluations with only a permissions matrix.
TEST_F(ExpViterbiConstrainedSequenceTest,
       ComputesSingleTransitionWithNoWeights) {
  // Prepare graph.
  SetUpOpWithDefaults();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 12.0, 13.0, 4.0,  //
                               1.0, 12.0, 13.0, 14.0,  //
                               15.0, 2.0, 3.0, 14.0,   //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

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

// This test examines evaluations with an empty weights matrix not of rank 2.
TEST_F(ExpViterbiConstrainedSequenceTest,
       ComputesSingleTransitionWithNonMatrixEmptyWeights) {
  // Prepare graph.
  SetUpOpWithDefaults();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 12.0, 13.0, 4.0,  //
                               1.0, 12.0, 13.0, 14.0,  //
                               15.0, 2.0, 3.0, 14.0,   //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

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
  AddInputFromArray<float>(TensorShape({0}), {});

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

// This test examines evaluations with a 2D score matrix (implicit batch 1).
TEST_F(ExpViterbiConstrainedSequenceTest,
       ComputesSingleTransitionWithSingleBatchItem) {
  // Prepare graph.
  SetUpOpWithDefaults();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({1, 4}),  //
                           {
                               10.0, 12.0, 13.0, 4.0,  //
                           });

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({1}), {1});

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

  // The sequence's highest score is 2, but OUT->2 is not ok, so it's 1.
  // Validate the output.
  std::vector<int32> expected_transitions({1});
  std::vector<int64> expected_offsets({0, 1});

  // Validate the output.
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_transitions));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_offsets));
}

// This test examines int64 input type and int32 output type.
TEST_F(ExpViterbiConstrainedSequenceTest, int64inint32out) {
  // Prepare graph.
  SetUpOpWithDefaults();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 12.0, 13.0, 4.0,  //
                               1.0, 12.0, 13.0, 14.0,  //
                               15.0, 2.0, 3.0, 14.0,   //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

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
  // Validate the output.
  std::vector<int32> expected_transitions({1, 3, 0});
  std::vector<int64> expected_offsets({0, 1, 2, 3});

  // Validate the output.
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_transitions));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_offsets));
}

// This test ensures the op can take a sequence length of type {{X},{Y},{Z}}
// (with an outer batch dimension).
TEST_F(ExpViterbiConstrainedSequenceTest, TwoDimensionalSequenceLengths) {
  // Prepare graph.
  SetUpOpWithDefaults();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 12.0, 13.0, 4.0,  //
                               1.0, 12.0, 13.0, 14.0,  //
                               15.0, 2.0, 3.0, 14.0,   //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3, 1}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({5, 5}),
                          {
                              // TO 0  TO 1  TO 2   TO 3  TO OUT
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

// This test ensures that final transitions that are forbidden by the permission
// matrix (final->null) are not taken.
TEST_F(ExpViterbiConstrainedSequenceTest,
       ComputesSingleTransitionWithNoWeightsConstrainedByEnd) {
  // Prepare graph.
  SetUpOpWithDefaults();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 12.0, 13.0, 4.0,  //
                               1.0, 12.0, 13.0, 14.0,  //
                               15.0, 2.0, 3.0, 14.0,   //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({5, 5}),
                          {
                              // TO 0 TO 1  TO 2  TO 3  TO OUT
                              true, true, true,  true, true,   // FROM 0
                              true, true, true,  true, false,  // FROM 1
                              true, true, true,  true, true,   // FROM 2
                              true, true, true,  true, true,   // FROM 3
                              true, true, false, true, false,  // FROM 'OUTSIDE'
                          });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  TF_ASSERT_OK(RunOpKernel());

  // The first sequence's highest score is 2, but OUT->2 is not ok; the next
  // highest is 1, but 1->OUT is not OK; the next highest is 0, which is OK.
  // The second sequence's highest score is 3, OUT->3 is OK and 3->OUT is OK.
  // The third sequence's highest score is 0, OUT->0 is OK and 0->OUT is OK.
  // Validate the output.
  std::vector<int32> expected_transitions({0, 3, 0});
  std::vector<int64> expected_offsets({0, 1, 2, 3});

  // Validate the output.
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_transitions));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_offsets));
}

// This test examines evaluations with only a weight matrix.
TEST_F(ExpViterbiConstrainedSequenceTest,
       ComputesSingleTransitionWithNoPermissions) {
  // Prepare graph.
  SetUpOpWithDefaults();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 7.0, 4.0,    //
                               1.0, 9.0, 11.0, 5.0,    //
                               100.0, 24.0, 3.0, 4.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({0, 0}), {});

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({5, 5}), {0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.1, 0.5, 0.5, 1.0, 1.0});

  TF_ASSERT_OK(RunOpKernel());

  // All scores should be multiplied by the last row in the weight tensor, so
  // the 'real' scores are:
  // 1: {1.0, 1.0, 3.5, 4.0}   (max is 3)
  // 2: {0.1, 4.5, 5.5, 5.0}   (max is 2)
  // 3: {10.0, 12.0, 1.5, 4.0} (max is 1)
  // Validate the output.
  std::vector<int32> expected_transitions({3, 2, 1});
  std::vector<int64> expected_offsets({0, 1, 2, 3});

  // Validate the output.
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_transitions));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_offsets));
}

// This test examines evaluations with an empty not rank 2 permissions matrix.
TEST_F(ExpViterbiConstrainedSequenceTest,
       ComputesSingleTransitionWithNonMatrixEmptyPermissions) {
  // Prepare graph.
  SetUpOpWithDefaults();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 7.0, 4.0,    //
                               1.0, 9.0, 11.0, 5.0,    //
                               100.0, 24.0, 3.0, 4.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({0, 0, 0}), {});

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({5, 5}), {0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.1, 0.5, 0.5, 1.0, 1.0});

  TF_ASSERT_OK(RunOpKernel());

  // All scores should be multiplied by the last row in the weight tensor, so
  // the 'real' scores are:
  // 1: {1.0, 1.0, 3.5, 4.0}   (max is 3)
  // 2: {0.1, 4.5, 5.5, 5.0}   (max is 2)
  // 3: {10.0, 12.0, 1.5, 4.0} (max is 1)
  // Validate the output.
  std::vector<int32> expected_transitions({3, 2, 1});
  std::vector<int64> expected_offsets({0, 1, 2, 3});

  // Validate the output.
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_transitions));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_offsets));
}

// This test ensures that final transitions are scored with the probability
// of ending the sequence on the transition (x->final->null).
TEST_F(ExpViterbiConstrainedSequenceTest,
       ComputesSingleTransitionWithNoPermissionsWeightedByEnd) {
  // Prepare graph.
  SetUpOpWithDefaults();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 7.0, 4.0,    //
                               1.0, 9.0, 11.0, 5.0,    //
                               100.0, 24.0, 3.0, 4.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({0, 0}), {});

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({5, 5}), {0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 0.1,  //
                                                 0.1, 0.5, 0.5, 1.0, 1.0});

  TF_ASSERT_OK(RunOpKernel());

  // All scores should be multiplied by the last row and the last column in the
  // score tensor, so the real scores are:
  // 1: {1.0, 1.0, 3.5, 0.4}   (max is 2)
  // 2: {0.1, 4.5, 5.5, 0.5}   (max is 2)
  // 3: {10.0, 12.0, 1.5, 0.4} (max is 1)
  // Validate the output.
  std::vector<int32> expected_transitions({2, 2, 1});
  std::vector<int64> expected_offsets({0, 1, 2, 3});

  // Validate the output.
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_transitions));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_offsets));
}

// This test ensures that final transitions are not scored with the probability
// of ending the sequence on the transition (x->final->null) if
// use_start_and_end_states is False.
TEST_F(ExpViterbiConstrainedSequenceTest,
       ComputesSingleTransitionWithNoPermissionsNotWeightedByEnd) {
  // Prepare graph.
  TF_ASSERT_OK(NodeDefBuilder("tested_op", "ConstrainedSequence")
                   .Attr("Tin", DT_INT32)
                   .Attr("use_viterbi", true)
                   .Attr("use_log_space", false)
                   .Attr("use_start_and_end_states", false)
                   .Input(FakeInput())
                   .Input(FakeInput())
                   .Input(FakeInput())
                   .Input(FakeInput())
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 7.0, 4.0,    //
                               1.0, 9.0, 11.0, 5.0,    //
                               100.0, 24.0, 3.0, 4.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({0, 0}), {});

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({4, 4}), {0.5, 0.5, 0.5, 0.5,  //
                                                 0.5, 0.5, 0.5, 0.5,  //
                                                 0.5, 0.5, 0.5, 0.5,  //
                                                 0.5, 0.5, 0.5, 0.5});

  TF_ASSERT_OK(RunOpKernel());

  // All scores should be multiplied by the last row and the last column in the
  // score tensor, so the real scores are:
  // 1: {5.0, 1.0, 3.5, 4.0}   (max is 0)
  // 2: {.5, 4.5, 5.5, 2.5}   (max is 2)
  // 3: {50.0, 12.0, 1.5,2.0} (max is 0)
  // Validate the output.
  std::vector<int32> expected_transitions({0, 2, 0});
  std::vector<int64> expected_offsets({0, 1, 2, 3});

  // Validate the output.
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_transitions));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_offsets));
}

// This test examines evaluations with both weight and permission matrices.
TEST_F(ExpViterbiConstrainedSequenceTest,
       ComputesSingleTransitionWithWeightsAndPermissions) {
  // Prepare graph.
  SetUpOpWithDefaults();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 12.0, 7.0, 4.0,   //
                               1.0, 9.0, 11.0, 5.0,    //
                               100.0, 24.0, 3.0, 4.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({5, 5}),
                          {
                              // TO 0 TO 1  TO 2  TO 3  TO OUT
                              true, true,  true, true, true,   // FROM 0
                              true, true,  true, true, true,   // FROM 1
                              true, true,  true, true, true,   // FROM 2
                              true, true,  true, true, true,   // FROM 'OUTSIDE'
                              true, false, true, true, false,  // FROM 'NULL'
                          });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({5, 5}), {0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  //
                                                 0.1, 0.5, 0.5, 1.0, 1.0});

  TF_ASSERT_OK(RunOpKernel());

  // All scores should be multiplied by the last row in the weight tensor, so
  // the 'real' scores are:
  // 1: {1.0, 1.0, 3.5, 4.0}   (max is 3). OUT->3 is OK.
  // 2: {0.1, 4.5, 5.5, 5.0}   (max is 2). OUT->2 is OK.
  // 3: {10.0, 12.0, 1.5, 4.0} (max is 1). OUT->1 is not OK, so go with 0.
  // Note that X->OUT is set to always be OK here.
  std::vector<int32> expected_transitions({3, 2, 0});
  std::vector<int64> expected_offsets({0, 1, 2, 3});

  // Validate the output.
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_transitions));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_offsets));
}

// This test examines multiple evaluations with both weight and permission
// matrices.
TEST_F(ExpViterbiConstrainedSequenceTest,
       ComputesMultipleTransitionsWithWeightsAndPermissions) {
  // Prepare graph.
  SetUpOpWithDefaults();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 2, 4}),  //
                           {{
                               10.0,  12.0, 7.0,  4.0,   // Batch 0, step 0
                               10.0,  10.0, 10.0, 10.0,  // Batch 0, step 1
                               1.0,   9.0,  11.0, 5.0,   // Batch 1, step 0
                               10.0,  15.0, 1.0,  12.0,  // Batch 1, step 1
                               100.0, 24.0, 3.0,  4.0,   // Batch 2, step 0
                               1.0,   11.0, 1.0,  10.0,  // Batch 2, step 1
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {2, 2, 2});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({5, 5}),
                          {
                              // TO 0 TO 1  TO 2  TO 3  TO NUL
                              true, true,  true, true,  true,   // FROM 0
                              true, true,  true, true,  false,  // FROM 1
                              true, false, true, false, true,   // FROM 2
                              true, true,  true, true,  true,   // FROM 3 (OUT)
                              true, false, true, true,  true,   // FROM 'NULL'
                          });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({5, 5}), {0.5, 0.5, 0.5, 0.5, 1.0,  // 0
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  // 1
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  // 2
                                                 0.5, 0.5, 1.0, 0.5, 1.0,  // 3
                                                 0.1, 0.5, 0.5, 1.0, 1.0});

  TF_ASSERT_OK(RunOpKernel());

  // STEP 1:
  // All scores should be multiplied by the last row in the weight tensor, so
  // the 'real' scores are:
  // B0: { 1.0, [NOTOK], 3.5, 4.0}
  // B1: { 0.1, [NOTOK], 5.5, 5.0}
  // B2: {10.0, [NOTOK], 1.5, 4.0}
  //
  // STEP 2:
  //  (Forbidden transitions are marked with '*')
  //
  // BATCH 0:
  //   Raw scores are: {10.0, 10.0, 10.0, 10.0}
  // from 0: New scores are {5.0, 5.0, 5.0, 5.0},  totals: {5, 0, 17.5, 20}
  // from 1: New scores are {5.0, 5.0,  0*, 5.0},  totals: {5, 0,    0, 20}
  // from 2: New scores are {5.0, 5.0, 5.0, 10.0}, totals: {5, 0, 17.5, 40}
  // from 3: New scores are {5.0, 5.0,  0*, 5.0},  totals: {5, 0,    0, 20}
  //   Top scores are 20, 20, 40, 20 from [3, 3, 3, 3].
  //   1->OUT is not valid.
  //   Final scores are  [20, 0, 40, 20] for a
  //   final state of [2] with a sequence of [3->2].
  //
  // BATCH 1:
  //   Raw scores are {10, 15, 1, 12}
  // from 0: Weighted score is {5, 5, 5, 5}, totals:    {0.5, 0, 27.5, 25}
  // from 1: Weighted score is {7.5, 7.5, 0*, 7.5}, t:  {0.75, 0, 0, 37.5}
  // from 2: Weighted score is {0.5, 0.5, 0.5, 1.0}, t: {0.05, 0, 2.75, 5}
  // from 3: Weighted score is {6, 6, 0*, 6}, totals:   {0.6,  0, 0, 30}
  //   Top scores are {27.5, 37.5, 5, 30} from [2, 3, 3, 3]
  //   1->OUT is not valid, so final scores are [27.5, 0, 5, 30] for a final
  //   state of [3] and a sequence of [3, 3]
  //
  // BATCH 2:
  //  Raw scores are {1.0, 11.0, 1.0, 10.0}
  // 2/0: Weighted score is {.5, .5, .5, .5}. t:    {5, 0, 0.75, 2}
  // 2/1: Weighted score is {5.5, 5.5, 0*, 5.5}. t: {55, 0, 0, 22}
  // 2/2: Weighted score is {.5, .5, .5, 1.0}. t:   {5, 0, 0.75, 4}
  // 2/3: Weighted score is {5, 5, 0*, 5}. t:       {50, 0, 0, 20}
  //  Top scores are {5, 55, 5, 50} from [0, 0, 0, 0]
  //  1->OUT is not valid, so final scores are [5, 0, 5, 50] for a final
  //  state of 3 and a sequence of [0, 3].

  std::vector<int32> expected_transitions({3, 2, 3, 3, 0, 3});
  std::vector<int64> expected_offsets({0, 2, 4, 6});

  // Validate the output.
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_transitions));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_offsets));
}

// This test examines multiple evaluations with both weight and permission
// matrices.
TEST_F(ExpViterbiConstrainedSequenceTest,
       ComputesMultipleTransitionsWithVaryingLengths) {
  // Prepare graph.
  SetUpOpWithDefaults();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 2, 4}),  //
                           {{
                               10.0,  12.0, 7.0,  4.0,   // Batch 0, step 0
                               10.0,  10.0, 10.0, 10.0,  // Batch 0, step 1
                               1.0,   9.0,  11.0, 5.0,   // Batch 1, step 0
                               10.0,  15.0, 1.0,  12.0,  // Batch 1, step 1
                               100.0, 24.0, 3.0,  4.0,   // Batch 2, step 0
                               1.0,   11.0, 1.0,  10.0,  // Batch 2, step 1
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {2, 1, 2});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({5, 5}),
                          {
                              // TO 0 TO 1  TO 2  TO 3  TO NUL
                              true, true,  true, true,  true,   // FROM 0
                              true, true,  true, true,  false,  // FROM 1
                              true, false, true, false, true,   // FROM 2
                              true, true,  true, true,  true,   // FROM 3 (OUT)
                              true, false, true, true,  true,   // FROM 'NULL'
                          });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({5, 5}), {0.5, 0.5, 0.5, 0.5, 1.0,  // 0
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  // 1
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  // 2
                                                 0.5, 0.5, 1.0, 0.5, 1.0,  // 3
                                                 0.1, 0.5, 0.5, 1.0, 1.0});

  TF_ASSERT_OK(RunOpKernel());

  // STEP 1:
  // All scores should be multiplied by the last row in the weight tensor, so
  // the 'real' scores are:
  // B0: { 1.0, [NOTOK], 3.5, 4.0}
  // B1: { 0.1, [NOTOK], 5.5, 5.0}
  // B2: {10.0, [NOTOK], 1.5, 4.0}
  //
  // STEP 2:
  //  (Forbidden transitions are marked with '*')
  //
  // BATCH 0:
  //   Raw scores are: {10.0, 10.0, 10.0, 10.0}
  // from 0: New scores are {5.0, 5.0, 5.0, 5.0},  totals: {5, 0, 17.5, 20}
  // from 1: New scores are {5.0, 5.0,  0*, 5.0},  totals: {5, 0,    0, 20}
  // from 2: New scores are {5.0, 5.0, 5.0, 10.0}, totals: {5, 0, 17.5, 40}
  // from 3: New scores are {5.0, 5.0,  0*, 5.0},  totals: {5, 0,    0, 20}
  //   Top scores are 20, 20, 40, 20 from [3, 3, 3, 3].
  //   1->OUT is not valid.
  //   Final scores are  [20, 0, 40, 20] for a
  //   final state of [2] with a sequence of [3->2].
  //
  // BATCH 1:
  //   End of sequence; no further action.
  //
  // BATCH 2:
  //  Raw scores are {1.0, 11.0, 1.0, 10.0}
  // 2/0: Weighted score is {.5, .5, .5, .5}. t:    {5, 0, 0.75, 2}
  // 2/1: Weighted score is {5.5, 5.5, 0*, 5.5}. t: {55, 0, 0, 22}
  // 2/2: Weighted score is {.5, .5, .5, 1.0}. t:   {5, 0, 0.75, 4}
  // 2/3: Weighted score is {5, 5, 0*, 5}. t:       {50, 0, 0, 20}
  //  Top scores are {5, 55, 5, 50} from [0, 0, 0, 0]
  //  1->OUT is not valid, so final scores are [5, 0, 5, 50] for a final
  //  state of 3 and a sequence of [0, 3].

  std::vector<int32> expected_transitions({3, 2, 2, 0, 3});
  std::vector<int64> expected_offsets({0, 2, 3, 5});

  // Validate the output.
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_transitions));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_offsets));
}

// This test examines evaluations with an all-zero weight matrix.
TEST_F(ExpViterbiConstrainedSequenceTest,
       ComputesSingleTransitionWithZeroedWeights) {
  // Prepare graph.
  SetUpOpWithDefaults();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 7.0, 4.0,    //
                               1.0, 9.0, 11.0, 5.0,    //
                               100.0, 24.0, 3.0, 4.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({0, 0}), {});

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({5, 5}), {
                                                    0.0, 0.0, 0.0, 0.0, 0.0,  //
                                                    0.0, 0.0, 0.0, 0.0, 0.0,  //
                                                    0.0, 0.0, 0.0, 0.0, 0.0,  //
                                                    0.0, 0.0, 0.0, 0.0, 0.0,  //
                                                    0.0, 0.0, 0.0, 0.0, 0.0,
                                                });

  TF_ASSERT_OK(RunOpKernel());

  // In the case of a tie between weights, the higher state number wins;
  // if all weights are zero, the states should all be 3.

  std::vector<int32> expected_transitions({3, 3, 3});
  std::vector<int64> expected_offsets({0, 1, 2, 3});

  // Validate the output.
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_transitions));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_offsets));
}

TEST_F(ExpViterbiConstrainedSequenceTest,
       ImpossibleSequencesResultInNegativeOnesIfAttrIsSet) {
  // Prepare graph.
  SetUpOpWithDefaults();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 2, 4}),  //
                           {{
                               10.0, 12.0, 13.0, 4.0,   //
                               1.0,  12.0, 13.0, 14.0,  //
                               15.0, 2.0,  3.0,  14.0,  //
                               10.0, 12.0, 13.0, 4.0,   //
                               1.0,  12.0, 13.0, 14.0,  //
                               15.0, 2.0,  3.0,  14.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {2, 2, 2});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({5, 5}),
                          {
                              // TO 0 TO 1  TO 2  TO 3  TO OUT
                              false, false, false, false, false,  // FROM 0
                              false, false, false, false, false,  // FROM 1
                              false, false, false, false, false,  // FROM 2
                              false, false, false, false, false,  // FROM 3
                              false, false, false, false, false,  // FROM 'OUT'
                          });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  TF_ASSERT_OK(RunOpKernel());

  // Validate the output.

  std::vector<int32> expected_transitions({-1, -1, -1, -1, -1, -1});
  std::vector<int64> expected_offsets({0, 2, 4, 6});

  // Validate the output.
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_transitions));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_offsets));
}

// This test ensures the op will throw an error if there are too few scores to
// finalize all the sequences.
TEST_F(ExpViterbiConstrainedSequenceTest, ErrorsIfGivenInsufficientScores) {
  // Prepare graph.
  SetUpOpWithDefaults();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 12.0, 13.0, 4.0,  //
                               1.0, 12.0, 13.0, 14.0,  //
                               15.0, 2.0, 3.0, 14.0,   //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 2, 1});

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

  auto result = RunOpKernel();
  EXPECT_FALSE(result.ok());
}

// This test ensures that the op correctly outputs a ragged tensor with type
// int32
TEST_F(ExpViterbiConstrainedSequenceTest, OutputsInt32RaggedTensor) {
  // Prepare graph.
  SetUpOpWithDefaults();

  AddInputFromArray<float>(
      TensorShape({3, 2, 4}),  //
      {{
          10.0,  12.0, 7.0,  4.0,   // Tr. to 3
          10.0,  10.0, 10.0, 10.0,  // Tr. 3 to 2 on wt.
          1.0,   9.0,  11.0, 5.0,   // Tr. to 2
          10.0,  15.0, 1.0,  12.0,  // Irrelevant (past end of sequence)
          100.0, 24.0, 3.0,  4.0,   // Tr. to 0
          1.0,   10.0, 1.0,  10.0,  // Tr. 0 to 3 (1 cannot tr. to NULL)
      }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {2, 1, 2});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({5, 5}),
                          {
                              // TO 0 TO 1  TO 2  TO 3  TO NUL
                              true, true,  true, true,  true,   // FROM 0
                              true, true,  true, true,  false,  // FROM 1
                              true, false, true, false, true,   // FROM 2
                              true, true,  true, true,  true,   // FROM 3 (OUT)
                              true, false, true, true,  true,   // FROM 'NULL'
                          });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({5, 5}), {0.5, 0.5, 0.5, 0.5, 1.0,  // 0
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  // 1
                                                 0.5, 0.5, 0.5, 0.5, 1.0,  // 2
                                                 0.5, 0.5, 1.0, 0.5, 1.0,  // 3
                                                 0.1, 0.5, 0.5, 1.0, 1.0});

  TF_ASSERT_OK(RunOpKernel());

  std::vector<int32> expected_transitions({3, 2, 2, 0, 3});
  std::vector<int64> expected_offsets({0, 2, 3, 5});

  // Validate the output.
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_transitions));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_offsets));
}

}  // namespace tensorflow
