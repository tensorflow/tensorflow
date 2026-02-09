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


// TODO(b/122968457): There are a bunch of tests that only validate !ok instead
// of looking for specific error messages; fix that.

class LogViterbiConstrainedSequenceTest : public tensorflow::OpsTestBase {
 public:
  void SetUpOpWithDefaults() {
    // Prepare graph.
    TF_ASSERT_OK(NodeDefBuilder("tested_op", "ConstrainedSequence")
                     .Attr("Tin", DT_INT32)
                     .Attr("use_viterbi", true)
                     .Attr("use_log_space", true)
                     .Attr("use_start_and_end_states", true)
                     .Input(FakeInput())
                     .Input(FakeInput())
                     .Input(FakeInput())
                     .Input(FakeInput())
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

// This test examines evaluations with only a permissions matrix.
TEST_F(LogViterbiConstrainedSequenceTest,
       ComputesSingleTransitionWithNoWeights) {
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
TEST_F(LogViterbiConstrainedSequenceTest,
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
TEST_F(LogViterbiConstrainedSequenceTest,
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
TEST_F(LogViterbiConstrainedSequenceTest, int64inint32out) {
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
TEST_F(LogViterbiConstrainedSequenceTest, TwoDimensionalSequenceLengths) {
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
TEST_F(LogViterbiConstrainedSequenceTest,
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
TEST_F(LogViterbiConstrainedSequenceTest,
       ComputesSingleTransitionWithNoPermissions) {
  // Prepare graph.
  SetUpOpWithDefaults();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 7.0, 4.0,   //
                               1.0, 9.0, 11.0, 5.0,   //
                               -12.0, 3.0, 3.0, 4.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({0, 0}), {});

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({5, 5}), {0.0,  0.0, 0.0, 0.0, 0.0,  //
                                                 0.0,  0.0, 0.0, 0.0, 0.0,  //
                                                 0.0,  0.0, 0.0, 0.0, 0.0,  //
                                                 0.0,  0.0, 0.0, 0.0, 0.0,  //
                                                 10.0, 5.0, 3.0, 1.0, 0.0});

  TF_ASSERT_OK(RunOpKernel());

  // All scores should be summed with the last row in the weight tensor, so
  // the 'real' scores are:
  // 1: {20.0, 7.0, 10.0, 5.0}   (max is 0)
  // 2: {11.0, 14.0, 14.0, 6.0}  (max is 2, due to tiebreaker.)
  // 3: {-2.0, 8.0, 6.0, 5.0}    (max is 1)
  // Validate the output.
  std::vector<int32> expected_transitions({0, 2, 1});
  std::vector<int64> expected_offsets({0, 1, 2, 3});

  // Validate the output.
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_transitions));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_offsets));
}

// This test examines evaluations with an empty not rank 2 permissions matrix.
TEST_F(LogViterbiConstrainedSequenceTest,
       ComputesSingleTransitionWithNonMatrixEmptyPermissions) {
  // Prepare graph.
  SetUpOpWithDefaults();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 7.0, 4.0,   //
                               1.0, 9.0, 11.0, 5.0,   //
                               -12.0, 3.0, 3.0, 4.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({0, 0}), {});

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({5, 5}), {0.0,  0.0, 0.0, 0.0, 0.0,  //
                                                 0.0,  0.0, 0.0, 0.0, 0.0,  //
                                                 0.0,  0.0, 0.0, 0.0, 0.0,  //
                                                 0.0,  0.0, 0.0, 0.0, 0.0,  //
                                                 10.0, 5.0, 3.0, 1.0, 0.0});

  TF_ASSERT_OK(RunOpKernel());

  // All scores should be summed with the last row in the weight tensor, so
  // the 'real' scores are:
  // 1: {20.0, 7.0, 10.0, 5.0}   (max is 0)
  // 2: {11.0, 14.0, 14.0, 6.0}  (max is 2, due to tiebreaker.)
  // 3: {-2.0, 8.0, 6.0, 5.0}    (max is 1)
  // Validate the output.
  std::vector<int32> expected_transitions({0, 2, 1});
  std::vector<int64> expected_offsets({0, 1, 2, 3});

  // Validate the output.
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_transitions));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_offsets));
}

// This test ensures that final transitions are scored with the probability
// of ending the sequence on the transition (x->final->null).
TEST_F(LogViterbiConstrainedSequenceTest,
       ComputesSingleTransitionWithNoPermissionsWeightedByEnd) {
  // Prepare graph.
  SetUpOpWithDefaults();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 7.0, 4.0,   //
                               1.0, 9.0, 11.0, 5.0,   //
                               -12.0, 3.0, 3.0, 4.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({0, 0}), {});

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({5, 5}), {0.0,  0.0, 0.0, 0.0, -15.0,  //
                                                 0.0,  0.0, 0.0, 0.0, 0.0,    //
                                                 0.0,  0.0, 0.0, 0.0, 0.0,    //
                                                 0.0,  0.0, 0.0, 0.0, 0.0,    //
                                                 10.0, 5.0, 3.0, 1.0, 0.0});

  TF_ASSERT_OK(RunOpKernel());

  // All scores should be summed with the last row in the weight tensor, so
  // the 'real' scores are:
  // 1: {5.0, 7.0, 10.0, 5.0}     (max is 2 - state 1->null adds -15.)
  // 2: {11.0, 14.0, 14.0, 6.0}   (max is 2, due to tiebreaker.)
  // 3: {-2.0, 8.0, 6.0, 5.0}     (max is 1)
  // Validate the output.
  std::vector<int32> expected_transitions({2, 2, 1});
  std::vector<int64> expected_offsets({0, 1, 2, 3});

  // Validate the output.
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_transitions));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_offsets));
}

// This test examines evaluations with both weight and permission matrices.
TEST_F(LogViterbiConstrainedSequenceTest,
       ComputesSingleTransitionWithWeightsAndPermissions) {
  // Prepare graph.
  SetUpOpWithDefaults();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               10.0, 2.0, 7.0, 4.0,   //
                               1.0, 9.0, 11.0, 5.0,   //
                               -12.0, 3.0, 3.0, 4.0,  //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({5, 5}),
                          {
                              // TO 0 TO 1  TO 2  TO 3  TO OUT
                              true,  true, true, true, true,   // FROM 0
                              true,  true, true, true, true,   // FROM 1
                              true,  true, true, true, true,   // FROM 2
                              true,  true, true, true, true,   // FROM 'OUTSIDE'
                              false, true, true, true, false,  // FROM 'NULL'
                          });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({5, 5}), {0.0,  0.0, 0.0, 0.0, 0.0,  //
                                                 0.0,  0.0, 0.0, 0.0, 0.0,  //
                                                 0.0,  0.0, 0.0, 0.0, 0.0,  //
                                                 0.0,  0.0, 0.0, 0.0, 0.0,  //
                                                 10.0, 5.0, 3.0, 1.0, 0.0});

  TF_ASSERT_OK(RunOpKernel());

  // All scores should be summed with the last row in the weight tensor, so
  // the 'real' scores are:
  // 1: {20.0, 7.0, 10.0, 5.0}   (max is 0, but NUL->0 is forbidden, so 2.)
  // 2: {11.0, 14.0, 14.0, 6.0}  (max is 2, due to tiebreaker.)
  // 3: {-2.0, 8.0, 6.0, 5.0}    (max is 1)
  // Validate the output.
  std::vector<int32> expected_transitions({2, 2, 1});
  std::vector<int64> expected_offsets({0, 1, 2, 3});

  // Validate the output.
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_transitions));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_offsets));
}

// This test examines multiple evaluations with both weight and permission
// matrices.
TEST_F(LogViterbiConstrainedSequenceTest,
       ComputesMultipleTransitionsWithWeightsAndPermissions) {
  // Prepare graph.
  SetUpOpWithDefaults();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({2, 2, 4}),  //
                           {{
                               10.0, 12.0, 7.0, 4.0,    // Batch 0, step 0
                               13.0, 12.0, 11.0, 10.0,  // Batch 0, step 1
                               7.0, 9.0, 11.0, 5.0,     // Batch 1, step 0
                               10.0, 15.0, 1.0, 12.0,   // Batch 1, step 1
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({2}), {2, 2});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({5, 5}),
                          {
                              // TO 0 TO 1  TO 2  TO 3  TO NUL
                              true, true,  true, true,  true,   // FROM 0
                              true, true,  true, true,  true,   // FROM 1
                              true, false, true, false, false,  // FROM 2
                              true, true,  true, true,  true,   // FROM 3 (OUT)
                              true, false, true, true,  true,   // FROM 'NULL'
                          });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({5, 5}),
                           {-1.0, 1.0,  -2.0, 2.0,  0.0,  // 0
                            3.0,  -3.0, 4.0,  -4.0, 0.0,  // 1
                            5.0,  -5.0, 6.0,  -6.0, 0.0,  // 2
                            -7.0, 7.0,  -8.0, 8.0,  0.0,  // 3
                            0.0,  1.0,  2.0,  3.0,  0.0});

  TF_ASSERT_OK(RunOpKernel());

  // STEP 1:
  // All scores should be summed with the last row in the weight tensor, so the
  // 'real' scores are:
  // B0: { 10.0, [NOTOK],  9.0,  7.0}
  // B1: {  7.0, [NOTOK], 13.0,  8.0}
  //
  // STEP 2:
  //  (Forbidden transitions are marked with '*' and X stands for the lowest
  //   possible score.)
  //
  // BATCH 0:
  //   Raw scores are: {13.0, 12.0, 11.0, 10.0}
  //
  // Final state 0: (13.0) Weighted scores are {12.0, 16.0, 18.0, 6.0}
  //      New totals are {22, X, 27, 18} [max 27 from 2]
  //
  // Final state 1: (12.0) Weighted scores are {13.0, 9.0, X, 19.0},
  //      New totals are {23, X, X, 26} [max 26 from 3]
  //
  // Final state 2: (11.0) Weighted scores are {9, 15, 21, 3},
  //      New totals are {19, X, 30, 10} [max 30 from 2]
  //
  // Final state 3: (10.0) Weighted scores are {12, 6, X, 18},
  //      New totals are {19, X, X, 25} [max 25 from 3]
  //
  //   Top scores are [27, 26, 30, 25] from [2, 3, 2, 3].
  //   2->OUT is X, so final scores are  [27, 26, X, 25] for a
  //   final state of [0] with a sequence of [2->0].
  //
  //
  // BATCH 1:
  //   Previous scores are {7, X, 13, 8}
  //   Raw scores are {10, 15, 1, 12}
  //
  // Final state 0: Weighted score is {9, 18, 15, 3}
  //      New totals are    {16,  X, 28, 11} [max 28 from 2]
  //
  // Final state 1: Weighted score is {16, 12, 10, 22}
  //      New totals are    {23,  X, X*, 30} [max 30 from 3]
  //
  // Final state 2: Weighted score is {-1, 5,  7, -7}
  //      New totals are    {6,  X, 20, 1} [max 20 from 2]
  //
  // Final state 3: Weighted score is {14, 8, 6, 20}
  //      New totals are    {21,  X, X*, 28} [max 28 from 3]
  //
  //   Top scores are [28, 30, 20, 28] from [2, 3, 2, 3].
  //   2->OUT is not valid, so final scores are  [28, 30, X*, 28] for a
  //   final state of [1] with a sequence of [3->1].
  //

  std::vector<int32> expected_transitions({2, 0, 3, 1});
  std::vector<int64> expected_offsets({0, 2, 4});

  // Validate the output.
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_transitions));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_offsets));
}

// This test examines multiple evaluations with both weight and permission
// matrices.
TEST_F(LogViterbiConstrainedSequenceTest,
       ComputesMultipleTransitionsWithVaryingLengths) {
  // Prepare graph.
  SetUpOpWithDefaults();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({2, 2, 4}),  //
                           {{
                               10.0, 12.0, 7.0, 4.0,   // Batch 0, step 0
                               0.0, 0.0, 0.0, 0.0,     // PAD
                               7.0, 9.0, 11.0, 5.0,    // Batch 1, step 0
                               10.0, 15.0, 1.0, 12.0,  // Batch 1, step 1
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({2}), {1, 2});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({5, 5}),
                          {
                              // TO 0 TO 1  TO 2  TO 3  TO NUL
                              true, true,  true, true,  true,   // FROM 0
                              true, true,  true, true,  true,   // FROM 1
                              true, false, true, false, false,  // FROM 2
                              true, true,  true, true,  true,   // FROM 3 (OUT)
                              true, false, true, true,  true,   // FROM 'NULL'
                          });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({5, 5}),
                           {-1.0, 1.0,  -2.0, 2.0,  0.0,  // 0
                            3.0,  -3.0, 4.0,  -4.0, 0.0,  // 1
                            5.0,  -5.0, 6.0,  -6.0, 0.0,  // 2
                            -7.0, 7.0,  -8.0, 8.0,  0.0,  // 3
                            0.0,  1.0,  2.0,  3.0,  0.0});

  TF_ASSERT_OK(RunOpKernel());

  // STEP 1:
  // All scores should be summed with the last row in the weight tensor, so the
  // 'real' scores are:
  // B0: { 10.0, [NOTOK],  9.0,  7.0}
  // B1: {  7.0, [NOTOK], 13.0,  8.0}
  //
  // STEP 2:
  //  (Forbidden transitions are marked with '*' and X stands for the lowest
  //   possible score.)
  //
  // BATCH 0:
  //  Batch 0 is complete.
  //
  // BATCH 1:
  //   Previous scores are {7, X, 13, 8}
  //   Raw scores are {10, 15, 1, 12}
  //
  // Final state 0: Weighted score is {9, 18, 15, 3}
  //      New totals are    {16,  X, 28, 11} [max 28 from 2]
  //
  // Final state 1: Weighted score is {16, 12, 10, 22}
  //      New totals are    {23,  X, X*, 30} [max 30 from 3]
  //
  // Final state 2: Weighted score is {-1, 5,  7, -7}
  //      New totals are    {6,  X, 20, 1} [max 20 from 2]
  //
  // Final state 3: Weighted score is {14, 8, 6, 20}
  //      New totals are    {21,  X, X*, 28} [max 28 from 3]
  //
  //   Top scores are [28, 30, 20, 28] from [2, 3, 2, 3].
  //   2->OUT is not valid, so final scores are  [28, 30, X*, 28] for a
  //   final state of [1] with a sequence of [3->1].
  //

  std::vector<int32> expected_transitions({0, 3, 1});
  std::vector<int64> expected_offsets({0, 1, 3});

  // Validate the output.
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_transitions));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_offsets));
}

// This test examines evaluations with a fully negative input set.
TEST_F(LogViterbiConstrainedSequenceTest,
       ComputesSingleTransitionWithNegativeInputs) {
  // Prepare graph.
  SetUpOpWithDefaults();

  // Add the scores input.
  AddInputFromArray<float>(TensorShape({3, 1, 4}),  //
                           {{
                               -10.0, -12.0, -13.0, -4.0,  //
                               -1.0, -12.0, -13.0, -14.0,  //
                               -15.0, -2.0, -3.0, -14.0,   //
                           }});

  // Add the sequence_lengths input.
  AddInputFromArray<int>(TensorShape({3}), {1, 1, 1});

  // Add the allowed_transitions input.
  AddInputFromArray<bool>(TensorShape({5, 5}),
                          {
                              // TO 0 TO 1  TO 2  TO 3  TO OUT
                              true, true, true, true, true,  // FROM 0
                              true, true, true, true, true,  // FROM 1
                              true, true, true, true, true,  // FROM 2
                              true, true, true, true, true,  // FROM 3
                              true, true, true, true, true,  // FROM 'OUTSIDE'
                          });

  // Add the transition_weights input.
  AddInputFromArray<float>(TensorShape({0, 0}), {});

  TF_ASSERT_OK(RunOpKernel());

  std::vector<int32> expected_transitions({3, 0, 1});
  std::vector<int64> expected_offsets({0, 1, 2, 3});

  // Validate the output.
  EXPECT_THAT(*GetOutput(0), VectorEq(expected_transitions));
  EXPECT_THAT(*GetOutput(1), VectorEq(expected_offsets));
}

TEST_F(LogViterbiConstrainedSequenceTest,
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
TEST_F(LogViterbiConstrainedSequenceTest, ErrorsIfGivenInsufficientScores) {
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

}  // namespace tensorflow
