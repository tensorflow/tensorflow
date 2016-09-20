/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include <limits>

#include "tensorflow/contrib/linear_optimizer/kernels/hinge-loss.h"
#include "tensorflow/contrib/linear_optimizer/kernels/logistic-loss.h"
#include "tensorflow/contrib/linear_optimizer/kernels/smooth-hinge-loss.h"
#include "tensorflow/contrib/linear_optimizer/kernels/squared-loss.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

// TODO(sibyl-Aix6ihai): add a test to show the improvements of the Newton
// modification detailed in readme.md

TEST(LogisticLoss, ComputePrimalLoss) {
  LogisticLossUpdater loss_updater;
  EXPECT_NEAR(0.693147, loss_updater.ComputePrimalLoss(
                            0 /* wx */, 1 /* label */, 1 /* example weight */),
              1e-3);
  EXPECT_NEAR(0.0, loss_updater.ComputePrimalLoss(70 /* wx */, 1 /* label */,
                                                  1 /* example weight */),
              1e-3);
  EXPECT_NEAR(0.0, loss_updater.ComputePrimalLoss(-70 /* wx */, -1 /* label */,
                                                  1 /* example weight */),
              1e-3);
}

TEST(LogisticLoss, ComputeDualLoss) {
  LogisticLossUpdater loss_updater;
  EXPECT_NEAR(0.0,
              loss_updater.ComputeDualLoss(0 /* current dual */, 1 /* label */,
                                           1 /* example weight */),
              1e-3);
  EXPECT_NEAR(0.0,
              loss_updater.ComputeDualLoss(1 /* current dual */, 1 /* label */,
                                           1 /* example weight */),
              1e-3);
  EXPECT_NEAR(-0.693147, loss_updater.ComputeDualLoss(0.5 /* current dual */,
                                                      1 /* label */,
                                                      1 /* example weight */),
              1e-3);
}

TEST(LogisticLoss, ComputeUpdatedDual) {
  LogisticLossUpdater loss_updater;
  EXPECT_NEAR(0.479, loss_updater.ComputeUpdatedDual(
                         1 /* num partitions */, 1.0 /* label */,
                         1.0 /* example weight */, 0.5 /* current_dual */,
                         0.3 /* wx */, 10.0 /* weighted_example_norm */),
              1e-3);

  EXPECT_NEAR(-0.031, loss_updater.ComputeUpdatedDual(
                          2 /* num partitions */, -1.0 /* label */,
                          1.0 /* example weight */, 0.1 /* current_dual */,
                          -0.8 /* wx */, 10.0 /* weighted_example_norm */),
              1e-3);
}

TEST(SquaredLoss, ComputePrimalLoss) {
  SquaredLossUpdater loss_updater;
  EXPECT_NEAR(0.5, loss_updater.ComputePrimalLoss(0.0 /* wx */, 1.0 /* label */,
                                                  1.0 /* example weight */),
              1e-3);
  EXPECT_NEAR(40.5,
              loss_updater.ComputePrimalLoss(10.0 /* wx */, 1.0 /* label */,
                                             1.0 /* example weight */),
              1e-3);
  EXPECT_NEAR(0.125,
              loss_updater.ComputePrimalLoss(-0.5 /* wx */, -1.0 /* label */,
                                             1.0 /* example weight */),
              1e-3);
  EXPECT_NEAR(4.84,
              loss_updater.ComputePrimalLoss(1.2 /* wx */, -1.0 /* label */,
                                             2.0 /* example weight */),
              1e-3);
}

TEST(SquaredLoss, ComputeDualLoss) {
  SquaredLossUpdater loss_updater;
  EXPECT_NEAR(0.0, loss_updater.ComputeDualLoss(0.0 /* current dual */,
                                                -1.0 /* label */,
                                                1.0 /* example weight */),
              1e-3);
  EXPECT_NEAR(0.66, loss_updater.ComputeDualLoss(0.2 /* current dual */,
                                                 -1.0 /* label */,
                                                 3.0 /* example weight */),
              1e-3);
  EXPECT_NEAR(-0.375, loss_updater.ComputeDualLoss(1.5 /* current dual */,
                                                   1.0 /* label */,
                                                   1.0 /* example weight */),
              1e-3);
  EXPECT_NEAR(-1.125, loss_updater.ComputeDualLoss(0.5 /* current dual */,
                                                   1.0 /* label */,
                                                   3.0 /* example weight */),
              1e-3);
}

TEST(SquaredLoss, ComputeUpdatedDual) {
  SquaredLossUpdater loss_updater;
  EXPECT_NEAR(0.336, loss_updater.ComputeUpdatedDual(
                         1 /* num partitions */, 1.0 /* label */,
                         1.0 /* example weight */, 0.3 /* current_dual */,
                         0.3 /* wx */, 10.0 /* weighted_example_norm */),
              1e-3);

  EXPECT_NEAR(-0.427, loss_updater.ComputeUpdatedDual(
                          5 /* num partitions */, -1.0 /* label */,
                          1.0 /* example weight */, -0.4 /* current_dual */,
                          0.8 /* wx */, 10.0 /* weighted_example_norm */),
              1e-3);
}

TEST(HingeLoss, ComputePrimalLoss) {
  HingeLossUpdater loss_updater;
  EXPECT_NEAR(1.0, loss_updater.ComputePrimalLoss(0.0 /* wx */, 1.0 /* label */,
                                                  1.0 /* example weight */),
              1e-3);
  EXPECT_NEAR(0.0,
              loss_updater.ComputePrimalLoss(10.0 /* wx */, 1.0 /* label */,
                                             1.0 /* example weight */),
              1e-3);
  EXPECT_NEAR(0.5,
              loss_updater.ComputePrimalLoss(-0.5 /* wx */, -1.0 /* label */,
                                             1.0 /* example weight */),
              1e-3);
  EXPECT_NEAR(4.4,
              loss_updater.ComputePrimalLoss(1.2 /* wx */, -1.0 /* label */,
                                             2.0 /* example weight */),
              1e-3);
}

TEST(HingeLoss, ComputeDualLoss) {
  HingeLossUpdater loss_updater;
  EXPECT_NEAR(0.0, loss_updater.ComputeDualLoss(0.0 /* current dual */,
                                                -1.0 /* label */,
                                                1.0 /* example weight */),
              1e-3);
  EXPECT_NEAR(
      std::numeric_limits<double>::max(),
      loss_updater.ComputeDualLoss(0.2 /* current dual */, -1.0 /* label */,
                                   3.0 /* example weight */),
      1e-3);
  EXPECT_NEAR(
      std::numeric_limits<double>::max(),
      loss_updater.ComputeDualLoss(1.5 /* current dual */, 1.0 /* label */,
                                   1.0 /* example weight */),
      1e-3);
  EXPECT_NEAR(-1.5, loss_updater.ComputeDualLoss(0.5 /* current dual */,
                                                 1.0 /* label */,
                                                 3.0 /* example weight */),
              1e-3);
}

TEST(HingeLoss, ConvertLabel) {
  HingeLossUpdater loss_updater;
  float example_label = 1.0;
  Status status;

  // A label with value 1.0 should remain intact.
  TF_EXPECT_OK(loss_updater.ConvertLabel(&example_label));
  EXPECT_EQ(1.0, example_label);

  // A label with value 0.0 should be converted to -1.0.
  example_label = 0.0;
  TF_EXPECT_OK(loss_updater.ConvertLabel(&example_label));
  EXPECT_EQ(-1.0, example_label);

  // Any other initial value should throw an error.
  example_label = 0.5;
  status = loss_updater.ConvertLabel(&example_label);
  EXPECT_FALSE(status.ok());
}

TEST(HingeLoss, ComputeUpdatedDual) {
  HingeLossUpdater loss_updater;
  // When label=1.0, example_weight=1.0, current_dual=0.5, wx=0.3 and
  // weighted_example_norm=100.0, it turns out that the optimal value to update
  // the dual to is 0.507 which is within the permitted range and thus should be
  // the value returned.
  EXPECT_NEAR(0.507, loss_updater.ComputeUpdatedDual(
                         1 /* num partitions */, 1.0 /* label */,
                         1.0 /* example weight */, 0.5 /* current_dual */,
                         0.3 /* wx */, 100.0 /* weighted_example_norm */),
              1e-3);
  // When label=-1.0, example_weight=1.0, current_dual=0.4, wx=0.6,
  // weighted_example_norm=10.0 and num_loss_partitions=10, it turns out that
  // the optimal value to update the dual to is 0.384 which is within the
  // permitted range and thus should be the value returned.
  EXPECT_NEAR(-0.416, loss_updater.ComputeUpdatedDual(
                          10 /* num partitions */, -1.0 /* label */,
                          1.0 /* example weight */, -0.4 /* current_dual */,
                          0.6 /* wx */, 10.0 /* weighted_example_norm */),
              1e-3);
  // When label=1.0, example_weight=1.0, current_dual=-0.5, wx=0.3 and
  // weighted_example_norm=10.0, it turns out that the optimal value to update
  // the dual to is -0.43. However, this is outside the allowed [0.0, 1.0] range
  // and hence the closest permitted value (0.0) should be returned instead.
  EXPECT_NEAR(0.0, loss_updater.ComputeUpdatedDual(
                       1 /* num partitions */, 1.0 /* label */,
                       1.0 /* example weight */, -0.5 /* current_dual */,
                       0.3 /* wx */, 10.0 /* weighted_example_norm */),
              1e-3);

  // When label=-1.0, example_weight=2.0, current_dual=-1.0, wx=0.3 and
  // weighted_example_norm=10.0, it turns out that the optimal value to update
  // the dual to is -1.065. However, this is outside the allowed [-1.0, 0.0]
  // range and hence the closest permitted value (-1.0) should be returned
  // instead.
  EXPECT_NEAR(-1.0, loss_updater.ComputeUpdatedDual(
                        1 /* num partitions */, -1.0 /* label */,
                        2.0 /* example weight */, -1.0 /* current_dual */,
                        0.3 /* wx */, 10.0 /* weighted_example_norm */),
              1e-3);
}

TEST(SmoothHingeLoss, ComputePrimalLoss) {
  SmoothHingeLossUpdater loss_updater;
  EXPECT_NEAR(0.5, loss_updater.ComputePrimalLoss(0.0 /* wx */, 1.0 /* label */,
                                                  1.0 /* example weight */),
              1e-3);
  EXPECT_NEAR(0.0,
              loss_updater.ComputePrimalLoss(10.0 /* wx */, 1.0 /* label */,
                                             1.0 /* example weight */),
              1e-3);
  EXPECT_NEAR(0.125,
              loss_updater.ComputePrimalLoss(-0.5 /* wx */, -1.0 /* label */,
                                             1.0 /* example weight */),
              1e-3);
  EXPECT_NEAR(3.4,
              loss_updater.ComputePrimalLoss(1.2 /* wx */, -1.0 /* label */,
                                             2.0 /* example weight */),
              1e-3);
}

TEST(SmoothHingeLoss, ComputeDualLoss) {
  SmoothHingeLossUpdater loss_updater;
  EXPECT_NEAR(0.0, loss_updater.ComputeDualLoss(0.0 /* current dual */,
                                                -1.0 /* label */,
                                                1.0 /* example weight */),
              1e-3);
  EXPECT_NEAR(
      std::numeric_limits<double>::max(),
      loss_updater.ComputeDualLoss(0.2 /* current dual */, -1.0 /* label */,
                                   3.0 /* example weight */),
      1e-3);
  EXPECT_NEAR(
      std::numeric_limits<double>::max(),
      loss_updater.ComputeDualLoss(1.5 /* current dual */, 1.0 /* label */,
                                   1.0 /* example weight */),
      1e-3);
  EXPECT_NEAR(-1.125, loss_updater.ComputeDualLoss(0.5 /* current dual */,
                                                   1.0 /* label */,
                                                   3.0 /* example weight */),
              1e-3);
}

TEST(SmoothHingeLoss, ComputeUpdatedDual) {
  SmoothHingeLossUpdater loss_updater;
  EXPECT_NEAR(0.336, loss_updater.ComputeUpdatedDual(
                         1 /* num partitions */, 1.0 /* label */,
                         1.0 /* example weight */, 0.3 /* current_dual */,
                         0.3 /* wx */, 10.0 /* weighted_example_norm */),
              1e-3);

  EXPECT_NEAR(-0.427, loss_updater.ComputeUpdatedDual(
                          5 /* num partitions */, -1.0 /* label */,
                          1.0 /* example weight */, -0.4 /* current_dual */,
                          0.8 /* wx */, 10.0 /* weighted_example_norm */),
              1e-3);
}

}  // namespace
}  // namespace tensorflow
