/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/contrib/linear_optimizer/kernels/logistic-loss.h"
#include "tensorflow/contrib/linear_optimizer/kernels/squared-loss.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

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

// TODO(rohananil): Add tests for dual update.
// TODO(dbaylor): Add tests for squared loss.
// TODO(pmol): Add tests for hinge loss.

}  // namespace
}  // namespace tensorflow
