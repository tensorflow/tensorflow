// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and

#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/qnn_manager.h"

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/lrt/test/common.h"

namespace {

using ::lrt::qnn::QnnManager;
using ::lrt::qnn::SetupAll;

// NOTE: This tests that all of the dynamic loading works properly and
// the QNN SDK instance can be properly initialized and destroyed.

TEST(QnnSdkTest, SetupQnnManager) {
  QnnManager qnn;
  ASSERT_STATUS_OK(SetupAll(/*soc_model=*/std::nullopt, qnn));
}

TEST(QnnSdkTest, SetupQnnManagerWithSystem) {
  QnnManager qnn;
  ASSERT_STATUS_OK(
      SetupAll(/*soc_model=*/std::nullopt, qnn, /*load_system=*/true));
}

TEST(QnnSdkTest, SetupQnnManagerWithContext) {
  QnnManager qnn;
  ASSERT_STATUS_OK(SetupAll(/*soc_model=*/std::nullopt, qnn,
                            /*load_system=*/false, /*load_context=*/true));
}

}  // namespace
