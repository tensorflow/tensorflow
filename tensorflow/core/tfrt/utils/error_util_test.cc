/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/utils/error_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/platform/status.h"
#include "tfrt/support/error_util.h"  // from @tf_runtime

namespace tfrt {
namespace {

TEST(ErrorUtilTest, AllSupportedErrorConversion){
#define ERROR_TYPE(TFRT_ERROR, TF_ERROR)                                  \
  {                                                                       \
    tensorflow::Status status(tensorflow::error::TF_ERROR, "error_test"); \
    EXPECT_EQ(ConvertTfErrorCodeToTfrtErrorCode(status),                  \
              tfrt::ErrorCode::TFRT_ERROR);                               \
  }
#include "tensorflow/core/tfrt/utils/error_type.def"  // NOLINT
}

TEST(ErrorUtilTest, UnsupportedErrorConversion) {
  tensorflow::Status status(tensorflow::error::UNAUTHENTICATED, "error_test");
  EXPECT_EQ(ConvertTfErrorCodeToTfrtErrorCode(status),
            tfrt::ErrorCode::kUnknown);
}

}  // namespace
}  // namespace tfrt
