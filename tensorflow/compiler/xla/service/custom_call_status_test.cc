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

#include "tensorflow/compiler/xla/service/custom_call_status_internal.h"
#include "tensorflow/compiler/xla/service/custom_call_status_test_c_caller.h"
#include "tensorflow/tsl/platform/test.h"

TEST(XlaCustomCallStatusTest, DefaultIsSuccess) {
  XlaCustomCallStatus status;

  ASSERT_EQ(xla::CustomCallStatusGetMessage(&status), std::nullopt);
}

TEST(XlaCustomCallStatusTest, SetSuccess) {
  XlaCustomCallStatus status;
  XlaCustomCallStatusSetSuccess(&status);

  ASSERT_EQ(xla::CustomCallStatusGetMessage(&status), std::nullopt);
}

TEST(XlaCustomCallStatusTest, SetSuccessAfterFailure) {
  XlaCustomCallStatus status;
  XlaCustomCallStatusSetFailure(&status, "error", 5);
  XlaCustomCallStatusSetSuccess(&status);

  ASSERT_EQ(xla::CustomCallStatusGetMessage(&status), std::nullopt);
}

TEST(XlaCustomCallStatusTest, SetFailure) {
  XlaCustomCallStatus status;
  XlaCustomCallStatusSetFailure(&status, "error", 5);

  ASSERT_EQ(xla::CustomCallStatusGetMessage(&status), "error");
}

TEST(XlaCustomCallStatusTest, SetFailureAfterSuccess) {
  XlaCustomCallStatus status;
  XlaCustomCallStatusSetSuccess(&status);
  XlaCustomCallStatusSetFailure(&status, "error", 5);

  ASSERT_EQ(xla::CustomCallStatusGetMessage(&status), "error");
}

TEST(XlaCustomCallStatusTest, SetFailureTruncatesErrorAtGivenLength) {
  XlaCustomCallStatus status;
  XlaCustomCallStatusSetFailure(&status, "error", 4);

  ASSERT_EQ(xla::CustomCallStatusGetMessage(&status), "erro");
}

TEST(XlaCustomCallStatusTest, SetFailureTruncatesErrorAtNullTerminator) {
  XlaCustomCallStatus status;
  XlaCustomCallStatusSetFailure(&status, "error", 100);

  ASSERT_EQ(xla::CustomCallStatusGetMessage(&status), "error");
}

// Test that the API works when called from pure C code.

TEST(XlaCustomCallStatusTest, CSetSuccess) {
  XlaCustomCallStatus status;
  CSetSuccess(&status);

  ASSERT_EQ(xla::CustomCallStatusGetMessage(&status), std::nullopt);
}

TEST(XlaCustomCallStatusTest, CSetFailure) {
  XlaCustomCallStatus status;
  CSetFailure(&status, "error", 5);

  ASSERT_EQ(xla::CustomCallStatusGetMessage(&status), "error");
}
