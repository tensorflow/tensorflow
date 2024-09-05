/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/tf_status_helper.h"

#include "tsl/platform/errors.h"
#include "tsl/platform/test.h"

namespace tsl {
namespace {

TEST(StatusHelper, TestStatusHelper) {
  TSL_Status* s = TSL_NewStatus();
  absl::Status cc_status(absl::InvalidArgumentError("some error"));
  cc_status.SetPayload("key1", absl::Cord("value1"));
  cc_status.SetPayload("key2", absl::Cord("value2"));
  Set_TF_Status_from_Status(s, cc_status);
  ASSERT_EQ(TSL_INVALID_ARGUMENT, TSL_GetCode(s));
  ASSERT_EQ(std::string("some error"), TSL_Message(s));

  absl::Status another_cc_status(StatusFromTF_Status(s));
  ASSERT_FALSE(another_cc_status.ok());
  ASSERT_EQ(std::string("some error"), another_cc_status.message());
  ASSERT_EQ(error::INVALID_ARGUMENT, another_cc_status.code());
  // Ensure the payloads are not lost during conversions
  ASSERT_EQ(cc_status.GetPayload("key1"), another_cc_status.GetPayload("key1"));
  ASSERT_EQ(cc_status.GetPayload("key2"), another_cc_status.GetPayload("key2"));
  TSL_DeleteStatus(s);
}

}  // namespace
}  // namespace tsl
