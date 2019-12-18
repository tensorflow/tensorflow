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

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(StatusHelper, TestStatusHelper) {
  TF_Status* s = TF_NewStatus();
  Status cc_status(errors::InvalidArgument("some error"));
  Set_TF_Status_from_Status(s, cc_status);
  ASSERT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(s));
  ASSERT_EQ(std::string("some error"), TF_Message(s));

  Status another_cc_status(StatusFromTF_Status(s));
  ASSERT_FALSE(another_cc_status.ok());
  ASSERT_EQ(std::string("some error"), another_cc_status.error_message());
  ASSERT_EQ(error::INVALID_ARGUMENT, another_cc_status.code());
  TF_DeleteStatus(s);
}

}  // namespace
}  // namespace tensorflow
