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

#include "tensorflow/c/tf_status.h"

#include <utility>

#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(TF_Status, PayloadsSet) {
  TF_Status* tf_status = TF_NewStatus();
  TF_SetStatus(tf_status, TF_CANCELLED, "Error Message");
  TF_SetPayload(tf_status, "a", "1");
  TF_SetPayload(tf_status, "b", "2");
  TF_SetPayload(tf_status, "c", "3");

  const std::unordered_map<std::string, std::string> payloads =
      errors::GetPayloads(tf_status->status);
  EXPECT_EQ(payloads.size(), 3);
  EXPECT_EQ(payloads.at("a"), "1");
  EXPECT_EQ(payloads.at("b"), "2");
  EXPECT_EQ(payloads.at("c"), "3");
  TF_DeleteStatus(tf_status);
}

}  // namespace
}  // namespace tensorflow
