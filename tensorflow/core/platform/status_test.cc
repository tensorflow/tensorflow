
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
#include "tensorflow/core/platform/status.h"

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(ToStringTest, PayloadsArePrinted) {
  Status status = errors::Aborted("Aborted Error Message");
  status.SetPayload("payload_key",
                    absl::StrFormat("payload_value %c%c%c", 1, 2, 3));

  EXPECT_EQ(status.ToString(),
            "Aborted: Aborted Error Message [payload_key='payload_value "
            "\\x01\\x02\\x03']");
}

TEST(ToStringTest, MatchesAbslStatus) {
  Status status = errors::Aborted("Aborted Error Message");
  status.SetPayload("payload_key",
                    absl::StrFormat("payload_value %c%c%c", 1, 2, 3));

  absl::Status absl_status =
      absl::Status(absl::StatusCode::kAborted, status.error_message());
  absl_status.SetPayload("payload_key", absl::Cord(absl::StrFormat(
                                            "payload_value %c%c%c", 1, 2, 3)));

  // TODO(b/194924033): Error Codes do not match capitalization
  EXPECT_EQ(status.ToString().substr(7), absl_status.ToString().substr(7));
}
}  // namespace tensorflow
