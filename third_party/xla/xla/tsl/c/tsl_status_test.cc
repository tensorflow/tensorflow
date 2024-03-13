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

#include "xla/tsl/c/tsl_status.h"

#include <string>
#include <unordered_map>
#include <utility>

#include "xla/tsl/c/tsl_status_internal.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/test.h"

namespace tsl {
namespace {

TEST(TSL_Status, PayloadsSet) {
  TSL_Status* tsl_status = TSL_NewStatus();
  TSL_SetStatus(tsl_status, TSL_CANCELLED, "Error Message");
  TSL_SetPayload(tsl_status, "a", "1");
  TSL_SetPayload(tsl_status, "b", "2");
  TSL_SetPayload(tsl_status, "c", "3");

  std::unordered_map<std::string, std::string> payloads;
  TSL_ForEachPayload(
      tsl_status,
      [](const char* key, const char* value, void* capture) {
        std::unordered_map<std::string, std::string>* payloads =
            static_cast<std::unordered_map<std::string, std::string>*>(capture);
        payloads->emplace(key, value);
      },
      &payloads);
  EXPECT_EQ(payloads.size(), 3);
  EXPECT_EQ(payloads.at("a"), "1");
  EXPECT_EQ(payloads.at("b"), "2");
  EXPECT_EQ(payloads.at("c"), "3");
  TSL_DeleteStatus(tsl_status);
}

}  // namespace
}  // namespace tsl
