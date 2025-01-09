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
// limitations under the License.

#include "tensorflow/lite/experimental/litert/c/litert_logging.h"

#include <gtest/gtest.h>  // NOLINT: Need when ANDROID_API_LEVEL >= 26
#include "tensorflow/lite/experimental/litert/c/litert_common.h"

TEST(Layout, Creation) {
  LiteRtLogger logger;
  ASSERT_EQ(LiteRtCreateLogger(&logger), kLiteRtStatusOk);
  LiteRtDestroyLogger(logger);
}

TEST(Layout, MinLogging) {
  LiteRtLogger logger;
  ASSERT_EQ(LiteRtCreateLogger(&logger), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetMinLoggerSeverity(logger, LITERT_SILENT), kLiteRtStatusOk);
  LiteRtLogSeverity min_severity;
  ASSERT_EQ(LiteRtGetMinLoggerSeverity(logger, &min_severity), kLiteRtStatusOk);
  ASSERT_EQ(min_severity, LITERT_SILENT);
  LiteRtDestroyLogger(logger);
}
