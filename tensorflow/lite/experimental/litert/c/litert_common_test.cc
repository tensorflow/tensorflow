// Copyright 2025 Google LLC.
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

#include "tensorflow/lite/experimental/litert/c/litert_common.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

using testing::Eq;
using testing::Gt;
using testing::Lt;
using testing::StrEq;

TEST(GetStatusString, Works) {
  EXPECT_THAT(LiteRtGetStatusString(kLiteRtStatusOk), StrEq("kLiteRtStatusOk"));
  EXPECT_THAT(LiteRtGetStatusString(kLiteRtStatusErrorInvalidArgument),
              StrEq("kLiteRtStatusErrorInvalidArgument"));
  EXPECT_THAT(LiteRtGetStatusString(kLiteRtStatusErrorMemoryAllocationFailure),
              StrEq("kLiteRtStatusErrorMemoryAllocationFailure"));
  EXPECT_THAT(LiteRtGetStatusString(kLiteRtStatusErrorRuntimeFailure),
              StrEq("kLiteRtStatusErrorRuntimeFailure"));
  EXPECT_THAT(LiteRtGetStatusString(kLiteRtStatusErrorMissingInputTensor),
              StrEq("kLiteRtStatusErrorMissingInputTensor"));
  EXPECT_THAT(LiteRtGetStatusString(kLiteRtStatusErrorUnsupported),
              StrEq("kLiteRtStatusErrorUnsupported"));
  EXPECT_THAT(LiteRtGetStatusString(kLiteRtStatusErrorNotFound),
              StrEq("kLiteRtStatusErrorNotFound"));
  EXPECT_THAT(LiteRtGetStatusString(kLiteRtStatusErrorTimeoutExpired),
              StrEq("kLiteRtStatusErrorTimeoutExpired"));
  EXPECT_THAT(LiteRtGetStatusString(kLiteRtStatusErrorFileIO),
              StrEq("kLiteRtStatusErrorFileIO"));
  EXPECT_THAT(LiteRtGetStatusString(kLiteRtStatusErrorInvalidFlatbuffer),
              StrEq("kLiteRtStatusErrorInvalidFlatbuffer"));
  EXPECT_THAT(LiteRtGetStatusString(kLiteRtStatusErrorDynamicLoading),
              StrEq("kLiteRtStatusErrorDynamicLoading"));
  EXPECT_THAT(LiteRtGetStatusString(kLiteRtStatusErrorSerialization),
              StrEq("kLiteRtStatusErrorSerialization"));
  EXPECT_THAT(LiteRtGetStatusString(kLiteRtStatusErrorCompilation),
              StrEq("kLiteRtStatusErrorCompilation"));
  EXPECT_THAT(LiteRtGetStatusString(kLiteRtStatusErrorIndexOOB),
              StrEq("kLiteRtStatusErrorIndexOOB"));
  EXPECT_THAT(LiteRtGetStatusString(kLiteRtStatusErrorInvalidIrType),
              StrEq("kLiteRtStatusErrorInvalidIrType"));
  EXPECT_THAT(LiteRtGetStatusString(kLiteRtStatusErrorInvalidGraphInvariant),
              StrEq("kLiteRtStatusErrorInvalidGraphInvariant"));
  EXPECT_THAT(LiteRtGetStatusString(kLiteRtStatusErrorGraphModification),
              StrEq("kLiteRtStatusErrorGraphModification"));
  EXPECT_THAT(LiteRtGetStatusString(kLiteRtStatusErrorInvalidToolConfig),
              StrEq("kLiteRtStatusErrorInvalidToolConfig"));
  EXPECT_THAT(LiteRtGetStatusString(kLiteRtStatusLegalizeNoMatch),
              StrEq("kLiteRtStatusLegalizeNoMatch"));
  EXPECT_THAT(LiteRtGetStatusString(kLiteRtStatusErrorInvalidLegalization),
              StrEq("kLiteRtStatusErrorInvalidLegalization"));
  EXPECT_THAT(LiteRtGetStatusString(kLiteRtStatusErrorWrongVersion),
              StrEq("kLiteRtStatusErrorWrongVersion"));
}

}  // namespace
