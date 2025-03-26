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

extern "C" {

const char* LiteRtGetStatusString(LiteRtStatus status) {
  switch (status) {
    // NOLINTNEXTLINE(preprocessor-macros)
#define LITERT_STATUS_STR_CASE(STATUS) \
  case STATUS:                         \
    return #STATUS;
    LITERT_STATUS_STR_CASE(kLiteRtStatusOk);
    LITERT_STATUS_STR_CASE(kLiteRtStatusErrorInvalidArgument);
    LITERT_STATUS_STR_CASE(kLiteRtStatusErrorMemoryAllocationFailure);
    LITERT_STATUS_STR_CASE(kLiteRtStatusErrorRuntimeFailure);
    LITERT_STATUS_STR_CASE(kLiteRtStatusErrorMissingInputTensor);
    LITERT_STATUS_STR_CASE(kLiteRtStatusErrorUnsupported);
    LITERT_STATUS_STR_CASE(kLiteRtStatusErrorNotFound);
    LITERT_STATUS_STR_CASE(kLiteRtStatusErrorTimeoutExpired);
    LITERT_STATUS_STR_CASE(kLiteRtStatusErrorFileIO);
    LITERT_STATUS_STR_CASE(kLiteRtStatusErrorInvalidFlatbuffer);
    LITERT_STATUS_STR_CASE(kLiteRtStatusErrorDynamicLoading);
    LITERT_STATUS_STR_CASE(kLiteRtStatusErrorSerialization);
    LITERT_STATUS_STR_CASE(kLiteRtStatusErrorCompilation);
    LITERT_STATUS_STR_CASE(kLiteRtStatusErrorIndexOOB);
    LITERT_STATUS_STR_CASE(kLiteRtStatusErrorInvalidIrType);
    LITERT_STATUS_STR_CASE(kLiteRtStatusErrorInvalidGraphInvariant);
    LITERT_STATUS_STR_CASE(kLiteRtStatusErrorGraphModification);
    LITERT_STATUS_STR_CASE(kLiteRtStatusErrorInvalidToolConfig);
    LITERT_STATUS_STR_CASE(kLiteRtStatusLegalizeNoMatch);
    LITERT_STATUS_STR_CASE(kLiteRtStatusErrorInvalidLegalization);
    LITERT_STATUS_STR_CASE(kLiteRtStatusErrorWrongVersion);
    LITERT_STATUS_STR_CASE(kLiteRtStatusErrorUnknown);
#undef LITERT_STATUS_STR_CASE
  }
}

}  // extern "C"
