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

#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

namespace litert {

ErrorStatusBuilder::operator absl::Status() const noexcept {
  switch (error_.Status()) {
    case kLiteRtStatusOk:
      return absl::OkStatus();
    case kLiteRtStatusErrorInvalidArgument:
      return absl::InvalidArgumentError(error_.Message());
    case kLiteRtStatusErrorMemoryAllocationFailure:
      return absl::ResourceExhaustedError(error_.Message());
    case kLiteRtStatusErrorRuntimeFailure:
      return absl::InternalError(error_.Message());
    case kLiteRtStatusErrorMissingInputTensor:
      return absl::InvalidArgumentError(error_.Message());
    case kLiteRtStatusErrorUnsupported:
      return absl::UnimplementedError(error_.Message());
    case kLiteRtStatusErrorNotFound:
      return absl::NotFoundError(error_.Message());
    case kLiteRtStatusErrorTimeoutExpired:
      return absl::DeadlineExceededError(error_.Message());
    case kLiteRtStatusErrorWrongVersion:
      return absl::FailedPreconditionError(error_.Message());
    case kLiteRtStatusErrorUnknown:
      return absl::UnknownError(error_.Message());
    case kLiteRtStatusErrorFileIO:
      return absl::UnavailableError(error_.Message());
    case kLiteRtStatusErrorInvalidFlatbuffer:
      return absl::InvalidArgumentError(error_.Message());
    case kLiteRtStatusErrorDynamicLoading:
      return absl::UnavailableError(error_.Message());
    case kLiteRtStatusErrorSerialization:
      return absl::InternalError(error_.Message());
    case kLiteRtStatusErrorCompilation:
      return absl::InternalError(error_.Message());
    case kLiteRtStatusErrorIndexOOB:
      return absl::OutOfRangeError(error_.Message());
    case kLiteRtStatusErrorInvalidIrType:
      return absl::InvalidArgumentError(error_.Message());
    case kLiteRtStatusErrorInvalidGraphInvariant:
      return absl::InvalidArgumentError(error_.Message());
    case kLiteRtStatusErrorGraphModification:
      return absl::InternalError(error_.Message());
    case kLiteRtStatusErrorInvalidToolConfig:
      return absl::InvalidArgumentError(error_.Message());
    case kLiteRtStatusLegalizeNoMatch:
      return absl::NotFoundError(error_.Message());
    case kLiteRtStatusErrorInvalidLegalization:
      return absl::InvalidArgumentError(error_.Message());
    default:
      return absl::UnknownError(error_.Message());
  }
}

}  // namespace litert
