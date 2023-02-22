/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_ERRORS_H_
#define TENSORFLOW_CORE_PLATFORM_ERRORS_H_

#include <sstream>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/tsl/platform/errors.h"

namespace tensorflow {
namespace errors {

// NOLINTBEGIN(misc-unused-using-decls)
// Maps UNIX errors into a Status.
using error::OK;
using tsl::errors::Aborted;
using tsl::errors::AbortedWithPayloads;
using tsl::errors::AlreadyExists;
using tsl::errors::AlreadyExistsWithPayloads;
using tsl::errors::AppendToMessage;
using tsl::errors::Cancelled;
using tsl::errors::CancelledWithPayloads;
using tsl::errors::CopyPayloads;
using tsl::errors::Create;
using tsl::errors::CreateWithUpdatedMessage;
using tsl::errors::DataLoss;
using tsl::errors::DataLossWithPayloads;
using tsl::errors::DeadlineExceeded;
using tsl::errors::DeadlineExceededWithPayloads;
using tsl::errors::FailedPrecondition;
using tsl::errors::FailedPreconditionWithPayloads;
using tsl::errors::FormatColocationNodeForError;
using tsl::errors::FormatFunctionForError;
using tsl::errors::FormatNodeNameForError;
using tsl::errors::FormatNodeNamesForError;
using tsl::errors::FormatOriginalNodeLocationForError;
using tsl::errors::GetPayloads;
using tsl::errors::InsertPayloads;
using tsl::errors::Internal;
using tsl::errors::InternalWithPayloads;
using tsl::errors::InvalidArgument;
using tsl::errors::InvalidArgumentWithPayloads;
using tsl::errors::IOError;
using tsl::errors::IsAborted;
using tsl::errors::IsAlreadyExists;
using tsl::errors::IsCancelled;
using tsl::errors::IsDataLoss;
using tsl::errors::IsDeadlineExceeded;
using tsl::errors::IsFailedPrecondition;
using tsl::errors::IsInternal;
using tsl::errors::IsInvalidArgument;
using tsl::errors::IsNotFound;
using tsl::errors::IsOutOfRange;
using tsl::errors::IsPermissionDenied;
using tsl::errors::IsResourceExhausted;
using tsl::errors::IsUnauthenticated;
using tsl::errors::IsUnavailable;
using tsl::errors::IsUnimplemented;
using tsl::errors::IsUnknown;
using tsl::errors::NotFound;
using tsl::errors::NotFoundWithPayloads;
using tsl::errors::OutOfRange;
using tsl::errors::OutOfRangeWithPayloads;
using tsl::errors::PermissionDenied;
using tsl::errors::PermissionDeniedWithPayloads;
using tsl::errors::ReplaceErrorFromNonCommunicationOps;
using tsl::errors::ResourceExhausted;
using tsl::errors::ResourceExhaustedWithPayloads;
using tsl::errors::Unauthenticated;
using tsl::errors::UnauthenticatedWithPayloads;
using tsl::errors::Unavailable;
using tsl::errors::UnavailableWithPayloads;
using tsl::errors::Unimplemented;
using tsl::errors::UnimplementedWithPayloads;
using tsl::errors::Unknown;
using tsl::errors::UnknownPayloads;
namespace internal {
using tsl::errors::internal::PrepareForStrCat;
}
// NOLINTEND(misc-unused-using-decls)

}  // namespace errors
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_ERRORS_H_
