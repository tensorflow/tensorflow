/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/bigtable/kernels/bigtable_lib.h"

namespace tensorflow {

Status GrpcStatusToTfStatus(const ::grpc::Status& status) {
  if (status.ok()) {
    return Status::OK();
  }
  auto grpc_code = status.error_code();
  if (status.error_code() == ::grpc::StatusCode::ABORTED ||
      status.error_code() == ::grpc::StatusCode::UNAVAILABLE ||
      status.error_code() == ::grpc::StatusCode::OUT_OF_RANGE) {
    grpc_code = ::grpc::StatusCode::INTERNAL;
  }
  return Status(static_cast<::tensorflow::error::Code>(grpc_code),
                strings::StrCat("Error reading from Cloud Bigtable: ",
                                status.error_message()));
}

namespace {
::tensorflow::error::Code GcpErrorCodeToTfErrorCode(
    ::google::cloud::StatusCode code) {
  switch (code) {
    case ::google::cloud::StatusCode::kOk:
      return ::tensorflow::error::OK;
    case ::google::cloud::StatusCode::kCancelled:
      return ::tensorflow::error::CANCELLED;
    case ::google::cloud::StatusCode::kUnknown:
      return ::tensorflow::error::UNKNOWN;
    case ::google::cloud::StatusCode::kInvalidArgument:
      return ::tensorflow::error::INVALID_ARGUMENT;
    case ::google::cloud::StatusCode::kDeadlineExceeded:
      return ::tensorflow::error::DEADLINE_EXCEEDED;
    case ::google::cloud::StatusCode::kNotFound:
      return ::tensorflow::error::NOT_FOUND;
    case ::google::cloud::StatusCode::kAlreadyExists:
      return ::tensorflow::error::ALREADY_EXISTS;
    case ::google::cloud::StatusCode::kPermissionDenied:
      return ::tensorflow::error::PERMISSION_DENIED;
    case ::google::cloud::StatusCode::kUnauthenticated:
      return ::tensorflow::error::UNAUTHENTICATED;
    case ::google::cloud::StatusCode::kResourceExhausted:
      return ::tensorflow::error::RESOURCE_EXHAUSTED;
    case ::google::cloud::StatusCode::kFailedPrecondition:
      return ::tensorflow::error::FAILED_PRECONDITION;
    case ::google::cloud::StatusCode::kAborted:
      return ::tensorflow::error::ABORTED;
    case ::google::cloud::StatusCode::kOutOfRange:
      return ::tensorflow::error::OUT_OF_RANGE;
    case ::google::cloud::StatusCode::kUnimplemented:
      return ::tensorflow::error::UNIMPLEMENTED;
    case ::google::cloud::StatusCode::kInternal:
      return ::tensorflow::error::INTERNAL;
    case ::google::cloud::StatusCode::kUnavailable:
      return ::tensorflow::error::UNAVAILABLE;
    case ::google::cloud::StatusCode::kDataLoss:
      return ::tensorflow::error::DATA_LOSS;
  }
}
}  // namespace

Status GcpStatusToTfStatus(const ::google::cloud::Status& status) {
  if (status.ok()) {
    return Status::OK();
  }
  return Status(
      GcpErrorCodeToTfErrorCode(status.code()),
      strings::StrCat("Error reading from Cloud Bigtable: ", status.message()));
}

string RegexFromStringSet(const std::vector<string>& strs) {
  CHECK(!strs.empty()) << "The list of strings to turn into a regex was empty.";
  std::unordered_set<string> uniq(strs.begin(), strs.end());
  if (uniq.size() == 1) {
    return *uniq.begin();
  }
  return str_util::Join(uniq, "|");
}

}  // namespace tensorflow
