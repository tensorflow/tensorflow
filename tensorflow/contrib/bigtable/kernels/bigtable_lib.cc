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
  return Status(
      static_cast<::tensorflow::error::Code>(status.error_code()),
      strings::StrCat("Error reading from BigTable: ", status.error_message(),
                      " (Details: ", status.error_details(), ")"));
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
