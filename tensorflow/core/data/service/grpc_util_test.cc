/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/grpc_util.h"

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace grpc_util {

TEST(GrpcUtil, WrapInvalidArgument) {
  grpc::Status s(grpc::StatusCode::INVALID_ARGUMENT, "test message");
  Status wrapped = WrapError("wrapping message", s);
  ASSERT_EQ(wrapped, errors::InvalidArgument("wrapping message: test message"));
}

TEST(GrpcUtil, WrapOk) {
  grpc::Status s;
  Status wrapped = WrapError("wrapping message", s);
  ASSERT_EQ(wrapped, errors::Internal("Expected a non-ok grpc status. Wrapping "
                                      "message: wrapping message"));
}

}  // namespace grpc_util
}  // namespace data
}  // namespace tensorflow
