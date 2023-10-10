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

#ifndef TENSORFLOW_CORE_PLATFORM_PROTOBUF_H_
#define TENSORFLOW_CORE_PLATFORM_PROTOBUF_H_

#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/protobuf.h"

namespace tensorflow {
namespace protobuf = tsl::protobuf;  // NOLINT(misc-unused-alias-decls)
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::kProtobufInt64Typename;
using tsl::kProtobufUint64Typename;
using tsl::ParseFromTString;
using tsl::ParseProtoUnlimited;
using tsl::protobuf_int64;
using tsl::protobuf_uint64;
using tsl::ProtobufStringToString;
using tsl::SerializeToTString;
using tsl::SetProtobufStringSwapAllowed;
using tsl::TStringOutputStream;
// NOLINTEND(misc-unused-using-decls)
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_PROTOBUF_H_
