/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_TSL_PLATFORM_STATUS_TO_FROM_PROTO_H_
#define TENSORFLOW_TSL_PLATFORM_STATUS_TO_FROM_PROTO_H_

#include "tsl/platform/status.h"
#include "tsl/protobuf/status.pb.h"

namespace tsl {

// TODO(b/250921378): Merge this file with `status.h` once we figure out how to
// fix the following error with the MacOS build:
//
// ImportError:
// dlopen(/org_tensorflow/tensorflow/python/platform/_pywrap_tf2.so, 2):
// Symbol not found: tensorflow11StatusProtoC1EPN6protobuf5ArenaEb

// Converts a `Status` to a `StatusProto`.
tensorflow::StatusProto StatusToProto(const Status& s);

#if defined(PLATFORM_GOOGLE)
// Constructs a `Status` from a `StatusProto`.
Status StatusFromProto(
    const tensorflow::StatusProto& proto,
    absl::SourceLocation loc = absl::SourceLocation::current());
#else
Status StatusFromProto(const tensorflow::StatusProto& proto);
#endif
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_STATUS_TO_FROM_PROTO_H_
