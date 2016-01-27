/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_DEFAULT_PROTOBUF_H_
#define TENSORFLOW_CORE_PLATFORM_DEFAULT_PROTOBUF_H_

// IWYU pragma: private, include "third_party/tensorflow/core/platform/protobuf.h"
// IWYU pragma: friend third_party/tensorflow/core/platform/protobuf.h

#include "google/protobuf/descriptor.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/json_util.h"
#include "google/protobuf/util/type_resolver_util.h"

namespace tensorflow {
namespace protobuf = ::google::protobuf;
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_DEFAULT_PROTOBUF_H_
