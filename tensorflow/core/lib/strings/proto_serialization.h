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
#ifndef TENSORFLOW_CORE_LIB_STRINGS_PROTO_SERIALIZATION_H_
#define TENSORFLOW_CORE_LIB_STRINGS_PROTO_SERIALIZATION_H_

#include "tsl/lib/strings/proto_serialization.h"

namespace tensorflow {
// NOLINTBEGIN(misc-unused-using-decls)
using ::tsl::AreSerializedProtosEqual;
using ::tsl::DeterministicProtoHash64;
using ::tsl::SerializeToBufferDeterministic;
using ::tsl::SerializeToStringDeterministic;
// NOLINTEND(misc-unused-using-decls)
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_STRINGS_PROTO_SERIALIZATION_H_
