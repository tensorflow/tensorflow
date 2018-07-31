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

#ifndef TENSORFLOW_CORE_UTIL_PROTO_DESCRIPTORS_H_
#define TENSORFLOW_CORE_UTIL_PROTO_DESCRIPTORS_H_

#include <memory>
#include <string>

#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
class Env;
class Status;

// Get a `DescriptorPool` object from the named `descriptor_source`.
// `descriptor_source` may be a path to a file accessible to TensorFlow, in
// which case it is parsed as a `FileDescriptorSet` and used to build the
// `DescriptorPool`.
//
// `owned_desc_pool` will be filled in with the same pointer as `desc_pool` if
// the caller should take ownership.
extern tensorflow::Status GetDescriptorPool(
    tensorflow::Env* env, string const& descriptor_source,
    tensorflow::protobuf::DescriptorPool const** desc_pool,
    std::unique_ptr<tensorflow::protobuf::DescriptorPool>* owned_desc_pool);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_PROTO_DESCRIPTORS_H_
