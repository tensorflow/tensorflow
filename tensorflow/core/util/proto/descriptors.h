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
#include "tensorflow/core/platform/status.h"

namespace tsl {
class Env;
}  // namespace tsl
namespace tensorflow {
using tsl::Env;

// Gets a `DescriptorPool` object from the `descriptor_source`. This may be:
//
// 1) An empty string  or "local://", in which case the local descriptor pool
// created for proto definitions linked to the binary is returned.
//
// 2) A file path, in which case the descriptor pool is created from the
// contents of the file, which is expected to contain a `FileDescriptorSet`
// serialized as a string. The descriptor pool ownership is transferred to the
// caller via `owned_desc_pool`.
//
// 3) A "bytes://<bytes>", in which case the descriptor pool is created from
// `<bytes>`, which is expected to be a `FileDescriptorSet` serialized as a
// string. The descriptor pool ownership is transferred to the caller via
// `owned_desc_pool`.
//
// Custom schemas can be supported by registering a handler with the
// `DescriptorPoolRegistry`.
Status GetDescriptorPool(
    Env* env, string const& descriptor_source,
    protobuf::DescriptorPool const** desc_pool,
    std::unique_ptr<protobuf::DescriptorPool>* owned_desc_pool);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_PROTO_DESCRIPTORS_H_
