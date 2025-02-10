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

#include "tensorflow/core/util/proto/descriptors.h"

#include "absl/strings/match.h"
#include "absl/strings/strip.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/proto/descriptor_pool_registry.h"

namespace tensorflow {
namespace {

absl::Status CreatePoolFromSet(
    const protobuf::FileDescriptorSet& set,
    std::unique_ptr<protobuf::DescriptorPool>* out_pool) {
  *out_pool = absl::make_unique<protobuf::DescriptorPool>();
  for (const auto& file : set.file()) {
    if ((*out_pool)->BuildFile(file) == nullptr) {
      return errors::InvalidArgument("Failed to load FileDescriptorProto: ",
                                     file.DebugString());
    }
  }
  return absl::OkStatus();
}

// Build a `DescriptorPool` from the named file or URI. The file or URI
// must be available to the current TensorFlow environment.
//
// The file must contain a serialized `FileDescriptorSet`. See
// `GetDescriptorPool()` for more information.
absl::Status GetDescriptorPoolFromFile(
    tensorflow::Env* env, const string& filename,
    std::unique_ptr<protobuf::DescriptorPool>* owned_desc_pool) {
  absl::Status st = env->FileExists(filename);
  if (!st.ok()) {
    return st;
  }
  // Read and parse the FileDescriptorSet.
  protobuf::FileDescriptorSet descs;
  std::unique_ptr<ReadOnlyMemoryRegion> buf;
  st = env->NewReadOnlyMemoryRegionFromFile(filename, &buf);
  if (!st.ok()) {
    return st;
  }
  if (!descs.ParseFromArray(buf->data(), buf->length())) {
    return errors::InvalidArgument(
        "descriptor_source contains invalid FileDescriptorSet: ", filename);
  }
  return CreatePoolFromSet(descs, owned_desc_pool);
}

absl::Status GetDescriptorPoolFromBinary(
    const string& source,
    std::unique_ptr<protobuf::DescriptorPool>* owned_desc_pool) {
  if (!absl::StartsWith(source, "bytes://")) {
    return errors::InvalidArgument(absl::StrCat(
        "Source does not represent serialized file descriptor set proto. ",
        "This may be due to a missing dependency on the file containing ",
        "REGISTER_DESCRIPTOR_POOL(\"", source, "\", ...);"));
  }
  // Parse the FileDescriptorSet.
  protobuf::FileDescriptorSet proto;
  if (!proto.ParseFromString(string(absl::StripPrefix(source, "bytes://")))) {
    return errors::InvalidArgument(absl::StrCat(
        "Source does not represent serialized file descriptor set proto. ",
        "This may be due to a missing dependency on the file containing ",
        "REGISTER_DESCRIPTOR_POOL(\"", source, "\", ...);"));
  }
  return CreatePoolFromSet(proto, owned_desc_pool);
}

}  // namespace

absl::Status GetDescriptorPool(
    Env* env, string const& descriptor_source,
    protobuf::DescriptorPool const** desc_pool,
    std::unique_ptr<protobuf::DescriptorPool>* owned_desc_pool) {
  // Attempt to lookup the pool in the registry.
  auto pool_fn = DescriptorPoolRegistry::Global()->Get(descriptor_source);
  if (pool_fn != nullptr) {
    return (*pool_fn)(desc_pool, owned_desc_pool);
  }

  // If there is no pool function registered for the given source, let the
  // runtime find the file or URL.
  absl::Status status =
      GetDescriptorPoolFromFile(env, descriptor_source, owned_desc_pool);
  if (status.ok()) {
    *desc_pool = owned_desc_pool->get();
    return absl::OkStatus();
  }

  status = GetDescriptorPoolFromBinary(descriptor_source, owned_desc_pool);
  *desc_pool = owned_desc_pool->get();
  return status;
}

}  // namespace tensorflow
