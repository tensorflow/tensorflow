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

#ifndef TENSORFLOW_FRAMEWORK_RESOURCE_HANDLE_H_
#define TENSORFLOW_FRAMEWORK_RESOURCE_HANDLE_H_

#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class ResourceHandleProto;

// Class representing a handle to a tensorflow resource. Handles are
// not valid across executions, but can be serialized back and forth from within
// a single run.
//
// This is the native C++ class equivalent of ResourceHandleProto.  They are
// separate so that kernels do not need to depend on protos.
class ResourceHandle {
 public:
  ResourceHandle();
  ResourceHandle(const ResourceHandleProto& proto);
  ~ResourceHandle();

  // Unique name for the device containing the resource.
  const string& device() const { return device_; }
  void set_device(const string& device) { device_ = device; }

  // Container in which this resource is placed.
  const string& container() const { return container_; }
  void set_container(const string& container) { container_ = container; }

  // Unique name of this resource.
  const string& name() const { return name_; }
  void set_name(const string& name) { name_ = name; }

  // Hash code for the type of the resource. Is only valid in the same device
  // and in the same execution.
  uint64 hash_code() const { return hash_code_; }
  void set_hash_code(uint64 hash_code) { hash_code_ = hash_code; }

  // For debug-only, the name of the type pointed to by this handle, if
  // available.
  const string& maybe_type_name() const { return maybe_type_name_; }
  void set_maybe_type_name(const string& value) { maybe_type_name_ = value; }

  // Conversion to and from ResourceHandleProto
  void AsProto(ResourceHandleProto* proto) const;
  void FromProto(const ResourceHandleProto& proto);

  // Serialization via ResourceHandleProto
  string SerializeAsString() const;
  bool ParseFromString(const string& s);

  string DebugString() const;

  // GUID for anonymous resources. Resources with this shared_name will have
  // their shared_name replaced with a GUID at creation time
  static constexpr const char* ANONYMOUS_NAME =
      "cd2c89b7-88b7-44c8-ad83-06c2a9158347";

 public:
  string device_;
  string container_;
  string name_;
  uint64 hash_code_ = 0;
  string maybe_type_name_;
};

// For backwards compatibility for when this was a proto
string ProtoDebugString(const ResourceHandle& handle);

// Encodes a list of ResourceHandle protos in the given StringListEncoder.
void EncodeResourceHandleList(const ResourceHandle* p, int64 n,
                              std::unique_ptr<port::StringListEncoder> e);

// Decodes a list of ResourceHandle protos from the given StringListDecoder.
bool DecodeResourceHandleList(std::unique_ptr<port::StringListDecoder> d,
                              ResourceHandle* ps, int64 n);

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_RESOURCE_HANDLE_H_
