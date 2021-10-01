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

#include <string>

#include "tensorflow/core/framework/resource_base.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/intrusive_ptr.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/managed_stack_trace.h"

namespace tensorflow {

class ResourceHandleProto;

// Class representing a handle to a tensorflow resource. Handles are
// not valid across executions, but can be serialized back and forth from within
// a single run (except for those created from MakeRefCountingHandle i.e. whose
// resource_ field is not empty).
//
// This is the native C++ class equivalent of ResourceHandleProto.  They are
// separate so that kernels do not need to depend on protos.
class ResourceHandle {
 public:
  ResourceHandle();
  ResourceHandle(const ResourceHandleProto& proto);
  ~ResourceHandle();

  // Unique name for the device containing the resource.
  const std::string& device() const { return device_; }

  void set_device(const std::string& device) { device_ = device; }

  // Container in which this resource is placed.
  const std::string& container() const { return container_; }
  void set_container(const std::string& container) { container_ = container; }

  // Unique name of this resource.
  const std::string& name() const { return name_; }
  void set_name(const std::string& name) { name_ = name; }

  // Hash code for the type of the resource. Is only valid in the same device
  // and in the same execution.
  uint64 hash_code() const { return hash_code_; }
  void set_hash_code(uint64 hash_code) { hash_code_ = hash_code; }

  // For debug-only, the name of the type pointed to by this handle, if
  // available.
  const std::string& maybe_type_name() const { return maybe_type_name_; }
  void set_maybe_type_name(const std::string& value) {
    maybe_type_name_ = value;
  }

  // Data types and shapes for the underlying resource.
  std::vector<DtypeAndPartialTensorShape> dtypes_and_shapes() const {
    return dtypes_and_shapes_;
  }
  void set_dtypes_and_shapes(
      const std::vector<DtypeAndPartialTensorShape>& dtypes_and_shapes) {
    dtypes_and_shapes_ = dtypes_and_shapes;
  }

  void set_definition_stack_trace(
      const absl::optional<ManagedStackTrace>& definition_stack_trace) {
    definition_stack_trace_ = definition_stack_trace;
  }

  const absl::optional<ManagedStackTrace>& definition_stack_trace() const {
    return definition_stack_trace_;
  }

  // Conversion to and from ResourceHandleProto
  void AsProto(ResourceHandleProto* proto) const;
  void FromProto(const ResourceHandleProto& proto);

  // Serialization via ResourceHandleProto
  std::string SerializeAsString() const;
  bool ParseFromString(const std::string& s);

  std::string DebugString() const;

  std::string SummarizeValue() const { return "Resource Tensor"; }

  // GUID for anonymous resources. Resources with this shared_name will have
  // their shared_name replaced with a GUID at creation time
  static constexpr const char* ANONYMOUS_NAME =
      "cd2c89b7-88b7-44c8-ad83-06c2a9158347";

  // Creates a `ResourceHandle` that holds a pointer to a resource and takes
  // ownership of it. Normally a `ResourceHandle` only contains the name (and
  // some other metadata) of the resource. When created via this function,
  // the handle will own the resource, in the sense that it will destroy the
  // resource automatically when the resource is no longer needed. It does this
  // via automatic ref-counting on the resource: when the handle is copied, it
  // will call `Ref` on the resource (remember that all resources inherit from
  // `ResourceBase` which inherits from `RefCounted`), and when the handle is
  // destroyed, it will call `Unref` on the resource. When the last handle goes
  // out of scope, the resource's ref-count will go down to zero and the
  // resource will be destroyed. When calling this function, the `resource`
  // argument should have a ref-count of one (which is the case when the
  // resource is newly created).
  //
  // For those familiar with `ResourceMgr`, when you create a handle by the
  // `MakeResourceHandle` function in resource_mgr.h, the handle doesn't hold a
  // strong reference to the resource, and the resource is owned by the
  // resource manager whose strong reference must be manually deleted by
  // calling `ResourceMgr::Delete`. In contrast, a handle created by this
  // function holds a strong reference to the resource. The resource manager
  // does not hold a strong reference to the resource.
  template <typename T>
  static ResourceHandle MakeRefCountingHandle(
      T* resource, const string& device_name,
      const std::vector<DtypeAndPartialTensorShape>& dtypes_and_shapes = {},
      const absl::optional<ManagedStackTrace>& definition_stack_trace = {}) {
    return MakeRefCountingHandle(resource, device_name, TypeIndex::Make<T>(),
                                 dtypes_and_shapes, definition_stack_trace);
  }

  static ResourceHandle MakeRefCountingHandle(
      ResourceBase* resource, const string& device_name,
      const TypeIndex& type_index,
      const std::vector<DtypeAndPartialTensorShape>& dtypes_and_shapes = {},
      const absl::optional<ManagedStackTrace>& definition_stack_trace = {});

  // Pointer to the resource.
  const core::IntrusivePtr<ResourceBase>& resource() const { return resource_; }

  // Gets the resource pointer in `handle` as `T*`, or an error if the actual
  // resource type is not `T`.
  template <typename T>
  StatusOr<T*> GetResource() const {
    TF_RETURN_IF_ERROR(ValidateType<T>());
    return down_cast<T*>(resource_.get());
  }

  // Returns True if the resource handle is ref-counting.
  // See MakeRefCountingHandle.
  bool IsRefCounting() const { return resource_.get() != nullptr; }

  // Validates that the resource type in `handle` is `T`.
  template <typename T>
  Status ValidateType() const {
    return ValidateType(TypeIndex::Make<T>());
  }

  Status ValidateType(const TypeIndex& type_index) const;

  // Generates unique IDs (e.g. for names of anonymous variables)
  static int64_t GenerateUniqueId();

 private:
  std::string device_;
  std::string container_;
  std::string name_;
  uint64 hash_code_ = 0;
  std::string maybe_type_name_;
  std::vector<DtypeAndPartialTensorShape> dtypes_and_shapes_;
  absl::optional<ManagedStackTrace> definition_stack_trace_;
  // A smart pointer to the actual resource. When this field is not empty, the
  // handle is in a "ref-counting" mode, owning the resource; otherwise it's in
  // a "weak-ref" mode, only containing the name of the resource (conceptually a
  // weak reference).
  core::IntrusivePtr<ResourceBase> resource_;
  static std::atomic<int64_t> current_id_;
};

// For backwards compatibility for when this was a proto
std::string ProtoDebugString(const ResourceHandle& handle);

// Encodes a list of ResourceHandle protos in the given StringListEncoder.
void EncodeResourceHandleList(const ResourceHandle* p, int64_t n,
                              std::unique_ptr<port::StringListEncoder> e);

// Decodes a list of ResourceHandle protos from the given StringListDecoder.
bool DecodeResourceHandleList(std::unique_ptr<port::StringListDecoder> d,
                              ResourceHandle* ps, int64_t n);

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_RESOURCE_HANDLE_H_
