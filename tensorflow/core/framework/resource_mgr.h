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

#ifndef TENSORFLOW_CORE_FRAMEWORK_RESOURCE_MGR_H_
#define TENSORFLOW_CORE_FRAMEWORK_RESOURCE_MGR_H_

#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

// Forward declaration to avoid introducing a dependency on headers in
// "tensorflow/core/graph/...".
class GraphDefBuilder;
class Node;

// A ResourceMgr instance keeps track of named and typed resources
// grouped into containers.
//
// Each resource must be represented as a sub-class of ResourceBase,
// which is reference counted explicitly.  Each named resource is
// registered with ResourceMgr under a named "container" name. At any
// time, there is at most one instance of a resource given the container
// name, the resource type and the resource name.
//
// All resources for a given container can be dropped by one call of
// Cleanup().
//
// E.g.,
//   struct MyVar : public ResourceBase {
//     mutex mu;
//     Tensor val;
//   }
//
//   ResourceMgr rm;
//
//   // Create a var.
//   MyVar* my_var = new MyVar;
//   my_var->val = Tensor(DT_FLOAT, my_shape);
//   my_var->val.flat<float>().setZeros();   // 0 initialized.
//   ctx->SetStatus(rm.Create("my_container", "my_name", my_var));
//
//   // += a variable.
//   MyVar* my_var = nullptr;
//   Status s = rm.Lookup("my_container", "my_name", &my_var);
//   if (s.ok()) {
//     my_var->val.flat<float>() += grad;
//   }
//   my_var->Unref();   // Or use ScopedUnref().
//   ctx->SetStatus(s);
class ResourceBase : public core::RefCounted {
 public:
  // Returns a debug string for *this.
  virtual std::string DebugString() const = 0;

  // Returns memory used by this resource.
  virtual int64 MemoryUsed() const { return 0; }

  // Writes a representation of this resource into `builder`, so that executing
  // `*out` will recreate this resource.
  virtual Status AsGraphDef(GraphDefBuilder* builder, Node** out) const {
    return errors::Unimplemented("AsGraphDef not implemented for resource ",
                                 DebugString());
  }
};

// Container used for per-step resources.
class ScopedStepContainer {
 public:
  // step_id: the unique ID of this step. Doesn't have to be sequential, just
  // has to be unique.
  // cleanup: callback to delete a container of this name.
  // prefix: optional string prefix to disambiguate step containers.
  ScopedStepContainer(const int64 step_id,
                      std::function<void(const string&)> cleanup)
      : step_id_(step_id),
        container_(strings::StrCat("__per_step_", step_id)),
        cleanup_(cleanup),
        dirty_(false) {}

  ScopedStepContainer(const int64 step_id,
                      std::function<void(const string&)> cleanup,
                      const std::string& prefix)
      : step_id_(step_id),
        container_(strings::StrCat("__", prefix, "_per_step_", step_id)),
        cleanup_(cleanup),
        dirty_(false) {}

  ~ScopedStepContainer() { CleanUp(); }

  void CleanUp() TF_NO_THREAD_SAFETY_ANALYSIS {
    // NOTE(mrry): Avoid acquiring the mutex in the case that the container is
    // clean.
    if (dirty_) {
      mutex_lock ml(mu_);
      cleanup_(container_);
      dirty_ = false;
    }
  }

  // Pass through functions for resource lookup and creation. We do this to
  // ensure that we can appropriately set the dirty_ bit in the
  // ScopedStepContainer if the name of the container is used to create
  // resources.

  // Pass through to MakeResourceHandle with the container name
  template <typename T>
  ResourceHandle MakeResourceHandle(
      const std::string& name, const DeviceBase& device) TF_MUST_USE_RESULT;
  // Pass through to ResourceMgr::Create with the container name
  template <typename T>
  Status Create(ResourceMgr* rm, const std::string& name,
                T* resource) TF_MUST_USE_RESULT;
  // Pass through to ResourceMgr::Delete with the container name
  template <typename T>
  Status Delete(ResourceMgr* rm, const std::string& name) TF_MUST_USE_RESULT;
  // Pass through to ResourceMgr::Lookup with the container name
  template <typename T>
  Status Lookup(ResourceMgr* rm, const std::string& name,
                T** resource) const TF_MUST_USE_RESULT;
  // Pass through to ResourceMgr::LookupOrCreate with the container name
  template <typename T>
  Status LookupOrCreate(ResourceMgr* rm, const std::string& name, T** resource,
                        std::function<Status(T**)> creator) TF_MUST_USE_RESULT;
  int64 StepId() const { return step_id_; }

 private:
  const int64 step_id_;
  const std::string container_;
  const std::function<void(const string&)> cleanup_;
  mutex mu_;
  mutable std::atomic<bool> dirty_ TF_GUARDED_BY(mu_);
};

class ResourceMgr {
 public:
  ResourceMgr();
  explicit ResourceMgr(const std::string& default_container);
  ~ResourceMgr();

  // Returns the default container name for *this.
  const std::string& default_container() const { return default_container_; }

  // Creates a resource "name" in the "container".  The caller transfers
  // the ownership of one ref on "resource" to *this, regardless of whether this
  // operation succeeds or fails.
  //
  // REQUIRES: std::is_base_of<ResourceBase, T>
  // REQUIRES: resource != nullptr.
  template <typename T>
  Status Create(const std::string& container, const std::string& name,
                T* resource) TF_MUST_USE_RESULT;

  // If "container" has a resource "name", returns it in "*resource" and
  // the caller takes the ownership of one ref on "*resource".
  //
  // REQUIRES: std::is_base_of<ResourceBase, T>
  // REQUIRES: resource != nullptr
  template <typename T, bool use_dynamic_cast = false>
  Status Lookup(const std::string& container, const std::string& name,
                T** resource) const TF_MUST_USE_RESULT;

  // If the resource manager has a resource matching "handle", returns it in
  // "*resource" and the caller takes the ownership of one ref on "*resource".
  //
  // REQUIRES: resource != nullptr
  Status Lookup(const ResourceHandle& handle,
                ResourceBase** resource) const TF_MUST_USE_RESULT;

  // Similar to Lookup, but looks up multiple resources at once, with only a
  // single lock acquisition.  If containers_and_names[i] is uninitialized
  // then this function does not modify resources[i].
  template <typename T, bool use_dynamic_cast = false>
  Status LookupMany(absl::Span<std::pair<const string*, const string*> const>
                        containers_and_names,
                    std::vector<std::unique_ptr<T, core::RefCountDeleter>>*
                        resources) const TF_MUST_USE_RESULT;

  // If "container" has a resource "name", returns it in
  // "*resource". Otherwise, invokes creator() to create the resource.
  // The caller takes the ownership of one ref on "*resource".
  //
  // WARNING: creator() must not call any methods on ResourceMgr during its
  // execution, because a non-reentrant lock is held during the creator() call
  // in order to guarantee atomicity of LookupOrCreate().
  //
  // REQUIRES: std::is_base_of<ResourceBase, T>
  // REQUIRES: resource != nullptr
  template <typename T, bool use_dynamic_cast = false>
  Status LookupOrCreate(const std::string& container, const std::string& name,
                        T** resource,
                        std::function<Status(T**)> creator) TF_MUST_USE_RESULT;

  // Deletes the resource "name" from the "container".
  //
  // REQUIRES: std::is_base_of<ResourceBase, T>
  template <typename T>
  Status Delete(const std::string& container,
                const std::string& name) TF_MUST_USE_RESULT;

  // Deletes the resource pointed by "handle".
  Status Delete(const ResourceHandle& handle) TF_MUST_USE_RESULT;

  // Deletes all resources from the "container" and removes the container.
  Status Cleanup(const std::string& container) TF_MUST_USE_RESULT;

  // Deletes all resources in all containers.
  void Clear();

  // Returns a text description for all resources.
  std::string DebugString() const;

 private:
  typedef std::pair<uint64, StringPiece> Key;
  struct KeyHash {
    std::size_t operator()(const Key& k) const {
      return Hash64(k.second.data(), k.second.size(), k.first);
    }
  };
  struct KeyEqual {
    bool operator()(const Key& x, const Key& y) const {
      return (x.second == y.second) && (x.first == y.first);
    }
  };
  struct ResourceAndName {
    core::RefCountPtr<ResourceBase> resource;
    std::unique_ptr<string> name;

    ResourceAndName();
    ResourceAndName(ResourceBase* resource, std::string name);
    ResourceAndName(ResourceAndName&& other) noexcept;
    ~ResourceAndName();

    ResourceAndName& operator=(ResourceAndName&&) noexcept;

   private:
    TF_DISALLOW_COPY_AND_ASSIGN(ResourceAndName);
  };
  typedef std::unordered_map<Key, ResourceAndName, KeyHash, KeyEqual> Container;

  const std::string default_container_;
  mutable mutex mu_;
  std::unordered_map<string, Container*> containers_ TF_GUARDED_BY(mu_);

  template <typename T, bool use_dynamic_cast = false>
  Status LookupInternal(const std::string& container, const std::string& name,
                        T** resource) const
      TF_SHARED_LOCKS_REQUIRED(mu_) TF_MUST_USE_RESULT;
  Status LookupInternal(const std::string& container, uint64 type_hash_code,
                        const std::string& name, ResourceBase** resource) const
      TF_SHARED_LOCKS_REQUIRED(mu_) TF_MUST_USE_RESULT;

  Status DoCreate(const std::string& container, TypeIndex type,
                  const std::string& name, ResourceBase* resource)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) TF_MUST_USE_RESULT;

  Status DoLookup(const std::string& container, TypeIndex type,
                  const std::string& name, ResourceBase** resource) const
      TF_SHARED_LOCKS_REQUIRED(mu_) TF_MUST_USE_RESULT;
  Status DoLookup(const std::string& container, uint64 type_hash_code,
                  const std::string& type_name,
                  const std::string& resource_name,
                  ResourceBase** resource) const
      TF_SHARED_LOCKS_REQUIRED(mu_) TF_MUST_USE_RESULT;

  Status DoDelete(const std::string& container, uint64 type_hash_code,
                  const std::string& resource_name,
                  const std::string& type_name) TF_MUST_USE_RESULT;
  Status DoDelete(const std::string& container, TypeIndex type,
                  const std::string& resource_name) TF_MUST_USE_RESULT;

  // Inserts the type name for 'hash_code' into the hash_code to type name map.
  Status InsertDebugTypeName(uint64 hash_code, const std::string& type_name)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) TF_MUST_USE_RESULT;

  // Returns the type name for the 'hash_code'.
  // Returns "<unknown>" if a resource with such a type was never inserted into
  // the container.
  const char* DebugTypeName(uint64 hash_code) const
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Map from type hash_code to type name.
  std::unordered_map<uint64, string> debug_type_names_ TF_GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(ResourceMgr);
};

// Makes a resource handle with the specified type for a given container /
// name.
ResourceHandle MakeResourceHandle(
    const std::string& container, const std::string& name,
    const DeviceBase& device, const TypeIndex& type_index,
    const std::vector<DtypeAndPartialTensorShape>& dtypes_and_shapes = {},
    const absl::optional<ManagedStackTrace>& definition_stack_trace = {})
    TF_MUST_USE_RESULT;

template <typename T>
ResourceHandle MakeResourceHandle(
    OpKernelContext* ctx, const std::string& container, const std::string& name,
    const std::vector<DtypeAndPartialTensorShape>& dtypes_and_shapes = {},
    const absl::optional<ManagedStackTrace>& definition_stack_trace = {}) {
  return MakeResourceHandle(container.empty()
                                ? ctx->resource_manager()->default_container()
                                : container,
                            name, *ctx->device(), TypeIndex::Make<T>(),
                            dtypes_and_shapes, definition_stack_trace);
}

template <typename T>
ResourceHandle MakeResourceHandle(
    OpKernelConstruction* ctx, const std::string& container,
    const std::string& name,
    const std::vector<DtypeAndPartialTensorShape>& dtypes_and_shapes = {},
    const absl::optional<ManagedStackTrace>& definition_stack_trace = {}) {
  return MakeResourceHandle(container.empty()
                                ? ctx->resource_manager()->default_container()
                                : container,
                            name, *ctx->device(), TypeIndex::Make<T>(),
                            dtypes_and_shapes, definition_stack_trace);
}

Status MakeResourceHandleToOutput(OpKernelContext* context, int output_index,
                                  const std::string& container,
                                  const std::string& name,
                                  const TypeIndex& type_index);

// Returns a resource handle from a numbered op input.
const ResourceHandle& HandleFromInput(OpKernelContext* ctx, int input);
Status HandleFromInput(OpKernelContext* ctx, StringPiece input,
                       ResourceHandle* handle);

// Create a resource pointed by a given resource handle.
//
// If successful, the caller transfers the ownership of one ref on `resource` to
// `ctx->resource_mgr()`.
template <typename T>
Status CreateResource(OpKernelContext* ctx, const ResourceHandle& p, T* value);

// Looks up a resource pointed by a given resource handle.
//
// If the lookup is successful, the caller takes the ownership of one ref on
// `*value`, and must call its `Unref()` method when it has finished using it.
template <typename T, bool use_dynamic_cast = false>
Status LookupResource(OpKernelContext* ctx, const ResourceHandle& p, T** value);

// Looks up a resource pointed by a given resource handle.
//
// Prefer usage of LookupResource taking `core::RefCountPtr` to avoid
// requiring the caller to explicitly call `Unref()`.
template <typename T>
Status LookupResource(OpKernelContext* ctx, const ResourceHandle& p,
                      core::RefCountPtr<T>* value);

// Looks up multiple resources pointed by a sequence of resource handles.  If
// p[i] is uninitialized then values[i] is unmodified.
template <typename T>
Status LookupResources(OpKernelContext* ctx, absl::Span<ResourceHandle const> p,
                       std::vector<core::RefCountPtr<T>>* values);

// Looks up or creates a resource.
//
// If successful, the caller takes the ownership of one ref on `*value`, and
// must call its `Unref()` method when it has finished using it. If the
// `creator` is invoked, its reference on the created resource is transferred
// to `ctx->resource_mgr()`.
//
// Prefer usage of LookupOrCreateResource taking `core::RefCountPtr` to avoid
// requiring the caller to explicitly call `Unref()`.
template <typename T>
Status LookupOrCreateResource(OpKernelContext* ctx, const ResourceHandle& p,
                              T** value, std::function<Status(T**)> creator);

// Looks up or creates a resource.
template <typename T>
Status LookupOrCreateResource(OpKernelContext* ctx, const ResourceHandle& p,
                              core::RefCountPtr<T>* value,
                              std::function<Status(T**)> creator);

// Destroys a resource pointed by a given resource handle.
template <typename T>
Status DeleteResource(OpKernelContext* ctx, const ResourceHandle& p);

// Same as above, but uses the hash code of the type directly.
// The type name information will be missing in the debug output when the
// resource is not present in the container.
Status DeleteResource(OpKernelContext* ctx, const ResourceHandle& p);

// Policy helper to decide which container/shared_name to use for a
// stateful kernel that accesses shared resource.
class ContainerInfo {
 public:
  // Analyze the node attribute of 'ndef' and decides the container and
  // resource name the kernel should use for accessing the shared
  // resource.
  //
  // 'ndef' is expected to have node attribute "container" and
  // "shared_name". Returns non-OK if they are not provided or they are
  // invalid.
  //
  // The policy is as following:
  // * If the attribute "container" is non-empty, it is used as is.
  //   Otherwise, uses the resource manager's default container.
  // * If the attribute "shared_name" is non-empty, it is used as is.
  //   Otherwise, if "use_node_name_as_default" is true, the kernel's
  //   node name is used as the resource name. Otherwise, a string
  //   unique to this process is used.
  Status Init(ResourceMgr* rmgr, const NodeDef& ndef,
              bool use_node_name_as_default);
  Status Init(ResourceMgr* rmgr, const NodeDef& ndef) {
    return Init(rmgr, ndef, false);
  }

  // The policy decides that the kernel should access the resource in
  // resource_manager(), the resource is in the container() and its
  // name is name().  If resource_is_private_to_kernel() is true, the
  // kernel should delete the resource when the kernel is deleted.
  ResourceMgr* resource_manager() const { return rmgr_; }
  const std::string& container() const { return container_; }
  const std::string& name() const { return name_; }
  bool resource_is_private_to_kernel() const {
    return resource_is_private_to_kernel_;
  }

  // Returns a readable string for *this.
  std::string DebugString() const;

 private:
  ResourceMgr* rmgr_ = nullptr;
  std::string container_;
  std::string name_;
  bool resource_is_private_to_kernel_ = false;
};

// Helper for kernels to obtain 'resource' from the
// ctx->resource_manager().
//
// "input_name" specifies the kernel's ref input which gives a string
// tensor with two elements, which specifies the container and
// resource name.
//
// Returns OK if the resource is found and transfers one ref of
// *resource to the caller. Otherwise, returns an error.
template <typename T>
Status GetResourceFromContext(OpKernelContext* ctx,
                              const std::string& input_name, T** resource);

// Utility op kernel to check if a handle to resource type T is initialized.
template <typename T>
class IsResourceInitialized : public OpKernel {
 public:
  explicit IsResourceInitialized(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override;
};

// Registers an op which produces just a resource handle to a resource of the
// specified type. The type will be a part of the generated op name.
// TODO(apassos): figure out how to get non-cpu-allocated tensors to work
// through constant folding so this doesn't have to be marked as stateful.
#define REGISTER_RESOURCE_HANDLE_OP(Type) \
  REGISTER_OP(#Type "HandleOp")           \
      .Attr("container: string = ''")     \
      .Attr("shared_name: string = ''")   \
      .Output("resource: resource")       \
      .SetIsStateful()                    \
      .SetShapeFn(tensorflow::shape_inference::ScalarShape)

// Utility op kernel to produce a handle to a resource of type T.
template <typename T>
class ResourceHandleOp : public OpKernel {
 public:
  explicit ResourceHandleOp(OpKernelConstruction* context);

  void Compute(OpKernelContext* ctx) override;

  bool IsExpensive() override { return false; }

 private:
  std::string container_;
  std::string name_;
  mutex mutex_;
  Tensor resource_;
  std::atomic<bool> initialized_{false};
};

// Utility op kernel to produce a handle to a resource of type T.
template <typename T>
class ResourceHandlesOp : public OpKernel {
 public:
  explicit ResourceHandlesOp(OpKernelConstruction* context);

  void Compute(OpKernelContext* ctx) override;

  bool IsExpensive() override { return false; }

 private:
  std::vector<string> containers_;
  std::vector<string> names_;
  mutex mutex_;
  std::vector<Tensor> resources_;
  std::atomic<bool> initialized_{false};
};

// Registers a kernel for an op which produces a handle to a resource of the
// specified type.
#define REGISTER_RESOURCE_HANDLE_KERNEL(Type)                        \
  REGISTER_KERNEL_BUILDER(Name(#Type "HandleOp").Device(DEVICE_CPU), \
                          ResourceHandleOp<Type>)

// This class is used to guarantee that an anonymous resource is deleted
// (irrespective of whether a resource deleter op is called explicitly or
// the execution encounters an error before the op runs).
//
// This is achieved by wrapping an instance of this class into a variant
// tensor which is passed as an input to a resource deleter op. If the
// execution encounters an error before the op runs, the tensor will be
// destroyed, essentially triggering the iterator deletion.
// NOTE: This is not a feature-complete implementation of the DT_VARIANT
// specification. In particular, we cannot serialize the `ResourceMgr`
// object, so the `Encode()` and `Decode()` methods are not implemented.
class ResourceDeleter {
 public:
  ResourceDeleter() : deleter_() {}

  ResourceDeleter(ResourceHandle handle, ResourceMgr* resource_manager)
      : deleter_(std::make_shared<Helper>(handle, resource_manager)) {}

  ResourceDeleter(ResourceDeleter&& rhs) : deleter_(std::move(rhs.deleter_)) {
    VLOG(3) << "ResourceDeleter move constructor called.";
  }

  ResourceDeleter(const ResourceDeleter& rhs) : deleter_(rhs.deleter_) {
    VLOG(3) << "ResourceDeleter copy constructor called.";
  }

  ResourceDeleter& operator=(const ResourceDeleter& rhs) = delete;

  ResourceDeleter& operator=(ResourceDeleter&& rhs) = default;

  virtual ~ResourceDeleter() {
    VLOG(3) << "ResourceDeleter destructor called.";
  }

  void Encode(VariantTensorData*) const {
    LOG(ERROR) << "The Encode() method is not implemented for ResourceDeleter "
                  "objects.";
  }

  bool Decode(const VariantTensorData&) {
    LOG(ERROR) << "The Decode() method is not implemented for ResourceDeleter "
                  "objects";
    return false;  // Not supported.
  }

 private:
  // Helper that performs reference counting for the parent class and deletes
  // the iterator resource when the refcount goes to zero.
  //
  // NOTE: The object is borrowing a pointer to the resource manager.
  // Consequently, the tensor containing this object should not escape the
  // function in which was created (so that it is guaranteed that the resource
  // manager will outlive it).
  struct Helper {
    Helper(ResourceHandle handle, ResourceMgr* resource_manager)
        : handle(handle), resource_manager(resource_manager) {}

    Helper(const Helper& rhs) = delete;
    Helper(Helper&& rhs) = delete;

    ~Helper() {
      VLOG(3) << "Deleting Resource: " << handle.DebugString();
      resource_manager->Delete(handle).IgnoreError();
    }

    ResourceHandle handle;
    ResourceMgr* resource_manager;  // not owned
  };

  std::shared_ptr<Helper> deleter_;
};

// Implementation details below.

template <typename T>
void CheckDeriveFromResourceBase() {
  static_assert(std::is_base_of<ResourceBase, T>::value,
                "T must derive from ResourceBase");
}

template <typename T>
Status ResourceMgr::Create(const std::string& container,
                           const std::string& name, T* resource) {
  CheckDeriveFromResourceBase<T>();
  CHECK(resource != nullptr);
  mutex_lock l(mu_);
  return DoCreate(container, TypeIndex::Make<T>(), name, resource);
}

template <typename T, bool use_dynamic_cast>
Status ResourceMgr::Lookup(const std::string& container,
                           const std::string& name, T** resource) const {
  CheckDeriveFromResourceBase<T>();
  tf_shared_lock l(mu_);
  return LookupInternal<T, use_dynamic_cast>(container, name, resource);
}

template <typename T, bool use_dynamic_cast>
Status ResourceMgr::LookupMany(
    absl::Span<std::pair<const string*, const string*> const>
        containers_and_names,
    std::vector<std::unique_ptr<T, core::RefCountDeleter>>* resources) const {
  CheckDeriveFromResourceBase<T>();
  tf_shared_lock l(mu_);
  resources->resize(containers_and_names.size());
  for (size_t i = 0; i < containers_and_names.size(); ++i) {
    T* resource;
    Status s = LookupInternal<T, use_dynamic_cast>(
        *containers_and_names[i].first, *containers_and_names[i].second,
        &resource);
    if (s.ok()) {
      (*resources)[i].reset(resource);
    }
  }
  return Status::OK();
}

// Simple wrapper to allow conditional dynamic / static casts.
template <typename T, bool use_dynamic_cast>
struct TypeCastFunctor {
  static T* Cast(ResourceBase* r) { return static_cast<T*>(r); }
};

template <typename T>
struct TypeCastFunctor<T, true> {
  static T* Cast(ResourceBase* r) { return dynamic_cast<T*>(r); }
};

template <typename T, bool use_dynamic_cast>
Status ResourceMgr::LookupInternal(const std::string& container,
                                   const std::string& name,
                                   T** resource) const {
  ResourceBase* found = nullptr;
  Status s = DoLookup(container, TypeIndex::Make<T>(), name, &found);
  if (s.ok()) {
    // It's safe to down cast 'found' to T* since
    // typeid(T).hash_code() is part of the map key.
    *resource = TypeCastFunctor<T, use_dynamic_cast>::Cast(found);
  }
  return s;
}

template <typename T, bool use_dynamic_cast>
Status ResourceMgr::LookupOrCreate(const std::string& container,
                                   const std::string& name, T** resource,
                                   std::function<Status(T**)> creator) {
  CheckDeriveFromResourceBase<T>();
  *resource = nullptr;
  Status s;
  {
    tf_shared_lock l(mu_);
    s = LookupInternal<T, use_dynamic_cast>(container, name, resource);
    if (s.ok()) return s;
  }
  mutex_lock l(mu_);
  s = LookupInternal<T, use_dynamic_cast>(container, name, resource);
  if (s.ok()) return s;
  TF_RETURN_IF_ERROR(creator(resource));
  s = DoCreate(container, TypeIndex::Make<T>(), name, *resource);
  if (!s.ok()) {
    return errors::Internal("LookupOrCreate failed unexpectedly");
  }
  (*resource)->Ref();
  return s;
}

template <typename T>
Status ResourceMgr::Delete(const std::string& container,
                           const std::string& name) {
  CheckDeriveFromResourceBase<T>();
  return DoDelete(container, TypeIndex::Make<T>(), name);
}

template <typename T>
Status GetResourceFromContext(OpKernelContext* ctx,
                              const std::string& input_name, T** resource) {
  DataType dtype;
  TF_RETURN_IF_ERROR(ctx->input_dtype(input_name, &dtype));
  if (dtype == DT_RESOURCE) {
    const Tensor* handle;
    TF_RETURN_IF_ERROR(ctx->input(input_name, &handle));
    return LookupResource(ctx, handle->scalar<ResourceHandle>()(), resource);
  }
  std::string container;
  std::string shared_name;
  {
    mutex* mu;
    TF_RETURN_IF_ERROR(ctx->input_ref_mutex(input_name, &mu));
    mutex_lock l(*mu);
    Tensor tensor;
    TF_RETURN_IF_ERROR(ctx->mutable_input(input_name, &tensor, true));
    if (tensor.NumElements() != 2) {
      return errors::InvalidArgument(
          "Resource handle must have 2 elements, but had shape: ",
          tensor.shape().DebugString());
    }
    container = tensor.flat<tstring>()(0);
    shared_name = tensor.flat<tstring>()(1);
  }
  return ctx->resource_manager()->Lookup(container, shared_name, resource);
}

namespace internal {

Status ValidateDevice(OpKernelContext* ctx, const ResourceHandle& p);

template <typename T>
Status ValidateDeviceAndType(OpKernelContext* ctx, const ResourceHandle& p) {
  TF_RETURN_IF_ERROR(internal::ValidateDevice(ctx, p));
  auto type_index = TypeIndex::Make<T>();
  if (type_index.hash_code() != p.hash_code()) {
    return errors::InvalidArgument(
        "Trying to access resource using the wrong type. Expected ",
        p.maybe_type_name(), " got ", type_index.name());
  }
  return Status::OK();
}

}  // namespace internal

// Creates the resource pointed at by "p". The caller transfers the ownership of
// one ref on "*value" to the resource manager in "ctx", regardless of whether
// this operation succeeds or fails.
template <typename T>
Status CreateResource(OpKernelContext* ctx, const ResourceHandle& p, T* value) {
  TF_RETURN_IF_ERROR(internal::ValidateDeviceAndType<T>(ctx, p));
  return ctx->resource_manager()->Create(p.container(), p.name(), value);
}

// If the resource manager in "ctx" has a resource matching "p", returns it in
// "*value" and the caller takes the ownership of one ref on "*value"
template <typename T, bool use_dynamic_cast>
Status LookupResource(OpKernelContext* ctx, const ResourceHandle& p,
                      T** value) {
  TF_RETURN_IF_ERROR(internal::ValidateDeviceAndType<T>(ctx, p));
  return ctx->resource_manager()->Lookup<T, use_dynamic_cast>(p.container(),
                                                              p.name(), value);
}

// If the resource manager in "ctx" has a resource matching "p", returns it in
// "*value" and the caller takes the ownership of one ref on "*value"
Status LookupResource(OpKernelContext* ctx, const ResourceHandle& p,
                      ResourceBase** value);

// If the resource manager in "ctx" has a resource matching "p", returns it in
// "*value".
template <typename T>
Status LookupResource(OpKernelContext* ctx, const ResourceHandle& p,
                      core::RefCountPtr<T>* value) {
  T* raw_ptr = nullptr;
  TF_RETURN_IF_ERROR(LookupResource<T, false>(ctx, p, &raw_ptr));
  value->reset(raw_ptr);

  return Status::OK();
}

// Similar to Lookup, but looks up multiple resources at once, with only a
// single lock acquisition.
template <typename T>
Status LookupResources(OpKernelContext* ctx,
                       absl::Span<ResourceHandle const* const> p,
                       std::vector<core::RefCountPtr<T>>* values) {
  std::vector<std::pair<const string*, const string*>> containers_and_names(
      p.size());
  for (size_t i = 0; i < p.size(); ++i) {
    TF_RETURN_IF_ERROR(internal::ValidateDeviceAndType<T>(ctx, *p[i]));
    containers_and_names[i] = {&p[i]->container(), &p[i]->name()};
  }
  return ctx->resource_manager()->LookupMany(containers_and_names, values);
}

// If the resource manager in "ctx" has a resource pointed at by "p", returns
// it in "*value". Otherwise, invokes creator() to create the resource.
// The caller takes the ownership of one ref on "*value".
//
// WARNING: creator() must not call any methods on the resource manager during
// its execution, because a non-reentrant lock is held during the creator() call
// in order to guarantee atomicity of LookupOrCreateResource().
template <typename T>
Status LookupOrCreateResource(OpKernelContext* ctx, const ResourceHandle& p,
                              T** value, std::function<Status(T**)> creator) {
  TF_RETURN_IF_ERROR(internal::ValidateDeviceAndType<T>(ctx, p));
  return ctx->resource_manager()->LookupOrCreate(p.container(), p.name(), value,
                                                 creator);
}

// If the resource manager in "ctx" has a resource pointed at by "p", returns
// it in "*value". Otherwise, invokes creator() to create the resource.
//
// WARNING: creator() must not call any methods on the resource manager during
// its execution, because a non-reentrant lock is held during the creator() call
// in order to guarantee atomicity of LookupOrCreateResource().
template <typename T>
Status LookupOrCreateResource(OpKernelContext* ctx, const ResourceHandle& p,
                              core::RefCountPtr<T>* value,
                              std::function<Status(T**)> creator) {
  T* raw_ptr = nullptr;
  TF_RETURN_IF_ERROR(LookupOrCreateResource<T>(ctx, p, &raw_ptr, creator));
  value->reset(raw_ptr);

  return Status::OK();
}

// Deletes the resource pointed by "p", using the resource manager in "ctx".
template <typename T>
Status DeleteResource(OpKernelContext* ctx, const ResourceHandle& p) {
  TF_RETURN_IF_ERROR(internal::ValidateDeviceAndType<T>(ctx, p));
  return ctx->resource_manager()->Delete<T>(p.container(), p.name());
}

// Deletes the resource pointed by "p", using the resource manager in "ctx".
Status DeleteResource(OpKernelContext* ctx, const ResourceHandle& p);

template <typename T>
void IsResourceInitialized<T>::Compute(OpKernelContext* ctx) {
  Tensor* output;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &output));
  T* object;
  bool found;
  if (LookupResource(ctx, HandleFromInput(ctx, 0), &object).ok()) {
    found = true;
    object->Unref();
  } else {
    found = false;
  }

  output->flat<bool>()(0) = found;
}

template <typename T>
ResourceHandleOp<T>::ResourceHandleOp(OpKernelConstruction* context)
    : OpKernel(context) {
  OP_REQUIRES_OK(context, context->GetAttr("container", &container_));
  OP_REQUIRES_OK(context, context->GetAttr("shared_name", &name_));
}

template <typename T>
void ResourceHandleOp<T>::Compute(OpKernelContext* ctx) {
  if (name_ == ResourceHandle::ANONYMOUS_NAME) {
    AllocatorAttributes attr;
    attr.set_on_host(true);
    Tensor handle;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}), &handle, attr));
    handle.scalar<ResourceHandle>()() =
        MakeResourceHandle<T>(ctx, container_, name_);
    ctx->set_output(0, handle);
  } else {
    if (!initialized_.load()) {
      mutex_lock ml(mutex_);
      // Checking again to see if another thread has initialized the resource.
      if (!initialized_.load()) {
        AllocatorAttributes attr;
        attr.set_on_host(true);
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}),
                                               &resource_, attr));
        resource_.scalar<ResourceHandle>()() =
            MakeResourceHandle<T>(ctx, container_, name_);
        initialized_.store(true);
      }
    }
    ctx->set_output(0, resource_);
  }
}

template <typename T>
ResourceHandlesOp<T>::ResourceHandlesOp(OpKernelConstruction* context)
    : OpKernel(context) {
  int n;
  OP_REQUIRES_OK(context, context->GetAttr("N", &n));
  OP_REQUIRES_OK(context, context->GetAttr("containers", &containers_));
  OP_REQUIRES_OK(context, context->GetAttr("shared_names", &names_));
  OP_REQUIRES(
      context, containers_.size() == n,
      errors::InvalidArgument("Number of containers (", containers_.size(),
                              ") must be equal to N (", n, ")"));
  OP_REQUIRES(context, names_.size() == n,
              errors::InvalidArgument("Number of names (", containers_.size(),
                                      ") must be equal to N (", n, ")"));
  resources_.resize(n);
}

template <typename T>
void ResourceHandlesOp<T>::Compute(OpKernelContext* ctx) {
  if (!initialized_.load()) {
    mutex_lock ml(mutex_);
    // Checking again to see if another thread has initialized the resource.
    if (!initialized_.load()) {
      AllocatorAttributes attr;
      attr.set_on_host(true);
      for (size_t i = 0; i < resources_.size(); ++i) {
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}),
                                               &resources_[i], attr));
        ResourceHandle h =
            MakeResourceHandle<T>(ctx, containers_[i], names_[i]);
        resources_[i].template scalar<ResourceHandle>()() = h;
      }
      initialized_.store(true);
    }
  }
  for (size_t i = 0; i < resources_.size(); ++i) {
    ctx->set_output(i, resources_[i]);
  }
}

template <typename T>
ResourceHandle ScopedStepContainer::MakeResourceHandle(
    const std::string& name, const DeviceBase& device) {
  mutex_lock ml(mu_);
  dirty_ = true;
  return tensorflow::MakeResourceHandle(container_, name, device,
                                        TypeIndex::Make<T>(), {});
}

template <typename T>
Status ScopedStepContainer::Lookup(ResourceMgr* rm, const std::string& name,
                                   T** resource) const {
  return rm->Lookup<T>(container_, name, resource);
}

template <typename T>
Status ScopedStepContainer::LookupOrCreate(ResourceMgr* rm,
                                           const std::string& name,
                                           T** resource,
                                           std::function<Status(T**)> creator) {
  mutex_lock ml(mu_);
  dirty_ = true;
  return rm->LookupOrCreate<T>(container_, name, resource, creator);
}

template <typename T>
Status ScopedStepContainer::Create(ResourceMgr* rm, const std::string& name,
                                   T* resource) {
  mutex_lock ml(mu_);
  dirty_ = true;
  return rm->Create<T>(container_, name, resource);
}

template <typename T>
Status ScopedStepContainer::Delete(ResourceMgr* rm, const std::string& name) {
  return rm->Delete<T>(container_, name);
}

}  //  end namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_RESOURCE_MGR_H_
