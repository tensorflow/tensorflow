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

#ifndef TENSORFLOW_FRAMEWORK_RESOURCE_MGR_H_
#define TENSORFLOW_FRAMEWORK_RESOURCE_MGR_H_

#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

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
//   my_var.val = Tensor(DT_FLOAT, my_shape);
//   my_var.val.flat<float>().setZeros();   // 0 initialized.
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
  virtual string DebugString() = 0;

  // Returns memory used by this resource.
  virtual int64 MemoryUsed() const { return 0; };
};

// Container used for per-step resources.
class ScopedStepContainer {
 public:
  // step_id: the unique ID of this step. Doesn't have to be sequential, just
  // has to be unique.
  // cleanup: callback to delete a container of this name.
  ScopedStepContainer(const int64 step_id,
                      std::function<void(const string&)> cleanup)
      : name_(strings::StrCat("__per_step_", step_id)), cleanup_(cleanup) {}
  ~ScopedStepContainer() { cleanup_(name_); }

  const string& name() const { return name_; }

 private:
  const string name_;
  const std::function<void(const string&)> cleanup_;
};

class ResourceMgr {
 public:
  ResourceMgr();
  explicit ResourceMgr(const string& default_container);
  ~ResourceMgr();

  // Returns the default container name for *this.
  const string& default_container() const { return default_container_; }

  // Creates a resource "name" in the "container".  The caller transfers
  // the ownership of one ref on "resource" to *this
  //
  // REQUIRES: std::is_base_of<ResourceBase, T>
  // REQUIRES: resource != nullptr.
  template <typename T>
  Status Create(const string& container, const string& name,
                T* resource) TF_MUST_USE_RESULT;

  // If "container" has a resource "name", returns it in "*resource" and
  // the caller takes the ownership of one ref on "*resource".
  //
  // REQUIRES: std::is_base_of<ResourceBase, T>
  // REQUIRES: resource != nullptr
  template <typename T>
  Status Lookup(const string& container, const string& name,
                T** resource) const TF_MUST_USE_RESULT;

  // If "container" has a resource "name", returns it in
  // "*resource". Otherwise, invokes creator() to create the resource.
  // The caller takes the ownership of one ref on "*resource".
  //
  // REQUIRES: std::is_base_of<ResourceBase, T>
  // REQUIRES: resource != nullptr
  template <typename T>
  Status LookupOrCreate(const string& container, const string& name,
                        T** resource,
                        std::function<Status(T**)> creator) TF_MUST_USE_RESULT;

  // Deletes the resource "name" from the "container".
  //
  // REQUIRES: std::is_base_of<ResourceBase, T>
  template <typename T>
  Status Delete(const string& container, const string& name) TF_MUST_USE_RESULT;

  // Deletes the resource pointed by "handle".
  Status Delete(const ResourceHandle& handle) TF_MUST_USE_RESULT;

  // Deletes all resources from the "container" and removes the container.
  Status Cleanup(const string& container) TF_MUST_USE_RESULT;

  // Deletes all resources in all containers.
  void Clear();

  // Returns a text description for all resources.
  string DebugString() const;

 private:
  typedef std::pair<uint64, string> Key;
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
  typedef std::unordered_map<Key, ResourceBase*, KeyHash, KeyEqual> Container;

  const string default_container_;
  mutable mutex mu_;
  std::unordered_map<string, Container*> containers_ GUARDED_BY(mu_);

  Status DoCreate(const string& container, TypeIndex type, const string& name,
                  ResourceBase* resource) TF_MUST_USE_RESULT;
  Status DoLookup(const string& container, TypeIndex type, const string& name,
                  ResourceBase** resource) const TF_MUST_USE_RESULT;
  Status DoDelete(const string& container, uint64 type_hash_code,
                  const string& resource_name,
                  const string& type_name) TF_MUST_USE_RESULT;
  Status DoDelete(const string& container, TypeIndex type,
                  const string& resource_name) TF_MUST_USE_RESULT;

  // Inserts the type name for 'hash_code' into the hash_code to type name map.
  Status InsertDebugTypeName(uint64 hash_code, const string& type_name)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) TF_MUST_USE_RESULT;

  // Returns the type name for the 'hash_code'.
  // Returns "<unknown>" if a resource with such a type was never inserted into
  // the container.
  const char* DebugTypeName(uint64 hash_code) const
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Map from type hash_code to type name.
  std::unordered_map<uint64, string> debug_type_names_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(ResourceMgr);
};

// Makes a resource handle with the specified type for a given container /
// name.
ResourceHandle MakeResourceHandle(OpKernelContext* ctx, const string& container,
                                  const string& name,
                                  const TypeIndex& type_index);

template <typename T>
ResourceHandle MakeResourceHandle(OpKernelContext* ctx, const string& container,
                                  const string& name) {
  return MakeResourceHandle(ctx, container, name, MakeTypeIndex<T>());
}

Status MakeResourceHandleToOutput(OpKernelContext* context, int output_index,
                                  const string& container, const string& name,
                                  const TypeIndex& type_index);

template <typename T>
ResourceHandle MakePerStepResourceHandle(OpKernelContext* ctx,
                                         const string& name);

// Returns a resource handle from a numbered op input.
ResourceHandle HandleFromInput(OpKernelContext* ctx, int input);
Status HandleFromInput(OpKernelContext* ctx, StringPiece input,
                       ResourceHandle* handle);

// Create a resource pointed by a given resource handle.
template <typename T>
Status CreateResource(OpKernelContext* ctx, const ResourceHandle& p, T* value);

// Looks up a resource pointed by a given resource handle.
template <typename T>
Status LookupResource(OpKernelContext* ctx, const ResourceHandle& p, T** value);

// Looks up or creates a resource.
template <typename T>
Status LookupOrCreateResource(OpKernelContext* ctx, const ResourceHandle& p,
                              T** value, std::function<Status(T**)> creator);

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
  const string& container() const { return container_; }
  const string& name() const { return name_; }
  bool resource_is_private_to_kernel() const {
    return resource_is_private_to_kernel_;
  }

  // Returns a readable string for *this.
  string DebugString() const;

 private:
  ResourceMgr* rmgr_ = nullptr;
  string container_;
  string name_;
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
Status GetResourceFromContext(OpKernelContext* ctx, const string& input_name,
                              T** resource);

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
#define REGISTER_RESOURCE_HANDLE_OP(Type)                   \
  REGISTER_OP(#Type "HandleOp")                             \
      .Attr("container: string = ''")                       \
      .Attr("shared_name: string = ''")                     \
      .Output("resource: resource")                         \
      .SetIsStateful()                                      \
      .SetShapeFn(tensorflow::shape_inference::ScalarShape) \
      .Doc("Creates a handle to a " #Type)

// Utility op kernel to produce a handle to a resource of type T.
template <typename T>
class ResourceHandleOp : public OpKernel {
 public:
  explicit ResourceHandleOp(OpKernelConstruction* context);

  void Compute(OpKernelContext* ctx) override;

 private:
  string container_;
  string name_;
};

// Registers a kernel for an op which produces a handle to a resource of the
// specified type.
#define REGISTER_RESOURCE_HANDLE_KERNEL(Type)                        \
  REGISTER_KERNEL_BUILDER(Name(#Type "HandleOp").Device(DEVICE_CPU), \
                          ResourceHandleOp<Type>)

// Implementation details below.

template <typename T>
void CheckDeriveFromResourceBase() {
  static_assert(std::is_base_of<ResourceBase, T>::value,
                "T must derive from ResourceBase");
}

template <typename T>
Status ResourceMgr::Create(const string& container, const string& name,
                           T* resource) {
  CheckDeriveFromResourceBase<T>();
  CHECK(resource != nullptr);
  return DoCreate(container, MakeTypeIndex<T>(), name, resource);
}

template <typename T>
Status ResourceMgr::Lookup(const string& container, const string& name,
                           T** resource) const {
  CheckDeriveFromResourceBase<T>();
  ResourceBase* found = nullptr;
  Status s = DoLookup(container, MakeTypeIndex<T>(), name, &found);
  if (s.ok()) {
    // It's safe to down cast 'found' to T* since
    // typeid(T).hash_code() is part of the map key.
    *resource = static_cast<T*>(found);
  }
  return s;
}

template <typename T>
Status ResourceMgr::LookupOrCreate(const string& container, const string& name,
                                   T** resource,
                                   std::function<Status(T**)> creator) {
  Status s;
  *resource = nullptr;
  while (*resource == nullptr) {
    s = Lookup(container, name, resource);
    if (s.ok()) break;
    s = creator(resource);
    if (!s.ok()) break;
    s = Create(container, name, *resource);
    if (s.ok()) {
      (*resource)->Ref();
      break;
    }
    // Rare event. Concurrent racy creation. Redo the lookup.
    *resource = nullptr;
  }
  return s;
}

template <typename T>
Status ResourceMgr::Delete(const string& container, const string& name) {
  CheckDeriveFromResourceBase<T>();
  return DoDelete(container, MakeTypeIndex<T>(), name);
}

template <typename T>
Status GetResourceFromContext(OpKernelContext* ctx, const string& input_name,
                              T** resource) {
  DataType dtype;
  TF_RETURN_IF_ERROR(ctx->input_dtype(input_name, &dtype));
  if (dtype == DT_RESOURCE) {
    const Tensor* handle;
    TF_RETURN_IF_ERROR(ctx->input(input_name, &handle));
    return LookupResource(ctx, handle->scalar<ResourceHandle>()(), resource);
  }
  string container;
  string shared_name;
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
    container = tensor.flat<string>()(0);
    shared_name = tensor.flat<string>()(1);
  }
  return ctx->resource_manager()->Lookup(container, shared_name, resource);
}

template <typename T>
ResourceHandle MakePerStepResourceHandle(OpKernelContext* ctx,
                                         const string& name) {
  return MakeResourceHandle<T>(ctx, ctx->step_container()->name(), name);
}

namespace internal {

Status ValidateDevice(OpKernelContext* ctx, const ResourceHandle& p);

template <typename T>
Status ValidateDeviceAndType(OpKernelContext* ctx, const ResourceHandle& p) {
  TF_RETURN_IF_ERROR(internal::ValidateDevice(ctx, p));
  auto type_index = MakeTypeIndex<T>();
  if (type_index.hash_code() != p.hash_code()) {
    return errors::InvalidArgument(
        "Trying to access resource using the wrong type. Expected ",
        p.maybe_type_name(), " got ", type_index.name());
  }
  return Status::OK();
}

}  // namespace internal

template <typename T>
Status CreateResource(OpKernelContext* ctx, const ResourceHandle& p, T* value) {
  TF_RETURN_IF_ERROR(internal::ValidateDeviceAndType<T>(ctx, p));
  return ctx->resource_manager()->Create(p.container(), p.name(), value);
}

template <typename T>
Status LookupResource(OpKernelContext* ctx, const ResourceHandle& p,
                      T** value) {
  TF_RETURN_IF_ERROR(internal::ValidateDeviceAndType<T>(ctx, p));
  return ctx->resource_manager()->Lookup(p.container(), p.name(), value);
}

template <typename T>
Status LookupOrCreateResource(OpKernelContext* ctx, const ResourceHandle& p,
                              T** value, std::function<Status(T**)> creator) {
  TF_RETURN_IF_ERROR(internal::ValidateDeviceAndType<T>(ctx, p));
  return ctx->resource_manager()->LookupOrCreate(p.container(), p.name(), value,
                                                 creator);
}

template <typename T>
Status DeleteResource(OpKernelContext* ctx, const ResourceHandle& p) {
  TF_RETURN_IF_ERROR(internal::ValidateDeviceAndType<T>(ctx, p));
  return ctx->resource_manager()->Delete<T>(p.container(), p.name());
}

Status DeleteResource(OpKernelContext* ctx, const ResourceHandle& p);

template <typename T>
void IsResourceInitialized<T>::Compute(OpKernelContext* ctx) {
  Tensor* output;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &output));
  T* unused;
  output->flat<bool>()(0) =
      LookupResource(ctx, HandleFromInput(ctx, 0), &unused).ok();
}

template <typename T>
ResourceHandleOp<T>::ResourceHandleOp(OpKernelConstruction* context)
    : OpKernel(context) {
  OP_REQUIRES_OK(context, context->GetAttr("container", &container_));
  OP_REQUIRES_OK(context, context->GetAttr("shared_name", &name_));
}

template <typename T>
void ResourceHandleOp<T>::Compute(OpKernelContext* ctx) {
  Tensor* output = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
  output->scalar<ResourceHandle>()() =
      MakeResourceHandle<T>(ctx, container_, name_);
}

}  //  end namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_RESOURCE_MGR_H_
