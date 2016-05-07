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

#ifndef TENSORFLOW_FRAMEWORK_RESOURCE_MGR_H_
#define TENSORFLOW_FRAMEWORK_RESOURCE_MGR_H_

#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
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
//   my_val.val.flat<float>().setZeros();   // 0 initialized.
//   ctx->SetStatus(rm.Create("my_container", "my_name", my_val));
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

  // Deletes all resources from the "container" and removes the container.
  Status Cleanup(const string& container) TF_MUST_USE_RESULT;

  // Deletes all resources in all containers.
  void Clear();

  // Returns a text description for all resources.
  string DebugString() const;

 private:
  typedef std::pair<TypeIndex, string> Key;
  struct KeyHash {
    std::size_t operator()(const Key& k) const {
      return Hash64(k.second.data(), k.second.size(), k.first.hash_code());
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
  Status DoDelete(const string& container, TypeIndex type,
                  const string& name) TF_MUST_USE_RESULT;

  TF_DISALLOW_COPY_AND_ASSIGN(ResourceMgr);
};

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

}  //  end namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_RESOURCE_MGR_H_
