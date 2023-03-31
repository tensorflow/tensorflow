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

#include "tensorflow/core/framework/resource_mgr.h"

#include <atomic>

#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/demangle.h"
#include "tensorflow/core/platform/stacktrace.h"

namespace tensorflow {

ResourceHandle MakeResourceHandle(
    const string& container, const string& name, const DeviceBase& device,
    const TypeIndex& type_index,
    const std::vector<DtypeAndPartialTensorShape>& dtypes_and_shapes,
    const absl::optional<ManagedStackTrace>& definition_stack_trace) {
  ResourceHandle result;
  result.set_device(device.name());
  result.set_container(container);
  result.set_definition_stack_trace(definition_stack_trace);
  if (name == ResourceHandle::ANONYMOUS_NAME) {
    result.set_name(
        strings::StrCat("_AnonymousVar", ResourceHandle::GenerateUniqueId()));
  } else {
    result.set_name(name);
  }
  result.set_hash_code(type_index.hash_code());
  result.set_maybe_type_name(type_index.name());
  result.set_dtypes_and_shapes(dtypes_and_shapes);
  return result;
}

Status MakeResourceHandleToOutput(OpKernelContext* context, int output_index,
                                  const string& container, const string& name,
                                  const TypeIndex& type_index) {
  Tensor* handle;
  TF_RETURN_IF_ERROR(
      context->allocate_output(output_index, TensorShape({}), &handle));
  handle->scalar<ResourceHandle>()() =
      MakeResourceHandle(container, name, *context->device(), type_index);
  return OkStatus();
}

namespace internal {

Status ValidateDevice(OpKernelContext* ctx, const ResourceHandle& p) {
  if (DeviceNameUtils::GetDeviceNameFromStreamDeviceName(
          ctx->device()->attributes().name()) !=
      DeviceNameUtils::GetDeviceNameFromStreamDeviceName(p.device())) {
    return errors::InvalidArgument(
        "Trying to access resource ", p.name(), " located in device ",
        p.device(), " from device ", ctx->device()->attributes().name());
  }
  return OkStatus();
}

}  // end namespace internal

Status ResourceMgr::InsertDebugTypeName(uint64 hash_code,
                                        const string& type_name) {
  auto iter = debug_type_names_.emplace(hash_code, type_name);
  if (iter.first->second != type_name) {
    return errors::AlreadyExists("Duplicate hash code found for type ",
                                 type_name);
  }
  return OkStatus();
}

const char* ResourceMgr::DebugTypeName(uint64 hash_code) const {
  auto type_name_iter = debug_type_names_.find(hash_code);
  if (type_name_iter == debug_type_names_.end()) {
    return "<unknown>";
  } else {
    return type_name_iter->second.c_str();
  }
}

ResourceMgr::ResourceAndName::ResourceAndName() : name(nullptr) {}

ResourceMgr::ResourceAndName::ResourceAndName(const string& name)
    : name(absl::make_unique<string>(name)) {}

core::RefCountPtr<ResourceBase> ResourceMgr::ResourceAndName::GetResource()
    const {
  if (absl::holds_alternative<core::RefCountPtr<ResourceBase>>(resource)) {
    ResourceBase* ptr =
        absl::get<core::RefCountPtr<ResourceBase>>(resource).get();
    ptr->Ref();
    return core::RefCountPtr<ResourceBase>(ptr);
  } else if (absl::holds_alternative<core::WeakPtr<ResourceBase>>(resource)) {
    return absl::get<core::WeakPtr<ResourceBase>>(resource).GetNewRef();
  } else {
    return nullptr;
  }
}

ResourceMgr::ResourceAndName::ResourceAndName(
    ResourceAndName&& other) noexcept {
  name = std::move(other.name);
  resource = std::move(other.resource);
}

ResourceMgr::ResourceAndName::~ResourceAndName() {}

ResourceMgr::ResourceAndName& ResourceMgr::ResourceAndName::operator=(
    ResourceAndName&& other) noexcept {
  name = std::move(other.name);
  resource = std::move(other.resource);
  return *this;
}

ResourceMgr::ResourceMgr() : default_container_("localhost") {}

ResourceMgr::ResourceMgr(const string& default_container)
    : default_container_(default_container) {}

ResourceMgr::~ResourceMgr() { Clear(); }

void ResourceMgr::Clear() {
  // We do the deallocation outside of the lock to avoid a potential deadlock
  // in case any of the destructors access the resource manager.
  absl::flat_hash_map<string, Container*> tmp_containers;
  {
    mutex_lock l(mu_);
    tmp_containers = std::move(containers_);
  }
  for (const auto& p : tmp_containers) {
    delete p.second;
  }
  tmp_containers.clear();
}

string ResourceMgr::DebugString() const {
  mutex_lock l(mu_);
  struct Line {
    const string* container;
    const string type;
    const string* resource;
    const string detail;
  };
  std::vector<Line> lines;
  for (const auto& p : containers_) {
    const string& container = p.first;
    for (const auto& q : *p.second) {
      const Key& key = q.first;
      const char* type = DebugTypeName(key.first);
      const core::RefCountPtr<ResourceBase> resource = q.second.GetResource();
      Line l{&container, port::Demangle(type), q.second.name.get(),
             resource ? resource->DebugString() : "<nullptr>"};
      lines.push_back(l);
    }
  }
  std::vector<string> text;
  text.reserve(lines.size());
  for (const Line& line : lines) {
    text.push_back(strings::Printf(
        "%-20s | %-40s | %-40s | %-s", line.container->c_str(),
        line.type.c_str(), line.resource->c_str(), line.detail.c_str()));
  }
  std::sort(text.begin(), text.end());
  return absl::StrJoin(text, "\n");
}

Status ResourceMgr::DoCreate(const string& container_name, TypeIndex type,
                             const string& name, ResourceBase* resource,
                             bool owns_resource) {
  Container* container = [&]() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    Container** ptr = &containers_[container_name];
    if (*ptr == nullptr) {
      *ptr = new Container;
    }
    return *ptr;
  }();

  // NOTE: Separating out the construction of the map key and value so that the
  // key can contain a StringPiece that borrows from the string in the value.
  ResourceAndName resource_and_name(name);

  StringPiece borrowed_name(*resource_and_name.name);

  if (owns_resource) {
    resource_and_name.resource = core::RefCountPtr<ResourceBase>(resource);
  } else {
    auto cleanup_fn = [this, container, type, borrowed_name]() {
      mutex_lock l(mu_);
      auto iter = container->find({type.hash_code(), borrowed_name});
      if (iter != container->end()) {
        container->erase(iter);
      }
    };
    resource_and_name.resource =
        core::WeakPtr<ResourceBase>(resource, cleanup_fn);
  }

  Container::value_type key_and_value(Key(type.hash_code(), borrowed_name),
                                      std::move(resource_and_name));

  auto st = container->insert(std::move(key_and_value));
  if (st.second) {
    TF_RETURN_IF_ERROR(InsertDebugTypeName(type.hash_code(), type.name()));
    return OkStatus();
  }
  return errors::AlreadyExists("Resource ", container_name, "/", name, "/",
                               type.name());
}

Status ResourceMgr::Lookup(const ResourceHandle& handle,
                           ResourceBase** resource) const {
  tf_shared_lock l(mu_);
  return DoLookup(handle.container(), handle.hash_code(),
                  /*type_name=*/"ResourceBase", handle.name(), resource);
}

Status ResourceMgr::DoLookup(const string& container, TypeIndex type,
                             const string& name,
                             ResourceBase** resource) const {
  return DoLookup(container, type.hash_code(), type.name(), name, resource);
}

Status ResourceMgr::DoLookup(const string& container, uint64 type_hash_code,
                             const string& type_name,
                             const string& resource_name,
                             ResourceBase** resource) const {
  const Container* b = gtl::FindPtrOrNull(containers_, container);
  if (b == nullptr) {
    return errors::NotFound("Container ", container,
                            " does not exist. (Could not find resource: ",
                            container, "/", resource_name, ")");
  }
  auto iter = b->find({type_hash_code, resource_name});
  if (iter == b->end()) {
    return errors::NotFound("Resource ", container, "/", resource_name, "/",
                            type_name, " does not exist.");
  }
  ResourceBase* ptr = iter->second.GetResource().release();
  if (ptr == nullptr) {
    return errors::NotFound("Resource ", container, "/", resource_name, "/",
                            type_name, " has been destroyed.");
  }
  *resource = ptr;
  return OkStatus();
}

Status ResourceMgr::PopResourceAndName(const string& container,
                                       uint64 type_hash_code,
                                       const string& resource_name,
                                       const string& type_name,
                                       ResourceAndName& resource_and_name) {
  mutex_lock l(mu_);
  Container* b = gtl::FindPtrOrNull(containers_, container);
  if (b == nullptr) {
    return errors::NotFound("Container ", container, " does not exist.");
  }
  auto iter = b->find({type_hash_code, resource_name});
  if (iter == b->end()) {
    return errors::NotFound("Resource ", container, "/", resource_name, "/",
                            type_name, " does not exist.");
  }
  std::swap(resource_and_name, iter->second);
  b->erase(iter);
  return OkStatus();
}

Status ResourceMgr::DoDelete(const string& container, uint64 type_hash_code,
                             const string& resource_name,
                             const string& type_name) {
  ResourceAndName resource_and_name;
  TF_RETURN_IF_ERROR(PopResourceAndName(
      container, type_hash_code, resource_name, type_name, resource_and_name));

  if (absl::holds_alternative<core::WeakPtr<ResourceBase>>(
          resource_and_name.resource)) {
    return errors::Internal(
        "Cannot delete an unowned Resource ", container, "/", resource_name,
        "/", type_name, " from ResourceMgr. ",
        "This indicates ref-counting ResourceHandle is exposed to weak "
        "ResourceHandle code paths.");
  }
  return OkStatus();
}

Status ResourceMgr::DoDelete(const string& container, TypeIndex type,
                             const string& resource_name) {
  return DoDelete(container, type.hash_code(), resource_name, type.name());
}

Status ResourceMgr::Delete(const ResourceHandle& handle) {
  return DoDelete(handle.container(), handle.hash_code(), handle.name(),
                  "<unknown>");
}

Status ResourceMgr::Cleanup(const string& container) {
  {
    tf_shared_lock l(mu_);
    if (!gtl::FindOrNull(containers_, container)) {
      // Nothing to cleanup.
      return OkStatus();
    }
  }
  Container* b = nullptr;
  {
    mutex_lock l(mu_);
    auto iter = containers_.find(container);
    if (iter == containers_.end()) {
      // Nothing to cleanup, it's OK (concurrent cleanup).
      return OkStatus();
    }
    b = iter->second;
    containers_.erase(iter);
  }
  CHECK(b != nullptr);
  delete b;
  return OkStatus();
}

static bool IsValidContainerName(StringPiece s) {
  using ::tensorflow::strings::Scanner;
  return Scanner(s)
      .One(Scanner::LETTER_DIGIT_DOT)
      .Any(Scanner::LETTER_DIGIT_DASH_DOT_SLASH)
      .Eos()
      .GetResult();
}

Status ContainerInfo::Init(ResourceMgr* rmgr, const NodeDef& ndef,
                           bool use_node_name_as_default) {
  CHECK(rmgr);
  rmgr_ = rmgr;
  string attr_container;
  TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "container", &attr_container));
  if (!attr_container.empty() && !IsValidContainerName(attr_container)) {
    return errors::InvalidArgument("container contains invalid characters: ",
                                   attr_container);
  }
  string attr_shared_name;
  TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "shared_name", &attr_shared_name));
  if (!attr_shared_name.empty() && (attr_shared_name[0] == '_')) {
    return errors::InvalidArgument("shared_name cannot start with '_':",
                                   attr_shared_name);
  }
  if (!attr_container.empty()) {
    container_ = attr_container;
  } else {
    container_ = rmgr_->default_container();
  }
  if (!attr_shared_name.empty()) {
    name_ = attr_shared_name;
  } else if (use_node_name_as_default) {
    name_ = ndef.name();
  } else {
    resource_is_private_to_kernel_ = true;
    static std::atomic<int64_t> counter(0);
    name_ = strings::StrCat("_", counter.fetch_add(1), "_", ndef.name());
  }
  return OkStatus();
}

string ContainerInfo::DebugString() const {
  return strings::StrCat("[", container(), ",", name(), ",",
                         resource_is_private_to_kernel() ? "private" : "public",
                         "]");
}

// TODO(b/228388547) users of this method should be migrated to the one below.
const ResourceHandle& HandleFromInput(OpKernelContext* ctx, int input) {
  return ctx->input(input).flat<ResourceHandle>()(0);
}

Status HandleFromInput(OpKernelContext* ctx, StringPiece input,
                       ResourceHandle* handle) {
  const Tensor* tensor;
  TF_RETURN_IF_ERROR(ctx->input(input, &tensor));
  if (tensor->NumElements() == 0) {
    return errors::InvalidArgument("Empty resouce handle");
  }
  *handle = tensor->flat<ResourceHandle>()(0);
  return OkStatus();
}

Status LookupResource(OpKernelContext* ctx, const ResourceHandle& p,
                      ResourceBase** value) {
  TF_RETURN_IF_ERROR(internal::ValidateDevice(ctx, p));
  if (p.IsRefCounting()) {
    TF_ASSIGN_OR_RETURN(*value, p.GetResource<ResourceBase>());
    (*value)->Ref();
    return OkStatus();
  }
  return ctx->resource_manager()->Lookup(p, value);
}

Status DeleteResource(OpKernelContext* ctx, const ResourceHandle& p) {
  TF_RETURN_IF_ERROR(internal::ValidateDevice(ctx, p));
  if (p.IsRefCounting()) {
    return OkStatus();
  }
  return ctx->resource_manager()->Delete(p);
}

}  //  end namespace tensorflow
