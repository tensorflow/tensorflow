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

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/demangle.h"

namespace tensorflow {

namespace internal {

Status ValidateDevice(OpKernelContext* ctx, const ResourceHandle& p) {
  if (ctx->device()->attributes().name() != p.device()) {
    return errors::InvalidArgument(
        "Trying to access resource located in device ", p.device(),
        " from device ", ctx->device()->attributes().name());
  }
  return Status::OK();
}

}  // end namespace internal

Status ResourceMgr::InsertDebugTypeName(uint64 hash_code,
                                        const string& type_name) {
  auto iter = debug_type_names_.emplace(hash_code, type_name);
  if (iter.first->second != type_name) {
    return errors::AlreadyExists("Duplicate hash code found for type ",
                                 type_name);
  }
  return Status::OK();
}

const char* ResourceMgr::DebugTypeName(uint64 hash_code) const {
  auto type_name_iter = debug_type_names_.find(hash_code);
  if (type_name_iter == debug_type_names_.end()) {
    return "<unknown>";
  } else {
    return type_name_iter->second.c_str();
  }
}

ResourceMgr::ResourceMgr() : default_container_("localhost") {}

ResourceMgr::ResourceMgr(const string& default_container)
    : default_container_(default_container) {}

ResourceMgr::~ResourceMgr() { Clear(); }

void ResourceMgr::Clear() {
  mutex_lock l(mu_);
  for (const auto& p : containers_) {
    for (const auto& q : *p.second) {
      q.second->Unref();
    }
    delete p.second;
  }
  containers_.clear();
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
      const string& resource = key.second;
      Line l{&container, port::Demangle(type), &resource,
             q.second->DebugString()};
      lines.push_back(l);
    }
  }
  std::vector<string> text;
  for (const Line& line : lines) {
    text.push_back(strings::Printf(
        "%-20s | %-40s | %-40s | %-s", line.container->c_str(),
        line.type.c_str(), line.resource->c_str(), line.detail.c_str()));
  }
  std::sort(text.begin(), text.end());
  return str_util::Join(text, "\n");
}

Status ResourceMgr::DoCreate(const string& container, TypeIndex type,
                             const string& name, ResourceBase* resource) {
  {
    mutex_lock l(mu_);
    Container** b = &containers_[container];
    if (*b == nullptr) {
      *b = new Container;
    }
    if ((*b)->insert({{type.hash_code(), name}, resource}).second) {
      TF_RETURN_IF_ERROR(InsertDebugTypeName(type.hash_code(), type.name()));
      return Status::OK();
    }
  }
  resource->Unref();
  return errors::AlreadyExists("Resource ", container, "/", name, "/",
                               type.name());
}

Status ResourceMgr::DoLookup(const string& container, TypeIndex type,
                             const string& name,
                             ResourceBase** resource) const {
  mutex_lock l(mu_);
  const Container* b = gtl::FindPtrOrNull(containers_, container);
  if (b == nullptr) {
    return errors::NotFound("Container ", container, " does not exist.");
  }
  auto r = gtl::FindPtrOrNull(*b, {type.hash_code(), name});
  if (r == nullptr) {
    return errors::NotFound("Resource ", container, "/", name, "/", type.name(),
                            " does not exist.");
  }
  *resource = const_cast<ResourceBase*>(r);
  (*resource)->Ref();
  return Status::OK();
}

Status ResourceMgr::DoDelete(const string& container, uint64 type_hash_code,
                             const string& resource_name,
                             const string& type_name) {
  ResourceBase* base = nullptr;
  {
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
    base = iter->second;
    b->erase(iter);
  }
  CHECK(base != nullptr);
  base->Unref();
  return Status::OK();
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
  Container* b = nullptr;
  {
    mutex_lock l(mu_);
    auto iter = containers_.find(container);
    if (iter == containers_.end()) {
      // Nothing to cleanup, it's OK.
      return Status::OK();
    }
    b = iter->second;
    containers_.erase(iter);
  }
  CHECK(b != nullptr);
  for (const auto& p : *b) {
    p.second->Unref();
  }
  delete b;
  return Status::OK();
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
    static std::atomic<int64> counter(0);
    name_ = strings::StrCat("_", counter.fetch_add(1), "_", ndef.name());
  }
  return Status::OK();
}

string ContainerInfo::DebugString() const {
  return strings::StrCat("[", container(), ",", name(), ",",
                         resource_is_private_to_kernel() ? "private" : "public",
                         "]");
}

ResourceHandle HandleFromInput(OpKernelContext* ctx, int input) {
  return ctx->input(input).flat<ResourceHandle>()(0);
}

Status DeleteResource(OpKernelContext* ctx, const ResourceHandle& p) {
  TF_RETURN_IF_ERROR(internal::ValidateDevice(ctx, p));
  return ctx->resource_manager()->Delete(p);
}

}  //  end namespace tensorflow
