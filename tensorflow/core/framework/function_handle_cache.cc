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
#include "tensorflow/core/framework/function_handle_cache.h"

#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace tensorflow {
namespace data {

FunctionHandleCache::FunctionHandleCache(FunctionLibraryRuntime* lib)
    : FunctionHandleCache(lib, std::numeric_limits<int64>::max()) {}

FunctionHandleCache::FunctionHandleCache(FunctionLibraryRuntime* lib,
                                         int64 capacity)
    : lib_(lib),
      state_handle_(
          strings::Printf("%lld", static_cast<long long>(random::New64()))),
      capacity_(capacity) {}

FunctionHandleCache::~FunctionHandleCache() {
  Status s = Clear();
  if (!s.ok()) {
    LOG(ERROR) << "Failed to clear function handle cache: " << s.ToString();
  }
}

Status FunctionHandleCache::Instantiate(
    const string& function_name, AttrSlice attrs,
    FunctionLibraryRuntime::InstantiateOptions options,
    FunctionLibraryRuntime::Handle* handle) {
  string key = Canonicalize(function_name, attrs, options);
  {
    mutex_lock l(mu_);
    if (Lookup(key, handle)) return Status::OK();
  }
  // We release the lock to avoid holding it across function instantiations. As
  // a consequence, multiple callers may execute the block below for the same
  // key. We account for this by calling `Lookup()` again to check if the key
  // already exists.
  options.state_handle = state_handle_;
  TF_RETURN_IF_ERROR(lib_->Instantiate(function_name, attrs, options, handle));

  mutex_lock l(mu_);
  FunctionLibraryRuntime::Handle tmp_handle;
  if (Lookup(key, &tmp_handle)) {
    TF_RETURN_IF_ERROR(lib_->ReleaseHandle(*handle));
    *handle = tmp_handle;
    return Status::OK();
  }
  // At this point we know that the key does not exist.
  if (lru_list_.size() >= capacity_) {
    string lru_key = lru_list_.back();
    Status s = lib_->ReleaseHandle(handles_[lru_key].handle);
    if (!s.ok()) {
      TF_RETURN_IF_ERROR(lib_->ReleaseHandle(*handle));
      return s;
    }
    handles_.erase(lru_key);
    lru_list_.pop_back();
  }
  lru_list_.push_front(key);
  handles_.emplace(std::make_pair(key, Entry{*handle, lru_list_.begin()}));
  return Status::OK();
}

bool FunctionHandleCache::Lookup(const string& key,
                                 FunctionLibraryRuntime::Handle* handle) {
  Entry* entry = gtl::FindOrNull(handles_, key);
  if (entry == nullptr) return false;
  lru_list_.splice(lru_list_.begin(), lru_list_, entry->lru_iterator);
  entry->lru_iterator = lru_list_.begin();
  *handle = entry->handle;
  return true;
}

Status FunctionHandleCache::Clear() {
  mutex_lock l(mu_);
  while (!lru_list_.empty()) {
    string lru_key = lru_list_.back();
    TF_RETURN_IF_ERROR(lib_->ReleaseHandle(handles_[lru_key].handle));
    handles_.erase(lru_key);
    lru_list_.pop_back();
  }
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow
