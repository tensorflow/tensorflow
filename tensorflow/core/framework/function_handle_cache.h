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
#ifndef TENSORFLOW_CORE_FRAMEWORK_FUNCTION_HANDLE_CACHE_H_
#define TENSORFLOW_CORE_FRAMEWORK_FUNCTION_HANDLE_CACHE_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/function.h"

namespace tensorflow {
namespace data {

// Thread-safe data structure for caching function instantiations that uses LRU
// policy for replacement.
class FunctionHandleCache {
 public:
  explicit FunctionHandleCache(FunctionLibraryRuntime* lib);
  FunctionHandleCache(FunctionLibraryRuntime* lib, int64 capacity);

  ~FunctionHandleCache();

  // Looks up the function to be instantiated in the cache first. If present,
  // returns handle from there. Otherwise, instantiates a new function
  // and stores handle in the cache.
  Status Instantiate(const string& function_name, AttrSlice attrs,
                     FunctionLibraryRuntime::InstantiateOptions options,
                     FunctionLibraryRuntime::Handle* handle);

  // Releases all the handles in the cache, clearing out the state for all
  // functions involved.
  Status Clear();

 private:
  struct Entry {
    FunctionLibraryRuntime::Handle handle;
    std::list<string>::iterator lru_iterator;
  };

  // If the given key exists, returns true, updates the LRU state, and sets
  // `handle` to point to the cached handle. Otherwise, returns false and the
  // LRU state and `handle` are unchanged.
  bool Lookup(const string& key, FunctionLibraryRuntime::Handle* handle)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  mutex mu_;
  FunctionLibraryRuntime* lib_ = nullptr;  // not owned
  const string state_handle_;
  absl::flat_hash_map<string, Entry> handles_ TF_GUARDED_BY(mu_);
  std::list<string> lru_list_ TF_GUARDED_BY(mu_);
  const int64 capacity_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_FUNCTION_HANDLE_CACHE_H_
