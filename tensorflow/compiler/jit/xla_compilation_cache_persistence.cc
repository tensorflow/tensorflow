/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/xla_compilation_cache_persistence.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/path.h"

namespace tensorflow {

constexpr char kXlaSerializedCacheKeySeparator[] = "__";

namespace {
XlaCompilationCacheSaverCreatorFn* saver_creator_fn =
    new XlaCompilationCacheSaverCreatorFn();

XlaCompilationCacheLoaderCreatorFn* loader_creator_fn =
    new XlaCompilationCacheLoaderCreatorFn();

mutex creator_mutex(LINKER_INITIALIZED);
}  // namespace

std::unique_ptr<XlaCompilationCacheSaver> CreateXlaCompilationCacheSaver() {
  mutex_lock lock(creator_mutex);
  if (!*saver_creator_fn) {
    return nullptr;
  }
  return (*saver_creator_fn)();
}

std::unique_ptr<XlaCompilationCacheLoader> CreateXlaCompilationCacheLoader() {
  mutex_lock lock(creator_mutex);
  if (!*loader_creator_fn) {
    return nullptr;
  }
  return (*loader_creator_fn)();
}

Status RegisterXlaCompilationCacheSaver(
    XlaCompilationCacheSaverCreatorFn&& creator) {
  mutex_lock lock(creator_mutex);
  auto& fn = (*saver_creator_fn);
  if (fn) {
    return errors::Internal("A saver was already registered.");
  }
  fn = std::move(creator);
  return Status();
}

Status RegisterXlaCompilationCacheLoader(
    XlaCompilationCacheLoaderCreatorFn&& creator) {
  mutex_lock lock(creator_mutex);
  auto& fn = (*loader_creator_fn);
  if (fn) {
    return errors::Internal("A loader was already registered.");
  }
  fn = std::move(creator);
  return Status();
}

void UnregisterXlaCompilationCacheSaver() {
  mutex_lock lock(creator_mutex);
  *saver_creator_fn = {};
}

void UnregisterXlaCompilationCacheLoader() {
  mutex_lock lock(creator_mutex);
  *loader_creator_fn = {};
}

std::string XlaSerializedCacheKeyToString(const XlaSerializedCacheKey& key) {
  return absl::StrCat(key.prefix(), kXlaSerializedCacheKeySeparator,
                      key.signature_fingerprint(),
                      kXlaSerializedCacheKeySeparator,
                      key.cluster_fingerprint(),
                      kXlaSerializedCacheKeySeparator, key.device_type());
}

Status XlaCompilationCacheFileSaver::Save(
    const XlaSerializedCacheEntry& entry) {
  Env* env = Env::Default();
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(directory_));
  const std::string file_name =
      absl::StrCat(XlaSerializedCacheKeyToString(entry.key()), ".pb");
  const std::string file_path = io::JoinPath(directory_, file_name);
  if (!allow_overwrite_ && env->FileExists(file_path).ok()) {
    return errors::AlreadyExists("Can not overwrite existing file ", file_path);
  }
  switch (mode_) {
    case kBINARY:
      return WriteBinaryProto(env, file_path, entry);
    case kTEXT:
      return WriteTextProto(env, file_path, entry);
  }
}

StatusOr<absl::optional<XlaSerializedCacheEntry>>
XlaCompilationCacheFileLoader::TryLoad(const XlaSerializedCacheKey& key) {
  absl::optional<XlaSerializedCacheEntry> entry;
  const std::string file_name =
      absl::StrCat(XlaSerializedCacheKeyToString(key), ".pb");
  const std::string file_path = io::JoinPath(directory_, file_name);
  Env* env = Env::Default();
  if (env->FileExists(file_path).ok()) {
    TF_RETURN_IF_ERROR(ReadTextOrBinaryProto(env, file_path, &entry.emplace()));
    return entry;
  } else {
    return entry;
  }
}

}  // namespace tensorflow
