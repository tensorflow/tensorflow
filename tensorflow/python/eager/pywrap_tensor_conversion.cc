/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/python/eager/pywrap_tensor_conversion.h"

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/core/lib/monitoring/counter.h"

namespace tensorflow {

namespace {

static monitoring::Counter<0>* const kScalarCacheHits =
    monitoring::Counter<0>::New(
        "/tensorflow/eager/python/scalar_cache_hits",
        "Number of times a scalar TFE_TensorHandle was retrieved from cache");
static monitoring::Counter<0>* const kScalarCacheMisses =
    monitoring::Counter<0>::New(
        "/tensorflow/eager/python/scalar_cache_misses",
        "Number of times a scalar TFE_TensorHandle was not available in cache");

}  // namespace

TFE_TensorHandleCache* TFE_TensorHandleCache::Get() {
  // TODO: b/169790439 - link with Context (in context.py) instead of having
  // a static global?
  static auto* cache = new TFE_TensorHandleCache();
  return cache;
}

TFE_TensorHandle* TFE_TensorHandleCache::Lookup(
    PyObject* value, DataType dtype, TFE_Context* ctx,
    absl::string_view device_name) const {
  CHECK(value != nullptr);  // Crash OK
#ifdef Py_GIL_DISABLED
  absl::MutexLock lock(mu_);
#endif  // Py_GIL_DISABLED
  const auto it =
      cache_.find(LookupKey{PyObjectPtr{value}, dtype, ctx, device_name});
  if (it == cache_.end()) {
    kScalarCacheMisses->GetCell()->IncrementBy(1);
    return nullptr;
  }

  kScalarCacheHits->GetCell()->IncrementBy(1);
  TFE_TensorHandle* h = it->second;
  unwrap(h)->Ref();
  return h;
}

void TFE_TensorHandleCache::Insert(PyObject* value, DataType dtype,
                                   TFE_Context* ctx,
                                   absl::string_view device_name,
                                   TFE_TensorHandle* h) {
  CHECK(value != nullptr);  // Crash OK
  CHECK(h != nullptr);      // Crash OK
#ifdef Py_GIL_DISABLED
  absl::MutexLock lock(mu_);
#endif  // Py_GIL_DISABLED
  auto [it, inserted] = cache_.try_emplace(
      LookupKey{PyObjectPtr{value}, dtype, ctx, device_name}, h);
  if (inserted) {
    Py_INCREF(value);
    unwrap(h)->Ref();
  }
}

void TFE_TensorHandleCache::Clear() { DecrefUnrefAll(); }

}  // namespace tensorflow
