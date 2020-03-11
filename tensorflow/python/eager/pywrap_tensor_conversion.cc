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

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

auto* scalar_cache_hits = tensorflow::monitoring::Counter<0>::New(
    "/tensorflow/eager/python/scalar_cache_hits",
    "Number of times a scalar TFE_TensorHandle was retrieved from cache");
auto* scalar_cache_misses = tensorflow::monitoring::Counter<0>::New(
    "/tensorflow/eager/python/scalar_cache_misses",
    "Number of times a scalar TFE_TensorHandle was not available in cache");

TFE_TensorHandleCache* TFE_TensorHandleCache::Get() {
  // TODO(slebedev): link with Context (in context.py) instead of having
  // a static global?
  static auto* cache = new TFE_TensorHandleCache();
  return cache;
}

TFE_TensorHandle* TFE_TensorHandleCache::Lookup(
    PyObject* value, tensorflow::DataType dtype,
    absl::string_view device_name) const {
  CHECK_NOTNULL(value);
  const auto& it = cache.find(Key{PyObjectPtr{value}, dtype, device_name});
  if (it == cache.end()) {
    scalar_cache_misses->GetCell()->IncrementBy(1);
    return nullptr;
  }

  scalar_cache_hits->GetCell()->IncrementBy(1);
  auto* h = it->second;
  return new TFE_TensorHandle{
      std::unique_ptr<AbstractTensorHandleInterface>(h->handle->Copy())};
}

void TFE_TensorHandleCache::Insert(PyObject* value, tensorflow::DataType dtype,
                                   absl::string_view device_name,
                                   TFE_TensorHandle* h) {
  Py_INCREF(value);
  cache.emplace(
      Key{PyObjectPtr{value}, dtype, device_name},
      new TFE_TensorHandle{
          std::unique_ptr<AbstractTensorHandleInterface>(h->handle->Copy())});
}

void TFE_TensorHandleCache::Clear() {
  DecrefUnrefAll();
  cache.clear();
}

}  // namespace tensorflow
