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

#include "tensorflow/core/util/tensor_slice_reader_cache.h"

#include <utility>

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

namespace checkpoint {

TensorSliceReaderCacheWrapper::TensorSliceReaderCacheWrapper() = default;
TensorSliceReaderCacheWrapper::~TensorSliceReaderCacheWrapper() {
  delete cache_;
  cache_ = nullptr;
}

const TensorSliceReader* TensorSliceReaderCacheWrapper::GetReader(
    const string& filepattern,
    TensorSliceReader::OpenTableFunction open_function,
    int preferred_shard) const {
  mutex_lock l(mu_);
  if (!cache_) {
    cache_ = new TensorSliceReaderCache;
  }
  return cache_->GetReader(filepattern, std::move(open_function),
                           preferred_shard);
}

TensorSliceReaderCache::TensorSliceReaderCache() = default;

TensorSliceReaderCache::~TensorSliceReaderCache() {
  for (const auto& pair : readers_) {
    delete pair.second.second;
  }
}

const TensorSliceReader* TensorSliceReaderCache::GetReader(
    const string& filepattern,
    TensorSliceReader::OpenTableFunction open_function, int preferred_shard) {
#if defined(__GXX_RTTI) || defined(_CPPRTTI)
  TensorSliceReaderCache::OpenFuncType* func_ptr =
      open_function.target<TensorSliceReaderCache::OpenFuncType>();
#else
  TensorSliceReaderCache::OpenFuncType* func_ptr = nullptr;
#endif

  if (!func_ptr) {
    LOG(WARNING) << "Caching disabled because the open function is a lambda or "
                    "RTTI is not enabled in this build.";
    return nullptr;
  }

  TensorSliceReader* reader = nullptr;

  // --- First critical section ---
  {
    mutex_lock l(mu_);

    // Wait if another thread is already trying to open the same files.
    while (still_opening_.find(filepattern) != still_opening_.end()) {
      cv_.wait(l);
    }

    // Check if cached
    auto it = readers_.find(filepattern);
    if (it != readers_.end()) {
      auto cached_val = it->second;
      if (cached_val.first == *func_ptr) {
        VLOG(1) << "Using cached TensorSliceReader for " << filepattern << ": "
                << cached_val.second;
        return cached_val.second;
      } else {
        LOG(WARNING) << "Caching disabled because the checkpoint file "
                     << "is being opened with two different open functions: "
                     << filepattern;
        return nullptr;
      }
    }

    // Mark as being opened
    still_opening_.insert(filepattern);
  }  // lock released here safely (RAII)

  // --- Create reader outside the lock ---
  TensorSliceReader* tmp_reader =
      new TensorSliceReader(filepattern, open_function, preferred_shard);

  // --- Second critical section ---
  {
    mutex_lock l(mu_);
    if (tmp_reader->status().ok()) {
      reader = tmp_reader;
      readers_[filepattern] = std::make_pair(*func_ptr, reader);
      VLOG(1) << "Cached TensorSliceReader for " << filepattern << ": "
              << reader;
    } else {
      delete tmp_reader;
    }
    CHECK_EQ(size_t{1}, still_opening_.erase(filepattern));
    cv_.notify_all();
  }  // lock released automatically

  return reader;
}

}  // namespace checkpoint

}  // namespace tensorflow
