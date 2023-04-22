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

TensorSliceReaderCacheWrapper::TensorSliceReaderCacheWrapper() {}
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

TensorSliceReaderCache::TensorSliceReaderCache() {}

TensorSliceReaderCache::~TensorSliceReaderCache() {
  for (const auto& pair : readers_) {
    delete pair.second.second;
  }
}

const TensorSliceReader* TensorSliceReaderCache::GetReader(
    const string& filepattern,
    TensorSliceReader::OpenTableFunction open_function, int preferred_shard) {
  mutex_lock l(mu_);

#if defined(__GXX_RTTI) || defined(_CPPRTTI)
  // Get the function pointer from the open_function value.
  TensorSliceReaderCache::OpenFuncType* func_ptr =
      open_function.target<TensorSliceReaderCache::OpenFuncType>();
#else   // __GXX_RTTI
  // When RTTI is disabled, we will hard-code func_ptr to be zero,
  // since we cannot figure out the target type for open_function.
  // TODO(jiayq): find a more elegant way to possibly enable cache again.
  TensorSliceReaderCache::OpenFuncType* func_ptr = nullptr;
#endif  // _GXX_RTTI

  if (!func_ptr) {
    // We could not get the pointer, no caching is possible.
    LOG(WARNING) << "Caching disabled because the open function is a lambda or "
                    "RTTI is not enabled in this build.";
    return nullptr;
  }

  // Wait if another thread is already trying to open the same files.
  while (still_opening_.find(filepattern) != still_opening_.end()) {
    cv_.wait(l);
  }

  TensorSliceReader* reader = nullptr;
  if (readers_.find(filepattern) == readers_.end()) {
    VLOG(1) << "Creating new TensorSliceReader for " << filepattern;
    still_opening_.insert(filepattern);
    // Release the lock temporary as constructing TensorSliceReader is
    // expensive.
    mu_.unlock();
    TensorSliceReader* tmp_reader(
        new TensorSliceReader(filepattern, open_function, preferred_shard));
    // Acquire the lock again.
    mu_.lock();
    if (tmp_reader->status().ok()) {
      reader = tmp_reader;
      readers_[filepattern] = std::make_pair(*func_ptr, reader);
    } else {
      delete tmp_reader;
    }
    CHECK_EQ(size_t{1}, still_opening_.erase(filepattern));
    VLOG(1) << "Cached TensorSliceReader for " << filepattern << ": " << reader;
  } else {
    auto cached_val = readers_[filepattern];
    if (cached_val.first == *func_ptr) {
      reader = cached_val.second;
      VLOG(1) << "Using cached TensorSliceReader for " << filepattern << ": "
              << reader;
    } else {
      LOG(WARNING) << "Caching disabled because the checkpoint file "
                   << "is being opened with two different open functions: "
                   << filepattern;
    }
  }

  cv_.notify_all();
  return reader;
}

}  // namespace checkpoint

}  // namespace tensorflow
