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

// The utility to read checkpoints for google brain tensor ops and v3
// checkpoints for dist_belief.

#ifndef TENSORFLOW_CORE_UTIL_TENSOR_SLICE_READER_CACHE_H_
#define TENSORFLOW_CORE_UTIL_TENSOR_SLICE_READER_CACHE_H_

#include <set>
#include <unordered_map>
#include <utility>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_slice_reader.h"

namespace tensorflow {

namespace checkpoint {

class TensorSliceReaderCache;

// Wrapper to a lazily allocated TensorSliceReaderCache.
class TensorSliceReaderCacheWrapper {
 public:
  TensorSliceReaderCacheWrapper();
  ~TensorSliceReaderCacheWrapper();

  // Same as TensorSliceReaderCache::GetReader().
  const TensorSliceReader* GetReader(
      const string& filepattern,
      TensorSliceReader::OpenTableFunction open_function,
      int preferred_shard) const;

 private:
  mutable mutex mu_;
  mutable TensorSliceReaderCache* cache_ = nullptr;
};

// A cache of TensorSliceReaders.
class TensorSliceReaderCache {
 public:
  TensorSliceReaderCache();
  ~TensorSliceReaderCache();

  // Returns the TensorSliceReader corresponding to 'filepattern' and the
  // open_function.  May return nullptr if we can not create a new
  // TensorSliceReader for the filepattern/open_function combination.
  const TensorSliceReader* GetReader(
      const string& filepattern,
      TensorSliceReader::OpenTableFunction open_function, int preferred_shard);

 private:
  // Need to use a regular function type in the key map as std::function does
  // not support ==.
  typedef absl::Status (*OpenFuncType)(const string&,
                                       TensorSliceReader::Table**);

  // Protects attributes below.
  mutex mu_;

  // Maps of opened readers.
  std::unordered_map<string, std::pair<OpenFuncType, TensorSliceReader*>>
      readers_;

  // Set of keys that a previous GetReader() call is still trying to populate.
  std::set<string> still_opening_;

  // Condition variable to notify when a reader has been created.
  condition_variable cv_;
};

}  // namespace checkpoint

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_TENSOR_SLICE_READER_CACHE_H_
