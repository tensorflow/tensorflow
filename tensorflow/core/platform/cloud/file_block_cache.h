/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_CLOUD_FILE_BLOCK_CACHE_H_
#define TENSORFLOW_CORE_PLATFORM_CLOUD_FILE_BLOCK_CACHE_H_

#include <functional>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class FileBlockCache;

/// FileBlockCacheStatsInterface allows for instrumentation of the block cache.
///
/// FileBlockCacheStatsInterface and its subclasses must be safe to use from
/// multiple threads concurrently.
///
/// WARNING! This is an experimental interface that may change or go away at any
/// time.
class FileBlockCacheStatsInterface {
 public:
  /// Configure is called to provide instrumentation hooks.
  ///
  /// Note: Configure can be called multiple times (e.g. if the block cache is
  /// re-initialized).
  virtual void Configure(const FileBlockCache* block_cache) = 0;

  /// RecordBlockLoadRequest is called to record the size of a hit block.
  virtual void RecordCacheHitBlockSize(size_t bytes_transferred) = 0;

  /// RecordBlockLoadRequest is called to record the size of a missed block.
  virtual void RecordCacheMissBlockSize(size_t bytes_transferred) = 0;

  virtual ~FileBlockCacheStatsInterface() = default;
};

/// \brief A block cache of file contents, keyed by {filename, offset}.
///
/// This class should be shared by read-only random access files on a remote
/// filesystem (e.g. GCS).
class FileBlockCache {
 public:
  /// The callback executed when a block is not found in the cache, and needs to
  /// be fetched from the backing filesystem. This callback is provided when the
  /// cache is constructed. The returned Status should be OK as long as the
  /// read from the remote filesystem succeeded (similar to the semantics of the
  /// read(2) system call).
  typedef std::function<Status(const string& filename, size_t offset,
                               size_t buffer_size, char* buffer,
                               size_t* bytes_transferred)>
      BlockFetcher;

  virtual ~FileBlockCache() {}

  /// Read `n` bytes from `filename` starting at `offset` into `out`. This
  /// method will return:
  ///
  /// 1) The error from the remote filesystem, if the read from the remote
  ///    filesystem failed.
  /// 2) PRECONDITION_FAILED if the read from the remote filesystem succeeded,
  ///    but the read returned a partial block, and the LRU cache contained a
  ///    block at a higher offset (indicating that the partial block should have
  ///    been a full block).
  /// 3) OUT_OF_RANGE if the read from the remote filesystem succeeded, but
  ///    the file contents do not extend past `offset` and thus nothing was
  ///    placed in `out`.
  /// 4) OK otherwise (i.e. the read succeeded, and at least one byte was placed
  ///    in `out`).
  virtual Status Read(const string& filename, size_t offset, size_t n,
                      char* buffer, size_t* bytes_transferred) = 0;

  // Validate the given file signature with the existing file signature in the
  // cache. Returns true if the signature doesn't change or the file did not
  // exist before. If the signature changes, update the existing signature with
  // the new one and remove the file from cache.
  virtual bool ValidateAndUpdateFileSignature(const string& filename,
                                              int64 file_signature) = 0;

  /// Remove all cached blocks for `filename`.
  virtual void RemoveFile(const string& filename) = 0;

  /// Remove all cached data.
  virtual void Flush() = 0;

  /// Accessors for cache parameters.
  virtual size_t block_size() const = 0;
  virtual size_t max_bytes() const = 0;
  virtual uint64 max_staleness() const = 0;

  /// The current size (in bytes) of the cache.
  virtual size_t CacheSize() const = 0;

  // Returns true if the cache is enabled. If false, the BlockFetcher callback
  // is always executed during Read.
  virtual bool IsCacheEnabled() const = 0;

  void SetStats(FileBlockCacheStatsInterface* stats) {
    if (stats == nullptr) {
      LOG(ERROR)
          << "Attempted to monitor a NULL stats object. This may prevent the "
             "corresponding monitoring data from being exported";
      return;
    }
    cache_stats_ = stats;
    cache_stats_->Configure(this);
  }

 protected:
  FileBlockCacheStatsInterface* cache_stats_ = nullptr;  // Not owned.
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_FILE_BLOCK_CACHE_H_
