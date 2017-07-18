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
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

/// \brief An LRU block cache of file contents, keyed by {filename, offset}.
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
  typedef std::function<Status(const string&, size_t, size_t,
                               std::vector<char>*)>
      BlockFetcher;

  FileBlockCache(size_t block_size, size_t max_bytes, uint64 max_staleness,
                 BlockFetcher block_fetcher, Env* env = Env::Default())
      : block_size_(block_size),
        max_bytes_(max_bytes),
        max_staleness_(max_staleness),
        block_fetcher_(block_fetcher),
        env_(env) {
    if (max_staleness_ > 0) {
      pruning_thread_.reset(env_->StartThread(ThreadOptions(), "TF_prune_FBC",
                                              [this] { Prune(); }));
    }
  }

  ~FileBlockCache() {
    if (pruning_thread_) {
      stop_pruning_thread_.Notify();
      // Destroying pruning_thread_ will block until Prune() receives the above
      // notification and returns.
      pruning_thread_.reset();
    }
  }

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
  Status Read(const string& filename, size_t offset, size_t n,
              std::vector<char>* out);

  /// Remove all cached blocks for `filename`.
  void RemoveFile(const string& filename) LOCKS_EXCLUDED(mu_);

  /// Accessors for cache parameters.
  size_t block_size() const { return block_size_; }
  size_t max_bytes() const { return max_bytes_; }
  uint64 max_staleness() const { return max_staleness_; }

  /// The current size (in bytes) of the cache.
  size_t CacheSize() const LOCKS_EXCLUDED(mu_);

 private:
  /// The size of the blocks stored in the LRU cache, as well as the size of the
  /// reads from the underlying filesystem.
  const size_t block_size_;
  /// The maximum number of bytes (sum of block sizes) allowed in the LRU cache.
  const size_t max_bytes_;
  /// The maximum staleness of any block in the LRU cache, in seconds.
  const uint64 max_staleness_;
  /// The callback to read a block from the underlying filesystem.
  const BlockFetcher block_fetcher_;
  /// The Env from which we read timestamps.
  Env* const env_;  // not owned

  /// \brief The key type for the file block cache.
  ///
  /// The file block cache key is a {filename, offset} pair.
  typedef std::pair<string, size_t> Key;

  /// \brief A block of a file.
  ///
  /// A file block consists of the block data, the block's current position in
  /// the LRU cache, and the timestamp (seconds since epoch) at which the block
  /// was cached.
  struct Block {
    /// The block data.
    std::vector<char> data;
    /// A list iterator pointing to the block's position in the LRU list.
    std::list<Key>::iterator lru_iterator;
    /// A list iterator pointing to the block's position in the LRA list.
    std::list<Key>::iterator lra_iterator;
    /// The timestamp (seconds since epoch) at which the block was cached.
    uint64 timestamp;
  };

  /// \brief The block map type for the file block cache.
  ///
  /// The block map is an ordered map from Key to Block.
  typedef std::map<Key, std::unique_ptr<Block>> BlockMap;

  /// Prune the cache by removing files with expired blocks.
  void Prune() LOCKS_EXCLUDED(mu_);

  /// Remove all blocks of a file, with mu_ already held.
  void RemoveFile_Locked(const string& filename) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  /// Remove the block `entry` from the block map and LRU list, and update the
  /// cache size accordingly.
  void RemoveBlock(BlockMap::iterator entry) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  /// The cache pruning thread that removes files with expired blocks.
  std::unique_ptr<Thread> pruning_thread_;

  /// Notification for stopping the cache pruning thread.
  Notification stop_pruning_thread_;

  /// Guards access to the block map, LRU list, and cached byte count.
  mutable mutex mu_;

  /// The block map (map from Key to Block).
  BlockMap block_map_ GUARDED_BY(mu_);

  /// The LRU list of block keys. The front of the list identifies the most
  /// recently accessed block.
  std::list<Key> lru_list_ GUARDED_BY(mu_);

  /// The LRA (least recently added) list of block keys. The front of the list
  /// identifies the most recently added block.
  std::list<Key> lra_list_ GUARDED_BY(mu_);

  /// The combined number of bytes in all of the cached blocks.
  size_t cache_size_ GUARDED_BY(mu_) = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_FILE_BLOCK_CACHE_H_
