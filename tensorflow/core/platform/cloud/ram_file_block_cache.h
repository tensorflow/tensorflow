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

#ifndef TENSORFLOW_CORE_PLATFORM_CLOUD_RAM_FILE_BLOCK_CACHE_H_
#define TENSORFLOW_CORE_PLATFORM_CLOUD_RAM_FILE_BLOCK_CACHE_H_

#include <functional>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/platform/cloud/file_block_cache.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

/// \brief An LRU block cache of file contents, keyed by {filename, offset}.
///
/// This class should be shared by read-only random access files on a remote
/// filesystem (e.g. GCS).
class RamFileBlockCache : public FileBlockCache {
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

  RamFileBlockCache(size_t block_size, size_t max_bytes, uint64 max_staleness,
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
    VLOG(1) << "GCS file block cache is "
            << (IsCacheEnabled() ? "enabled" : "disabled");
  }

  ~RamFileBlockCache() override {
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
  Status Read(const string& filename, size_t offset, size_t n, char* buffer,
              size_t* bytes_transferred) override;

  // Validate the given file signature with the existing file signature in the
  // cache. Returns true if the signature doesn't change or the file doesn't
  // exist before. If the signature changes, update the existing signature with
  // the new one and remove the file from cache.
  bool ValidateAndUpdateFileSignature(const string& filename,
                                      int64 file_signature) override
      TF_LOCKS_EXCLUDED(mu_);

  /// Remove all cached blocks for `filename`.
  void RemoveFile(const string& filename) override TF_LOCKS_EXCLUDED(mu_);

  /// Remove all cached data.
  void Flush() override TF_LOCKS_EXCLUDED(mu_);

  /// Accessors for cache parameters.
  size_t block_size() const override { return block_size_; }
  size_t max_bytes() const override { return max_bytes_; }
  uint64 max_staleness() const override { return max_staleness_; }

  /// The current size (in bytes) of the cache.
  size_t CacheSize() const override TF_LOCKS_EXCLUDED(mu_);

  // Returns true if the cache is enabled. If false, the BlockFetcher callback
  // is always executed during Read.
  bool IsCacheEnabled() const override {
    return block_size_ > 0 && max_bytes_ > 0;
  }

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

  /// \brief The state of a block.
  ///
  /// A block begins in the CREATED stage. The first thread will attempt to read
  /// the block from the filesystem, transitioning the state of the block to
  /// FETCHING. After completing, if the read was successful the state should
  /// be FINISHED. Otherwise the state should be ERROR. A subsequent read can
  /// re-fetch the block if the state is ERROR.
  enum class FetchState {
    CREATED,
    FETCHING,
    FINISHED,
    ERROR,
  };

  /// \brief A block of a file.
  ///
  /// A file block consists of the block data, the block's current position in
  /// the LRU cache, the timestamp (seconds since epoch) at which the block
  /// was cached, a coordination lock, and state & condition variables.
  ///
  /// Thread safety:
  /// The iterator and timestamp fields should only be accessed while holding
  /// the block-cache-wide mu_ instance variable. The state variable should only
  /// be accessed while holding the Block's mu lock. The data vector should only
  /// be accessed after state == FINISHED, and it should never be modified.
  ///
  /// In order to prevent deadlocks, never grab the block-cache-wide mu_ lock
  /// AFTER grabbing any block's mu lock. It is safe to grab mu without locking
  /// mu_.
  struct Block {
    /// The block data.
    std::vector<char> data;
    /// A list iterator pointing to the block's position in the LRU list.
    std::list<Key>::iterator lru_iterator;
    /// A list iterator pointing to the block's position in the LRA list.
    std::list<Key>::iterator lra_iterator;
    /// The timestamp (seconds since epoch) at which the block was cached.
    uint64 timestamp;
    /// Mutex to guard state variable
    mutex mu;
    /// The state of the block.
    FetchState state TF_GUARDED_BY(mu) = FetchState::CREATED;
    /// Wait on cond_var if state is FETCHING.
    condition_variable cond_var;
  };

  /// \brief The block map type for the file block cache.
  ///
  /// The block map is an ordered map from Key to Block.
  typedef std::map<Key, std::shared_ptr<Block>> BlockMap;

  /// Prune the cache by removing files with expired blocks.
  void Prune() TF_LOCKS_EXCLUDED(mu_);

  bool BlockNotStale(const std::shared_ptr<Block>& block)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  /// Look up a Key in the block cache.
  std::shared_ptr<Block> Lookup(const Key& key) TF_LOCKS_EXCLUDED(mu_);

  Status MaybeFetch(const Key& key, const std::shared_ptr<Block>& block)
      TF_LOCKS_EXCLUDED(mu_);

  /// Trim the block cache to make room for another entry.
  void Trim() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  /// Update the LRU iterator for the block at `key`.
  Status UpdateLRU(const Key& key, const std::shared_ptr<Block>& block)
      TF_LOCKS_EXCLUDED(mu_);

  /// Remove all blocks of a file, with mu_ already held.
  void RemoveFile_Locked(const string& filename)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  /// Remove the block `entry` from the block map and LRU list, and update the
  /// cache size accordingly.
  void RemoveBlock(BlockMap::iterator entry) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  /// The cache pruning thread that removes files with expired blocks.
  std::unique_ptr<Thread> pruning_thread_;

  /// Notification for stopping the cache pruning thread.
  Notification stop_pruning_thread_;

  /// Guards access to the block map, LRU list, and cached byte count.
  mutable mutex mu_;

  /// The block map (map from Key to Block).
  BlockMap block_map_ TF_GUARDED_BY(mu_);

  /// The LRU list of block keys. The front of the list identifies the most
  /// recently accessed block.
  std::list<Key> lru_list_ TF_GUARDED_BY(mu_);

  /// The LRA (least recently added) list of block keys. The front of the list
  /// identifies the most recently added block.
  ///
  /// Note: blocks are added to lra_list_ only after they have successfully been
  /// fetched from the underlying block store.
  std::list<Key> lra_list_ TF_GUARDED_BY(mu_);

  /// The combined number of bytes in all of the cached blocks.
  size_t cache_size_ TF_GUARDED_BY(mu_) = 0;

  // A filename->file_signature map.
  std::map<string, int64> file_signature_map_ TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_RAM_FILE_BLOCK_CACHE_H_
