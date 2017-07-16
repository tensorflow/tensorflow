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
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

/// \brief An LRU block cache of file contents.
///
/// This class should be used by read-only random access files on a remote
/// filesystem (e.g. GCS).
class FileBlockCache {
 public:
  /// The callback executed when a block is not found in the cache, and needs to
  /// be fetched from the backing filesystem. This callback is provided when the
  /// cache is constructed. The returned Status should be OK as long as the
  /// read from the remote filesystem succeeded (similar to the semantics of the
  /// read(2) system call).
  typedef std::function<Status(uint64, size_t, std::vector<char>*)>
      BlockFetcher;

  FileBlockCache(uint64 block_size, uint32 block_count, uint64 max_staleness,
                 BlockFetcher block_fetcher, Env* env = Env::Default())
      : block_size_(block_size),
        block_count_(block_count),
        max_staleness_(max_staleness),
        block_fetcher_(block_fetcher),
        env_(env) {}

  /// Read `n` bytes starting at `offset` into `out`. This method will return:
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
  Status Read(uint64 offset, size_t n, std::vector<char>* out);

 private:
  /// Trim the LRU cache until its size is at most `size` blocks.
  void TrimCache(size_t size) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  /// The size of the blocks stored in the LRU cache, as well as the size of the
  /// reads from the underlying filesystem.
  const uint64 block_size_;
  /// The maximum number of blocks allowed in the LRU cache.
  const uint32 block_count_;
  /// The maximum staleness of any block in the LRU cache, in seconds.
  const uint64 max_staleness_;
  /// The callback to read a block from the underlying filesystem.
  const BlockFetcher block_fetcher_;
  /// The Env from which we read timestamps.
  Env* const env_;  // not owned

  /// \brief A block of a file.
  ///
  /// A file block consists of the block data and the block's current position
  /// in the LRU cache.
  struct Block {
    /// The block data.
    std::vector<char> data;
    /// A list iterator pointing to the block's position in the LRU list.
    std::list<uint64>::iterator lru_iterator;
  };

  /// Guards access to the block map, LRU list, and cache timestamp.
  mutex mu_;

  /// The block map (map from offset in the file to Block object).
  std::map<uint64, std::unique_ptr<Block>> block_map_ GUARDED_BY(mu_);

  /// The LRU list of offsets in the file. The front of the list is the position
  /// of the most recently accessed block.
  std::list<uint64> lru_list_ GUARDED_BY(mu_);

  /// The most recent timestamp (in seconds since epoch) at which the block map
  /// transitioned from empty to non-empty.  A value of 0 means the block map is
  /// currently empty.
  uint64 timestamp_ GUARDED_BY(mu_) = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_FILE_BLOCK_CACHE_H_
