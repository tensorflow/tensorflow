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

#include "tensorflow/core/platform/cloud/file_block_cache.h"
#include <cstring>
#include <memory>
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

Status FileBlockCache::Read(uint64 offset, size_t n, std::vector<char>* out) {
  out->clear();
  if (n == 0) {
    return Status::OK();
  }
  if (block_size_ == 0 || block_count_ == 0) {
    // The cache is effectively disabled, so we pass the read through to the
    // fetcher without breaking it up into blocks.
    return block_fetcher_(offset, n, out);
  }
  // Calculate the block-aligned start and end of the read.
  uint64 start = block_size_ * (offset / block_size_);
  uint64 finish = block_size_ * ((offset + n) / block_size_);
  if (finish < offset + n) {
    finish += block_size_;
  }
  // Now iterate through the blocks, reading them one at a time. Reads are
  // locked so that only one block_fetcher call is active at any given time.
  mutex_lock lock(mu_);
  for (uint64 pos = start; pos < finish; pos += block_size_) {
    auto entry = block_map_.find(pos);
    if (entry == block_map_.end()) {
      // We need to fetch the block from the remote filesystem. Trim the LRU
      // cache if needed - we do this up front in order to avoid any period of
      // time during which the cache size exceeds its desired limit. The
      // tradeoff is that if the fetcher fails, the cache may evict a block
      // prematurely.
      while (lru_list_.size() >= block_count_) {
        block_map_.erase(lru_list_.back());
        lru_list_.pop_back();
      }
      std::unique_ptr<Block> block(new Block);
      TF_RETURN_IF_ERROR(block_fetcher_(pos, block_size_, &block->data));
      // Sanity check to detect interrupted reads leading to partial blocks: a
      // partial block must have a higher key than the highest existing key in
      // the block map.
      if (block->data.size() < block_size_ && !block_map_.empty() &&
          pos < block_map_.rbegin()->first) {
        // We expected to read a full block at this position.
        return errors::FailedPrecondition("File contents are inconsistent");
      }
      entry = block_map_.emplace(std::make_pair(pos, std::move(block))).first;
    } else {
      // Cache hit. Remove the block from the LRU list at its prior location.
      lru_list_.erase(entry->second->lru_iterator);
    }
    // Push the block to the front of the LRU list.
    lru_list_.push_front(pos);
    entry->second->lru_iterator = lru_list_.begin();
    // Copy the relevant portion of the block into the result buffer.
    const auto& data = entry->second->data;
    if (offset >= pos + data.size()) {
      // The requested offset is at or beyond the end of the file. This can
      // happen if `offset` is not block-aligned, and the read returns the last
      // block in the file, which does not extend all the way out to `offset`.
      return errors::OutOfRange("EOF at offset ", offset);
    }
    auto begin = data.begin();
    if (offset > pos) {
      // The block begins before the slice we're reading.
      begin += offset - pos;
    }
    auto end = data.end();
    if (pos + data.size() > offset + n) {
      // The block extends past the end of the slice we're reading.
      end -= (pos + data.size()) - (offset + n);
    }
    if (begin < end) {
      out->insert(out->end(), begin, end);
    }
  }
  return Status::OK();
}

}  // namespace tensorflow
