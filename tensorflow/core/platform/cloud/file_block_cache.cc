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

Status FileBlockCache::Read(const string& filename, size_t offset, size_t n,
                            std::vector<char>* out) {
  out->clear();
  if (n == 0) {
    return Status::OK();
  }
  if (block_size_ == 0 || max_bytes_ == 0) {
    // The cache is effectively disabled, so we pass the read through to the
    // fetcher without breaking it up into blocks.
    return block_fetcher_(filename, offset, n, out);
  }
  // Calculate the block-aligned start and end of the read.
  size_t start = block_size_ * (offset / block_size_);
  size_t finish = block_size_ * ((offset + n) / block_size_);
  if (finish < offset + n) {
    finish += block_size_;
  }
  mutex_lock lock(mu_);
  // Now iterate through the blocks, reading them one at a time. Reads are
  // locked so that only one block_fetcher call is active at any given time.
  for (size_t pos = start; pos < finish; pos += block_size_) {
    Key key = std::make_pair(filename, pos);
    auto entry = block_map_.find(key);
    // If we're enforcing max staleness and the block is stale, remove all of
    // the file's cached blocks so we reload them.
    if (entry != block_map_.end() && max_staleness_ > 0 &&
        env_->NowSeconds() - entry->second->timestamp > max_staleness_) {
      RemoveFile_Locked(filename);
      entry = block_map_.end();
    }
    if (entry == block_map_.end()) {
      // We need to fetch the block from the remote filesystem. Trim the LRU
      // cache if needed - we do this up front in order to avoid any period of
      // time during which the cache size exceeds its desired limit. The
      // tradeoff is that if the fetcher fails, the cache may evict blocks
      // prematurely.
      while (!lru_list_.empty() && cache_size_ + block_size_ > max_bytes_) {
        RemoveBlock(block_map_.find(lru_list_.back()));
      }
      std::unique_ptr<Block> block(new Block);
      TF_RETURN_IF_ERROR(
          block_fetcher_(filename, pos, block_size_, &block->data));
      // Sanity check to detect interrupted reads leading to partial blocks: a
      // partial block must have a higher key than the highest existing key in
      // the block map for the file.
      if (block->data.size() < block_size_ && !block_map_.empty()) {
        Key fmax = std::make_pair(filename, std::numeric_limits<size_t>::max());
        auto fcmp = block_map_.upper_bound(fmax);
        if (fcmp != block_map_.begin() && key < (--fcmp)->first) {
          // We expected to read a full block at this position.
          return errors::Internal("File contents are inconsistent");
        }
      }
      // Record the block timestamp, update the cache size, and add the block to
      // the cache.
      block->timestamp = env_->NowSeconds();
      lra_list_.push_front(key);
      block->lra_iterator = lra_list_.begin();
      cache_size_ += block->data.size();
      entry = block_map_.emplace(std::make_pair(key, std::move(block))).first;
    } else {
      // Cache hit. Remove the block from the LRU list at its prior location.
      lru_list_.erase(entry->second->lru_iterator);
    }
    // Push the block to the front of the LRU list.
    lru_list_.push_front(key);
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
    if (data.size() < block_size_) {
      // The block was a partial block and thus signals EOF at its upper bound.
      break;
    }
  }
  return Status::OK();
}

size_t FileBlockCache::CacheSize() const {
  mutex_lock lock(mu_);
  return cache_size_;
}

void FileBlockCache::Prune() {
  while (!WaitForNotificationWithTimeout(&stop_pruning_thread_, 1000000)) {
    mutex_lock lock(mu_);
    uint64 now = env_->NowSeconds();
    while (!lra_list_.empty()) {
      auto it = block_map_.find(lra_list_.back());
      if (now - it->second->timestamp <= max_staleness_) {
        // The oldest block is not yet expired. Come back later.
        break;
      }
      // We need to make a copy of the filename here, since it could otherwise
      // be used within RemoveFile_Locked after `it` is deleted.
      RemoveFile_Locked(std::string(it->first.first));
    }
  }
}

void FileBlockCache::RemoveFile(const string& filename) {
  mutex_lock lock(mu_);
  RemoveFile_Locked(filename);
}

void FileBlockCache::RemoveFile_Locked(const string& filename) {
  Key begin = std::make_pair(filename, 0);
  auto it = block_map_.lower_bound(begin);
  while (it != block_map_.end() && it->first.first == filename) {
    auto next = std::next(it);
    RemoveBlock(it);
    it = next;
  }
}

void FileBlockCache::RemoveBlock(BlockMap::iterator entry) {
  lru_list_.erase(entry->second->lru_iterator);
  lra_list_.erase(entry->second->lra_iterator);
  cache_size_ -= entry->second->data.size();
  block_map_.erase(entry);
}

}  // namespace tensorflow
