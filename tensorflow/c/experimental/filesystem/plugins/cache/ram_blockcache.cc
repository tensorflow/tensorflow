/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/experimental/filesystem/plugins/cache/ram_blockcache.h"

#include <cstring>
#include <iostream>

#include "tensorflow/c/experimental/filesystem/plugins/cache/cleanup.h"

namespace tf_cache {

bool RamBlockCache::BlockNotStale(const std::shared_ptr<Block>& block) {
  std::lock_guard<std::mutex> l(block->mu);
  if (block->state != FetchState::FINISHED) {
    return true;  // No need to check for staleness.
  }
  if (max_staleness_ == 0) return true;  // Not enforcing staleness.
  return timer_seconds_() - block->timestamp <= max_staleness_;
}

std::shared_ptr<RamBlockCache::Block> RamBlockCache::Lookup(const Key& key) {
  std::lock_guard<std::mutex> lock(mu_);
  auto entry = block_map_.find(key);
  if (entry != block_map_.end()) {
    if (BlockNotStale(entry->second)) {
      if (cache_stats_ != nullptr) {
        cache_stats_->RecordCacheHitBlockSize(entry->second->data.size());
      }
      return entry->second;
    } else {
      // Remove the stale block and continue.
      RemoveFile_Locked(key.first);
    }
  }

  // Insert a new empty block, setting the bookkeeping to sentinel values
  // in order to update them as appropriate.
  auto new_entry = std::make_shared<Block>();
  lru_list_.push_front(key);
  lra_list_.push_front(key);
  new_entry->lru_iterator = lru_list_.begin();
  new_entry->lra_iterator = lra_list_.begin();
  new_entry->timestamp = timer_seconds_();
  block_map_.emplace(std::make_pair(key, new_entry));
  return new_entry;
}

// Remove blocks from the cache until we do not exceed our maximum size.
void RamBlockCache::Trim() {
  while (!lru_list_.empty() && cache_size_ > max_bytes_) {
    RemoveBlock(block_map_.find(lru_list_.back()));
  }
}

/// Move the block to the front of the LRU list if it isn't already there.
void RamBlockCache::UpdateLRU(const Key& key,
                              const std::shared_ptr<Block>& block,
                              TF_Status* status) {
  std::lock_guard<std::mutex> lock(mu_);
  if (block->timestamp == 0) {
    // The block was evicted from another thread. Allow it to remain evicted.
    TF_SetStatus(status, TF_OK, "");
    return;
  }
  if (block->lru_iterator != lru_list_.begin()) {
    lru_list_.erase(block->lru_iterator);
    lru_list_.push_front(key);
    block->lru_iterator = lru_list_.begin();
  }

  // Check for inconsistent state. If there is a block later in the same file
  // in the cache, and our current block is not block size, this likely means
  // we have inconsistent state within the cache. Note: it's possible some
  // incomplete reads may still go undetected.
  if (block->data.size() < block_size_) {
    Key fmax = std::make_pair(key.first, std::numeric_limits<size_t>::max());
    auto fcmp = block_map_.upper_bound(fmax);
    if (fcmp != block_map_.begin() && key < (--fcmp)->first) {
      TF_SetStatus(status, TF_INTERNAL,
                   "Block cache contents are inconsistent.");
      return;
    }
  }

  Trim();

  TF_SetStatus(status, TF_OK, "");
  return;
}

void RamBlockCache::MaybeFetch(const Key& key,
                               const std::shared_ptr<Block>& block,
                               TF_Status* status) {
  bool downloaded_block = false;
  auto reconcile_state =
      gtl::MakeCleanup([this, &downloaded_block, &key, &block] {
        // Perform this action in a cleanup callback to avoid locking mu_ after
        // locking block->mu.
        if (downloaded_block) {
          std::lock_guard<std::mutex> l(mu_);
          // Do not update state if the block is already to be evicted.
          if (block->timestamp != 0) {
            // Use capacity() instead of size() to account for all  memory
            // used by the cache.
            cache_size_ += block->data.capacity();
            // Put to beginning of LRA list.
            lra_list_.erase(block->lra_iterator);
            lra_list_.push_front(key);
            block->lra_iterator = lra_list_.begin();
            block->timestamp = timer_seconds_();
          }
        }
      });
  // Loop until either block content is successfully fetched, or our request
  // encounters an error.
  std::unique_lock<std::mutex> l(block->mu);
  TF_SetStatus(status, TF_OK, "");
  while (true) {
    switch (block->state) {
      case FetchState::ERROR:
      case FetchState::CREATED:
        block->state = FetchState::FETCHING;
        block->mu.unlock();  // Release the lock while making the API call.
        block->data.clear();
        block->data.resize(block_size_, 0);
        size_t bytes_transferred;
        block_fetcher_(key.first, key.second, block_size_, block->data.data(),
                       &bytes_transferred, status);
        if (cache_stats_ != nullptr) {
          cache_stats_->RecordCacheMissBlockSize(bytes_transferred);
        }
        block->mu.lock();  // Reacquire the lock immediately afterwards
        if (TF_GetCode(status) == TF_OK) {
          block->data.resize(bytes_transferred, 0);
          // Shrink the data capacity to the actual size used.
          // NOLINTNEXTLINE: shrink_to_fit() may not shrink the capacity.
          std::vector<char>(block->data).swap(block->data);
          downloaded_block = true;
          block->state = FetchState::FINISHED;
        } else {
          block->state = FetchState::ERROR;
        }
        block->cond_var.notify_all();
        return;
      case FetchState::FETCHING:
        block->cond_var.wait_for(l, std::chrono::seconds(60));
        if (block->state == FetchState::FINISHED) {
          TF_SetStatus(status, TF_OK, "");
          return;
        }
        // Re-loop in case of errors.
        break;
      case FetchState::FINISHED:
        TF_SetStatus(status, TF_OK, "");
        return;
    }
  }
  TF_SetStatus(
      status, TF_INTERNAL,
      "Control flow should never reach the end of RamFileBlockCache::Fetch.");
  return;
}

void RamBlockCache::Read(const std::string& filename, size_t offset, size_t n,
                         char* buffer, size_t* bytes_transferred,
                         TF_Status* status) {
  *bytes_transferred = 0;
  if (n == 0) {
    TF_SetStatus(status, TF_OK, "");
    return;
  }
  if (!IsCacheEnabled() || (n > max_bytes_)) {
    // The cache is effectively disabled, so we pass the read through to the
    // fetcher without breaking it up into blocks.
    block_fetcher_(filename, offset, n, buffer, bytes_transferred, status);
    return;
  }
  // Calculate the block-aligned start and end of the read.
  size_t start = block_size_ * (offset / block_size_);
  size_t finish = block_size_ * ((offset + n) / block_size_);
  if (finish < offset + n) {
    finish += block_size_;
  }
  size_t total_bytes_transferred = 0;
  // Now iterate through the blocks, reading them one at a time.
  for (size_t pos = start; pos < finish; pos += block_size_) {
    Key key = std::make_pair(filename, pos);
    // Look up the block, fetching and inserting it if necessary, and update the
    // LRU iterator for the key and block.
    std::shared_ptr<Block> block = Lookup(key);
    if (!block)
      std::cerr << "No block for key " << key.first << "@" << key.second;
    MaybeFetch(key, block, status);
    if (TF_GetCode(status) != TF_OK) return;
    UpdateLRU(key, block, status);
    if (TF_GetCode(status) != TF_OK) return;
    // Copy the relevant portion of the block into the result buffer.
    const auto& data = block->data;
    if (offset >= pos + data.size()) {
      // The requested offset is at or beyond the end of the file. This can
      // happen if `offset` is not block-aligned, and the read returns the last
      // block in the file, which does not extend all the way out to `offset`.
      *bytes_transferred = total_bytes_transferred;
      TF_SetStatus(status, TF_OUT_OF_RANGE,
                   ("EOF at offset " + std::to_string(offset) + " in file " +
                    filename + " at position " + std::to_string(pos) +
                    " with data size " + std::to_string(data.size()))
                       .c_str());
      return;
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
      size_t bytes_to_copy = end - begin;
      memcpy(&buffer[total_bytes_transferred], &*begin, bytes_to_copy);
      total_bytes_transferred += bytes_to_copy;
    }
    if (data.size() < block_size_) {
      // The block was a partial block and thus signals EOF at its upper bound.
      break;
    }
  }
  *bytes_transferred = total_bytes_transferred;
  TF_SetStatus(status, TF_OK, "");
  return;
}

bool RamBlockCache::ValidateAndUpdateFileSignature(const std::string& filename,
                                                   int64_t file_signature) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = file_signature_map_.find(filename);
  if (it != file_signature_map_.end()) {
    if (it->second == file_signature) {
      return true;
    }
    // Remove the file from cache if the signatures don't match.
    RemoveFile_Locked(filename);
    it->second = file_signature;
    return false;
  }
  file_signature_map_[filename] = file_signature;
  return true;
}

size_t RamBlockCache::CacheSize() const {
  std::lock_guard<std::mutex> lock(mu_);
  return cache_size_;
}

void RamBlockCache::Prune() {
  while (!WaitForNotificationWithTimeout(&stop_pruning_thread_, 1000000)) {
    std::lock_guard<std::mutex> lock(mu_);
    uint64_t now = timer_seconds_();
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

void RamBlockCache::Flush() {
  std::lock_guard<std::mutex> lock(mu_);
  block_map_.clear();
  lru_list_.clear();
  lra_list_.clear();
  cache_size_ = 0;
}

void RamBlockCache::RemoveFile(const std::string& filename) {
  std::lock_guard<std::mutex> lock(mu_);
  RemoveFile_Locked(filename);
}

void RamBlockCache::RemoveFile_Locked(const std::string& filename) {
  Key begin = std::make_pair(filename, 0);
  auto it = block_map_.lower_bound(begin);
  while (it != block_map_.end() && it->first.first == filename) {
    auto next = std::next(it);
    RemoveBlock(it);
    it = next;
  }
}

void RamBlockCache::RemoveBlock(BlockMap::iterator entry) {
  // This signals that the block is removed, and should not be inadvertently
  // reinserted into the cache in UpdateLRU.
  entry->second->timestamp = 0;
  lru_list_.erase(entry->second->lru_iterator);
  lra_list_.erase(entry->second->lra_iterator);
  cache_size_ -= entry->second->data.capacity();
  block_map_.erase(entry);
}

}  // namespace tf_cache
