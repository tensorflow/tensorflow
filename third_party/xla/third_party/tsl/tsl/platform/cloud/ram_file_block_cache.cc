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

#include "tsl/platform/cloud/ram_file_block_cache.h"

#include <cstring>
#include <memory>

#include "absl/cleanup/cleanup.h"
#include "tsl/platform/env.h"

namespace tsl {

bool RamFileBlockCache::BlockNotStale(const std::shared_ptr<Block>& block) {
  mutex_lock l(block->mu);
  if (block->state != FetchState::FINISHED) {
    return true;  // No need to check for staleness.
  }
  if (max_staleness_ == 0) return true;  // Not enforcing staleness.
  return env_->NowSeconds() - block->timestamp <= max_staleness_;
}

std::shared_ptr<RamFileBlockCache::Block> RamFileBlockCache::Lookup(
    const Key& key) {
  mutex_lock lock(mu_);
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
  new_entry->timestamp = env_->NowSeconds();
  block_map_.emplace(std::make_pair(key, new_entry));
  return new_entry;
}

// Remove blocks from the cache until we do not exceed our maximum size.
void RamFileBlockCache::Trim() {
  while (!lru_list_.empty() && cache_size_ > max_bytes_) {
    RemoveBlock(block_map_.find(lru_list_.back()));
  }
}

/// Move the block to the front of the LRU list if it isn't already there.
Status RamFileBlockCache::UpdateLRU(const Key& key,
                                    const std::shared_ptr<Block>& block) {
  mutex_lock lock(mu_);
  if (block->timestamp == 0) {
    // The block was evicted from another thread. Allow it to remain evicted.
    return OkStatus();
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
      return errors::Internal("Block cache contents are inconsistent.");
    }
  }

  Trim();

  return OkStatus();
}

Status RamFileBlockCache::MaybeFetch(const Key& key,
                                     const std::shared_ptr<Block>& block) {
  bool downloaded_block = false;
  auto reconcile_state =
      absl::MakeCleanup([this, &downloaded_block, &key, &block] {
        // Perform this action in a cleanup callback to avoid locking mu_ after
        // locking block->mu.
        if (downloaded_block) {
          mutex_lock l(mu_);
          // Do not update state if the block is already to be evicted.
          if (block->timestamp != 0) {
            // Use capacity() instead of size() to account for all  memory
            // used by the cache.
            cache_size_ += block->data.capacity();
            // Put to beginning of LRA list.
            lra_list_.erase(block->lra_iterator);
            lra_list_.push_front(key);
            block->lra_iterator = lra_list_.begin();
            block->timestamp = env_->NowSeconds();
          }
        }
      });
  // Loop until either block content is successfully fetched, or our request
  // encounters an error.
  mutex_lock l(block->mu);
  Status status = OkStatus();
  while (true) {
    switch (block->state) {
      case FetchState::ERROR:
        TF_FALLTHROUGH_INTENDED;
      case FetchState::CREATED:
        block->state = FetchState::FETCHING;
        block->mu.unlock();  // Release the lock while making the API call.
        block->data.clear();
        block->data.resize(block_size_, 0);
        size_t bytes_transferred;
        status.Update(block_fetcher_(key.first, key.second, block_size_,
                                     block->data.data(), &bytes_transferred));
        if (cache_stats_ != nullptr) {
          cache_stats_->RecordCacheMissBlockSize(bytes_transferred);
        }
        block->mu.lock();  // Reacquire the lock immediately afterwards
        if (status.ok()) {
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
        return status;
      case FetchState::FETCHING:
        block->cond_var.wait_for(l, std::chrono::seconds(60));
        if (block->state == FetchState::FINISHED) {
          return OkStatus();
        }
        // Re-loop in case of errors.
        break;
      case FetchState::FINISHED:
        return OkStatus();
    }
  }
  return errors::Internal(
      "Control flow should never reach the end of RamFileBlockCache::Fetch.");
}

Status RamFileBlockCache::Read(const string& filename, size_t offset, size_t n,
                               char* buffer, size_t* bytes_transferred) {
  *bytes_transferred = 0;
  if (n == 0) {
    return OkStatus();
  }
  if (!IsCacheEnabled() || (n > max_bytes_)) {
    // The cache is effectively disabled, so we pass the read through to the
    // fetcher without breaking it up into blocks.
    return block_fetcher_(filename, offset, n, buffer, bytes_transferred);
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
    DCHECK(block) << "No block for key " << key.first << "@" << key.second;
    TF_RETURN_IF_ERROR(MaybeFetch(key, block));
    TF_RETURN_IF_ERROR(UpdateLRU(key, block));
    // Copy the relevant portion of the block into the result buffer.
    const auto& data = block->data;
    if (offset >= pos + data.size()) {
      // The requested offset is at or beyond the end of the file. This can
      // happen if `offset` is not block-aligned, and the read returns the last
      // block in the file, which does not extend all the way out to `offset`.
      *bytes_transferred = total_bytes_transferred;
      return errors::OutOfRange("EOF at offset ", offset, " in file ", filename,
                                " at position ", pos, "with data size ",
                                data.size());
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
  return OkStatus();
}

bool RamFileBlockCache::ValidateAndUpdateFileSignature(const string& filename,
                                                       int64_t file_signature) {
  mutex_lock lock(mu_);
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

size_t RamFileBlockCache::CacheSize() const {
  mutex_lock lock(mu_);
  return cache_size_;
}

void RamFileBlockCache::Prune() {
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

void RamFileBlockCache::Flush() {
  mutex_lock lock(mu_);
  block_map_.clear();
  lru_list_.clear();
  lra_list_.clear();
  cache_size_ = 0;
}

void RamFileBlockCache::RemoveFile(const string& filename) {
  mutex_lock lock(mu_);
  RemoveFile_Locked(filename);
}

void RamFileBlockCache::RemoveFile_Locked(const string& filename) {
  Key begin = std::make_pair(filename, 0);
  auto it = block_map_.lower_bound(begin);
  while (it != block_map_.end() && it->first.first == filename) {
    auto next = std::next(it);
    RemoveBlock(it);
    it = next;
  }
}

void RamFileBlockCache::RemoveBlock(BlockMap::iterator entry) {
  // This signals that the block is removed, and should not be inadvertently
  // reinserted into the cache in UpdateLRU.
  entry->second->timestamp = 0;
  lru_list_.erase(entry->second->lru_iterator);
  lra_list_.erase(entry->second->lra_iterator);
  cache_size_ -= entry->second->data.capacity();
  block_map_.erase(entry);
}

}  // namespace tsl
