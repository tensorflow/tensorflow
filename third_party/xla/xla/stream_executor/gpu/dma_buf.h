/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_GPU_DMA_BUF_H_
#define XLA_STREAM_EXECUTOR_GPU_DMA_BUF_H_

#include <cstdint>
#include <utility>

namespace stream_executor::gpu {

// Owns an exported Linux dma-buf file descriptor for a device memory range.
// Closes the underlying file descriptor on destruction.
class DmaBuf {
 public:
  DmaBuf(int fd, uint64_t offset, uint64_t size)
      : fd_(fd), offset_(offset), size_(size) {}
  ~DmaBuf() { Close(); }

  DmaBuf(const DmaBuf&) = delete;
  DmaBuf& operator=(const DmaBuf&) = delete;

  DmaBuf(DmaBuf&& other) noexcept
      : fd_(std::exchange(other.fd_, -1)),
        offset_(other.offset_),
        size_(other.size_) {}
  DmaBuf& operator=(DmaBuf&& other) noexcept {
    if (this != &other) {
      Close();
      fd_ = std::exchange(other.fd_, -1);
      offset_ = other.offset_;
      size_ = other.size_;
    }
    return *this;
  }

  // Returns the Linux dma-buf file descriptor. Borrowed; do not close directly.
  int fd() const { return fd_; }

  // Returns the byte offset of the requested address from the start of the
  // dma-buf.
  uint64_t offset() const { return offset_; }

  // Returns the total exported size of the dma-buf in bytes, starting from
  // byte 0 of `fd()`.
  uint64_t size() const { return size_; }

 private:
  void Close();

  // Linux dma-buf file descriptor, or -1 if moved-from.
  int fd_;

  // Byte offset of the requested address within the dma-buf.
  uint64_t offset_;

  // Total exported size of the dma-buf in bytes.
  uint64_t size_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_DMA_BUF_H_
