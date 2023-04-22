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

// TODO(vrv): Switch this to an open-sourced version of Arena.

#ifndef TENSORFLOW_CORE_LIB_CORE_ARENA_H_
#define TENSORFLOW_CORE_LIB_CORE_ARENA_H_

#include <assert.h>

#include <vector>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace core {

// This class is "thread-compatible": different threads can access the
// arena at the same time without locking, as long as they use only
// const methods.
class Arena {
 public:
  // Allocates a thread-compatible arena with the specified block size.
  explicit Arena(const size_t block_size);
  ~Arena();

  char* Alloc(const size_t size) {
    return reinterpret_cast<char*>(GetMemory(size, 1));
  }

  char* AllocAligned(const size_t size, const size_t alignment) {
    return reinterpret_cast<char*>(GetMemory(size, alignment));
  }

  void Reset();

// This should be the worst-case alignment for any type.  This is
// good for IA-32, SPARC version 7 (the last one I know), and
// supposedly Alpha.  i386 would be more time-efficient with a
// default alignment of 8, but ::operator new() uses alignment of 4,
// and an assertion will fail below after the call to MakeNewBlock()
// if you try to use a larger alignment.
#ifdef __i386__
  static const int kDefaultAlignment = 4;
#else
  static constexpr int kDefaultAlignment = 8;
#endif

 protected:
  bool SatisfyAlignment(const size_t alignment);
  void MakeNewBlock(const uint32 alignment);
  void* GetMemoryFallback(const size_t size, const int align);
  void* GetMemory(const size_t size, const int align) {
    assert(remaining_ <= block_size_);                  // an invariant
    if (size > 0 && size < remaining_ && align == 1) {  // common case
      void* result = freestart_;
      freestart_ += size;
      remaining_ -= size;
      return result;
    }
    return GetMemoryFallback(size, align);
  }

  size_t remaining_;

 private:
  struct AllocatedBlock {
    char* mem;
    size_t size;
  };

  // Allocate new block of at least block_size, with the specified
  // alignment.
  // The returned AllocatedBlock* is valid until the next call to AllocNewBlock
  // or Reset (i.e. anything that might affect overflow_blocks_).
  AllocatedBlock* AllocNewBlock(const size_t block_size,
                                const uint32 alignment);

  const size_t block_size_;
  char* freestart_;  // beginning of the free space in most recent block
  char* freestart_when_empty_;  // beginning of the free space when we're empty
  // STL vector isn't as efficient as it could be, so we use an array at first
  size_t blocks_alloced_;  // how many of the first_blocks_ have been alloced
  AllocatedBlock first_blocks_[16];  // the length of this array is arbitrary
  // if the first_blocks_ aren't enough, expand into overflow_blocks_.
  std::vector<AllocatedBlock>* overflow_blocks_;

  void FreeBlocks();  // Frees all except first block

  TF_DISALLOW_COPY_AND_ASSIGN(Arena);
};

}  // namespace core
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_CORE_ARENA_H_
