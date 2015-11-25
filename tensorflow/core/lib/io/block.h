/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_LIB_IO_BLOCK_H_
#define TENSORFLOW_LIB_IO_BLOCK_H_

#include <stddef.h>
#include <stdint.h>
#include "tensorflow/core/lib/io/iterator.h"

namespace tensorflow {
namespace table {

struct BlockContents;

class Block {
 public:
  // Initialize the block with the specified contents.
  explicit Block(const BlockContents& contents);

  ~Block();

  size_t size() const { return size_; }
  Iterator* NewIterator();

 private:
  uint32 NumRestarts() const;

  const char* data_;
  size_t size_;
  uint32 restart_offset_;  // Offset in data_ of restart array
  bool owned_;             // Block owns data_[]

  // No copying allowed
  Block(const Block&);
  void operator=(const Block&);

  class Iter;
};

}  // namespace table
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_IO_BLOCK_H_
