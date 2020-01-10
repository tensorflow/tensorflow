// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

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
