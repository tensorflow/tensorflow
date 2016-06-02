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

#include "tensorflow/core/lib/io/table.h"

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/block.h"
#include "tensorflow/core/lib/io/format.h"
#include "tensorflow/core/lib/io/table_options.h"
#include "tensorflow/core/lib/io/two_level_iterator.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace table {

struct Table::Rep {
  ~Rep() { delete index_block; }

  Options options;
  Status status;
  RandomAccessFile* file;
  // XXX  uint64 cache_id;

  BlockHandle metaindex_handle;  // Handle to metaindex_block: saved from footer
  Block* index_block;
};

Status Table::Open(const Options& options, RandomAccessFile* file, uint64 size,
                   Table** table) {
  *table = NULL;
  if (size < Footer::kEncodedLength) {
    return errors::DataLoss("file is too short to be an sstable");
  }

  char footer_space[Footer::kEncodedLength];
  StringPiece footer_input;
  Status s = file->Read(size - Footer::kEncodedLength, Footer::kEncodedLength,
                        &footer_input, footer_space);
  if (!s.ok()) return s;

  Footer footer;
  s = footer.DecodeFrom(&footer_input);
  if (!s.ok()) return s;

  // Read the index block
  BlockContents contents;
  Block* index_block = NULL;
  if (s.ok()) {
    s = ReadBlock(file, footer.index_handle(), &contents);
    if (s.ok()) {
      index_block = new Block(contents);
    }
  }

  if (s.ok()) {
    // We've successfully read the footer and the index block: we're
    // ready to serve requests.
    Rep* rep = new Table::Rep;
    rep->options = options;
    rep->file = file;
    rep->metaindex_handle = footer.metaindex_handle();
    rep->index_block = index_block;
    // XXX    rep->cache_id = (options.block_cache ?
    // options.block_cache->NewId() : 0);
    *table = new Table(rep);
  } else {
    if (index_block) delete index_block;
  }

  return s;
}

Table::~Table() { delete rep_; }

static void DeleteBlock(void* arg, void* ignored) {
  delete reinterpret_cast<Block*>(arg);
}

// Convert an index iterator value (i.e., an encoded BlockHandle)
// into an iterator over the contents of the corresponding block.
Iterator* Table::BlockReader(void* arg, const StringPiece& index_value) {
  Table* table = reinterpret_cast<Table*>(arg);
  //  Cache* block_cache = table->rep_->options.block_cache;
  Block* block = NULL;
  //  Cache::Handle* cache_handle = NULL;

  BlockHandle handle;
  StringPiece input = index_value;
  Status s = handle.DecodeFrom(&input);
  // We intentionally allow extra stuff in index_value so that we
  // can add more features in the future.

  if (s.ok()) {
    BlockContents contents;
    s = ReadBlock(table->rep_->file, handle, &contents);
    if (s.ok()) {
      block = new Block(contents);
    }
  }

  Iterator* iter;
  if (block != NULL) {
    iter = block->NewIterator();
    iter->RegisterCleanup(&DeleteBlock, block, NULL);
  } else {
    iter = NewErrorIterator(s);
  }
  return iter;
}

Iterator* Table::NewIterator() const {
  return NewTwoLevelIterator(rep_->index_block->NewIterator(),
                             &Table::BlockReader, const_cast<Table*>(this));
}

Status Table::InternalGet(const StringPiece& k, void* arg,
                          void (*saver)(void*, const StringPiece&,
                                        const StringPiece&)) {
  Status s;
  Iterator* iiter = rep_->index_block->NewIterator();
  iiter->Seek(k);
  if (iiter->Valid()) {
    BlockHandle handle;
    Iterator* block_iter = BlockReader(this, iiter->value());
    block_iter->Seek(k);
    if (block_iter->Valid()) {
      (*saver)(arg, block_iter->key(), block_iter->value());
    }
    s = block_iter->status();
    delete block_iter;
  }
  if (s.ok()) {
    s = iiter->status();
  }
  delete iiter;
  return s;
}

uint64 Table::ApproximateOffsetOf(const StringPiece& key) const {
  Iterator* index_iter = rep_->index_block->NewIterator();
  index_iter->Seek(key);
  uint64 result;
  if (index_iter->Valid()) {
    BlockHandle handle;
    StringPiece input = index_iter->value();
    Status s = handle.DecodeFrom(&input);
    if (s.ok()) {
      result = handle.offset();
    } else {
      // Strange: we can't decode the block handle in the index block.
      // We'll just return the offset of the metaindex block, which is
      // close to the whole file size for this case.
      result = rep_->metaindex_handle.offset();
    }
  } else {
    // key is past the last key in the file.  Approximate the offset
    // by returning the offset of the metaindex block (which is
    // right near the end of the file).
    result = rep_->metaindex_handle.offset();
  }
  delete index_iter;
  return result;
}

}  // namespace table
}  // namespace tensorflow
