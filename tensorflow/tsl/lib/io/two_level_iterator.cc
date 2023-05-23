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

#include "tensorflow/tsl/lib/io/two_level_iterator.h"

#include "tensorflow/tsl/lib/io/block.h"
#include "tensorflow/tsl/lib/io/format.h"
#include "tensorflow/tsl/lib/io/iterator.h"
#include "tensorflow/tsl/lib/io/table.h"

namespace tsl {
namespace table {

namespace {

typedef Iterator* (*BlockFunction)(void*, const StringPiece&);

class TwoLevelIterator : public Iterator {
 public:
  TwoLevelIterator(Iterator* index_iter, BlockFunction block_function,
                   void* arg);

  ~TwoLevelIterator() override;

  void Seek(const StringPiece& target) override;
  void SeekToFirst() override;
  void Next() override;

  bool Valid() const override {
    return (data_iter_ == nullptr) ? false : data_iter_->Valid();
  }
  StringPiece key() const override {
    assert(Valid());
    return data_iter_->key();
  }
  StringPiece value() const override {
    assert(Valid());
    return data_iter_->value();
  }
  Status status() const override {
    // It'd be nice if status() returned a const Status& instead of a
    // Status
    if (!index_iter_->status().ok()) {
      return index_iter_->status();
    } else if (data_iter_ != nullptr && !data_iter_->status().ok()) {
      return data_iter_->status();
    } else {
      return status_;
    }
  }

 private:
  void SaveError(const Status& s) {
    if (status_.ok() && !s.ok()) status_ = s;
  }
  void SkipEmptyDataBlocksForward();
  void SetDataIterator(Iterator* data_iter);
  void InitDataBlock();

  BlockFunction block_function_;
  void* arg_;
  Status status_;
  Iterator* index_iter_;
  Iterator* data_iter_;  // May be NULL
  // If data_iter_ is non-NULL, then "data_block_handle_" holds the
  // "index_value" passed to block_function_ to create the data_iter_.
  string data_block_handle_;
};

TwoLevelIterator::TwoLevelIterator(Iterator* index_iter,
                                   BlockFunction block_function, void* arg)
    : block_function_(block_function),
      arg_(arg),
      index_iter_(index_iter),
      data_iter_(nullptr) {}

TwoLevelIterator::~TwoLevelIterator() {
  delete index_iter_;
  delete data_iter_;
}

void TwoLevelIterator::Seek(const StringPiece& target) {
  index_iter_->Seek(target);
  InitDataBlock();
  if (data_iter_ != nullptr) data_iter_->Seek(target);
  SkipEmptyDataBlocksForward();
}

void TwoLevelIterator::SeekToFirst() {
  index_iter_->SeekToFirst();
  InitDataBlock();
  if (data_iter_ != nullptr) data_iter_->SeekToFirst();
  SkipEmptyDataBlocksForward();
}

void TwoLevelIterator::Next() {
  assert(Valid());
  data_iter_->Next();
  SkipEmptyDataBlocksForward();
}

void TwoLevelIterator::SkipEmptyDataBlocksForward() {
  while (data_iter_ == nullptr || !data_iter_->Valid()) {
    // Move to next block
    if (!index_iter_->Valid()) {
      SetDataIterator(nullptr);
      return;
    }
    index_iter_->Next();
    InitDataBlock();
    if (data_iter_ != nullptr) data_iter_->SeekToFirst();
  }
}

void TwoLevelIterator::SetDataIterator(Iterator* data_iter) {
  if (data_iter_ != nullptr) {
    SaveError(data_iter_->status());
    delete data_iter_;
  }
  data_iter_ = data_iter;
}

void TwoLevelIterator::InitDataBlock() {
  if (!index_iter_->Valid()) {
    SetDataIterator(nullptr);
  } else {
    StringPiece handle = index_iter_->value();
    if (data_iter_ != nullptr && handle.compare(data_block_handle_) == 0) {
      // data_iter_ is already constructed with this iterator, so
      // no need to change anything
    } else {
      Iterator* iter = (*block_function_)(arg_, handle);
      data_block_handle_.assign(handle.data(), handle.size());
      SetDataIterator(iter);
    }
  }
}

}  // namespace

Iterator* NewTwoLevelIterator(Iterator* index_iter,
                              BlockFunction block_function, void* arg) {
  return new TwoLevelIterator(index_iter, block_function, arg);
}

}  // namespace table
}  // namespace tsl
