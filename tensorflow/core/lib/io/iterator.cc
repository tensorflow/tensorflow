// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the License_LevelDB.txt file. See the Authors_LevelDB.txt file for names of contributors.

// Modifications:
// Copyright 2015 The TensorFlow Authors. All Rights Reserved.
// ===========================================================================

#include "tensorflow/core/lib/io/iterator.h"

namespace tensorflow {
namespace table {

Iterator::Iterator() {
  cleanup_.function = nullptr;
  cleanup_.next = nullptr;
}

Iterator::~Iterator() {
  if (cleanup_.function != nullptr) {
    (*cleanup_.function)(cleanup_.arg1, cleanup_.arg2);
    for (Cleanup* c = cleanup_.next; c != nullptr;) {
      (*c->function)(c->arg1, c->arg2);
      Cleanup* next = c->next;
      delete c;
      c = next;
    }
  }
}

void Iterator::RegisterCleanup(CleanupFunction func, void* arg1, void* arg2) {
  assert(func != nullptr);
  Cleanup* c;
  if (cleanup_.function == nullptr) {
    c = &cleanup_;
  } else {
    c = new Cleanup;
    c->next = cleanup_.next;
    cleanup_.next = c;
  }
  c->function = func;
  c->arg1 = arg1;
  c->arg2 = arg2;
}

namespace {
class EmptyIterator : public Iterator {
 public:
  explicit EmptyIterator(const Status& s) : status_(s) {}
  bool Valid() const override { return false; }
  void Seek(const StringPiece& target) override {}
  void SeekToFirst() override {}
  void Next() override { assert(false); }
  StringPiece key() const override {
    assert(false);
    return StringPiece();
  }
  StringPiece value() const override {
    assert(false);
    return StringPiece();
  }
  Status status() const override { return status_; }

 private:
  Status status_;
};
}  // namespace

Iterator* NewEmptyIterator() { return new EmptyIterator(Status::OK()); }

Iterator* NewErrorIterator(const Status& status) {
  return new EmptyIterator(status);
}

}  // namespace table
}  // namespace tensorflow
