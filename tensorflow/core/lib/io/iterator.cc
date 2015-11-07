// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "tensorflow/core/lib/io/iterator.h"

namespace tensorflow {
namespace table {

Iterator::Iterator() {
  cleanup_.function = NULL;
  cleanup_.next = NULL;
}

Iterator::~Iterator() {
  if (cleanup_.function != NULL) {
    (*cleanup_.function)(cleanup_.arg1, cleanup_.arg2);
    for (Cleanup* c = cleanup_.next; c != NULL;) {
      (*c->function)(c->arg1, c->arg2);
      Cleanup* next = c->next;
      delete c;
      c = next;
    }
  }
}

void Iterator::RegisterCleanup(CleanupFunction func, void* arg1, void* arg2) {
  assert(func != NULL);
  Cleanup* c;
  if (cleanup_.function == NULL) {
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
  EmptyIterator(const Status& s) : status_(s) {}
  virtual bool Valid() const { return false; }
  virtual void Seek(const StringPiece& target) {}
  virtual void SeekToFirst() {}
  virtual void Next() { assert(false); }
  StringPiece key() const {
    assert(false);
    return StringPiece();
  }
  StringPiece value() const {
    assert(false);
    return StringPiece();
  }
  virtual Status status() const { return status_; }

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
