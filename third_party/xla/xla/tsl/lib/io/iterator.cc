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

#include "xla/tsl/lib/io/iterator.h"

namespace tsl {
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
  explicit EmptyIterator(const absl::Status& s) : status_(s) {}
  bool Valid() const override { return false; }
  void Seek(const absl::string_view& target) override {}
  void SeekToFirst() override {}
  void Next() override { assert(false); }
  absl::string_view key() const override {
    assert(false);
    return absl::string_view();
  }
  absl::string_view value() const override {
    assert(false);
    return absl::string_view();
  }
  absl::Status status() const override { return status_; }

 private:
  absl::Status status_;
};
}  // namespace

Iterator* NewEmptyIterator() { return new EmptyIterator(absl::OkStatus()); }

Iterator* NewErrorIterator(const absl::Status& status) {
  return new EmptyIterator(status);
}

}  // namespace table
}  // namespace tsl
