/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tsl/platform/protobuf.h"

namespace tsl {

const char* kProtobufInt64Typename = "::tensorflow::protobuf_int64";
const char* kProtobufUint64Typename = "::tensorflow::protobuf_uint64";

TStringOutputStream::TStringOutputStream(tstring* target) : target_(target) {}

bool TStringOutputStream::Next(void** data, int* size) {
  size_t old_size = target_->size();

  // Grow the string.
  if (old_size < target_->capacity()) {
    // Resize the string to match its capacity, since we can get away
    // without a memory allocation this way.
    target_->resize_uninitialized(target_->capacity());
  } else {
    // Size has reached capacity, try to double the size.
    if (old_size > std::numeric_limits<size_t>::max() / 2) {
      // Can not double the size otherwise it is going to cause integer
      // overflow in the expression below: old_size * 2 ";
      return false;
    }
    // Double the size, also make sure that the new size is at least
    // kMinimumSize.
    target_->resize_uninitialized(std::max(
        old_size * 2,
        (size_t)kMinimumSize + 0));  // "+ 0" works around GCC4 weirdness.
  }

  *data = target_->data() + old_size;
  *size = target_->size() - old_size;
  return true;
}

void TStringOutputStream::BackUp(int count) {
  target_->resize(target_->size() - count);
}

int64_t TStringOutputStream::ByteCount() const { return target_->size(); }

}  // namespace tsl
