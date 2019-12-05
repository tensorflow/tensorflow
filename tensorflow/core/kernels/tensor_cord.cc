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

#include "tensorflow/core/kernels/tensor_cord.h"

#include <cstring>

#include "tensorflow/core/framework/variant.h"

namespace tensorflow {

static_assert(Variant::CanInlineType<TensorCord>(),
              "TensorCord should be inlined into Variants");

TensorCord::CordRep::~CordRep() {
  if (!is_inline_ && rep_.external.releaser) {
    rep_.external.releaser(rep_.external.arg);
  }
}

TensorCord::~TensorCord() { Cleanup(); }

void TensorCord::Encode(VariantTensorData* data) const {
  data->metadata_string().clear();
  for (auto rep : Chunks()) {
    data->metadata_string().append(rep.data(), rep.size());
  }
}

bool TensorCord::Decode(VariantTensorData data) {
  auto* str = new string(std::move(data.metadata_string()));
  Cleanup();
  chunks_.push_back(new CordRep(absl::string_view(*str), &StringReleaser, str));
  return true;
}

TensorBuffer* TensorCord::TensorBufWithRef(Tensor* tensor) {
  TensorBuffer* buf = tensor->buf_;
  buf->Ref();
  return buf;
}

void TensorCord::TensorBufReleaser(void* tensor_buffer) {
  static_cast<TensorBuffer*>(tensor_buffer)->Unref();
}

void TensorCord::StringReleaser(void* str_ptr) {
  delete static_cast<string*>(str_ptr);
}

namespace {

// Helpers for STLStringResizeUninitialized
// HasMember is true_type or false_type, depending on whether or not
// T has a __resize_default_init member. Resize will call the
// __resize_default_init member if it exists, and will call the resize
// member otherwise.
template <typename string_type, typename = void>
struct ResizeUninitializedTraits {
  using HasMember = std::false_type;
  static void Resize(string_type* s, size_t new_size) { s->resize(new_size); }
};

// __resize_default_init is provided by libc++ >= 8.0.
template <typename string_type>
struct ResizeUninitializedTraits<
    string_type, absl::void_t<decltype(std::declval<string_type&>()
                                           .__resize_default_init(237))> > {
  using HasMember = std::true_type;
  static void Resize(string_type* s, size_t new_size) {
    s->__resize_default_init(new_size);
  }
};

// Resize string `s` to `new_size`, leaving the data uninitialized.
static inline void STLStringResizeUninitialized(string* s, size_t new_size) {
  ResizeUninitializedTraits<string>::Resize(s, new_size);
}

}  // namespace

TensorCord::operator string() const {
  string out;
  STLStringResizeUninitialized(&out, size());
  char* data = const_cast<char*>(out.data());
  for (auto* rep : chunks_) {
    auto view = rep->view();
    memcpy(data, view.data(), view.size());
    data += view.size();
  }
  DCHECK_EQ(data - out.data(), size());
  return out;
}

}  // namespace tensorflow
