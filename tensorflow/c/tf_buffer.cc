/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/tf_buffer.h"

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/status.h"

extern "C" {

TF_Buffer* TF_NewBuffer() { return new TF_Buffer{nullptr, 0, nullptr}; }

TF_Buffer* TF_NewBufferFromString(const void* proto, size_t proto_len) {
  void* copy = tensorflow::port::Malloc(proto_len);
  memcpy(copy, proto, proto_len);

  TF_Buffer* buf = new TF_Buffer;
  buf->data = copy;
  buf->length = proto_len;
  buf->data_deallocator = [](void* data, size_t length) {
    tensorflow::port::Free(data);
  };
  return buf;
}

void TF_DeleteBuffer(TF_Buffer* buffer) {
  if (buffer == nullptr) return;
  if (buffer->data_deallocator != nullptr) {
    (*buffer->data_deallocator)(const_cast<void*>(buffer->data),
                                buffer->length);
  }
  delete buffer;
}

TF_Buffer TF_GetBuffer(TF_Buffer* buffer) { return *buffer; }

}  // end extern "C"

namespace tensorflow {

Status MessageToBuffer(const tensorflow::protobuf::MessageLite& in,
                       TF_Buffer* out) {
  if (out->data != nullptr) {
    return errors::InvalidArgument("Passing non-empty TF_Buffer is invalid.");
  }
  const size_t proto_size = in.ByteSizeLong();
  void* buf = port::Malloc(proto_size);
  if (buf == nullptr) {
    return tensorflow::errors::ResourceExhausted(
        "Failed to allocate memory to serialize message of type '",
        in.GetTypeName(), "' and size ", proto_size);
  }
  if (!in.SerializeWithCachedSizesToArray(static_cast<uint8*>(buf))) {
    port::Free(buf);
    return errors::InvalidArgument(
        "Unable to serialize ", in.GetTypeName(),
        " protocol buffer, perhaps the serialized size (", proto_size,
        " bytes) is too large?");
  }
  out->data = buf;
  out->length = proto_size;
  out->data_deallocator = [](void* data, size_t length) { port::Free(data); };
  return OkStatus();
}

Status BufferToMessage(const TF_Buffer* in,
                       tensorflow::protobuf::MessageLite* out) {
  if (in == nullptr || !out->ParseFromArray(in->data, in->length)) {
    return errors::InvalidArgument("Unparseable ", out->GetTypeName(),
                                   " proto");
  }
  return OkStatus();
}

}  // namespace tensorflow
