#ifndef TENSORFLOW_PLATFORM_PROTOBUF_H_
#define TENSORFLOW_PLATFORM_PROTOBUF_H_

// Import whatever namespace protobuf comes from into the
// ::tensorflow::protobuf namespace.
//
// TensorFlow code should the ::tensorflow::protobuf namespace to refer
// to all protobuf APIs.

#include "tensorflow/core/platform/port.h"
#if defined(PLATFORM_GOOGLE)
#include "tensorflow/core/platform/google/protobuf.h"
#elif defined(PLATFORM_GOOGLE_ANDROID)
#include "tensorflow/core/platform/google/protobuf_android.h"
#else
#include "tensorflow/core/platform/default/protobuf.h"
#endif

namespace tensorflow {
// Parses a protocol buffer contained in a string in the binary wire format.
// Returns true on success. Note: Unlike protobuf's builtin ParseFromString,
// this function has no size restrictions on the total size of the encoded
// protocol buffer.
bool ParseProtoUnlimited(protobuf::Message* proto, const string& serialized);
bool ParseProtoUnlimited(protobuf::Message* proto, const void* serialized,
                         size_t size);
}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_PROTOBUF_H_
