/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_FLEXBUFFERS_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_FLEXBUFFERS_UTIL_H_

#include "flatbuffers/base.h"  // from @flatbuffers
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers

namespace tflite::xnnpack {
// We use this class defined with internal linkage as a key to prevent the
// following workaround to leak into other translation units.
struct FloatPointer {
  const float* ptr = nullptr;
};
}  // namespace tflite::xnnpack

namespace flexbuffers {

// TODO(b/359351192): switch to xnnpack builtin. This is a workaround until we
// are able to use just the value.
//
// We go around the access policy of the `Reference` class by specializing a
// template function that was not specialized for our use case.
//
// This is weakly tolerant to an update to the `Reference` class because:
//   - THIS IS MEANT TO BE TEMPORARY until we actually use the XNNPack
//     implementation of SDPA (and dependent on not needing data ptr).
//   - The flexbuffer spec is public and set, so the layout should not evolve
//     much.
//
// The alternative was to copy/paste the code to get to the map data and grab
// the pointer which basically means rewriting flexbuffer.h.
template <>
tflite::xnnpack::FloatPointer inline flexbuffers::Reference::As<
    tflite::xnnpack::FloatPointer>() const {
#if !FLATBUFFERS_LITTLEENDIAN
  // Flexbuffers are always stored in little endian order. Returning a pointer
  // to the float data on a big endian architecture is meaningless.
  return nullptr;
#else
  return {IsFloat() ? reinterpret_cast<const float*>(data_) : nullptr};
#endif
}

}  // namespace flexbuffers

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_FLEXBUFFERS_UTIL_H_
