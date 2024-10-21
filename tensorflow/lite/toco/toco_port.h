/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TOCO_TOCO_PORT_H_
#define TENSORFLOW_LITE_TOCO_TOCO_PORT_H_

// Portability layer for toco tool. Mainly, abstract filesystem access so we
// can build and use on google internal environments and on OSX.

#include <string>
#include "google/protobuf/text_format.h"
#include "tensorflow/lite/toco/format_port.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/platform.h"
#if defined(PLATFORM_GOOGLE)
#include "absl/strings/cord.h"
#endif  // PLATFORM_GOOGLE

#ifdef PLATFORM_GOOGLE
#define TFLITE_PROTO_NS proto2
#else
#define TFLITE_PROTO_NS google::protobuf
#endif

#ifdef __ANDROID__
#include <sstream>
namespace std {

template <typename T>
std::string to_string(T value)
{
    std::ostringstream os ;
    os << value ;
    return os.str() ;
}

#ifdef __ARM_ARCH_7A__
double round(double x);
#endif
}
#endif

namespace toco {
namespace port {

// Things like tests use other initialization routines that need control
// of flags. However, for testing we still want to use toco_port.h facilities.
// This function sets initialized flag trivially.
void InitGoogleWasDoneElsewhere();
void InitGoogle(const char* usage, int* argc, char*** argv, bool remove_flags);
void CheckInitGoogleIsDone(const char* message);

namespace file {
class Options {};
inline Options Defaults() {
  Options o;
  return o;
}
absl::Status GetContents(const std::string& filename, std::string* contents,
                         const Options& options);
absl::Status SetContents(const std::string& filename,
                         const std::string& contents, const Options& options);
std::string JoinPath(const std::string& a, const std::string& b);
absl::Status Writable(const std::string& filename);
absl::Status Readable(const std::string& filename, const Options& options);
absl::Status Exists(const std::string& filename, const Options& options);
}  // namespace file

// Copy `src` string to `dest`. User must ensure `dest` has enough space.
#if defined(PLATFORM_GOOGLE)
void CopyToBuffer(const ::absl::Cord& src, char* dest);
#endif  // PLATFORM_GOOGLE
void CopyToBuffer(const std::string& src, char* dest);

inline uint32 ReverseBits32(uint32 n) {
  n = ((n >> 1) & 0x55555555) | ((n & 0x55555555) << 1);
  n = ((n >> 2) & 0x33333333) | ((n & 0x33333333) << 2);
  n = ((n >> 4) & 0x0F0F0F0F) | ((n & 0x0F0F0F0F) << 4);
  return (((n & 0xFF) << 24) | ((n & 0xFF00) << 8) | ((n & 0xFF0000) >> 8) |
          ((n & 0xFF000000) >> 24));
}
}  // namespace port

inline bool ParseFromStringOverload(const std::string& in,
                                    TFLITE_PROTO_NS::Message* proto) {
  return TFLITE_PROTO_NS::TextFormat::ParseFromString(in, proto);
}

template <typename Proto>
bool ParseFromStringEitherTextOrBinary(const std::string& input_file_contents,
                                       Proto* proto) {
  if (proto->ParseFromString(input_file_contents)) {
    return true;
  }

  if (ParseFromStringOverload(input_file_contents, proto)) {
    return true;
  }

  return false;
}

}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_TOCO_PORT_H_
