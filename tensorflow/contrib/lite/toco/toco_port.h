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
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_TOCO_TOCO_PORT_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_TOCO_TOCO_PORT_H_

// Portability layer for toco tool. Mainly, abstract filesystem access so we
// can build and use on google internal environments and on OSX.

#include <string>
#include "tensorflow/contrib/lite/toco/format_port.h"
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

namespace toco {
namespace port {

class Status {
 public:
  Status() {}

  Status(bool ok, const string& message) : ok_(ok), message_(message) {}

  bool ok() const { return ok_; }

  const string error_message() const { return message_; }

 private:
  bool ok_ = false;
  string message_;
};

void InitGoogle(const char* usage, int* argc, char*** argv, bool remove_flags);
void CheckInitGoogleIsDone(const char* message);

namespace file {
class Options {};
inline Options Defaults() {
  Options o;
  return o;
}
Status GetContents(const string& filename, string* contents,
                   const Options& options);
Status SetContents(const string& filename, const string& contents,
                   const Options& options);
string JoinPath(const string& base, const string& filename);
Status Writable(const string& filename);
Status Readable(const string& filename, const Options& options);
Status Exists(const string& filename, const Options& options);
}  // namespace file

// Copy `src` string to `dest`. User must ensure `dest` has enough space.
#if defined(PLATFORM_GOOGLE)
void CopyToBuffer(const ::Cord& src, char* dest);
#endif  // PLATFORM_GOOGLE
void CopyToBuffer(const string& src, char* dest);
}  // namespace port
}  // namespace toco

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_TOCO_TOCO_PORT_H_
