/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_LIB_IO_INPUTSTREAM_INTERFACE_H_
#define XLA_TSL_LIB_IO_INPUTSTREAM_INTERFACE_H_

#include <string>

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/types.h"
#include "tsl/platform/cord.h"

namespace tsl {
namespace io {

// An interface that defines input streaming operations.
class InputStreamInterface {
 public:
  InputStreamInterface() {}
  virtual ~InputStreamInterface() {}

  // Reads the next bytes_to_read from the file. Typical return codes:
  //  * OK - in case of success.
  //  * OUT_OF_RANGE - not enough bytes remaining before end of file.
  virtual absl::Status ReadNBytes(int64_t bytes_to_read, tstring* result) = 0;

#if defined(TF_CORD_SUPPORT)
  // Reads the next bytes_to_read from the file. Typical return codes:
  //  * OK - in case of success.
  //  * OUT_OF_RANGE - not enough bytes remaining before end of file.
  virtual absl::Status ReadNBytes(int64_t bytes_to_read, absl::Cord* cord) {
    return errors::Unimplemented(
        "ReadNBytes(int64, absl::Cord*) is not implemented.");
  }
#endif

  // Skips bytes_to_skip before next ReadNBytes. bytes_to_skip should be >= 0.
  // Typical return codes:
  //  * OK - in case of success.
  //  * OUT_OF_RANGE - not enough bytes remaining before end of file.
  virtual absl::Status SkipNBytes(int64_t bytes_to_skip);

  // Return the offset of the current byte relative to the beginning of the
  // file.
  // If we Skip / Read beyond the end of the file, this should return the length
  // of the file.
  // If there are any errors, this must return -1.
  virtual int64_t Tell() const = 0;

  // Resets the stream to the beginning.
  virtual absl::Status Reset() = 0;
};

}  // namespace io
}  // namespace tsl

#endif  // XLA_TSL_LIB_IO_INPUTSTREAM_INTERFACE_H_
