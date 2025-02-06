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

#ifndef XLA_TSL_LIB_IO_BLOCK_BUILDER_H_
#define XLA_TSL_LIB_IO_BLOCK_BUILDER_H_

#include <stdint.h>

#include <vector>

#include "xla/tsl/platform/types.h"
#include "tsl/platform/stringpiece.h"

namespace tsl {
namespace table {

struct Options;

class BlockBuilder {
 public:
  explicit BlockBuilder(const Options* options);

  // Reset the contents as if the BlockBuilder was just constructed.
  void Reset();

  // REQUIRES: Finish() has not been called since the last call to Reset().
  // REQUIRES: key is larger than any previously added key
  void Add(absl::string_view key, absl::string_view value);

  // Finish building the block and return a slice that refers to the
  // block contents.  The returned slice will remain valid for the
  // lifetime of this builder or until Reset() is called.
  absl::string_view Finish();

  // Returns an estimate of the current (uncompressed) size of the block
  // we are building.
  size_t CurrentSizeEstimate() const;

  // Return true iff no entries have been added since the last Reset()
  bool empty() const { return buffer_.empty(); }

 private:
  const Options* options_;
  string buffer_;                 // Destination buffer
  std::vector<uint32> restarts_;  // Restart points
  int counter_;                   // Number of entries emitted since restart
  bool finished_;                 // Has Finish() been called?
  string last_key_;

  // No copying allowed
  BlockBuilder(const BlockBuilder&);
  void operator=(const BlockBuilder&);
};

}  // namespace table
}  // namespace tsl

#endif  // XLA_TSL_LIB_IO_BLOCK_BUILDER_H_
