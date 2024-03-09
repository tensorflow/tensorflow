/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_PACKED_LITERAL_READER_H_
#define XLA_PACKED_LITERAL_READER_H_

#include <memory>

#include "xla/literal.h"
#include "xla/statusor.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/env.h"

namespace xla {

// Reads packed data from a metadata-less file as requested by a user (who must
// know its internal format). These are yielded as (structured) literal values.
class PackedLiteralReader {
 public:
  // Ownership of file is passed to this instance -- this instance takes
  // responsibility for closing it.
  explicit PackedLiteralReader(tsl::RandomAccessFile* file);
  ~PackedLiteralReader();

  // Yields the next packed literal with shape "shape" as read from the
  // underlying file stream.
  //
  // Layout is optional. If it is not provided, no layout is set on the literal
  // that is produced.
  absl::StatusOr<Literal> Read(const Shape& shape,
                               const Layout* layout = nullptr);

  // Returns whether the input file has been fully exhausted; i.e. all available
  // packed literals have been read and we're at the end of the file.
  bool IsExhausted() const;

 private:
  tsl::RandomAccessFile* file_;  // We own and close in our destructor
  uint64_t offset_;              // Next file offset to read from

  PackedLiteralReader(const PackedLiteralReader&) = delete;
  PackedLiteralReader& operator=(const PackedLiteralReader&) = delete;
};

}  // namespace xla

#endif  // XLA_PACKED_LITERAL_READER_H_
