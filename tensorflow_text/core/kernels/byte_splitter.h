// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_BYTE_TOKENIZER_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_BYTE_TOKENIZER_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace tensorflow {
namespace text {

class ByteSplitter {
 public:
  // Creates an instance.
  ByteSplitter() { }

  // Tokenizes a string into bytes.
  //
  // Example:
  // input = "uñ"
  // bytes = [117, 195, 177]
  // start_offsets = [0, 1, 2]
  // end_offsets = [1, 2, 3]
  //
  // Args:
  //  * input: The string of an input.
  //  * bytes: The output bytes.
  //  * start_offsets: The start offsets of output bytes in the input text.
  //  * end_offsets: The end offsets of output bytes in the input text.
  // Note: the start offsets are inclusive and the end offsets are exclusive.
  void Split(const absl::string_view input,
             std::vector<unsigned char>* bytes,
             std::vector<int32_t>* start_offsets,
             std::vector<int32_t>* end_offsets) const;

  // Tokenizes a string into bytes.
  //
  // Example:
  // input = "uñ"
  // bytes = [117, 195, 177]
  // offsets = [0, 1, 2, 3]
  //
  // Args:
  //  * input: The string of an input.
  //  * bytes: The output bytes.
  //  * offsets: The offsets of output bytes in the input text. The size will
  //    be one plus the input. Each value is the mapped offset of each byte of
  //    the original input text. The final value maps the end.
  // Note: the start offsets are inclusive and the end offsets are exclusive.
  void Split(const absl::string_view input,
             std::vector<unsigned char>* bytes,
             std::vector<int32_t>* offsets) const;

  // Tokenizes a string into bytes.
  //
  // Example:
  // input = "uñ"
  // bytes = [117, 195, 177]
  //
  // Args:
  //  * input: The string of an input.
  //  * bytes: The output bytes.
  void Split(const absl::string_view input,
             std::vector<unsigned char>* bytes) const;

  // Splits a string by the given start and end offsets.
  //
  // Example:
  // input = "uñ"
  // start_offsets = [0, 1]
  // end_offsets = [1, 3]
  // string = ["u", "ñ"]
  //
  // Args:
  //  * input: The string of an input.
  //  * start_offsets: Input byte index where the new strings start (inclusive).
  //  * end_offsets: Input byte index where the new strings end. (exclusive)
  //
  // Return:
  //  The split substrings.
  absl::StatusOr<std::vector<absl::string_view>> SplitByOffsets(
      absl::string_view input,
      absl::Span<const int> start_offsets,
      absl::Span<const int> end_offsets) const;
};

}  // namespace text
}  // namespace tensorflow


#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_BYTE_TOKENIZER_H_
