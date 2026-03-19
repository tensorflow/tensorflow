/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_RUNTIME_LARGE_HLO_SNAPSHOT_SERIALIZATION_CODED_STREAM_ITERATORS_H_
#define XLA_RUNTIME_LARGE_HLO_SNAPSHOT_SERIALIZATION_CODED_STREAM_ITERATORS_H_

#include <cstddef>
#include <iterator>

#include "tsl/platform/protobuf.h"

// The main Motivations for these iterators is to be able to use the
// `Literal::Serialize` method with a `std::string` output stream.
// To use the native serialization of Literal, we must ensure that the size of
// each individual argument is less than 2 GiB, we don't want to limit the size
// of those. Also, as this is intended to be used with large snapshots, we're
// trying to minimize the memory usage. Using `SerializeAsString` or anything
// similar would create an additional copy of a (potentially very large)
// argument.

namespace xla {
// An output iterator over a tsl::protobuf::io::CodedOutputStream.
class CodedStreamOutputIterator {
 public:
  typedef std::output_iterator_tag iterator_category;
  typedef char value_type;
  typedef std::ptrdiff_t difference_type;
  typedef char* pointer;
  typedef char& reference;

  explicit CodedStreamOutputIterator(
      tsl::protobuf::io::CodedOutputStream* output_stream)
      : output_stream_(output_stream) {}

  // CodedOutputStream is buffered by default, thus it is being performant
  // even by writing a single byte.
  CodedStreamOutputIterator& operator=(char byte) {
    output_stream_->WriteRaw(&byte, 1);
    return *this;
  }

  CodedStreamOutputIterator& operator*() { return *this; }
  CodedStreamOutputIterator& operator++() { return *this; }
  CodedStreamOutputIterator operator++(int) { return *this; }

 private:
  tsl::protobuf::io::CodedOutputStream* output_stream_;
};

// An input iterator over a tsl::protobuf::io::CodedInputStream. The iterator
// can be limited to read a specific number of bytes to read delimited chunks of
// data.
class CodedStreamInputIterator {
 public:
  typedef std::input_iterator_tag iterator_category;
  typedef char value_type;
  typedef std::ptrdiff_t difference_type;
  typedef char* pointer;
  typedef char& reference;

  explicit CodedStreamInputIterator(
      tsl::protobuf::io::CodedInputStream* input_stream, int limit = -1)
      : input_stream_(input_stream), read_limit_(limit) {
    ReadNext();
  }

  CodedStreamInputIterator() : end_of_stream_(true) {}

  char operator*() const { return current_byte_; }

  CodedStreamInputIterator& operator++() {
    ReadNext();
    return *this;
  }

  CodedStreamInputIterator operator++(int) {
    CodedStreamInputIterator temp = *this;
    ReadNext();
    return temp;
  }

  bool operator==(const CodedStreamInputIterator& other) const {
    return end_of_stream_ == other.end_of_stream_ &&
           (end_of_stream_ || input_stream_ == other.input_stream_);
  }

  bool operator!=(const CodedStreamInputIterator& other) const {
    return !(*this == other);
  }

 private:
  void ReadNext() {
    if (end_of_stream_) {
      return;
    }

    if (read_limit_ != -1 && read_count_ >= read_limit_) {
      end_of_stream_ = true;
      return;
    }
    if (!input_stream_->ReadRaw(&current_byte_, 1)) {
      end_of_stream_ = true;
    } else {
      ++read_count_;
    }
  }

  tsl::protobuf::io::CodedInputStream* input_stream_;
  char current_byte_;
  bool end_of_stream_ = false;
  int read_limit_;
  int read_count_ = 0;
};

}  // namespace xla

#endif  // XLA_RUNTIME_LARGE_HLO_SNAPSHOT_SERIALIZATION_CODED_STREAM_ITERATORS_H_
