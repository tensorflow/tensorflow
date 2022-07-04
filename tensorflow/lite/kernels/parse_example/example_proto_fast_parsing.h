/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_PARSE_EXAMPLE_EXAMPLE_PROTO_FAST_PARSING_H_
#define TENSORFLOW_LITE_KERNELS_PARSE_EXAMPLE_EXAMPLE_PROTO_FAST_PARSING_H_
#include <algorithm>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/example_proto_fast_parsing.h"
#include "tensorflow/core/util/presized_cuckoo_map.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {
namespace example {

template <typename T>
using SmallVector = gtl::InlinedVector<T, 4>;

template <typename T>
class LimitedArraySlice {
 public:
  using value_type = T;

  LimitedArraySlice(T* begin, size_t num_elements)
      : current_(begin), begin_(begin), end_(begin + num_elements) {}

  // May return negative if there were push_back calls after slice was filled.
  int64_t EndDistance() const { return end_ - current_; }

  // Attempts to push value to the back of this. If the slice has
  // already been filled, this method has no effect on the underlying data, but
  // it changes the number returned by EndDistance into negative values.
  void push_back(T&& value) {
    if (EndDistance() > 0) *current_ = std::move(value);
    ++current_;
  }

  // "Constructs" an element at the back of this by resizing the slice, and
  // returns a mutable reference to the new last element.
  // REQUIRES: EndDistance() > 0.
  T& construct_at_end() {
    DCHECK_GT(EndDistance(), 0);
    return *(current_++);
  }

  // Returns a mutable reference to the last element in the slice.
  // REQUIRES: size() > 0.
  T& back() { return *(current_ - 1); }

  // Returns the number of elements in the slice.
  size_t size() const { return std::min(current_ - begin_, end_ - begin_); }

  // Attempts to resize the vector to the given size. It does so by advancing
  // the pointer to the current element, possibly beyond the end of the slice.
  // As a consequence, calling `size()` after `resize(x)` was called might
  // return a value less than `x`.
  void resize(size_t size) { current_ = begin_ + size; }

  // Returns the pointer to the underlying data buffer.
  T* data() { return begin_; }

 private:
  T* current_;
  T* begin_;
  T* end_;
};

template <typename A>
auto EnableAliasing(A* a) -> decltype(a->EnableAliasing(true), void()) {
  a->EnableAliasing(true);
}

template <typename A>
void EnableAliasing(A&& a) {}

uint8 PeekTag(protobuf::io::CodedInputStream* stream);

constexpr uint8 kVarintTag(uint32 tag) { return (tag << 3) | 0; }
constexpr uint8 kDelimitedTag(uint32 tag) { return (tag << 3) | 2; }
constexpr uint8 kFixed32Tag(uint32 tag) { return (tag << 3) | 5; }

namespace parsed {

// ParseDataType has to be called first, then appropriate ParseZzzzList.
class Feature {
 public:
  Feature() {}
  explicit Feature(StringPiece serialized) : serialized_(serialized) {}

  Status ParseDataType(DataType* dtype) {
    DCHECK(dtype != nullptr);
    if (serialized_.empty()) {
      *dtype = DT_INVALID;
      return OkStatus();
    }
    uint8 oneof_tag = static_cast<uint8>(*serialized_.data());
    serialized_.remove_prefix(1);
    switch (oneof_tag) {
      case kDelimitedTag(1):
        *dtype = DT_STRING;
        break;
      case kDelimitedTag(2):
        *dtype = DT_FLOAT;
        break;
      case kDelimitedTag(3):
        *dtype = DT_INT64;
        break;
      default:
        // Initialize variable to avoid compiler warning
        *dtype = DT_INVALID;
        return errors::InvalidArgument("Unsupported datatype.");
    }
    return OkStatus();
  }

  bool GetNumElementsInBytesList(int* num_elements) {
    protobuf::io::CodedInputStream stream(
        reinterpret_cast<const uint8*>(serialized_.data()), serialized_.size());
    EnableAliasing(&stream);
    uint32 length = 0;
    if (!stream.ReadVarint32(&length)) return false;
    auto limit = stream.PushLimit(length);
    *num_elements = 0;
    while (!stream.ExpectAtEnd()) {
      if (!stream.ExpectTag(kDelimitedTag(1))) return false;
      uint32 bytes_length = 0;
      if (!stream.ReadVarint32(&bytes_length)) return false;
      if (!stream.Skip(bytes_length)) return false;
      ++*num_elements;
    }
    stream.PopLimit(limit);
    return true;
  }

  // Helper methods
  tstring* construct_at_end(LimitedArraySlice<tstring>* bytes_list) {
    if (bytes_list->EndDistance() <= 0) {
      return nullptr;
    }
    return &bytes_list->construct_at_end();
  }
  tstring* construct_at_end(SmallVector<tstring>* bytes_list) {
    return &bytes_list->emplace_back();
  }

  template <typename Result>
  bool ParseBytesList(Result* bytes_list) {
    DCHECK(bytes_list != nullptr);

    protobuf::io::CodedInputStream stream(
        reinterpret_cast<const uint8*>(serialized_.data()), serialized_.size());

    EnableAliasing(&stream);

    uint32 length;
    if (!stream.ReadVarint32(&length)) return false;
    auto limit = stream.PushLimit(length);

    while (!stream.ExpectAtEnd()) {
      if (!stream.ExpectTag(kDelimitedTag(1))) return false;
      // parse string
      uint32 bytes_length;
      if (!stream.ReadVarint32(&bytes_length)) return false;
      tstring* bytes = construct_at_end(bytes_list);
      if (bytes == nullptr) return false;
      bytes->resize_uninitialized(bytes_length);
      if (!stream.ReadRaw(bytes->data(), bytes_length)) return false;
    }
    stream.PopLimit(limit);
    return true;
  }

  template <typename Result>
  bool ParseFloatList(Result* float_list) {
    DCHECK(float_list != nullptr);
    protobuf::io::CodedInputStream stream(
        reinterpret_cast<const uint8*>(serialized_.data()), serialized_.size());
    EnableAliasing(&stream);
    uint32 length;
    if (!stream.ReadVarint32(&length)) return false;
    auto limit = stream.PushLimit(length);

    if (!stream.ExpectAtEnd()) {
      uint8 peek_tag = PeekTag(&stream);
      if (peek_tag != kDelimitedTag(1) && peek_tag != kFixed32Tag(1)) {
        return false;
      }

      constexpr int32_t kNumFloatBytes = 4;
      if (peek_tag == kDelimitedTag(1)) {                       // packed
        if (!stream.ExpectTag(kDelimitedTag(1))) return false;  // packed tag
        uint32 packed_length;
        if (!stream.ReadVarint32(&packed_length)) return false;
        auto packed_limit = stream.PushLimit(packed_length);

        // Store the initial size to know the offset we have to start writing
        // data from before resizing the output "vector".
        const size_t initial_size = float_list->size();
        float_list->resize(initial_size + packed_length / kNumFloatBytes);

        // If the result data type is float and we are on a little endian
        // machine then we can simply memcpy the data from the proto into the
        // result vector.
        if (port::kLittleEndian &&
            sizeof(typename Result::value_type) == kNumFloatBytes) {
          // Calculate the length of the buffer available what can be less than
          // what we requested in resize in case of a LimitedArraySlice.
          const uint32 bytes_to_copy =
              std::min(static_cast<uint32>((float_list->size() - initial_size) *
                                           kNumFloatBytes),
                       packed_length);
          if (!stream.ReadRaw(float_list->data() + initial_size, bytes_to_copy))
            return false;
        } else {
          int64_t index = initial_size;
          while (!stream.ExpectAtEnd()) {
            uint32 buffer32;
            if (!stream.ReadLittleEndian32(&buffer32)) return false;
            if (index < float_list->size()) {
              float_list->data()[index] = absl::bit_cast<float>(buffer32);
              ++index;
            }
          }
        }

        stream.PopLimit(packed_limit);
      } else {  // non-packed
        const size_t initial_size = float_list->size();
        // 1 byte for the tag (`1` encoded as Variant32) and kNumFloatBytes for
        // the value.
        const int64_t num_elements =
            stream.BytesUntilLimit() / (1 + kNumFloatBytes);
        float_list->resize(initial_size + num_elements);
        int64_t index = initial_size;
        while (!stream.ExpectAtEnd()) {
          if (!stream.ExpectTag(kFixed32Tag(1))) return false;
          uint32 buffer32;
          if (!stream.ReadLittleEndian32(&buffer32)) return false;
          float_list->data()[index] = absl::bit_cast<float>(buffer32);
          ++index;
        }
      }
    }

    stream.PopLimit(limit);
    return true;
  }

  template <typename Result>
  bool ParseInt64List(Result* int64_list) {
    DCHECK(int64_list != nullptr);
    protobuf::io::CodedInputStream stream(
        reinterpret_cast<const uint8*>(serialized_.data()), serialized_.size());
    EnableAliasing(&stream);
    uint32 length;
    if (!stream.ReadVarint32(&length)) return false;
    auto limit = stream.PushLimit(length);

    if (!stream.ExpectAtEnd()) {
      uint8 peek_tag = PeekTag(&stream);
      if (peek_tag != kDelimitedTag(1) && peek_tag != kVarintTag(1)) {
        return false;
      }
      if (peek_tag == kDelimitedTag(1)) {                       // packed
        if (!stream.ExpectTag(kDelimitedTag(1))) return false;  // packed tag
        uint32 packed_length;
        if (!stream.ReadVarint32(&packed_length)) return false;
        auto packed_limit = stream.PushLimit(packed_length);

        while (!stream.ExpectAtEnd()) {
          protobuf_uint64 n;  // There is no API for int64
          if (!stream.ReadVarint64(&n)) return false;
          int64_list->push_back(static_cast<int64_t>(n));
        }

        stream.PopLimit(packed_limit);
      } else {  // non-packed
        while (!stream.ExpectAtEnd()) {
          if (!stream.ExpectTag(kVarintTag(1))) return false;
          protobuf_uint64 n;  // There is no API for int64
          if (!stream.ReadVarint64(&n)) return false;
          int64_list->push_back(static_cast<int64_t>(n));
        }
      }
    }
    stream.PopLimit(limit);
    return true;
  }

  StringPiece GetSerialized() const { return serialized_; }

 private:
  StringPiece serialized_;
};

using FeatureMapEntry = std::pair<StringPiece, Feature>;
using Example = std::vector<FeatureMapEntry>;

}  // namespace parsed

inline bool SkipExtraneousTag(protobuf::io::CodedInputStream* stream) {
  uint32 data;
  protobuf_uint64 dummy;
  switch (stream->ReadTag() & 0x7) {
    case 0:  // varint
      if (!stream->ReadVarint32(&data)) return false;
      return true;
    case 1:  // fixed64
      if (!stream->ReadLittleEndian64(&dummy)) return false;
      return true;
    case 2:  // length delimited
      if (!stream->ReadVarint32(&data)) return false;
      stream->Skip(data);
      return true;
    case 3:          // group begin
      return false;  // groups not supported.
    case 4:          // group end
      return false;  // groups not supported.
    case 5:          // fixed32
      if (!stream->ReadLittleEndian32(&data)) return false;
      return true;
  }
  return false;  // unrecognized tag type
}

bool ParseString(protobuf::io::CodedInputStream* stream, StringPiece* result);

bool ParseFeatureMapEntry(protobuf::io::CodedInputStream* stream,
                          parsed::FeatureMapEntry* feature_map_entry);

bool ParseFeatures(protobuf::io::CodedInputStream* stream,
                   parsed::Example* example);

bool ParseExample(protobuf::io::CodedInputStream* stream,
                  parsed::Example* example);

bool ParseExample(StringPiece serialized, parsed::Example* example);

using Config = FastParseExampleConfig;

// Enumeration for distinguishing feature types.
// Note: FastParseSequenceExample constructs a map that includes Type values,
// and relies on the fact that they are default-initialized to Dense.
enum class Type { Dense, Sparse, Ragged };

// Note: We use SparseBuffer for sparse, ragged, and dense_varlen features.
struct SparseBuffer {
  // Features are in one of the 3 vectors below depending on config's dtype.
  // Other 2 vectors remain empty.
  SmallVector<tstring> bytes_list;
  SmallVector<float> float_list;
  SmallVector<int64_t> int64_list;

  // Features of example i are elements with indices
  // from example_end_indices[i-1] to example_end_indices[i]-1 on the
  // appropriate xxxxx_list
  std::vector<size_t> example_end_indices;
};

struct SeededHasher {
  uint64 operator()(StringPiece s) const {
    return Hash64(s.data(), s.size(), seed);
  }
  uint64 seed{0xDECAFCAFFE};
};

// Use this in the "default" clause of switch statements when dispatching
// on a dtype variable that was checked by CheckConfigDataType():
inline void ReportUnexpectedDataType(DataType dtype) {
  DCHECK(false)
      << "Encountered unexpected DataType " << DataTypeString(dtype)
      << "in variable that should have been checked by CheckConfigDataType().";
}

template <typename T>
const SmallVector<T>& GetListFromBuffer(const SparseBuffer& buffer);

template <>
const SmallVector<int64_t>& GetListFromBuffer<int64_t>(
    const SparseBuffer& buffer);

template <>
const SmallVector<float>& GetListFromBuffer<float>(const SparseBuffer& buffer);

template <>
const SmallVector<tstring>& GetListFromBuffer<tstring>(
    const SparseBuffer& buffer);

template <typename T>
void CopyOrMoveBlock(const T* b, const T* e, T* t) {
  std::copy(b, e, t);
}
template <>
void CopyOrMoveBlock(const tstring* b, const tstring* e, tstring* t);

void CountSparseFeatures(
    const std::vector<std::vector<SparseBuffer>>& sparse_buffers, size_t d,
    size_t* total_num_features, size_t* max_num_features);

void CopySparseBufferToTensor(DataType dtype, size_t offset, SparseBuffer* src,
                              Tensor* dst);

// A struct used by FastParseSequenceExample to hold the serialized proto
// substrings for a single feature, plus some auxiliary information derived
// from those protos (such as the total value length).
struct FeatureProtos {
  // Proto substrings from each serialized SequenceExample that correspond
  // with this feature.  `protos_present` records whether the proto had a
  // value defined (even if that value is empty).
  std::vector<StringPiece> protos;
  std::vector<bool> protos_present;

  // Information derived from protos:
  size_t length;    // total length for ragged/sparse, max row length for dense.
  size_t num_rows;  // only populated for ragged sequence features.

  // Information from the config:
  Type type;  // Whether this feature is sparse, ragged, or dense.
  DataType dtype;
};

// Map from feature name to FeatureProtos for that feature.
using FeatureProtosMap = absl::flat_hash_map<StringPiece, FeatureProtos>;

string ExampleName(const gtl::ArraySlice<tstring> example_names, int n);

// Return the number of bytes elements parsed, or -1 on error. If out is null,
// this method simply counts the number of elements without any copying.
inline int ParseBytesFeature(protobuf::io::CodedInputStream* stream,
                             tstring* out) {
  int num_elements = 0;
  uint32 length;
  if (!stream->ExpectTag(kDelimitedTag(1)) || !stream->ReadVarint32(&length)) {
    return -1;
  }
  if (length > 0) {
    auto limit = stream->PushLimit(length);
    while (!stream->ExpectAtEnd()) {
      uint32 bytes_length;
      if (!stream->ExpectTag(kDelimitedTag(1)) ||
          !stream->ReadVarint32(&bytes_length)) {
        return -1;
      }
      if (out == nullptr) {
        stream->Skip(bytes_length);
      } else {
        out->resize_uninitialized(bytes_length);
        if (!stream->ReadRaw(out->data(), bytes_length)) {
          return -1;
        }
        out++;
      }
      num_elements++;
    }
    stream->PopLimit(limit);
  }
  return num_elements;
}

inline void PadFloatFeature(int num_to_pad, float* out) {
  for (int i = 0; i < num_to_pad; i++) {
    *out++ = 0.0;
  }
}

inline void PadInt64Feature(int num_to_pad, int64_t* out) {
  for (int i = 0; i < num_to_pad; i++) {
    *out++ = 0;
  }
}

// Return the number of float elements parsed, or -1 on error. If out is null,
// this method simply counts the number of elements without any copying.
inline int ParseFloatFeature(protobuf::io::CodedInputStream* stream,
                             float* out) {
  int num_elements = 0;
  uint32 length;
  if (!stream->ExpectTag(kDelimitedTag(2)) || !stream->ReadVarint32(&length)) {
    return -1;
  }
  if (length > 0) {
    auto limit = stream->PushLimit(length);
    uint8 peek_tag = PeekTag(stream);
    if (peek_tag == kDelimitedTag(1)) {  // packed
      uint32 packed_length;
      if (!stream->ExpectTag(kDelimitedTag(1)) ||
          !stream->ReadVarint32(&packed_length)) {
        return -1;
      }
      auto packed_limit = stream->PushLimit(packed_length);
      while (!stream->ExpectAtEnd()) {
        uint32 buffer32;
        if (!stream->ReadLittleEndian32(&buffer32)) {
          return -1;
        }
        if (out != nullptr) {
          *out++ = absl::bit_cast<float>(buffer32);
        }
        num_elements++;
      }
      stream->PopLimit(packed_limit);
    } else if (peek_tag == kFixed32Tag(1)) {
      while (!stream->ExpectAtEnd()) {
        uint32 buffer32;
        if (!stream->ExpectTag(kFixed32Tag(1)) ||
            !stream->ReadLittleEndian32(&buffer32)) {
          return -1;
        }
        if (out != nullptr) {
          *out++ = absl::bit_cast<float>(buffer32);
        }
        num_elements++;
      }
    } else {
      // Unknown tag.
      return -1;
    }
    stream->PopLimit(limit);
  }
  return num_elements;
}

// Return the number of int64 elements parsed, or -1 on error. If out is null,
// this method simply counts the number of elements without any copying.
inline int ParseInt64Feature(protobuf::io::CodedInputStream* stream,
                             int64_t* out) {
  int num_elements = 0;
  uint32 length;
  if (!stream->ExpectTag(kDelimitedTag(3)) || !stream->ReadVarint32(&length)) {
    return -1;
  }
  if (length > 0) {
    auto limit = stream->PushLimit(length);
    uint8 peek_tag = PeekTag(stream);
    if (peek_tag == kDelimitedTag(1)) {  // packed
      uint32 packed_length;
      if (!stream->ExpectTag(kDelimitedTag(1)) ||
          !stream->ReadVarint32(&packed_length)) {
        return -1;
      }
      auto packed_limit = stream->PushLimit(packed_length);
      while (!stream->ExpectAtEnd()) {
        protobuf_uint64 n;  // There is no API for int64
        if (!stream->ReadVarint64(&n)) {
          return -1;
        }
        if (out != nullptr) {
          *out++ = n;
        }
        num_elements++;
      }
      stream->PopLimit(packed_limit);
    } else if (peek_tag == kVarintTag(1)) {
      while (!stream->ExpectAtEnd()) {
        protobuf_uint64 n;  // There is no API for int64
        if (!stream->ExpectTag(kVarintTag(1)) || !stream->ReadVarint64(&n)) {
          return -1;
        }
        if (out != nullptr) {
          *out++ = n;
        }
        num_elements++;
      }
    } else {
      // Unknown tag.
      return -1;
    }
    stream->PopLimit(limit);
  }
  return num_elements;
}

// Parses the next feature on `stream` into `out` starting at `out_offset`.
// Updates `out_offset`, and returns the number of values added.
// Returns -1 if the next feature on `stream` doesn't match `dtype`.
inline int ParseFeature(DataType dtype, protobuf::io::CodedInputStream* stream,
                        Tensor* out, size_t* out_offset) {
  int delta;
  switch (dtype) {
    case DT_STRING:
      delta =
          ParseBytesFeature(stream, out->flat<tstring>().data() + *out_offset);
      break;
    case DT_FLOAT:
      delta =
          ParseFloatFeature(stream, out->flat<float>().data() + *out_offset);
      break;
    case DT_INT64:
      delta =
          ParseInt64Feature(stream, out->flat<int64_t>().data() + *out_offset);
      break;
    default:
      ReportUnexpectedDataType(dtype);
      delta = 0;
  }
  if (delta > 0) {
    *out_offset += delta;
  }
  return delta;
}

// Returns the length of the next feature on `stream`.
// Returns -1 if the next feature on `stream` doesn't match `dtype`.
inline int GetFeatureLength(DataType dtype,
                            protobuf::io::CodedInputStream* stream) {
  switch (dtype) {
    case DT_STRING:
      return ParseBytesFeature(stream, nullptr);
    case DT_FLOAT:
      return ParseFloatFeature(stream, nullptr);
    case DT_INT64:
      return ParseInt64Feature(stream, nullptr);
    default:
      ReportUnexpectedDataType(dtype);
      return -1;
  }
}

inline DataType ParseDataType(protobuf::io::CodedInputStream* stream) {
  uint8 peek_tag = PeekTag(stream);
  switch (peek_tag) {
    case kDelimitedTag(1):
      return DT_STRING;
    case kDelimitedTag(2):
      return DT_FLOAT;
    case kDelimitedTag(3):
      return DT_INT64;
    default:
      return DT_INVALID;
  }
}

inline bool SkipEmptyFeature(protobuf::io::CodedInputStream* stream,
                             DataType dtype) {
  switch (dtype) {
    case DT_STRING:
      if (!stream->ExpectTag(kDelimitedTag(1))) {
        return false;
      }
      break;
    case DT_FLOAT:
      if (!stream->ExpectTag(kDelimitedTag(2))) {
        return false;
      }
      break;
    case DT_INT64:
      if (!stream->ExpectTag(kDelimitedTag(3))) {
        return false;
      }
      break;
    default:
      return false;
  }
  uint32 length;
  return stream->ReadVarint32(&length) && length == 0;
}

}  // namespace example
}  // namespace tensorflow

#endif  // TENSORFLOW_LITE_KERNELS_PARSE_EXAMPLE_EXAMPLE_PROTO_FAST_PARSING_H_
