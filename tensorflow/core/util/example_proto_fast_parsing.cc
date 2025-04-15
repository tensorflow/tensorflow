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
#include "tensorflow/core/util/example_proto_fast_parsing.h"

#include <algorithm>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/presized_cuckoo_map.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {
namespace example {

namespace {

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

uint8 PeekTag(protobuf::io::CodedInputStream* stream) {
  DCHECK(stream != nullptr);
  const void* ptr;
  int size;
  if (!stream->GetDirectBufferPointer(&ptr, &size)) return 0;
  return *static_cast<const uint8*>(ptr);
}

constexpr uint8 kVarintTag(uint32 tag) { return (tag << 3) | 0; }
constexpr uint8 kDelimitedTag(uint32 tag) { return (tag << 3) | 2; }
constexpr uint8 kFixed32Tag(uint32 tag) { return (tag << 3) | 5; }

namespace parsed {

// ParseDataType has to be called first, then appropriate ParseZzzzList.
class Feature {
 public:
  Feature() = default;
  explicit Feature(absl::string_view serialized) : serialized_(serialized) {}

  absl::Status ParseDataType(DataType* dtype) {
    DCHECK(dtype != nullptr);
    if (serialized_.empty()) {
      *dtype = DT_INVALID;
      return absl::OkStatus();
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
    return absl::OkStatus();
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

  absl::string_view GetSerialized() const { return serialized_; }

 private:
  // TODO(lew): Pair of uint8* would be more natural.
  absl::string_view serialized_;
};

using FeatureMapEntry = std::pair<absl::string_view, Feature>;
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

bool ParseString(protobuf::io::CodedInputStream* stream,
                 absl::string_view* result) {
  DCHECK(stream != nullptr);
  DCHECK(result != nullptr);
  uint32 length;
  if (!stream->ReadVarint32(&length)) return false;
  if (length == 0) {
    *result = absl::string_view(nullptr, 0);
    return true;
  }
  const void* stream_alias;
  int stream_size;
  if (!stream->GetDirectBufferPointer(&stream_alias, &stream_size)) {
    return false;
  }
  if (static_cast<uint32>(stream_size) < length) return false;
  *result = absl::string_view(static_cast<const char*>(stream_alias), length);
  stream->Skip(length);
  return true;
}

bool ParseFeatureMapEntry(protobuf::io::CodedInputStream* stream,
                          parsed::FeatureMapEntry* feature_map_entry) {
  DCHECK(stream != nullptr);
  DCHECK(feature_map_entry != nullptr);
  uint32 length;
  if (!stream->ReadVarint32(&length)) return false;
  auto limit = stream->PushLimit(length);

  // Protobufs allow an arbitrary order for the key and value fields.
  for (int n = 0; n < 2; ++n) {
    const uint32_t tag = stream->ReadTag();
    switch (tag) {
      case kDelimitedTag(1):
        if (!ParseString(stream, &feature_map_entry->first)) return false;
        break;

      case kDelimitedTag(2): {
        absl::string_view feature_string_piece;
        if (!ParseString(stream, &feature_string_piece)) return false;
        feature_map_entry->second = parsed::Feature(feature_string_piece);
        break;
      }

      default:
        return false;
    }
  }

  if (!stream->ExpectAtEnd()) return false;
  stream->PopLimit(limit);
  return true;
}

bool ParseFeatures(protobuf::io::CodedInputStream* stream,
                   parsed::Example* example) {
  DCHECK(stream != nullptr);
  DCHECK(example != nullptr);
  uint32 length;
  if (!stream->ReadVarint32(&length)) return false;
  auto limit = stream->PushLimit(length);
  while (!stream->ExpectAtEnd()) {
    parsed::FeatureMapEntry feature_map_entry;
    if (!stream->ExpectTag(kDelimitedTag(1))) return false;
    if (!ParseFeatureMapEntry(stream, &feature_map_entry)) return false;
    example->push_back(std::move(feature_map_entry));
  }
  stream->PopLimit(limit);
  return true;
}

bool ParseExample(protobuf::io::CodedInputStream* stream,
                  parsed::Example* example) {
  DCHECK(stream != nullptr);
  DCHECK(example != nullptr);
  // Loop over the input stream which may contain multiple serialized Example
  // protos merged together as strings. This behavior is consistent with Proto's
  // ParseFromString when string representations are concatenated.
  while (!stream->ExpectAtEnd()) {
    if (!stream->ExpectTag(kDelimitedTag(1))) {
      if (!SkipExtraneousTag(stream)) return false;
    } else {
      if (!ParseFeatures(stream, example)) return false;
    }
  }
  return true;
}

bool ParseExample(absl::string_view serialized, parsed::Example* example) {
  DCHECK(example != nullptr);
  protobuf::io::CodedInputStream stream(
      reinterpret_cast<const uint8*>(serialized.data()), serialized.size());
  EnableAliasing(&stream);
  return ParseExample(&stream, example);
}

}  // namespace

bool TestFastParse(const string& serialized, Example* example) {
  DCHECK(example != nullptr);
  parsed::Example parsed_example;
  if (!ParseExample(serialized, &parsed_example)) return false;
  auto& features = *example->mutable_features();
  size_t parsed_example_size = parsed_example.size();
  for (size_t i = 0; i < parsed_example_size; ++i) {
    // This is a logic that standard protobuf parsing is implementing.
    // I.e. last entry in the map overwrites all the previous ones.
    parsed::FeatureMapEntry& name_and_feature =
        parsed_example[parsed_example_size - i - 1];
    string name(name_and_feature.first);
    if ((*features.mutable_feature()).count(name) > 0) continue;

    auto& value = (*features.mutable_feature())[name];
    DataType dtype;
    if (!name_and_feature.second.ParseDataType(&dtype).ok()) return false;
    switch (dtype) {
      case DT_INVALID:
        break;
      case DT_STRING: {
        SmallVector<tstring> list;
        if (!name_and_feature.second.ParseBytesList(&list)) return false;
        auto* result_list = value.mutable_bytes_list();
        for (auto& bytes : list) {
          result_list->add_value(bytes.data(), bytes.size());
        }
        break;
      }
      case DT_FLOAT: {
        SmallVector<float> list;
        if (!name_and_feature.second.ParseFloatList(&list)) return false;
        auto* result_list = value.mutable_float_list();
        for (float f : list) {
          result_list->add_value(f);
        }
        break;
      }
      case DT_INT64: {
        SmallVector<int64_t> list;
        if (!name_and_feature.second.ParseInt64List(&list)) return false;
        auto* result_list = value.mutable_int64_list();
        for (int64_t i : list) {
          result_list->add_value(i);
        }
        break;
      }
      default:
        LOG(FATAL) << "Should not happen.";
    }
  }
  return true;
}

// -----------------------------------------------------------------------------

namespace {

using Config = FastParseExampleConfig;

void ParallelFor(const std::function<void(size_t)>& f, size_t n,
                 thread::ThreadPool* thread_pool) {
  if (n == 0) return;
  if (thread_pool == nullptr) {
    for (size_t i = 0; i < n; ++i) {
      f(i);
    }
  } else {
    BlockingCounter counter(n - 1);
    for (size_t i = 1; i < n; ++i) {
      thread_pool->Schedule([i, &f, &counter] {
        f(i);
        counter.DecrementCount();
      });
    }
    f(0);
    counter.Wait();
  }
}

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
  uint64 operator()(absl::string_view s) const {
    return Hash64(s.data(), s.size(), seed);
  }
  uint64 seed{0xDECAFCAFFE};
};

void LogDenseFeatureDataLoss(absl::string_view feature_name) {
  LOG(WARNING) << "Data loss! Feature '" << feature_name
               << "' is present in multiple concatenated "
                  "tf.Examples. Ignoring all but last one.";
  static auto* duplicated_dense_feature = monitoring::Counter<0>::New(
      "/tensorflow/core/util/example_proto_fast_parsing/"
      "duplicated_dense_feature",
      "Dense feature appears twice in a tf.Example");
  duplicated_dense_feature->GetCell()->IncrementBy(1);
}

void LogSparseFeatureDataLoss(absl::string_view feature_name) {
  LOG(WARNING) << "Data loss! Feature '" << feature_name
               << "' is present in multiple concatenated "
                  "tf.Examples. Ignoring all but last one.";
  static auto* duplicated_sparse_feature = monitoring::Counter<0>::New(
      "/tensorflow/core/util/example_proto_fast_parsing/"
      "duplicated_sparse_feature",
      "Sparse feature appears twice in a tf.Example");
  duplicated_sparse_feature->GetCell()->IncrementBy(1);
}

absl::Status FastParseSerializedExample(
    const tstring& serialized_example, const tstring& example_name,
    const size_t example_index, const Config& config,
    const PresizedCuckooMap<std::pair<size_t, Type>>& config_index,
    SeededHasher hasher, std::vector<Tensor>* output_dense,
    std::vector<SparseBuffer>* output_varlen_dense,
    std::vector<SparseBuffer>* output_sparse,
    std::vector<SparseBuffer>* output_ragged,
    PerExampleFeatureStats* output_stats) {
  DCHECK(output_dense != nullptr);
  DCHECK(output_sparse != nullptr);
  DCHECK(output_ragged != nullptr);
  parsed::Example parsed_example;
  if (!ParseExample(serialized_example, &parsed_example)) {
    return errors::InvalidArgument("Could not parse example input, value: '",
                                   serialized_example, "'");
  }
  std::vector<int64_t> sparse_feature_last_example(config.sparse.size(), -1);
  std::vector<int64_t> dense_feature_last_example(config.dense.size(), -1);
  std::vector<int64_t> ragged_feature_last_example(config.ragged.size(), -1);

  // Handle features present in the example.
  const size_t parsed_example_size = parsed_example.size();

  if (output_stats) {
    // TODO(b/111553342): This may over-count the number of features if there
    // are duplicate keys in the feature map. Consider deduplicating the keys
    // before computing the count.
    output_stats->features_count = parsed_example_size;
  }

  for (size_t i = 0; i < parsed_example_size; ++i) {
    // This is a logic that standard protobuf parsing is implementing.
    // I.e. last entry in the map overwrites all the previous ones.
    parsed::FeatureMapEntry& name_and_feature =
        parsed_example[parsed_example_size - i - 1];

    const absl::string_view feature_name = name_and_feature.first;
    parsed::Feature& feature = name_and_feature.second;

    std::pair<size_t, Type> d_and_type;
    uint64 h = hasher(feature_name);
    if (!config_index.Find(h, &d_and_type)) continue;

    size_t d = d_and_type.first;
    bool is_dense = d_and_type.second == Type::Dense;
    bool is_ragged = d_and_type.second == Type::Ragged;

    {
      // Testing for PresizedCuckooMap collision.
      // TODO(lew): Use dense_hash_map and avoid this and hasher creation.
      const tstring& config_feature_name =
          is_dense ? config.dense[d].feature_name
                   : (is_ragged ? config.ragged[d].feature_name
                                : config.sparse[d].feature_name);
      if (feature_name != config_feature_name) continue;
    }

    auto example_error = [&](absl::string_view suffix) {
      return errors::InvalidArgument("Name: ", example_name,
                                     ", Key: ", feature_name,
                                     ", Index: ", example_index, ".  ", suffix);
    };

    auto parse_error = [&] {
      return example_error("Can't parse serialized Example.");
    };

    DataType example_dtype;
    TF_RETURN_IF_ERROR(feature.ParseDataType(&example_dtype));

    if (is_dense) {
      if (example_dtype == DT_INVALID) continue;

      // If feature was already visited, skip.
      // Compare comment at the beginning of the loop.
      if (dense_feature_last_example[d] == example_index) {
        LogDenseFeatureDataLoss(feature_name);
        continue;
      }
      dense_feature_last_example[d] = example_index;

      if (example_dtype != config.dense[d].dtype) {
        return example_error(strings::StrCat(
            "Data types don't match. Data type: ",
            DataTypeString(example_dtype),
            " but expected type: ", DataTypeString(config.dense[d].dtype)));
      }
      if (!config.dense[d].variable_length) {
        Tensor& out = (*output_dense)[d];

        const std::size_t num_elements = config.dense[d].elements_per_stride;
        if (output_stats) {
          // TODO(b/111553342): If desirable, we could add support for counting
          // elements in the features that aren't parsed, but this could add
          // considerable runtime cost.
          output_stats->feature_values_count += num_elements;
        }

        const std::size_t offset = example_index * num_elements;

        auto shape_error = [&](size_t size, absl::string_view type_str) {
          return example_error(strings::StrCat(
              "Number of ", type_str,
              " values != expected.  "
              "Values size: ",
              size,
              " but output shape: ", config.dense[d].shape.DebugString()));
        };

        switch (config.dense[d].dtype) {
          case DT_INT64: {
            auto out_p = out.flat<int64_t>().data() + offset;
            LimitedArraySlice<int64_t> slice(out_p, num_elements);
            if (!feature.ParseInt64List(&slice)) return parse_error();
            if (slice.EndDistance() != 0) {
              return shape_error(num_elements - slice.EndDistance(), "int64");
            }
            break;
          }
          case DT_FLOAT: {
            auto out_p = out.flat<float>().data() + offset;
            LimitedArraySlice<float> slice(out_p, num_elements);
            if (!feature.ParseFloatList(&slice)) return parse_error();
            if (slice.EndDistance() != 0) {
              return shape_error(num_elements - slice.EndDistance(), "float");
            }
            break;
          }
          case DT_STRING: {
            auto out_p = out.flat<tstring>().data() + offset;
            LimitedArraySlice<tstring> slice(out_p, num_elements);
            if (!feature.ParseBytesList(&slice)) return parse_error();
            if (slice.EndDistance() != 0) {
              return shape_error(num_elements - slice.EndDistance(), "bytes");
            }
            break;
          }
          default:
            LOG(FATAL) << "Should not happen.";
        }
      } else {  // if variable length
        SparseBuffer& out = (*output_varlen_dense)[d];

        const std::size_t num_elements = config.dense[d].elements_per_stride;

        if (example_dtype != DT_INVALID &&
            example_dtype != config.dense[d].dtype) {
          return example_error(strings::StrCat(
              "Data types don't match. ",
              "Expected type: ", DataTypeString(config.dense[d].dtype)));
        }

        auto shape_error = [&](size_t size, absl::string_view type_str) {
          return example_error(strings::StrCat(
              "Number of ", type_str,
              " values is not a multiple of stride length. Saw ", size,
              " values but output shape is: ",
              config.dense[d].shape.DebugString()));
        };

        switch (config.dense[d].dtype) {
          case DT_INT64: {
            if (example_dtype != DT_INVALID) {
              if (!feature.ParseInt64List(&out.int64_list)) {
                return parse_error();
              }
              if (out.int64_list.size() % num_elements != 0) {
                return shape_error(out.int64_list.size(), "int64");
              }
            }
            out.example_end_indices.push_back(out.int64_list.size());
            break;
          }
          case DT_FLOAT: {
            if (example_dtype != DT_INVALID) {
              if (!feature.ParseFloatList(&out.float_list)) {
                return parse_error();
              }
              if (out.float_list.size() % num_elements != 0) {
                return shape_error(out.float_list.size(), "float");
              }
            }
            out.example_end_indices.push_back(out.float_list.size());
            break;
          }
          case DT_STRING: {
            if (example_dtype != DT_INVALID) {
              if (!feature.ParseBytesList(&out.bytes_list)) {
                return parse_error();
              }
              if (out.bytes_list.size() % num_elements != 0) {
                return shape_error(out.bytes_list.size(), "bytes");
              }
            }
            out.example_end_indices.push_back(out.bytes_list.size());
            break;
          }
          default:
            LOG(FATAL) << "Should not happen.";
        }

        if (output_stats) {
          // Use `out.example_end_indices` to determine the feature-value count
          // for this feature, because the preceding switch statement pushes
          // the length of the appropriate feature list to that vector.
          // TODO(b/111553342): If desirable, we could add support for counting
          // elements in the features that aren't parsed, but this could add
          // considerable runtime cost.
          const size_t out_examples_count = out.example_end_indices.size();
          if (out_examples_count == 1) {
            output_stats->feature_values_count += out.example_end_indices[0];
          } else {
            output_stats->feature_values_count +=
                out.example_end_indices[out_examples_count - 1] -
                out.example_end_indices[out_examples_count - 2];
          }
        }
      }
    } else {
      // Feature is sparse or ragged.
      auto& last_example =
          is_ragged ? ragged_feature_last_example : sparse_feature_last_example;

      // If feature was already visited, skip.
      // Compare comment at the beginning of the loop.
      if (last_example[d] == example_index) {
        LogSparseFeatureDataLoss(feature_name);
        continue;
      }
      last_example[d] = example_index;

      // Handle sparse features.
      SparseBuffer& out = is_ragged ? (*output_ragged)[d] : (*output_sparse)[d];
      DataType feature_dtype =
          is_ragged ? config.ragged[d].dtype : config.sparse[d].dtype;
      if (example_dtype != DT_INVALID && example_dtype != feature_dtype) {
        return example_error(
            strings::StrCat("Data types don't match. ",
                            "Expected type: ", DataTypeString(feature_dtype),
                            ", Actual type: ", DataTypeString(example_dtype)));
      }

      switch (feature_dtype) {
        case DT_INT64: {
          if (example_dtype != DT_INVALID) {
            if (!feature.ParseInt64List(&out.int64_list)) {
              return parse_error();
            }
          }
          out.example_end_indices.push_back(out.int64_list.size());
          break;
        }
        case DT_FLOAT: {
          if (example_dtype != DT_INVALID) {
            if (!feature.ParseFloatList(&out.float_list)) {
              return parse_error();
            }
          }
          out.example_end_indices.push_back(out.float_list.size());
          break;
        }
        case DT_STRING: {
          if (example_dtype != DT_INVALID) {
            if (!feature.ParseBytesList(&out.bytes_list)) {
              return parse_error();
            }
          }
          out.example_end_indices.push_back(out.bytes_list.size());
          break;
        }
        default:
          LOG(FATAL) << "Should not happen.";
      }

      if (output_stats) {
        // Use `out.example_end_indices` to determine the feature-value count
        // for this feature, because the preceding switch statement pushes
        // the length of the appropriate feature list to that vector.
        // TODO(b/111553342): If desirable, we could add support for counting
        // elements in the features that aren't parsed, but this could add
        // considerable runtime cost.
        const size_t out_examples_count = out.example_end_indices.size();
        if (out_examples_count == 1) {
          output_stats->feature_values_count += out.example_end_indices[0];
        } else {
          output_stats->feature_values_count +=
              out.example_end_indices[out_examples_count - 1] -
              out.example_end_indices[out_examples_count - 2];
        }
      }
    }
  }

  // Handle missing dense features for fixed strides.
  for (size_t d = 0; d < config.dense.size(); ++d) {
    if (config.dense[d].variable_length) continue;
    if (dense_feature_last_example[d] == example_index) continue;
    if (config.dense[d].default_value.NumElements() == 0) {
      return errors::InvalidArgument(
          "Name: ", example_name, ", Feature: ", config.dense[d].feature_name,
          " (data type: ", DataTypeString(config.dense[d].dtype), ")",
          " is required but could not be found.");
    }
    const Tensor& in = config.dense[d].default_value;
    Tensor& out = (*output_dense)[d];
    const std::size_t num_elements = in.shape().num_elements();
    const std::size_t offset = example_index * num_elements;

    switch (config.dense[d].dtype) {
      case DT_INT64: {
        std::copy_n(in.flat<int64_t>().data(), num_elements,
                    out.flat<int64_t>().data() + offset);
        break;
      }
      case DT_FLOAT: {
        std::copy_n(in.flat<float>().data(), num_elements,
                    out.flat<float>().data() + offset);
        break;
      }
      case DT_STRING: {
        std::copy_n(in.flat<tstring>().data(), num_elements,
                    out.flat<tstring>().data() + offset);
        break;
      }
      default:
        LOG(FATAL) << "Should not happen.";
    }
  }

  // Handle missing varlen dense features.
  for (size_t d = 0; d < config.dense.size(); ++d) {
    if (!config.dense[d].variable_length) continue;
    if (dense_feature_last_example[d] == example_index) continue;
    SparseBuffer& out = (*output_varlen_dense)[d];
    size_t prev_example_end_index =
        out.example_end_indices.empty() ? 0 : out.example_end_indices.back();
    out.example_end_indices.push_back(prev_example_end_index);
  }

  // Handle missing sparse features.
  for (size_t d = 0; d < config.sparse.size(); ++d) {
    if (sparse_feature_last_example[d] == example_index) continue;
    SparseBuffer& out = (*output_sparse)[d];
    size_t prev_example_end_index =
        out.example_end_indices.empty() ? 0 : out.example_end_indices.back();
    out.example_end_indices.push_back(prev_example_end_index);
  }

  // Handle missing ragged features.
  for (size_t d = 0; d < config.ragged.size(); ++d) {
    if (ragged_feature_last_example[d] == example_index) continue;
    SparseBuffer& out = (*output_ragged)[d];
    size_t prev_example_end_index =
        out.example_end_indices.empty() ? 0 : out.example_end_indices.back();
    out.example_end_indices.push_back(prev_example_end_index);
  }

  return absl::OkStatus();
}

absl::Status CheckConfigDataType(DataType dtype) {
  switch (dtype) {
    case DT_INT64:
    case DT_FLOAT:
    case DT_STRING:
      return absl::OkStatus();
    default:
      return errors::InvalidArgument("Invalid config dtype: ",
                                     DataTypeString(dtype));
  }
}

// Use this in the "default" clause of switch statements when dispatching
// on a dtype variable that was checked by CheckConfigDataType():
inline void ReportUnexpectedDataType(DataType dtype) {
  DCHECK(false)
      << "Encountered unexpected DataType " << DataTypeString(dtype)
      << "in variable that should have been checked by CheckConfigDataType().";
}

absl::Status CheckConfigDataTypes(const Config& config) {
  // Check config so we can safely CHECK(false) in switches on config.*.dtype
  for (auto& c : config.sparse) {
    TF_RETURN_IF_ERROR(CheckConfigDataType(c.dtype));
  }
  for (auto& c : config.dense) {
    TF_RETURN_IF_ERROR(CheckConfigDataType(c.dtype));
  }
  for (auto& c : config.ragged) {
    TF_RETURN_IF_ERROR(CheckConfigDataType(c.dtype));
    if (!(c.splits_dtype == DT_INT32 || c.splits_dtype == DT_INT64)) {
      return errors::InvalidArgument("Invalid ragged_split_type: ",
                                     DataTypeString(c.splits_dtype));
    }
  }
  return absl::OkStatus();
}

template <typename T>
const SmallVector<T>& GetListFromBuffer(const SparseBuffer& buffer);

template <>
const SmallVector<int64_t>& GetListFromBuffer<int64_t>(
    const SparseBuffer& buffer) {
  return buffer.int64_list;
}
template <>
const SmallVector<float>& GetListFromBuffer<float>(const SparseBuffer& buffer) {
  return buffer.float_list;
}
template <>
const SmallVector<tstring>& GetListFromBuffer<tstring>(
    const SparseBuffer& buffer) {
  return buffer.bytes_list;
}

template <typename T>
void CopyOrMoveBlock(const T* b, const T* e, T* t) {
  std::copy(b, e, t);
}
template <>
void CopyOrMoveBlock(const tstring* b, const tstring* e, tstring* t) {
  std::move(b, e, t);
}

template <typename T>
void FillAndCopyVarLen(
    const int d, const size_t num_elements,
    const size_t num_elements_per_minibatch, const Config& config,
    const std::vector<std::vector<SparseBuffer>>& varlen_dense_buffers,
    Tensor* values) {
  const Tensor& default_value = config.dense[d].default_value;

  // Copy-fill the tensors (creating the zero/fill-padding)
  std::fill(values->flat<T>().data(), values->flat<T>().data() + num_elements,
            default_value.flat<T>()(0));

  // Data is [batch_size, max_num_elements, data_stride_size]
  //   and num_elements_per_minibatch = max_num_elements * data_stride_size
  auto data = values->flat<T>().data();

  // Iterate over minibatch elements
  for (size_t i = 0; i < varlen_dense_buffers.size(); ++i) {
    const SparseBuffer& buffer = varlen_dense_buffers[i][d];
    // Number of examples being stored in this buffer
    const auto& end_indices = buffer.example_end_indices;
    const size_t examples_in_buffer = end_indices.size();
    // const size_t stride_size = config.dense[d].elements_per_stride;

    const auto& list = GetListFromBuffer<T>(buffer);
    auto list_ptr = list.begin();

    size_t elements_tally = 0;
    // Iterate through all the examples stored in this buffer.
    for (size_t j = 0; j < examples_in_buffer; ++j) {
      // Number of elements stored for this example.
      const size_t num_elems = end_indices[j] - elements_tally;
      CopyOrMoveBlock(list_ptr, list_ptr + num_elems, data);
      // Move forward this many elements in the varlen buffer.
      list_ptr += num_elems;
      // Move forward to the next minibatch entry in the values output.
      data += num_elements_per_minibatch;
      elements_tally = end_indices[j];
    }
    DCHECK(elements_tally == list.size());
  }
}

// Thin vector like interface wrapper around a Tensor. This enable us to
// directly populate a tensor during parsing instead of having to first create a
// vactor and then copy the data over.
template <typename T>
class TensorVector {
 public:
  using value_type = T;

  const Tensor& tensor() {
    if (!tensor_.has_value()) {
      resize(0);
    }
    return *tensor_;
  }

  int64_t size() const {
    return tensor_.has_value() ? tensor_->NumElements() : 0;
  }
  void resize(int64_t new_size) {
    DCHECK(!tensor_.has_value());
    tensor_ = Tensor(DataTypeToEnum<T>::v(), TensorShape({new_size}));
    data_ = tensor_->flat<T>().data();
  }
  T* data() { return data_; }
  const T* data() const { return data_; }

 private:
  // Use absl::optional to avoid calling the default constructor of Tensor
  // unnecessarily.
  std::optional<Tensor> tensor_;

  // Cached pointer to the raw data inside the tensor.
  T* data_ = nullptr;
};

void CountSparseFeatures(
    const std::vector<std::vector<SparseBuffer>>& sparse_buffers, size_t d,
    size_t* total_num_features, size_t* max_num_features) {
  for (auto& sparse_values_tmp : sparse_buffers) {
    const std::vector<size_t>& end_indices =
        sparse_values_tmp[d].example_end_indices;
    *total_num_features += end_indices.back();
    *max_num_features = std::max(*max_num_features, end_indices[0]);
    for (size_t i = 1; i < end_indices.size(); ++i) {
      size_t example_size = end_indices[i] - end_indices[i - 1];
      *max_num_features = std::max(*max_num_features, example_size);
    }
  }
}

void CopySparseBufferToTensor(DataType dtype, size_t offset, SparseBuffer* src,
                              Tensor* dst) {
  switch (dtype) {
    case DT_INT64: {
      std::copy(src->int64_list.begin(), src->int64_list.end(),
                dst->flat<int64_t>().data() + offset);
      break;
    }
    case DT_FLOAT: {
      std::copy(src->float_list.begin(), src->float_list.end(),
                dst->flat<float>().data() + offset);
      break;
    }
    case DT_STRING: {
      std::move(src->bytes_list.begin(), src->bytes_list.end(),
                dst->flat<tstring>().data() + offset);
      break;
    }
    default:
      ReportUnexpectedDataType(dtype);
  }
}

}  // namespace

absl::Status FastParseExample(const Config& config,
                              absl::Span<const tstring> serialized,
                              absl::Span<const tstring> example_names,
                              thread::ThreadPool* thread_pool, Result* result) {
  DCHECK(result != nullptr);
  // Check config so we can safely CHECK(false) in switches on config.*.dtype
  TF_RETURN_IF_ERROR(CheckConfigDataTypes(config));

  if (config.collect_feature_stats) {
    result->feature_stats.resize(serialized.size());
  }

  size_t config_size =
      config.dense.size() + config.sparse.size() + config.ragged.size();
  SeededHasher hasher;
  // Build config index.
  PresizedCuckooMap<std::pair<size_t, Type>> config_index(config_size);
  bool ok = true;
  for (size_t i = 0; i < 1000; ++i) {
    for (size_t d = 0; d < config.dense.size(); ++d) {
      ok &= config_index.InsertUnique(hasher(config.dense[d].feature_name),
                                      {d, Type::Dense});
    }
    for (size_t d = 0; d < config.sparse.size(); ++d) {
      ok &= config_index.InsertUnique(hasher(config.sparse[d].feature_name),
                                      {d, Type::Sparse});
    }
    for (size_t d = 0; d < config.ragged.size(); ++d) {
      ok &= config_index.InsertUnique(hasher(config.ragged[d].feature_name),
                                      {d, Type::Ragged});
    }
    if (ok) break;
    LOG(WARNING) << "Collision found. This should happen only if you have "
                    "around 2^32 entries in your config.";
    hasher.seed++;
    config_index.Clear(config_size);
    ok = true;
  }
  if (!ok) {
    return errors::Internal(
        "Could not avoid collision. This should not happen.");
  }

  // Allocate dense output for fixed length dense values
  // (variable-length dense and sparse and ragged have to be buffered).
  std::vector<Tensor> fixed_dense_values(config.dense.size());
  for (size_t d = 0; d < config.dense.size(); ++d) {
    if (config.dense[d].variable_length) continue;
    TensorShape out_shape;
    out_shape.AddDim(serialized.size());
    for (const int64_t dim : config.dense[d].shape.dim_sizes()) {
      out_shape.AddDim(dim);
    }
    fixed_dense_values[d] = Tensor(config.dense[d].dtype, out_shape);
  }

  // This parameter affects performance in a big and data-dependent way.
  const size_t kMiniBatchSizeBytes = 50000;

  // Calculate number of minibatches.
  // In main regime make each minibatch around kMiniBatchSizeBytes bytes.
  // Apply 'special logic' below for small and big regimes.
  const size_t num_minibatches = [&] {
    size_t result = 0;
    size_t minibatch_bytes = 0;
    for (size_t i = 0; i < serialized.size(); i++) {
      if (minibatch_bytes == 0) {  // start minibatch
        result++;
      }
      minibatch_bytes += serialized[i].size() + 1;
      if (minibatch_bytes > kMiniBatchSizeBytes) {
        minibatch_bytes = 0;
      }
    }
    // 'special logic'
    const size_t min_minibatches = std::min<size_t>(8, serialized.size());
    const size_t max_minibatches = 64;
    return std::max<size_t>(min_minibatches,
                            std::min<size_t>(max_minibatches, result));
  }();

  auto first_example_of_minibatch = [&](size_t minibatch) -> size_t {
    return (serialized.size() * minibatch) / num_minibatches;
  };

  // TODO(lew): A big performance low-hanging fruit here is to improve
  //   num_minibatches calculation to take into account actual amount of work
  //   needed, as the size in bytes is not perfect. Linear combination of
  //   size in bytes and average number of features per example is promising.
  //   Even better: measure time instead of estimating, but this is too costly
  //   in small batches.
  //   Maybe accept outside parameter #num_minibatches?

  // Do minibatches in parallel.
  std::vector<std::vector<SparseBuffer>> sparse_buffers(num_minibatches);
  std::vector<std::vector<SparseBuffer>> varlen_dense_buffers(num_minibatches);
  std::vector<std::vector<SparseBuffer>> ragged_buffers(num_minibatches);
  std::vector<absl::Status> status_of_minibatch(num_minibatches);
  auto ProcessMiniBatch = [&](size_t minibatch) {
    sparse_buffers[minibatch].resize(config.sparse.size());
    varlen_dense_buffers[minibatch].resize(config.dense.size());
    ragged_buffers[minibatch].resize(config.ragged.size());
    size_t start = first_example_of_minibatch(minibatch);
    size_t end = first_example_of_minibatch(minibatch + 1);
    for (size_t e = start; e < end; ++e) {
      PerExampleFeatureStats* stats = nullptr;
      if (config.collect_feature_stats) {
        stats = &result->feature_stats[e];
      }
      status_of_minibatch[minibatch] = FastParseSerializedExample(
          serialized[e],
          (!example_names.empty() ? example_names[e] : "<unknown>"), e, config,
          config_index, hasher, &fixed_dense_values,
          &varlen_dense_buffers[minibatch], &sparse_buffers[minibatch],
          &ragged_buffers[minibatch], stats);
      if (!status_of_minibatch[minibatch].ok()) break;
    }
  };

  ParallelFor(ProcessMiniBatch, num_minibatches, thread_pool);

  for (absl::Status& status : status_of_minibatch) {
    TF_RETURN_IF_ERROR(status);
  }

  result->sparse_indices.reserve(config.sparse.size());
  result->sparse_values.reserve(config.sparse.size());
  result->sparse_shapes.reserve(config.sparse.size());
  result->dense_values.reserve(config.dense.size());
  result->ragged_values.reserve(config.ragged.size());
  result->ragged_splits.reserve(config.ragged.size());

  for (size_t d = 0; d < config.dense.size(); ++d) {
    result->dense_values.push_back(std::move(fixed_dense_values[d]));
  }

  // Merge SparseBuffers from all minibatches for every config.sparse.
  auto MergeSparseMinibatches = [&](size_t d) {
    // Loop over minibatches
    size_t total_num_features = 0;
    size_t max_num_features = 0;
    CountSparseFeatures(sparse_buffers, d, &total_num_features,
                        &max_num_features);

    TensorShape indices_shape;
    indices_shape.AddDim(total_num_features);
    indices_shape.AddDim(2);
    result->sparse_indices.emplace_back(DT_INT64, indices_shape);
    Tensor* indices = &result->sparse_indices.back();

    TensorShape values_shape;
    values_shape.AddDim(total_num_features);
    result->sparse_values.emplace_back(config.sparse[d].dtype, values_shape);
    Tensor* values = &result->sparse_values.back();

    result->sparse_shapes.emplace_back(DT_INT64, TensorShape({2}));
    auto shapes_shape_t = result->sparse_shapes.back().vec<int64_t>();
    shapes_shape_t(0) = serialized.size();
    shapes_shape_t(1) = max_num_features;

    size_t offset = 0;
    for (size_t i = 0; i < sparse_buffers.size(); ++i) {
      SparseBuffer& buffer = sparse_buffers[i][d];

      // Update indices.
      size_t delta = 0;

      if (indices->NumElements() > 0) {
        int64* ix_p = &indices->matrix<int64_t>()(offset, 0);
        size_t example_index = first_example_of_minibatch(i);
        for (size_t example_end_index : buffer.example_end_indices) {
          size_t feature_index = 0;
          for (; delta < example_end_index; ++delta) {
            // Column 0: example index
            *ix_p = example_index;
            // Column 1: the feature index buffer example
            *(ix_p + 1) = feature_index;
            ix_p += 2;
            ++feature_index;
          }
          ++example_index;
        }
      }

      CopySparseBufferToTensor(config.sparse[d].dtype, offset, &buffer, values);
      offset += delta;
    }
  };

  // Merge SparseBuffers from all minibatches for every config.ragged.
  auto MergeRaggedMinibatches = [&](size_t d) {
    // Loop over minibatches
    size_t total_num_features = 0;
    size_t max_num_features = 0;
    CountSparseFeatures(ragged_buffers, d, &total_num_features,
                        &max_num_features);

    TensorShape row_splits_shape;
    row_splits_shape.AddDim(serialized.size() + 1);
    result->ragged_splits.emplace_back(config.ragged[d].splits_dtype,
                                       row_splits_shape);
    Tensor* row_splits = &result->ragged_splits.back();
    if (config.ragged[d].splits_dtype == DT_INT64) {
      row_splits->flat<int64_t>()(0) = 0;
    } else {
      row_splits->flat<int32>()(0) = 0;
    }

    TensorShape values_shape;
    values_shape.AddDim(total_num_features);
    result->ragged_values.emplace_back(config.ragged[d].dtype, values_shape);
    Tensor* values = &result->ragged_values.back();

    size_t values_offset = 0;
    size_t splits_offset = 0;
    for (size_t i = 0; i < ragged_buffers.size(); ++i) {
      SparseBuffer& buffer = ragged_buffers[i][d];
      if (buffer.example_end_indices.empty()) continue;

      // Update row_splits.  row_splits are formed by concatenating the example
      // end_indices (adjusting each to start after the previous one ends).
      if (config.ragged[d].splits_dtype == DT_INT64) {
        int64* row_splits_out = &row_splits->flat<int64_t>()(splits_offset);
        int64_t start = *row_splits_out;
        for (size_t example_end_index : buffer.example_end_indices) {
          *++row_splits_out = start + example_end_index;
        }
      } else {
        int32* row_splits_out = &row_splits->flat<int32>()(splits_offset);
        int32_t start = *row_splits_out;
        for (size_t example_end_index : buffer.example_end_indices) {
          *++row_splits_out = start + example_end_index;
        }
      }

      CopySparseBufferToTensor(config.ragged[d].dtype, values_offset, &buffer,
                               values);
      values_offset += buffer.example_end_indices.back();
      splits_offset += buffer.example_end_indices.size();
    }
  };

  // Merge SparseBuffers from all minibatches for every config.dense having
  // variable_length.
  auto MergeDenseVarLenMinibatches = [&](size_t d) {
    if (!config.dense[d].variable_length) return;

    // Loop over minibatches
    size_t max_num_features = 0;
    for (auto& dense_values_tmp : varlen_dense_buffers) {
      std::vector<size_t>& end_indices =
          dense_values_tmp[d].example_end_indices;
      max_num_features = std::max(max_num_features, end_indices[0]);
      for (size_t i = 1; i < end_indices.size(); ++i) {
        size_t example_size = end_indices[i] - end_indices[i - 1];
        max_num_features = std::max(max_num_features, example_size);
      }
    }

    const size_t stride_size = config.dense[d].elements_per_stride;
    const size_t max_num_elements = max_num_features / stride_size;
    TensorShape values_shape;
    DCHECK_EQ(max_num_features % config.dense[d].elements_per_stride, 0);
    const size_t batch_size = serialized.size();
    values_shape.AddDim(batch_size);
    values_shape.AddDim(max_num_elements);
    for (int i = 1; i < config.dense[d].shape.dims(); ++i) {
      values_shape.AddDim(config.dense[d].shape.dim_size(i));
    }
    Tensor values(config.dense[d].dtype, values_shape);
    result->dense_values[d] = values;
    const size_t num_elements = values.NumElements();

    // Nothing to write, exit early.
    if (num_elements == 0) return;

    const size_t num_elements_per_minibatch = num_elements / batch_size;

    switch (config.dense[d].dtype) {
      case DT_INT64: {
        FillAndCopyVarLen<int64_t>(d, num_elements, num_elements_per_minibatch,
                                   config, varlen_dense_buffers, &values);
        break;
      }
      case DT_FLOAT: {
        FillAndCopyVarLen<float>(d, num_elements, num_elements_per_minibatch,
                                 config, varlen_dense_buffers, &values);
        break;
      }
      case DT_STRING: {
        FillAndCopyVarLen<tstring>(d, num_elements, num_elements_per_minibatch,
                                   config, varlen_dense_buffers, &values);
        break;
      }
      default:
        ReportUnexpectedDataType(config.dense[d].dtype);
    }
  };

  for (size_t d = 0; d < config.dense.size(); ++d) {
    MergeDenseVarLenMinibatches(d);
  }

  for (size_t d = 0; d < config.sparse.size(); ++d) {
    MergeSparseMinibatches(d);
  }

  for (size_t d = 0; d < config.ragged.size(); ++d) {
    MergeRaggedMinibatches(d);
  }

  return absl::OkStatus();
}

absl::Status FastParseSingleExample(const Config& config,
                                    absl::string_view serialized,
                                    Result* result) {
  DCHECK(result != nullptr);
  // Check config so we can safely CHECK(false) in switches on config.*.dtype
  TF_RETURN_IF_ERROR(CheckConfigDataTypes(config));

  PerExampleFeatureStats* stats = nullptr;
  if (config.collect_feature_stats) {
    result->feature_stats.emplace_back();
    stats = &result->feature_stats.back();
  }

  // TODO(mrry): Cache the construction of this map at Op construction time.
  size_t config_size =
      config.dense.size() + config.sparse.size() + config.ragged.size();
  SeededHasher hasher;
  // Build config index.
  PresizedCuckooMap<std::pair<size_t, Type>> config_index(config_size);
  bool ok = true;
  for (size_t i = 0; i < 1000; ++i) {
    for (size_t d = 0; d < config.dense.size(); ++d) {
      ok &= config_index.InsertUnique(hasher(config.dense[d].feature_name),
                                      {d, Type::Dense});
    }
    for (size_t d = 0; d < config.sparse.size(); ++d) {
      ok &= config_index.InsertUnique(hasher(config.sparse[d].feature_name),
                                      {d, Type::Sparse});
    }
    for (size_t d = 0; d < config.ragged.size(); ++d) {
      ok &= config_index.InsertUnique(hasher(config.ragged[d].feature_name),
                                      {d, Type::Ragged});
    }
    if (ok) break;
    LOG(WARNING) << "Collision found. This should happen only if you have "
                    "around 2^32 entries in your config.";
    hasher.seed++;
    config_index.Clear(config_size);
    ok = true;
  }
  if (!ok) {
    return errors::Internal(
        "Could not avoid collision. This should not happen.");
  }

  result->sparse_indices.reserve(config.sparse.size());
  result->sparse_values.reserve(config.sparse.size());
  result->sparse_shapes.reserve(config.sparse.size());
  result->dense_values.reserve(config.dense.size());
  result->ragged_values.reserve(config.ragged.size());
  result->ragged_splits.reserve(config.ragged.size());

  // Allocate dense output tensors.
  for (size_t d = 0; d < config.dense.size(); ++d) {
    if (!config.dense[d].variable_length) {
      TensorShape values_shape;
      if (!config.dense[d].shape.AsTensorShape(&values_shape)) {
        return errors::Internal(
            "Fixed-length shape was not a statically defined shape.");
      }
      result->dense_values.emplace_back(config.dense[d].dtype, values_shape);
    } else {
      // Variable-length tensor will be allocated later.
      result->dense_values.emplace_back();
    }
  }

  // Allocate sparse output tensors.
  for (size_t d = 0; d < config.sparse.size(); ++d) {
    // The dense_shape is always a vector of length 1.
    result->sparse_shapes.emplace_back(DT_INT64, TensorShape({1}));
    // Variable-length tensors will be allocated later.
    result->sparse_indices.emplace_back();
    result->sparse_values.emplace_back();
  }

  // Allocate ragged output tensors.
  for (size_t d = 0; d < config.ragged.size(); ++d) {
    // Variable-length values tensors will be allocated later.
    result->ragged_values.emplace_back();
    // Splits tensors are empty (unused) for single (scalar) inputs.
    const auto splits_dtype = config.ragged[d].splits_dtype;
    result->ragged_splits.emplace_back(splits_dtype, TensorShape({0}));
  }

  parsed::Example parsed_example;
  if (!ParseExample(serialized, &parsed_example)) {
    return errors::InvalidArgument("Could not parse example input, value: '",
                                   serialized, "'");
  }
  std::vector<bool> sparse_feature_already_seen(config.sparse.size(), false);
  std::vector<bool> dense_feature_already_seen(config.dense.size(), false);
  std::vector<bool> ragged_feature_already_seen(config.ragged.size(), false);

  if (stats) {
    // TODO(b/111553342): This may over-count the number of features if there
    // are duplicate keys in the feature map. Consider deduplicating the keys
    // before computing the count.
    stats->features_count = parsed_example.size();
  }

  // Handle features present in the example.
  const size_t parsed_example_size = parsed_example.size();
  for (size_t i = 0; i < parsed_example_size; ++i) {
    // This is a logic that standard protobuf parsing is implementing.
    // I.e. last entry in the map overwrites all the previous ones.
    parsed::FeatureMapEntry& name_and_feature =
        parsed_example[parsed_example_size - i - 1];

    const absl::string_view feature_name = name_and_feature.first;
    parsed::Feature& feature = name_and_feature.second;

    std::pair<size_t, Type> d_and_type;
    uint64 h = hasher(feature_name);
    if (!config_index.Find(h, &d_and_type)) continue;

    size_t d = d_and_type.first;
    bool is_dense = d_and_type.second == Type::Dense;
    bool is_sparse = d_and_type.second == Type::Sparse;

    {
      // Testing for PresizedCuckooMap collision.
      // TODO(lew): Use dense_hash_map and avoid this and hasher creation.
      const tstring& config_feature_name =
          is_dense ? config.dense[d].feature_name
                   : (is_sparse ? config.sparse[d].feature_name
                                : config.ragged[d].feature_name);
      if (feature_name != config_feature_name) continue;
    }

    auto example_error = [feature_name](absl::string_view suffix) {
      return errors::InvalidArgument("Key: ", feature_name, ".  ", suffix);
    };

    auto parse_error = [feature_name](absl::string_view description) {
      return errors::InvalidArgument(
          "Key: ", feature_name,
          ".  Can't parse serialized Example: ", description);
    };

    DataType example_dtype;
    TF_RETURN_IF_ERROR(feature.ParseDataType(&example_dtype));
    if (example_dtype == DT_INVALID) continue;

    if (is_dense && !config.dense[d].variable_length) {
      // If feature was already visited, skip.
      // Compare comment at the beginning of the loop.
      if (dense_feature_already_seen[d]) {
        LogDenseFeatureDataLoss(feature_name);
        continue;
      }
      dense_feature_already_seen[d] = true;

      if (example_dtype != config.dense[d].dtype) {
        return example_error(strings::StrCat(
            "Data types don't match. Data type: ",
            DataTypeString(example_dtype),
            " but expected type: ", DataTypeString(config.dense[d].dtype)));
      }

      Tensor* out = &result->dense_values[d];
      const std::size_t num_elements = config.dense[d].elements_per_stride;
      if (stats) {
        // TODO(b/111553342): If desirable, we could add support for counting
        // elements in the features that aren't parsed, but this could add
        // considerable runtime cost.
        stats->feature_values_count += num_elements;
      }
      switch (example_dtype) {
        case DT_INT64: {
          auto out_p = out->flat<int64_t>().data();
          LimitedArraySlice<int64_t> slice(out_p, num_elements);
          if (!feature.ParseInt64List(&slice))
            return parse_error("Parsing int64_list failed.");
          if (slice.EndDistance() != 0) {
            return parse_error("Some int64_list slice was not parsed.");
          }
          break;
        }
        case DT_FLOAT: {
          auto out_p = out->flat<float>().data();
          LimitedArraySlice<float> slice(out_p, num_elements);
          if (!feature.ParseFloatList(&slice))
            return parse_error("Parsing float_list failed.");
          if (slice.EndDistance() != 0) {
            return parse_error("Some float_list slice was not parsed.");
          }
          break;
        }
        case DT_STRING: {
          auto out_p = out->flat<tstring>().data();
          LimitedArraySlice<tstring> slice(out_p, num_elements);
          if (!feature.ParseBytesList(&slice))
            return parse_error("Parsing bytes_list failed.");
          if (slice.EndDistance() != 0) {
            return parse_error("Some bytes_list slice was not parsed.");
          }
          break;
        }
        default:
          ReportUnexpectedDataType(example_dtype);
      }

    } else {  // if variable length
      SmallVector<tstring> bytes_list;
      TensorVector<float> float_list;
      SmallVector<int64_t> int64_list;

      const size_t num_elements_divisor =
          is_dense ? config.dense[d].elements_per_stride : 1;
      size_t num_elements;

      if (is_dense) {
        // If feature was already visited, skip.
        // Compare comment at the beginning of the loop.
        if (dense_feature_already_seen[d]) {
          LogDenseFeatureDataLoss(feature_name);
          continue;
        }
        dense_feature_already_seen[d] = true;
        if (example_dtype != config.dense[d].dtype) {
          return example_error(strings::StrCat(
              "Data types don't match. Data type: ",
              DataTypeString(example_dtype),
              " but expected type: ", DataTypeString(config.dense[d].dtype)));
        }
      } else {
        // Feature is sparse or ragged.
        auto& feature_already_seen = is_sparse ? sparse_feature_already_seen
                                               : ragged_feature_already_seen;
        auto& feature_dtype =
            is_sparse ? config.sparse[d].dtype : config.ragged[d].dtype;
        // If feature was already visited, skip.
        // Compare comment at the beginning of the loop.
        if (feature_already_seen[d]) {
          LogSparseFeatureDataLoss(feature_name);
          continue;
        }
        feature_already_seen[d] = true;

        // Handle sparse features.
        if (example_dtype != DT_INVALID && example_dtype != feature_dtype) {
          return example_error(strings::StrCat(
              "Data types don't match. ",
              "Expected type: ", DataTypeString(feature_dtype),
              ", Actual type: ", DataTypeString(example_dtype)));
        }
      }

      switch (example_dtype) {
        case DT_INT64: {
          // TODO(mrry): Use the fact that the `int64_list` is packed to read
          // out the length and pre-allocate the output tensor.
          if (!feature.ParseInt64List(&int64_list))
            return parse_error("Parsing int64_list failed.");
          num_elements = int64_list.size();
          break;
        }
        case DT_FLOAT: {
          if (!feature.ParseFloatList(&float_list))
            return parse_error("Parsing float_list failed.");
          num_elements = float_list.size();
          break;
        }
        case DT_STRING: {
          int actual_num_elements = 0;
          if (!feature.GetNumElementsInBytesList(&actual_num_elements)) {
            return parse_error("Could not get num elements in bytes_list.");
          }
          bytes_list.reserve(actual_num_elements);
          if (!feature.ParseBytesList(&bytes_list))
            return parse_error("Parsing bytes_list failed.");
          num_elements = bytes_list.size();
          break;
        }
        default:
          num_elements = 0;
          ReportUnexpectedDataType(example_dtype);
      }

      if (num_elements % num_elements_divisor != 0) {
        return absl::InvalidArgumentError(absl::Substitute(
            "Error while parsing feature with key $0: number "
            "of elements should be divisible by $1, found $2 instead",
            feature_name, num_elements_divisor, num_elements));
      }

      if (stats) {
        stats->feature_values_count += num_elements;
      }

      Tensor* out;
      DataType out_dtype;
      TensorShape out_shape;
      if (is_dense) {
        out_shape.AddDim(num_elements / num_elements_divisor);
        for (int i = 1; i < config.dense[d].shape.dims(); ++i) {
          out_shape.AddDim(config.dense[d].shape.dim_size(i));
        }

        out = &result->dense_values[d];
        out_dtype = config.dense[d].dtype;
      } else if (is_sparse) {
        Tensor* out_indices = &result->sparse_indices[d];
        Tensor* out_dense_shape = &result->sparse_shapes[d];

        // TODO(mrry): Investigate the possibility of not materializing
        // the indices (and perhaps dense_shape) until they are needed.
        *out_indices = Tensor(
            DT_INT64, TensorShape({static_cast<int64_t>(num_elements), 1}));
        auto indices_flat = out_indices->flat<int64_t>();
        for (size_t i = 0; i < num_elements; ++i) {
          indices_flat(i) = static_cast<int64_t>(i);
        }

        *out_dense_shape = Tensor(DT_INT64, TensorShape({1}));
        auto shapes_shape_t = out_dense_shape->vec<int64_t>();
        shapes_shape_t(0) = num_elements;

        out = &result->sparse_values[d];
        out_dtype = config.sparse[d].dtype;
        out_shape.AddDim(num_elements);
      } else {
        out = &result->ragged_values[d];
        out_dtype = config.ragged[d].dtype;
        out_shape.AddDim(num_elements);
      }

      switch (example_dtype) {
        case DT_INT64: {
          *out = Tensor(out_dtype, out_shape);
          CopyOrMoveBlock(int64_list.begin(), int64_list.end(),
                          out->flat<int64_t>().data());
          break;
        }
        case DT_FLOAT: {
          if (!out->CopyFrom(float_list.tensor(), out_shape)) {
            return parse_error(absl::StrCat("Size of float_list is ",
                                            float_list.tensor().dims(),
                                            ", expected ", out_shape.dims()));
          }
          break;
        }
        case DT_STRING: {
          *out = Tensor(out_dtype, out_shape);
          CopyOrMoveBlock(bytes_list.begin(), bytes_list.end(),
                          out->flat<tstring>().data());
          break;
        }
        default:
          ReportUnexpectedDataType(example_dtype);
      }
    }
  }

  // Handle missing dense features.
  for (size_t d = 0; d < config.dense.size(); ++d) {
    if (!dense_feature_already_seen[d]) {
      if (!config.dense[d].variable_length) {
        // Handle missing fixed-length dense feature.
        if (config.dense[d].default_value.NumElements() == 0) {
          return errors::InvalidArgument(
              "Feature: ", config.dense[d].feature_name,
              " (data type: ", DataTypeString(config.dense[d].dtype), ")",
              " is required but could not be found.");
        }
        result->dense_values[d] = config.dense[d].default_value;
      } else {
        // Handle missing varlen dense feature.
        TensorShape empty_shape;
        empty_shape.AddDim(0);
        for (int i = 1; i < config.dense[d].shape.dims(); ++i) {
          empty_shape.AddDim(config.dense[d].shape.dim_size(i));
        }
        result->dense_values[d] = Tensor(config.dense[d].dtype, empty_shape);
      }
    }
  }

  // Handle missing sparse features.
  for (size_t d = 0; d < config.sparse.size(); ++d) {
    if (!sparse_feature_already_seen[d]) {
      result->sparse_indices[d] = Tensor(DT_INT64, TensorShape({0, 1}));
      result->sparse_values[d] =
          Tensor(config.sparse[d].dtype, TensorShape({0}));
      result->sparse_shapes[d].vec<int64_t>()(0) = 0;
    }
  }

  // Handle missing ragged features.
  for (size_t d = 0; d < config.ragged.size(); ++d) {
    if (!ragged_feature_already_seen[d]) {
      result->ragged_values[d] =
          Tensor(config.ragged[d].dtype, TensorShape({0}));
    }
  }

  return absl::OkStatus();
}

// Private helper functions for FastParseSequenceExample.
namespace {

// A struct used by FastParseSequenceExample to hold the serialized proto
// substrings for a single feature, plus some auxiliary information derived
// from those protos (such as the total value length).
struct FeatureProtos {
  // Proto substrings from each serialized SequenceExample that correspond
  // with this feature.  `protos_present` records whether the proto had a
  // value defined (even if that value is empty).
  std::vector<absl::string_view> protos;
  std::vector<bool> protos_present;

  // Information derived from protos:
  size_t length;    // total length for ragged/sparse, max row length for dense.
  size_t num_rows;  // only populated for ragged sequence features.

  // Information from the config:
  Type type;  // Whether this feature is sparse, ragged, or dense.
  DataType dtype;
};

// Map from feature name to FeatureProtos for that feature.
using FeatureProtosMap = absl::flat_hash_map<absl::string_view, FeatureProtos>;

string ExampleName(const absl::Span<const tstring> example_names, int n) {
  return example_names.empty() ? "<unknown>" : example_names[n];
}

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

// Reads an example proto, and extracts a StringPiece pointer to each feature.
absl::Status ExtractFeaturesFromSequenceExamples(
    const absl::Span<const tstring> examples,
    const absl::Span<const tstring> example_names,
    FeatureProtosMap* context_features, FeatureProtosMap* sequence_features) {
  for (int d = 0; d < examples.size(); d++) {
    const tstring& example = examples[d];
    protobuf::io::CodedInputStream stream(
        reinterpret_cast<const uint8*>(example.data()), example.size());
    // Not clear what this does. Why not stream.EnableAliasing()?
    EnableAliasing(&stream);

    // Extract pointers to all features within this serialized example.
    while (!stream.ExpectAtEnd()) {
      FeatureProtosMap* features = nullptr;
      if (stream.ExpectTag(kDelimitedTag(1))) {
        // Context
        features = context_features;
      } else if (stream.ExpectTag(kDelimitedTag(2))) {
        // Sequence
        features = sequence_features;
      } else if (!SkipExtraneousTag(&stream)) {
        return errors::InvalidArgument(
            "Invalid protocol message input, example id: ",
            ExampleName(example_names, d));
      }
      if (features != nullptr) {
        uint32 length;
        if (!stream.ReadVarint32(&length)) {
          return errors::InvalidArgument(
              "Invalid protocol message input, example id: ",
              ExampleName(example_names, d));
        }
        auto limit = stream.PushLimit(length);
        while (!stream.ExpectAtEnd()) {
          absl::string_view key, value;
          uint32 length;
          if (!stream.ExpectTag(kDelimitedTag(1)) ||
              !stream.ReadVarint32(&length)) {
            return errors::InvalidArgument(
                "Invalid protocol message input, example id: ",
                ExampleName(example_names, d));
          }
          auto limit = stream.PushLimit(length);
          if (!stream.ExpectTag(kDelimitedTag(1)) ||
              !ParseString(&stream, &key) ||
              !stream.ExpectTag(kDelimitedTag(2)) ||
              !ParseString(&stream, &value) || !stream.ExpectAtEnd()) {
            return errors::InvalidArgument(
                "Invalid protocol message input, example id: ",
                ExampleName(example_names, d));
          }
          stream.PopLimit(limit);
          // Only save if this feature was requested.
          auto feature_iter = features->find(key);
          if (feature_iter != features->end()) {
            auto& feature = feature_iter->second;
            feature.protos[d] = value;
            feature.protos_present[d] = true;
          }
        }
        stream.PopLimit(limit);
      }
    }
  }
  return absl::OkStatus();
}

// Populates context_features[k].length based on context_features[k].protos
// (for all k).
absl::Status GetContextFeatureLengths(
    const absl::Span<const tstring> example_names,
    FeatureProtosMap* context_features) {
  for (auto& c : *context_features) {
    FeatureProtos& feature = c.second;
    for (int d = 0; d < feature.protos.size(); ++d) {
      const auto& proto = feature.protos[d];
      if (proto.empty()) continue;
      protobuf::io::CodedInputStream stream(
          reinterpret_cast<const uint8*>(proto.data()), proto.size());
      EnableAliasing(&stream);
      int num_elements = GetFeatureLength(feature.dtype, &stream);
      if (num_elements < 0) {
        return errors::InvalidArgument(
            "Name: ", ExampleName(example_names, d),
            ", Context feature: ", c.first,
            ".  Data types don't match. Expected type: ",
            DataTypeString(feature.dtype));
      }
      switch (feature.type) {
        case Type::Sparse:  // intentional fall-through
        case Type::Ragged:
          feature.length += num_elements;
          break;
        case Type::Dense:
          feature.length =
              std::max(feature.length, static_cast<size_t>(num_elements));
          break;
      }
    }
  }
  return absl::OkStatus();
}

// Populates sequence_features[k].length and sequence_features[k].num_rows based
// on sequence_features[k].protos (for all k).
absl::Status GetSequenceFeatureLengths(
    const absl::Span<const tstring> example_names,
    FeatureProtosMap* sequence_features) {
  for (auto& c : *sequence_features) {
    FeatureProtos& feature = c.second;
    for (int d = 0; d < feature.protos.size(); ++d) {
      const auto& proto = feature.protos[d];
      if (proto.empty()) continue;

      size_t num_rows = 0;
      size_t num_elements = 0;
      protobuf::io::CodedInputStream stream(
          reinterpret_cast<const uint8*>(proto.data()), proto.size());
      EnableAliasing(&stream);
      while (!stream.ExpectAtEnd()) {
        uint32 feature_bytes;
        if (!stream.ExpectTag(kDelimitedTag(1)) ||
            !stream.ReadVarint32(&feature_bytes)) {
          return errors::InvalidArgument("Error in sequence feature ", c.first,
                                         " in example ",
                                         ExampleName(example_names, d));
        }
        if (feature_bytes > 2) {
          auto limit = stream.PushLimit(feature_bytes);
          int delta = GetFeatureLength(feature.dtype, &stream);
          if (delta < 0) {
            return errors::InvalidArgument(
                "Name: ", ExampleName(example_names, d),
                ", Feature list: ", c.first, ", Index: ", num_rows,
                ".  Data types don't match. Expected type: ",
                DataTypeString(feature.dtype));
          }
          num_elements += delta;
          stream.PopLimit(limit);
        } else if (feature_bytes == 2) {
          if (!SkipEmptyFeature(&stream, feature.dtype)) {
            return errors::InvalidArgument(
                "Name: ", ExampleName(example_names, d),
                ", Feature list: ", c.first, ", Index: ", num_rows,
                ".  Data types don't match. Expected type: ",
                DataTypeString(feature.dtype));
          }
        } else if (feature_bytes != 0) {
          return errors::InvalidArgument("Error in sequence feature ", c.first,
                                         " in example ",
                                         ExampleName(example_names, d));
        }
        ++num_rows;
      }
      switch (feature.type) {
        case Type::Sparse:
          feature.length += num_elements;
          break;
        case Type::Ragged:
          feature.length += num_elements;
          feature.num_rows += num_rows;
          break;
        case Type::Dense:
          feature.length = std::max(feature.length, num_elements);
          break;
      }
    }
  }
  return absl::OkStatus();
}

// Copies src into dst[dst_offset:dst_offset+src.size], and then increments
// dst_offset by src.size.
void CopyTensorIntoTensor(DataType dtype, const Tensor& src, Tensor* dst,
                          size_t* dst_offset) {
  size_t src_size = src.NumElements();
  switch (dtype) {
    case DT_INT64: {
      auto src_t = src.flat<int64_t>().data();
      std::copy(src_t, src_t + src_size,
                dst->flat<int64_t>().data() + *dst_offset);
      break;
    }
    case DT_FLOAT: {
      auto src_t = src.flat<float>().data();
      std::copy(src_t, src_t + src_size,
                dst->flat<float>().data() + *dst_offset);
      break;
    }
    case DT_STRING: {
      auto src_t = src.flat<tstring>().data();
      std::copy(src_t, src_t + src_size,
                dst->flat<tstring>().data() + *dst_offset);
      break;
    }
    default:
      ReportUnexpectedDataType(dtype);
  }
  *dst_offset += src_size;
}

// Parses dense features in `context_features`, and writes their parsed
// values to `context_results`.
absl::Status ParseContextDenseFeatures(
    const FeatureProtosMap& context_features,
    const FastParseExampleConfig& context_config,
    absl::Span<const tstring> example_names, bool is_batch, int num_examples,
    Allocator* allocator, Result* context_result) {
  for (int t = 0; t < context_config.dense.size(); ++t) {
    const auto& c = context_config.dense[t];
    const FeatureProtos& feature =
        context_features.find(c.feature_name)->second;
    TensorShape dense_shape, example_shape;
    DataType dtype = c.dtype;
    const size_t data_max_elements = feature.length;
    if (!c.shape.AsTensorShape(&example_shape) ||
        data_max_elements != example_shape.num_elements()) {
      return errors::InvalidArgument(
          "Inconsistent max number of elements for feature ", c.feature_name,
          ": expected ", example_shape.num_elements(), ", but found ",
          data_max_elements);
    }
    if (is_batch) {
      dense_shape.AddDim(num_examples);
    }
    for (const int dim : c.shape.dim_sizes()) {
      dense_shape.AddDim(dim);
    }
    context_result->dense_values[t] = Tensor(allocator, dtype, dense_shape);

    Tensor& out = context_result->dense_values[t];
    size_t out_offset = 0;

    // Fill in the values.
    for (int e = 0; e < num_examples; e++) {
      size_t num_elements = 0;
      const auto& feature_proto = feature.protos[e];
      if (!feature.protos_present[e]) {
        // Copy the default value, if present. If not, return an error.
        if (c.default_value.NumElements() == 0) {
          return errors::InvalidArgument(
              "Feature: ", c.feature_name,
              " (data type: ", DataTypeString(c.dtype), ")",
              " is required but could not be found.");
        }
        CopyTensorIntoTensor(dtype, c.default_value, &out, &out_offset);
        num_elements += c.default_value.NumElements();
      } else if (!feature_proto.empty()) {
        protobuf::io::CodedInputStream stream(
            reinterpret_cast<const uint8*>(feature_proto.data()),
            feature_proto.size());
        EnableAliasing(&stream);
        num_elements += ParseFeature(dtype, &stream, &out, &out_offset);
      }
      if (num_elements != data_max_elements) {
        return errors::InvalidArgument(
            "Unexpected number of elements in example ",
            ExampleName(example_names, e));
      }
    }
  }
  return absl::OkStatus();
}

// Parses sparse features in `context_features`, and writes their parsed
// values to `context_results`.
absl::Status ParseContextSparseFeatures(
    const FeatureProtosMap& context_features,
    const FastParseExampleConfig& context_config,
    absl::Span<const tstring> example_names, bool is_batch, int num_examples,
    Allocator* allocator, Result* context_result) {
  for (int t = 0; t < context_config.sparse.size(); ++t) {
    const auto& c = context_config.sparse[t];
    const FeatureProtos& feature =
        context_features.find(c.feature_name)->second;
    TensorShape indices_shape, values_shape;
    DataType dtype = c.dtype;
    size_t expected_num_elements = feature.length;
    indices_shape.AddDim(expected_num_elements);
    indices_shape.AddDim(is_batch ? 2 : 1);
    values_shape.AddDim(expected_num_elements);
    context_result->sparse_indices[t] =
        Tensor(allocator, DT_INT64, indices_shape);
    context_result->sparse_values[t] = Tensor(allocator, dtype, values_shape);
    context_result->sparse_shapes[t] =
        Tensor(allocator, DT_INT64, TensorShape({is_batch ? 2 : 1}));
    Tensor& out_values = context_result->sparse_values[t];
    size_t out_values_offset = 0;
    int64_t* out_indices =
        context_result->sparse_indices[t].flat<int64_t>().data();
    auto out_shape = context_result->sparse_shapes[t].vec<int64_t>();

    // Fill in the values.
    size_t num_elements = 0;
    size_t max_num_cols = 0;
    for (int e = 0; e < num_examples; e++) {
      const auto& feature_proto = feature.protos[e];
      if (feature_proto.empty()) continue;
      protobuf::io::CodedInputStream stream(
          reinterpret_cast<const uint8*>(feature_proto.data()),
          feature_proto.size());
      EnableAliasing(&stream);
      size_t num_added =
          ParseFeature(dtype, &stream, &out_values, &out_values_offset);
      num_elements += num_added;
      max_num_cols = std::max(max_num_cols, num_added);
      for (int i = 0; i < num_added; i++) {
        if (is_batch) *out_indices++ = e;
        *out_indices++ = i;
      }
    }
    if (num_elements != expected_num_elements) {
      return errors::InvalidArgument(
          "Unexpected total number of elements in feature ", c.feature_name);
    }
    if (is_batch) {
      out_shape(0) = num_examples;
      out_shape(1) = max_num_cols;
    } else {
      out_shape(0) = max_num_cols;
    }
  }
  return absl::OkStatus();
}

// Parses ragged features in `context_features`, and writes their parsed
// values to `context_results`.
absl::Status ParseContextRaggedFeatures(
    const FeatureProtosMap& context_features,
    const FastParseExampleConfig& context_config,
    absl::Span<const tstring> example_names, bool is_batch, int num_examples,
    Allocator* allocator, Result* context_result) {
  for (int t = 0; t < context_config.ragged.size(); ++t) {
    const auto& c = context_config.ragged[t];
    const FeatureProtos& feature =
        context_features.find(c.feature_name)->second;
    TensorShape values_shape, splits_shape;
    DataType dtype = c.dtype;
    DataType splits_dtype = c.splits_dtype;
    size_t expected_num_elements = feature.length;
    values_shape.AddDim(expected_num_elements);
    if (is_batch) {
      splits_shape.AddDim(num_examples + 1);
    }
    context_result->ragged_values[t] = Tensor(allocator, dtype, values_shape);
    context_result->ragged_splits[t] =
        Tensor(allocator, splits_dtype, splits_shape);
    Tensor& out_values = context_result->ragged_values[t];
    size_t out_values_offset = 0;
    int32* int32_splits =
        is_batch && splits_dtype == DT_INT32
            ? context_result->ragged_splits[t].vec<int32>().data()
            : nullptr;
    int64_t* int64_splits =
        is_batch && splits_dtype == DT_INT64
            ? context_result->ragged_splits[t].vec<int64_t>().data()
            : nullptr;
    if (int32_splits) {
      *int32_splits++ = 0;
    } else if (int64_splits) {
      *int64_splits++ = 0;
    }

    // Fill in the values.
    size_t split = 0;  // = total number of elements we've seen so far
    for (int e = 0; e < num_examples; e++) {
      const auto& feature_proto = feature.protos[e];
      if (!feature_proto.empty()) {
        protobuf::io::CodedInputStream stream(
            reinterpret_cast<const uint8*>(feature_proto.data()),
            feature_proto.size());
        EnableAliasing(&stream);
        size_t num_added =
            ParseFeature(dtype, &stream, &out_values, &out_values_offset);
        split += num_added;
      }
      if (int32_splits) {
        *int32_splits++ = split;
      } else if (int64_splits) {
        *int64_splits++ = split;
      }
    }
    if (split != expected_num_elements) {
      return errors::InvalidArgument(
          "Unexpected total number of elements in feature ", c.feature_name);
    }
    if (int32_splits || int64_splits) {
      int actual_splits =
          int32_splits
              ? int32_splits -
                    context_result->ragged_splits[t].vec<int32>().data()
              : int64_splits -
                    context_result->ragged_splits[t].vec<int64_t>().data();
      if (actual_splits != num_examples + 1) {
        return errors::InvalidArgument(
            "Unexpected number of examples for feature ", c.feature_name);
      }
    }
  }
  return absl::OkStatus();
}

// Parses dense features in `sequence_features`, and writes their parsed
// values to `sequence_result`.
absl::Status ParseSequenceDenseFeatures(
    const FeatureProtosMap& sequence_features,
    const FastParseExampleConfig& sequence_config,
    absl::Span<const tstring> example_names, bool is_batch, int num_examples,
    Allocator* allocator, Result* sequence_result,
    std::vector<Tensor>* dense_feature_lengths) {
  TensorShape dense_length_shape;
  if (is_batch) {
    dense_length_shape.AddDim(num_examples);
  }
  for (int t = 0; t < sequence_config.dense.size(); ++t) {
    const auto& c = sequence_config.dense[t];
    const FeatureProtos& feature =
        sequence_features.find(c.feature_name)->second;
    TensorShape dense_shape, row_shape;
    DataType dtype = c.dtype;
    const size_t expected_max_elements = feature.length;
    if (!c.shape.AsTensorShape(&row_shape) ||
        expected_max_elements !=
            (expected_max_elements / row_shape.num_elements()) *
                row_shape.num_elements()) {
      PartialTensorShape total_shape = row_shape;
      total_shape.InsertDim(0, -1);
      return errors::InvalidArgument(
          "Feature list '", c.feature_name,
          "' has an unexpected number of values.  Total values size: ",
          expected_max_elements,
          " is not consistent with output shape: ", total_shape.DebugString());
    }
    int64_t expected_max_rows =
        expected_max_elements / row_shape.num_elements();
    if (is_batch) {
      dense_shape.AddDim(num_examples);
    }
    dense_shape.AddDim(expected_max_rows);
    for (const int dim : sequence_config.dense[t].shape.dim_sizes()) {
      dense_shape.AddDim(dim);
    }
    sequence_result->dense_values[t] = Tensor(allocator, dtype, dense_shape);
    (*dense_feature_lengths)[t] =
        Tensor(allocator, DT_INT64, dense_length_shape);
    int64_t* out_lengths = (*dense_feature_lengths)[t].flat<int64_t>().data();

    tstring* out_bytes = nullptr;
    float* out_float = nullptr;
    int64_t* out_int64 = nullptr;
    switch (dtype) {
      case DT_STRING:
        out_bytes = sequence_result->dense_values[t].flat<tstring>().data();
        break;
      case DT_FLOAT:
        out_float = sequence_result->dense_values[t].flat<float>().data();
        break;
      case DT_INT64:
        out_int64 = sequence_result->dense_values[t].flat<int64_t>().data();
        break;
      default:
        ReportUnexpectedDataType(dtype);
    }

    // Fill in the values.
    for (int e = 0; e < num_examples; e++) {
      size_t num_elements = 0, num_rows = 0;
      const auto& feature_proto = feature.protos[e];
      if (!feature.protos_present[e]) {
        // Return an error if this feature was not allowed to be missing.
        // Otherwise, we'll pad as needed below.
        if (!c.variable_length) {
          return errors::InvalidArgument(
              "Name: ", ExampleName(example_names, e), ", Feature list '",
              c.feature_name,
              "' is required but could not be found.  "
              "Did you mean to include it in "
              "feature_list_dense_missing_assumed_empty or "
              "feature_list_dense_defaults?");
        }
      } else if (!feature_proto.empty()) {
        protobuf::io::CodedInputStream stream(
            reinterpret_cast<const uint8*>(feature_proto.data()),
            feature_proto.size());
        EnableAliasing(&stream);
        while (!stream.ExpectAtEnd()) {
          uint32 feature_length;
          if (!stream.ExpectTag(kDelimitedTag(1)) ||
              !stream.ReadVarint32(&feature_length)) {
            return errors::InvalidArgument("Error in sequence feature ",
                                           c.feature_name, " in example ",
                                           ExampleName(example_names, e));
          }
          auto limit = stream.PushLimit(feature_length);
          int num_added = 0;
          if (feature_length > 2) {
            switch (dtype) {
              case DT_STRING:
                num_added = ParseBytesFeature(&stream, out_bytes);
                out_bytes += num_added;
                break;
              case DT_FLOAT:
                num_added = ParseFloatFeature(&stream, out_float);
                out_float += num_added;
                break;
              case DT_INT64:
                num_added = ParseInt64Feature(&stream, out_int64);
                out_int64 += num_added;
                break;
              default:
                ReportUnexpectedDataType(dtype);
                num_added = 0;
            }
            if (num_added < 0) {
              // This should be unreachable -- we already scanned the feature in
              // GetSequenceFeatureLengths, and it hasn't changed since then.
              return errors::InvalidArgument("Error in sequence feature ",
                                             c.feature_name, " in example ",
                                             ExampleName(example_names, e));
            }
          }
          if (num_added != row_shape.num_elements()) {
            return errors::InvalidArgument(
                "Name: ", ExampleName(example_names, e),
                ", Key: ", c.feature_name, ", Index: ", num_rows,
                ".  Number of values != expected.  values size: ", num_added,
                " but output shape: ", row_shape.DebugString());
          }
          num_elements += num_added;
          num_rows++;
          stream.PopLimit(limit);
        }
      }
      *out_lengths++ = num_rows;
      // Pad as necessary.
      int num_to_pad = expected_max_elements - num_elements;
      switch (dtype) {
        case DT_STRING:
          out_bytes += num_to_pad;
          break;
        case DT_FLOAT:
          PadFloatFeature(num_to_pad, out_float);
          out_float += num_to_pad;
          break;
        case DT_INT64:
          PadInt64Feature(num_to_pad, out_int64);
          out_int64 += num_to_pad;
          break;
        default:
          ReportUnexpectedDataType(dtype);
      }
    }
  }
  return absl::OkStatus();
}

// Parses sparse features in `sequence_features`, and writes their parsed
// values to `sequence_result`.
absl::Status ParseSequenceSparseFeatures(
    const FeatureProtosMap& sequence_features,
    const FastParseExampleConfig& sequence_config,
    absl::Span<const tstring> example_names, bool is_batch, int num_examples,
    Allocator* allocator, Result* sequence_result) {
  for (int t = 0; t < sequence_config.sparse.size(); ++t) {
    const auto& c = sequence_config.sparse[t];
    const FeatureProtos& feature =
        sequence_features.find(c.feature_name)->second;
    TensorShape indices_shape, values_shape;
    DataType dtype = c.dtype;
    size_t expected_num_elements = feature.length;
    indices_shape.AddDim(expected_num_elements);
    indices_shape.AddDim(is_batch ? 3 : 2);
    values_shape.AddDim(expected_num_elements);
    sequence_result->sparse_indices[t] =
        Tensor(allocator, DT_INT64, indices_shape);
    sequence_result->sparse_values[t] = Tensor(allocator, dtype, values_shape);
    sequence_result->sparse_shapes[t] =
        Tensor(allocator, DT_INT64, TensorShape({is_batch ? 3 : 2}));

    tstring* out_bytes = nullptr;
    float* out_float = nullptr;
    int64_t* out_int64 = nullptr;
    switch (dtype) {
      case DT_STRING:
        out_bytes = sequence_result->sparse_values[t].flat<tstring>().data();
        break;
      case DT_FLOAT:
        out_float = sequence_result->sparse_values[t].flat<float>().data();
        break;
      case DT_INT64:
        out_int64 = sequence_result->sparse_values[t].flat<int64_t>().data();
        break;
      default:
        ReportUnexpectedDataType(dtype);
    }
    int64_t* out_indices =
        sequence_result->sparse_indices[t].flat<int64_t>().data();
    auto out_shape = sequence_result->sparse_shapes[t].vec<int64_t>();

    // Fill in the values.
    size_t num_elements = 0;
    size_t max_num_rows = 0;
    size_t max_num_cols = 0;
    for (int e = 0; e < num_examples; e++) {
      const auto& feature_proto = feature.protos[e];
      if (feature_proto.empty()) continue;
      protobuf::io::CodedInputStream stream(
          reinterpret_cast<const uint8*>(feature_proto.data()),
          feature_proto.size());
      EnableAliasing(&stream);
      size_t num_rows = 0;
      while (!stream.ExpectAtEnd()) {
        uint32 feature_length;
        if (!stream.ExpectTag(kDelimitedTag(1)) ||
            !stream.ReadVarint32(&feature_length)) {
          // This should be unreachable -- we already scanned the feature in
          // GetSequenceFeatureLengths, and it hasn't changed since then.
          return errors::InvalidArgument("Error in sequence feature ",
                                         c.feature_name, " in example ",
                                         ExampleName(example_names, e));
        }
        if (feature_length > 2) {
          auto limit = stream.PushLimit(feature_length);
          size_t num_added;
          switch (dtype) {
            case DT_STRING:
              num_added = ParseBytesFeature(&stream, out_bytes);
              out_bytes += num_added;
              break;
            case DT_FLOAT:
              num_added = ParseFloatFeature(&stream, out_float);
              out_float += num_added;
              break;
            case DT_INT64:
              num_added = ParseInt64Feature(&stream, out_int64);
              out_int64 += num_added;
              break;
            default:
              ReportUnexpectedDataType(dtype);
              num_added = 0;
          }
          num_elements += num_added;
          max_num_cols = std::max(max_num_cols, num_added);
          for (int i = 0; i < num_added; i++) {
            if (is_batch) *out_indices++ = e;
            *out_indices++ = num_rows;
            *out_indices++ = i;
          }
          stream.PopLimit(limit);
        } else if (feature_length == 2) {
          if (!SkipEmptyFeature(&stream, dtype)) {
            // This should be unreachable -- we already scanned the feature in
            // GetSequenceFeatureLengths, and it hasn't changed since then.
            return errors::InvalidArgument("Error in sequence feature ",
                                           c.feature_name, " in example ",
                                           ExampleName(example_names, e));
          }
        } else if (feature_length != 0) {
          // This should be unreachable -- we already scanned the feature in
          // GetSequenceFeatureLengths, and it hasn't changed since then.
          return errors::InvalidArgument("Error in sequence feature ",
                                         c.feature_name, " in example ",
                                         ExampleName(example_names, e));
        }
        num_rows++;
      }
      max_num_rows = std::max(max_num_rows, num_rows);
    }
    if (num_elements != expected_num_elements) {
      return errors::InvalidArgument(
          "Unexpected number of elements in feature ", c.feature_name);
    }
    if (is_batch) {
      out_shape(0) = num_examples;
      out_shape(1) = max_num_rows;
      out_shape(2) = max_num_cols;
    } else {
      out_shape(0) = max_num_rows;
      out_shape(1) = max_num_cols;
    }
  }
  return absl::OkStatus();
}

// Parses ragged features in `sequence_features`, and writes their parsed
// values to `sequence_result`.
absl::Status ParseSequenceRaggedFeatures(
    const FeatureProtosMap& sequence_features,
    const FastParseExampleConfig& sequence_config,
    absl::Span<const tstring> example_names, bool is_batch, int num_examples,
    Allocator* allocator, Result* sequence_result) {
  for (int t = 0; t < sequence_config.ragged.size(); ++t) {
    const auto& c = sequence_config.ragged[t];
    const FeatureProtos& feature =
        sequence_features.find(c.feature_name)->second;
    TensorShape values_shape, inner_splits_shape, outer_splits_shape;
    DataType dtype = c.dtype;
    DataType splits_dtype = c.splits_dtype;
    size_t expected_num_elements = feature.length;
    size_t expected_num_rows = feature.num_rows;
    values_shape.AddDim(expected_num_elements);
    inner_splits_shape.AddDim(expected_num_rows + 1);
    if (is_batch) {
      outer_splits_shape.AddDim(num_examples + 1);
    }
    sequence_result->ragged_values[t] = Tensor(allocator, dtype, values_shape);
    sequence_result->ragged_splits[t] =
        Tensor(allocator, splits_dtype, inner_splits_shape);
    sequence_result->ragged_outer_splits[t] =
        Tensor(allocator, splits_dtype, outer_splits_shape);
    Tensor& out_values = sequence_result->ragged_values[t];
    size_t out_values_offset = 0;
    int32* int32_inner_splits =
        splits_dtype == DT_INT32
            ? sequence_result->ragged_splits[t].vec<int32>().data()
            : nullptr;
    int64_t* int64_inner_splits =
        splits_dtype == DT_INT64
            ? sequence_result->ragged_splits[t].vec<int64_t>().data()
            : nullptr;
    int32* int32_outer_splits =
        is_batch && splits_dtype == DT_INT32
            ? sequence_result->ragged_outer_splits[t].vec<int32>().data()
            : nullptr;
    int64_t* int64_outer_splits =
        is_batch && splits_dtype == DT_INT64
            ? sequence_result->ragged_outer_splits[t].vec<int64_t>().data()
            : nullptr;
    if (int32_inner_splits) {
      *int32_inner_splits++ = 0;
    } else if (int64_inner_splits) {
      *int64_inner_splits++ = 0;
    }
    if (int32_outer_splits) {
      *int32_outer_splits++ = 0;
    } else if (int64_outer_splits) {
      *int64_outer_splits++ = 0;
    }

    // Fill in the values.
    size_t inner_split = 0;  // total number of elements we've seen so far
    size_t outer_split = 0;  // total number of rows we've seen so far
    for (int e = 0; e < num_examples; e++) {
      const auto& feature_proto = feature.protos[e];
      if (!feature_proto.empty()) {
        protobuf::io::CodedInputStream stream(
            reinterpret_cast<const uint8*>(feature_proto.data()),
            feature_proto.size());
        EnableAliasing(&stream);
        while (!stream.ExpectAtEnd()) {
          uint32 feature_length;
          if (!stream.ExpectTag(kDelimitedTag(1)) ||
              !stream.ReadVarint32(&feature_length)) {
            // This should be unreachable -- we already scanned the feature in
            // GetSequenceFeatureLengths, and it hasn't changed since then.
            return errors::InvalidArgument("Error in sequence feature ",
                                           c.feature_name, " in example ",
                                           ExampleName(example_names, e));
          }
          if (feature_length > 2) {
            auto limit = stream.PushLimit(feature_length);
            size_t num_added =
                ParseFeature(dtype, &stream, &out_values, &out_values_offset);
            inner_split += num_added;
            stream.PopLimit(limit);
          } else if (feature_length == 2) {
            if (!SkipEmptyFeature(&stream, dtype)) {
              // This should be unreachable -- we already scanned the feature in
              // GetSequenceFeatureLengths, and it hasn't changed since then.
              return errors::InvalidArgument("Error in sequence feature ",
                                             c.feature_name, " in example ",
                                             ExampleName(example_names, e));
            }
          } else if (feature_length != 0) {
            // This should be unreachable -- we already scanned the feature in
            // GetSequenceFeatureLengths, and it hasn't changed since then.
            return errors::InvalidArgument("Error in sequence feature ",
                                           c.feature_name, " in example ",
                                           ExampleName(example_names, e));
          }
          if (int32_inner_splits) {
            *int32_inner_splits++ = inner_split;
          } else if (int64_inner_splits) {
            *int64_inner_splits++ = inner_split;
          }
          outer_split++;
        }
      }
      if (int32_outer_splits) {
        *int32_outer_splits++ = outer_split;
      } else if (int64_outer_splits) {
        *int64_outer_splits++ = outer_split;
      }
    }
    if (outer_split != expected_num_rows) {
      return errors::InvalidArgument("Unexpected number of rows for feature ",
                                     c.feature_name);
    }
    if (inner_split != expected_num_elements) {
      return errors::InvalidArgument(
          "Unexpected number of elements for feature ", c.feature_name);
    }

    if (int32_inner_splits || int64_inner_splits) {
      const auto& inner_splits = sequence_result->ragged_splits[t];
      int num_inner_splits =
          int32_inner_splits
              ? int32_inner_splits - inner_splits.vec<int32>().data()
              : int64_inner_splits - inner_splits.vec<int64_t>().data();
      if (num_inner_splits != expected_num_rows + 1) {
        return errors::InvalidArgument("Unexpected number of rows for feature ",
                                       c.feature_name);
      }
    }
    if (int32_outer_splits || int64_outer_splits) {
      const auto& outer_splits = sequence_result->ragged_outer_splits[t];
      int num_outer_splits =
          int32_outer_splits
              ? int32_outer_splits - outer_splits.vec<int32>().data()
              : int64_outer_splits - outer_splits.vec<int64_t>().data();
      if (num_outer_splits != num_examples + 1) {
        return errors::InvalidArgument(
            "Unexpected number of examples for feature ", c.feature_name);
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

// TODO(sundberg): Use the threadpool to parallelize example parsing.
// TODO(b/111553342): Support extracting feature statistics from the examples.
absl::Status FastParseSequenceExample(
    const FastParseExampleConfig& context_config,
    const FastParseExampleConfig& sequence_config,
    absl::Span<const tstring> serialized,
    absl::Span<const tstring> example_names, thread::ThreadPool* thread_pool,
    Result* context_result, Result* sequence_result,
    std::vector<Tensor>* dense_feature_lengths, bool is_batch) {
  int num_examples = serialized.size();
  DCHECK(context_result != nullptr);
  DCHECK(sequence_result != nullptr);
  DCHECK(dense_feature_lengths != nullptr);
  size_t num_context_features = context_config.sparse.size() +
                                context_config.dense.size() +
                                context_config.ragged.size();
  FeatureProtosMap context_features;
  context_features.reserve(num_context_features);

  if (!example_names.empty() && example_names.size() != num_examples) {
    return errors::InvalidArgument(
        "example_names must be empty or have the correct number of elements");
  }
  for (auto& c : context_config.sparse) {
    TF_RETURN_IF_ERROR(CheckConfigDataType(c.dtype));
    FeatureProtos& feature = context_features[c.feature_name];
    feature.dtype = c.dtype;
    feature.length = 0;
    feature.type = Type::Sparse;
    feature.protos.resize(num_examples);
    feature.protos_present.resize(num_examples);
  }
  for (auto& c : context_config.ragged) {
    TF_RETURN_IF_ERROR(CheckConfigDataType(c.dtype));
    FeatureProtos& feature = context_features[c.feature_name];
    if (feature.type == Type::Sparse) {
      return errors::InvalidArgument("Context feature " + c.feature_name +
                                     " cannot be both ragged and sparse");
    }
    feature.dtype = c.dtype;
    feature.length = 0;
    feature.type = Type::Ragged;
    feature.protos.resize(num_examples);
    feature.protos_present.resize(num_examples);
  }
  for (auto& c : context_config.dense) {
    TF_RETURN_IF_ERROR(CheckConfigDataType(c.dtype));
    FeatureProtos& feature = context_features[c.feature_name];
    if (feature.type != Type::Dense) {
      return errors::InvalidArgument("Context feature " + c.feature_name +
                                     " cannot be both dense and sparse");
    }
    if (c.default_value.NumElements() > 0) {
      if (!c.shape.IsCompatibleWith(c.default_value.shape())) {
        return errors::InvalidArgument("Default value for context feature ",
                                       c.feature_name,
                                       " has an incorrect shape: saw ",
                                       c.default_value.shape().DebugString(),
                                       " but expected ", c.shape.DebugString());
      }
    }
    feature.dtype = c.dtype;
    feature.length = c.default_value.NumElements();
    feature.protos.resize(num_examples);
    feature.protos_present.resize(num_examples);
  }
  size_t num_sequence_features = sequence_config.sparse.size() +
                                 sequence_config.dense.size() +
                                 sequence_config.ragged.size();
  FeatureProtosMap sequence_features;
  sequence_features.reserve(num_sequence_features);
  for (auto& c : sequence_config.sparse) {
    TF_RETURN_IF_ERROR(CheckConfigDataType(c.dtype));
    FeatureProtos& feature = sequence_features[c.feature_name];
    feature.dtype = c.dtype;
    feature.length = 0;
    feature.type = Type::Sparse;
    feature.protos.resize(num_examples);
    feature.protos_present.resize(num_examples);
  }
  for (auto& c : sequence_config.ragged) {
    TF_RETURN_IF_ERROR(CheckConfigDataType(c.dtype));
    FeatureProtos& feature = sequence_features[c.feature_name];
    if (feature.type == Type::Sparse) {
      return errors::InvalidArgument("Sequence feature " + c.feature_name +
                                     " cannot be both ragged and sparse");
    }
    feature.dtype = c.dtype;
    feature.length = 0;
    feature.type = Type::Ragged;
    feature.protos.resize(num_examples);
    feature.protos_present.resize(num_examples);
  }
  for (auto& c : sequence_config.dense) {
    TF_RETURN_IF_ERROR(CheckConfigDataType(c.dtype));
    FeatureProtos& feature = sequence_features[c.feature_name];
    if (feature.type != Type::Dense) {
      return errors::InvalidArgument("Sequence feature " + c.feature_name +
                                     " cannot be both dense and sparse");
    }
    feature.dtype = c.dtype;
    feature.length = 0;
    feature.protos.resize(num_examples);
    feature.protos_present.resize(num_examples);
  }

  // Find the serialized proto substrings for each feature.
  TF_RETURN_IF_ERROR(ExtractFeaturesFromSequenceExamples(
      serialized, example_names, &context_features, &sequence_features));

  // Scan through the protos to determine how much memory we need to allocate.
  TF_RETURN_IF_ERROR(
      GetContextFeatureLengths(example_names, &context_features));
  TF_RETURN_IF_ERROR(
      GetSequenceFeatureLengths(example_names, &sequence_features));

  // Allocate memory.
  context_result->sparse_values.resize(context_config.sparse.size());
  context_result->sparse_indices.resize(context_config.sparse.size());
  context_result->sparse_shapes.resize(context_config.sparse.size());
  context_result->dense_values.resize(context_config.dense.size());
  context_result->ragged_values.resize(context_config.ragged.size());
  context_result->ragged_splits.resize(context_config.ragged.size());
  context_result->ragged_outer_splits.resize(context_config.ragged.size());
  sequence_result->sparse_values.resize(sequence_config.sparse.size());
  sequence_result->sparse_indices.resize(sequence_config.sparse.size());
  sequence_result->sparse_shapes.resize(sequence_config.sparse.size());
  sequence_result->dense_values.resize(sequence_config.dense.size());
  sequence_result->ragged_values.resize(sequence_config.ragged.size());
  sequence_result->ragged_splits.resize(sequence_config.ragged.size());
  sequence_result->ragged_outer_splits.resize(sequence_config.ragged.size());
  dense_feature_lengths->resize(sequence_config.dense.size());

  // NOTE(mrry): Cache the CPU allocator here and use it in Tensor construction,
  // to avoid lock contention in `tensorflow::cpu_allocator()`.
  Allocator* allocator = tensorflow::cpu_allocator();

  TF_RETURN_IF_ERROR(ParseContextDenseFeatures(
      context_features, context_config, example_names, is_batch, num_examples,
      allocator, context_result));
  TF_RETURN_IF_ERROR(ParseContextSparseFeatures(
      context_features, context_config, example_names, is_batch, num_examples,
      allocator, context_result));
  TF_RETURN_IF_ERROR(ParseContextRaggedFeatures(
      context_features, context_config, example_names, is_batch, num_examples,
      allocator, context_result));
  TF_RETURN_IF_ERROR(ParseSequenceDenseFeatures(
      sequence_features, sequence_config, example_names, is_batch, num_examples,
      allocator, sequence_result, dense_feature_lengths));
  TF_RETURN_IF_ERROR(ParseSequenceSparseFeatures(
      sequence_features, sequence_config, example_names, is_batch, num_examples,
      allocator, sequence_result));
  TF_RETURN_IF_ERROR(ParseSequenceRaggedFeatures(
      sequence_features, sequence_config, example_names, is_batch, num_examples,
      allocator, sequence_result));

  return absl::OkStatus();
}

}  // namespace example
}  // namespace tensorflow
