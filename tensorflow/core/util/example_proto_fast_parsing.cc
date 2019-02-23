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

#include <vector>

#include "absl/base/casts.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb_text.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/monitoring/counter.h"
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
  Feature() {}
  explicit Feature(StringPiece serialized) : serialized_(serialized) {}

  Status ParseDataType(DataType* dtype) {
    DCHECK(dtype != nullptr);
    if (serialized_.empty()) {
      *dtype = DT_INVALID;
      return Status::OK();
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
    return Status::OK();
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
      string bytes;
      if (!stream.ReadString(&bytes, bytes_length)) return false;
      bytes_list->push_back(std::move(bytes));
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

      if (peek_tag == kDelimitedTag(1)) {                       // packed
        if (!stream.ExpectTag(kDelimitedTag(1))) return false;  // packed tag
        uint32 packed_length;
        if (!stream.ReadVarint32(&packed_length)) return false;
        auto packed_limit = stream.PushLimit(packed_length);

        // If the result data type is float and we are on a little endian
        // machine then we can simply memcpy the data from the proto into the
        // result vector.
        constexpr int32 kNumFloatBytes = 4;
        if (port::kLittleEndian &&
            sizeof(typename Result::value_type) == kNumFloatBytes) {
          // Store the initial size to know the offset we have to start writing
          // data from before resizing the output "vector".
          const size_t initial_size = float_list->size();
          float_list->resize(initial_size + packed_length / kNumFloatBytes);
          // Calculate the length of the buffer available what can be less than
          // what we requested in resize in case of a LimitedArraySlice.
          const uint32 bytes_to_copy =
              std::min(static_cast<uint32>((float_list->size() - initial_size) *
                                           kNumFloatBytes),
                       packed_length);
          if (!stream.ReadRaw(float_list->data() + initial_size, bytes_to_copy))
            return false;
        } else {
          while (!stream.ExpectAtEnd()) {
            uint32 buffer32;
            if (!stream.ReadLittleEndian32(&buffer32)) return false;
            float_list->push_back(absl::bit_cast<float>(buffer32));
          }
        }

        stream.PopLimit(packed_limit);
      } else {  // non-packed
        while (!stream.ExpectAtEnd()) {
          if (!stream.ExpectTag(kFixed32Tag(1))) return false;
          uint32 buffer32;
          if (!stream.ReadLittleEndian32(&buffer32)) return false;
          float_list->push_back(absl::bit_cast<float>(buffer32));
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
          int64_list->push_back(static_cast<int64>(n));
        }

        stream.PopLimit(packed_limit);
      } else {  // non-packed
        while (!stream.ExpectAtEnd()) {
          if (!stream.ExpectTag(kVarintTag(1))) return false;
          protobuf_uint64 n;  // There is no API for int64
          if (!stream.ReadVarint64(&n)) return false;
          int64_list->push_back(static_cast<int64>(n));
        }
      }
    }
    stream.PopLimit(limit);
    return true;
  }

  StringPiece GetSerialized() const { return serialized_; }

 private:
  // TODO(lew): Pair of uint8* would be more natural.
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

bool ParseString(protobuf::io::CodedInputStream* stream, StringPiece* result) {
  DCHECK(stream != nullptr);
  DCHECK(result != nullptr);
  uint32 length;
  if (!stream->ReadVarint32(&length)) return false;
  if (length == 0) {
    *result = StringPiece(nullptr, 0);
    return true;
  }
  const void* stream_alias;
  int stream_size;
  if (!stream->GetDirectBufferPointer(&stream_alias, &stream_size)) {
    return false;
  }
  if (static_cast<uint32>(stream_size) < length) return false;
  *result = StringPiece(static_cast<const char*>(stream_alias), length);
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
  if (!stream->ExpectTag(kDelimitedTag(1))) return false;
  if (!ParseString(stream, &feature_map_entry->first)) return false;
  if (!stream->ExpectTag(kDelimitedTag(2))) return false;
  StringPiece feature_string_piece;
  if (!ParseString(stream, &feature_string_piece)) return false;
  feature_map_entry->second = parsed::Feature(feature_string_piece);
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

bool ParseExample(StringPiece serialized, parsed::Example* example) {
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
        SmallVector<string> list;
        if (!name_and_feature.second.ParseBytesList(&list)) return false;
        auto* result_list = value.mutable_bytes_list();
        for (auto& bytes : list) {
          auto* new_value = result_list->add_value();
          new_value->swap(bytes);
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
        SmallVector<int64> list;
        if (!name_and_feature.second.ParseInt64List(&list)) return false;
        auto* result_list = value.mutable_int64_list();
        for (int64 i : list) {
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

enum class Type { Sparse, Dense };

struct SparseBuffer {
  // Features are in one of the 3 vectors below depending on config's dtype.
  // Other 2 vectors remain empty.
  SmallVector<string> bytes_list;
  SmallVector<float> float_list;
  SmallVector<int64> int64_list;

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

template <typename T>
class LimitedArraySlice {
 public:
  using value_type = T;

  LimitedArraySlice(T* begin, size_t num_elements)
      : current_(begin), begin_(begin), end_(begin + num_elements) {}

  // May return negative if there were push_back calls after slice was filled.
  int64 EndDistance() const { return end_ - current_; }

  // Attempts to push value to the back of this. If the slice has
  // already been filled, this method has no effect on the underlying data, but
  // it changes the number returned by EndDistance into negative values.
  void push_back(T&& value) {
    if (EndDistance() > 0) *current_ = std::move(value);
    ++current_;
  }

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

void LogDenseFeatureDataLoss(StringPiece feature_name) {
  LOG(WARNING) << "Data loss! Feature '" << feature_name
               << "' is present in multiple concatenated "
                  "tf.Examples. Ignoring all but last one.";
  static auto* duplicated_dense_feature = monitoring::Counter<0>::New(
      "/tensorflow/core/util/example_proto_fast_parsing/"
      "duplicated_dense_feature",
      "Dense feature appears twice in a tf.Example");
  duplicated_dense_feature->GetCell()->IncrementBy(1);
}

void LogSparseFeatureDataLoss(StringPiece feature_name) {
  LOG(WARNING) << "Data loss! Feature '" << feature_name
               << "' is present in multiple concatenated "
                  "tf.Examples. Ignoring all but last one.";
  static auto* duplicated_sparse_feature = monitoring::Counter<0>::New(
      "/tensorflow/core/util/example_proto_fast_parsing/"
      "duplicated_sparse_feature",
      "Sparse feature appears twice in a tf.Example");
  duplicated_sparse_feature->GetCell()->IncrementBy(1);
}

Status FastParseSerializedExample(
    const string& serialized_example, const string& example_name,
    const size_t example_index, const Config& config,
    const PresizedCuckooMap<std::pair<size_t, Type>>& config_index,
    SeededHasher hasher, std::vector<Tensor>* output_dense,
    std::vector<SparseBuffer>* output_varlen_dense,
    std::vector<SparseBuffer>* output_sparse,
    PerExampleFeatureStats* output_stats) {
  DCHECK(output_dense != nullptr);
  DCHECK(output_sparse != nullptr);
  parsed::Example parsed_example;
  if (!ParseExample(serialized_example, &parsed_example)) {
    return errors::InvalidArgument("Could not parse example input, value: '",
                                   serialized_example, "'");
  }
  std::vector<int64> sparse_feature_last_example(config.sparse.size(), -1);
  std::vector<int64> dense_feature_last_example(config.dense.size(), -1);

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

    const StringPiece feature_name = name_and_feature.first;
    parsed::Feature& feature = name_and_feature.second;

    std::pair<size_t, Type> d_and_type;
    uint64 h = hasher(feature_name);
    if (!config_index.Find(h, &d_and_type)) continue;

    size_t d = d_and_type.first;
    bool is_dense = d_and_type.second == Type::Dense;

    {
      // Testing for PresizedCuckooMap collision.
      // TODO(lew): Use dense_hash_map and avoid this and hasher creation.
      const string& config_feature_name = is_dense
                                              ? config.dense[d].feature_name
                                              : config.sparse[d].feature_name;
      if (feature_name != config_feature_name) continue;
    }

    auto example_error = [&](StringPiece suffix) {
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

        auto shape_error = [&](size_t size, StringPiece type_str) {
          return example_error(strings::StrCat(
              "Number of ", type_str,
              " values != expected.  "
              "Values size: ",
              size,
              " but output shape: ", config.dense[d].shape.DebugString()));
        };

        switch (config.dense[d].dtype) {
          case DT_INT64: {
            auto out_p = out.flat<int64>().data() + offset;
            LimitedArraySlice<int64> slice(out_p, num_elements);
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
            auto out_p = out.flat<string>().data() + offset;
            LimitedArraySlice<string> slice(out_p, num_elements);
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

        auto shape_error = [&](size_t size, StringPiece type_str) {
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
      // If feature was already visited, skip.
      // Compare comment at the beginning of the loop.
      if (sparse_feature_last_example[d] == example_index) {
        LogSparseFeatureDataLoss(feature_name);
        continue;
      }
      sparse_feature_last_example[d] = example_index;

      // Handle sparse features.
      SparseBuffer& out = (*output_sparse)[d];
      if (example_dtype != DT_INVALID &&
          example_dtype != config.sparse[d].dtype) {
        return example_error(strings::StrCat(
            "Data types don't match. ",
            "Expected type: ", DataTypeString(config.sparse[d].dtype),
            ", Actual type: ", DataTypeString(example_dtype)));
      }

      switch (config.sparse[d].dtype) {
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
        std::copy_n(in.flat<int64>().data(), num_elements,
                    out.flat<int64>().data() + offset);
        break;
      }
      case DT_FLOAT: {
        std::copy_n(in.flat<float>().data(), num_elements,
                    out.flat<float>().data() + offset);
        break;
      }
      case DT_STRING: {
        std::copy_n(in.flat<string>().data(), num_elements,
                    out.flat<string>().data() + offset);
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

  return Status::OK();
}

Status CheckConfigDataType(DataType dtype) {
  switch (dtype) {
    case DT_INT64:
    case DT_FLOAT:
    case DT_STRING:
      return Status::OK();
    default:
      return errors::InvalidArgument("Invalid config dtype: ",
                                     DataTypeString(dtype));
  }
}

template <typename T>
const SmallVector<T>& GetListFromBuffer(const SparseBuffer& buffer);

template <>
const SmallVector<int64>& GetListFromBuffer<int64>(const SparseBuffer& buffer) {
  return buffer.int64_list;
}
template <>
const SmallVector<float>& GetListFromBuffer<float>(const SparseBuffer& buffer) {
  return buffer.float_list;
}
template <>
const SmallVector<string>& GetListFromBuffer<string>(
    const SparseBuffer& buffer) {
  return buffer.bytes_list;
}

template <typename T>
void CopyOrMoveBlock(const T* b, const T* e, T* t) {
  std::copy(b, e, t);
}
template <>
void CopyOrMoveBlock(const string* b, const string* e, string* t) {
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

}  // namespace

Status FastParseExample(const Config& config,
                        gtl::ArraySlice<string> serialized,
                        gtl::ArraySlice<string> example_names,
                        thread::ThreadPool* thread_pool, Result* result) {
  DCHECK(result != nullptr);
  // Check config so we can safely CHECK(false) in switches on config.*.dtype
  for (auto& c : config.sparse) {
    TF_RETURN_IF_ERROR(CheckConfigDataType(c.dtype));
  }
  for (auto& c : config.dense) {
    TF_RETURN_IF_ERROR(CheckConfigDataType(c.dtype));
  }

  if (config.collect_feature_stats) {
    result->feature_stats.resize(serialized.size());
  }

  size_t config_size = config.dense.size() + config.sparse.size();
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
    if (ok) break;
    LOG(WARNING) << "Collision found. This should happen only if you have "
                    "around 2^32 entries in your config.";
    hasher.seed++;
    config_index.Clear(config_size);
  }
  if (!ok) {
    return errors::Internal(
        "Could not avoid collision. This should not happen.");
  }

  // Allocate dense output for fixed length dense values
  // (variable-length dense and sparse have to be buffered).
  std::vector<Tensor> fixed_dense_values(config.dense.size());
  for (size_t d = 0; d < config.dense.size(); ++d) {
    if (config.dense[d].variable_length) continue;
    TensorShape out_shape;
    out_shape.AddDim(serialized.size());
    for (const int64 dim : config.dense[d].shape.dim_sizes()) {
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
  std::vector<Status> status_of_minibatch(num_minibatches);
  auto ProcessMiniBatch = [&](size_t minibatch) {
    sparse_buffers[minibatch].resize(config.sparse.size());
    varlen_dense_buffers[minibatch].resize(config.dense.size());
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
          &varlen_dense_buffers[minibatch], &sparse_buffers[minibatch], stats);
      if (!status_of_minibatch[minibatch].ok()) break;
    }
  };

  ParallelFor(ProcessMiniBatch, num_minibatches, thread_pool);

  for (Status& status : status_of_minibatch) {
    TF_RETURN_IF_ERROR(status);
  }

  for (size_t d = 0; d < config.dense.size(); ++d) {
    result->dense_values.push_back(std::move(fixed_dense_values[d]));
  }

  // Merge SparseBuffers from all minibatches for every config.sparse.
  auto MergeSparseMinibatches = [&](size_t d) {
    // Loop over minibatches
    size_t total_num_features = 0;
    size_t max_num_features = 0;
    for (auto& sparse_values_tmp : sparse_buffers) {
      const std::vector<size_t>& end_indices =
          sparse_values_tmp[d].example_end_indices;
      total_num_features += end_indices.back();
      max_num_features = std::max(max_num_features, end_indices[0]);
      for (size_t i = 1; i < end_indices.size(); ++i) {
        size_t example_size = end_indices[i] - end_indices[i - 1];
        max_num_features = std::max(max_num_features, example_size);
      }
    }

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
    auto shapes_shape_t = result->sparse_shapes.back().vec<int64>();
    shapes_shape_t(0) = serialized.size();
    shapes_shape_t(1) = max_num_features;

    size_t offset = 0;
    for (size_t i = 0; i < sparse_buffers.size(); ++i) {
      const SparseBuffer& buffer = sparse_buffers[i][d];

      // Update indices.
      int64* ix_p = &indices->matrix<int64>()(offset, 0);
      size_t delta = 0;
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

      // Copy values over.
      switch (config.sparse[d].dtype) {
        case DT_INT64: {
          std::copy(buffer.int64_list.begin(), buffer.int64_list.end(),
                    values->flat<int64>().data() + offset);
          break;
        }
        case DT_FLOAT: {
          std::copy(buffer.float_list.begin(), buffer.float_list.end(),
                    values->flat<float>().data() + offset);
          break;
        }
        case DT_STRING: {
          std::move(buffer.bytes_list.begin(), buffer.bytes_list.end(),
                    values->flat<string>().data() + offset);
          break;
        }
        default:
          LOG(FATAL) << "Should not happen.";
      }

      offset += delta;
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
        FillAndCopyVarLen<int64>(d, num_elements, num_elements_per_minibatch,
                                 config, varlen_dense_buffers, &values);
        break;
      }
      case DT_FLOAT: {
        FillAndCopyVarLen<float>(d, num_elements, num_elements_per_minibatch,
                                 config, varlen_dense_buffers, &values);
        break;
      }
      case DT_STRING: {
        FillAndCopyVarLen<string>(d, num_elements, num_elements_per_minibatch,
                                  config, varlen_dense_buffers, &values);
        break;
      }
      default:
        LOG(FATAL) << "Should not happen.";
    }
  };

  for (size_t d = 0; d < config.dense.size(); ++d) {
    MergeDenseVarLenMinibatches(d);
  }

  for (size_t d = 0; d < config.sparse.size(); ++d) {
    MergeSparseMinibatches(d);
  }

  return Status::OK();
}

Status FastParseSingleExample(const Config& config, const string& serialized,
                              Result* result) {
  DCHECK(result != nullptr);
  // Check config so we can safely CHECK(false) in switches on config.*.dtype
  for (auto& c : config.sparse) {
    TF_RETURN_IF_ERROR(CheckConfigDataType(c.dtype));
  }
  for (auto& c : config.dense) {
    TF_RETURN_IF_ERROR(CheckConfigDataType(c.dtype));
  }

  PerExampleFeatureStats* stats = nullptr;
  if (config.collect_feature_stats) {
    result->feature_stats.emplace_back();
    stats = &result->feature_stats.back();
  }

  // TODO(mrry): Cache the construction of this map at Op construction time.
  size_t config_size = config.dense.size() + config.sparse.size();
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
    if (ok) break;
    LOG(WARNING) << "Collision found. This should happen only if you have "
                    "around 2^32 entries in your config.";
    hasher.seed++;
    config_index.Clear(config_size);
  }
  if (!ok) {
    return errors::Internal(
        "Could not avoid collision. This should not happen.");
  }

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

  parsed::Example parsed_example;
  if (!ParseExample(serialized, &parsed_example)) {
    return errors::InvalidArgument("Could not parse example input, value: '",
                                   serialized, "'");
  }
  std::vector<bool> sparse_feature_already_seen(config.sparse.size(), false);
  std::vector<bool> dense_feature_already_seen(config.dense.size(), false);

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

    const StringPiece feature_name = name_and_feature.first;
    parsed::Feature& feature = name_and_feature.second;

    std::pair<size_t, Type> d_and_type;
    uint64 h = hasher(feature_name);
    if (!config_index.Find(h, &d_and_type)) continue;

    size_t d = d_and_type.first;
    bool is_dense = d_and_type.second == Type::Dense;

    {
      // Testing for PresizedCuckooMap collision.
      // TODO(lew): Use dense_hash_map and avoid this and hasher creation.
      const string& config_feature_name = is_dense
                                              ? config.dense[d].feature_name
                                              : config.sparse[d].feature_name;
      if (feature_name != config_feature_name) continue;
    }

    auto example_error = [feature_name](StringPiece suffix) {
      return errors::InvalidArgument("Key: ", feature_name, ".  ", suffix);
    };

    auto parse_error = [feature_name] {
      return errors::InvalidArgument("Key: ", feature_name,
                                     ".  Can't parse serialized Example.");
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
          auto out_p = out->flat<int64>().data();
          LimitedArraySlice<int64> slice(out_p, num_elements);
          if (!feature.ParseInt64List(&slice)) return parse_error();
          if (slice.EndDistance() != 0) {
            return parse_error();
          }
          break;
        }
        case DT_FLOAT: {
          auto out_p = out->flat<float>().data();
          LimitedArraySlice<float> slice(out_p, num_elements);
          if (!feature.ParseFloatList(&slice)) return parse_error();
          if (slice.EndDistance() != 0) {
            return parse_error();
          }
          break;
        }
        case DT_STRING: {
          auto out_p = out->flat<string>().data();
          LimitedArraySlice<string> slice(out_p, num_elements);
          if (!feature.ParseBytesList(&slice)) return parse_error();
          if (slice.EndDistance() != 0) {
            return parse_error();
          }
          break;
        }
        default:
          LOG(FATAL) << "Should not happen.";
      }

    } else {  // if variable length
      SparseBuffer out_temp;
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
        // If feature was already visited, skip.
        // Compare comment at the beginning of the loop.
        if (sparse_feature_already_seen[d]) {
          LogSparseFeatureDataLoss(feature_name);
          continue;
        }
        sparse_feature_already_seen[d] = true;

        // Handle sparse features.
        if (example_dtype != DT_INVALID &&
            example_dtype != config.sparse[d].dtype) {
          return example_error(strings::StrCat(
              "Data types don't match. ",
              "Expected type: ", DataTypeString(config.sparse[d].dtype),
              ", Actual type: ", DataTypeString(example_dtype)));
        }
      }

      switch (example_dtype) {
        case DT_INT64: {
          // TODO(mrry): Use the fact that the `int64_list` is packed to read
          // out the length and pre-allocate the output tensor.
          if (!feature.ParseInt64List(&out_temp.int64_list))
            return parse_error();
          num_elements = out_temp.int64_list.size();
          break;
        }
        case DT_FLOAT: {
          // TODO(mrry): Use the fact that the `float_list` is packed to read
          // out the length and pre-allocate the output tensor.
          if (!feature.ParseFloatList(&out_temp.float_list))
            return parse_error();
          num_elements = out_temp.float_list.size();
          break;
        }
        case DT_STRING: {
          int actual_num_elements = 0;
          if (!feature.GetNumElementsInBytesList(&actual_num_elements)) {
            return parse_error();
          }
          out_temp.bytes_list.reserve(actual_num_elements);
          if (!feature.ParseBytesList(&out_temp.bytes_list))
            return parse_error();
          num_elements = out_temp.bytes_list.size();
          break;
        }
        default:
          LOG(FATAL) << "Should not happen. " << DataTypeString(example_dtype);
      }

      if (num_elements % num_elements_divisor != 0) {
        return parse_error();
      }

      if (stats) {
        stats->feature_values_count += num_elements;
      }

      Tensor* out;
      if (is_dense) {
        TensorShape values_shape;
        values_shape.AddDim(num_elements / num_elements_divisor);
        for (int i = 1; i < config.dense[d].shape.dims(); ++i) {
          values_shape.AddDim(config.dense[d].shape.dim_size(i));
        }

        out = &result->dense_values[d];
        *out = Tensor(config.dense[d].dtype, values_shape);

      } else {
        Tensor* out_indices = &result->sparse_indices[d];
        Tensor* out_dense_shape = &result->sparse_shapes[d];
        out = &result->sparse_values[d];

        // TODO(mrry): Investigate the possibility of not materializing
        // the indices (and perhaps dense_shape) until they are needed.
        *out_indices = Tensor(
            DT_INT64, TensorShape({static_cast<int64>(num_elements), 1}));
        auto indices_flat = out_indices->flat<int64>();
        for (size_t i = 0; i < num_elements; ++i) {
          indices_flat(i) = static_cast<int64>(i);
        }

        *out_dense_shape = Tensor(DT_INT64, TensorShape({1}));
        auto shapes_shape_t = out_dense_shape->vec<int64>();
        shapes_shape_t(0) = num_elements;

        *out = Tensor(config.sparse[d].dtype,
                      TensorShape({static_cast<int64>(num_elements)}));
      }

      switch (example_dtype) {
        case DT_INT64: {
          CopyOrMoveBlock(out_temp.int64_list.begin(),
                          out_temp.int64_list.end(), out->flat<int64>().data());
          break;
        }
        case DT_FLOAT: {
          CopyOrMoveBlock(out_temp.float_list.begin(),
                          out_temp.float_list.end(), out->flat<float>().data());
          break;
        }
        case DT_STRING: {
          CopyOrMoveBlock(out_temp.bytes_list.begin(),
                          out_temp.bytes_list.end(),
                          out->flat<string>().data());
          break;
        }
        default:
          LOG(FATAL) << "Should not happen.";
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
      result->sparse_shapes[d].vec<int64>()(0) = 0;
    }
  }

  return Status::OK();
}

// Return the number of bytes elements parsed, or -1 on error. If out is null,
// this method simply counts the number of elements without any copying.
inline int ParseBytesFeature(protobuf::io::CodedInputStream* stream,
                             string* out) {
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
          !stream->ReadVarint32(&bytes_length) ||
          (out != nullptr && !stream->ReadString(out++, bytes_length))) {
        return -1;
      }
      if (out == nullptr) {
        stream->Skip(bytes_length);
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

inline void PadInt64Feature(int num_to_pad, int64* out) {
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
                             int64* out) {
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

// TODO(sundberg): Use the threadpool to parallelize example parsing.
// TODO(b/111553342): Support extracting feature statistics from the examples.
Status FastParseSequenceExample(
    const FastParseExampleConfig& context_config,
    const FastParseExampleConfig& feature_list_config,
    gtl::ArraySlice<string> serialized, gtl::ArraySlice<string> example_names,
    thread::ThreadPool* thread_pool, Result* context_result,
    Result* feature_list_result, std::vector<Tensor>* dense_feature_lengths) {
  int num_examples = serialized.size();
  DCHECK(context_result != nullptr);
  DCHECK(feature_list_result != nullptr);
  DCHECK(dense_feature_lengths != nullptr);
  std::map<StringPiece, bool> context_is_sparse;
  std::map<StringPiece, std::pair<DataType, size_t>>
      context_feature_type_and_lengths;
  if (!example_names.empty() && example_names.size() != num_examples) {
    return errors::InvalidArgument(
        "example_names must be empty or have the correct number of elements");
  }
  for (auto& c : context_config.sparse) {
    TF_RETURN_IF_ERROR(CheckConfigDataType(c.dtype));
    context_feature_type_and_lengths[c.feature_name] =
        std::make_pair(c.dtype, 0);
    context_is_sparse[c.feature_name] = true;
  }
  for (auto& c : context_config.dense) {
    if (context_is_sparse[c.feature_name]) {
      return errors::InvalidArgument("Context feature " + c.feature_name +
                                     " cannot be both dense and sparse");
    }
    TF_RETURN_IF_ERROR(CheckConfigDataType(c.dtype));
    context_feature_type_and_lengths[c.feature_name] =
        std::make_pair(c.dtype, c.default_value.NumElements());
    if (c.default_value.NumElements() > 0) {
      if (!c.shape.IsCompatibleWith(c.default_value.shape())) {
        return errors::InvalidArgument("Default value for context feature ",
                                       c.feature_name,
                                       " has an incorrect shape: saw ",
                                       c.default_value.shape().DebugString(),
                                       " but expected ", c.shape.DebugString());
      }
    }
    context_is_sparse[c.feature_name] = false;
  }
  std::map<StringPiece, bool> sequence_is_sparse;
  std::map<StringPiece, std::pair<DataType, size_t>>
      sequence_feature_type_and_lengths;
  for (auto& c : feature_list_config.sparse) {
    TF_RETURN_IF_ERROR(CheckConfigDataType(c.dtype));
    sequence_feature_type_and_lengths[c.feature_name] =
        std::make_pair(c.dtype, 0);
    sequence_is_sparse[c.feature_name] = true;
  }
  for (auto& c : feature_list_config.dense) {
    if (sequence_is_sparse[c.feature_name]) {
      return errors::InvalidArgument("Sequence feature " + c.feature_name +
                                     " cannot be both dense and sparse");
    }
    TF_RETURN_IF_ERROR(CheckConfigDataType(c.dtype));
    sequence_feature_type_and_lengths[c.feature_name] =
        std::make_pair(c.dtype, 0);
    sequence_is_sparse[c.feature_name] = false;
  }

  std::vector<std::map<StringPiece, StringPiece>> all_context_features(
      num_examples);
  std::vector<std::map<StringPiece, StringPiece>> all_sequence_features(
      num_examples);
  const string kUnknown = "<unknown>";
  for (int d = 0; d < num_examples; d++) {
    const string& example = serialized[d];
    const string& example_name =
        example_names.empty() ? kUnknown : example_names[d];
    auto* context_features = &all_context_features[d];
    auto* sequence_features = &all_sequence_features[d];

    protobuf::io::CodedInputStream stream(
        reinterpret_cast<const uint8*>(example.data()), example.size());
    // Not clear what this does. Why not stream.EnableAliasing()?
    EnableAliasing(&stream);

    // Extract pointers to all features within this serialized example.
    while (!stream.ExpectAtEnd()) {
      std::map<StringPiece, StringPiece>* features = nullptr;
      const std::map<StringPiece, std::pair<DataType, size_t>>* config =
          nullptr;
      if (stream.ExpectTag(kDelimitedTag(1))) {
        // Context
        features = context_features;
        config = &context_feature_type_and_lengths;
      } else if (stream.ExpectTag(kDelimitedTag(2))) {
        // Sequence
        features = sequence_features;
        config = &sequence_feature_type_and_lengths;
      } else if (!SkipExtraneousTag(&stream)) {
        return errors::InvalidArgument(
            "Invalid protocol message input, example id: ", example_name);
      }
      if (features != nullptr) {
        uint32 length;
        if (!stream.ReadVarint32(&length)) {
          return errors::InvalidArgument(
              "Invalid protocol message input, example id: ", example_name);
        }
        auto limit = stream.PushLimit(length);
        while (!stream.ExpectAtEnd()) {
          StringPiece key, value;
          uint32 length;
          if (!stream.ExpectTag(kDelimitedTag(1)) ||
              !stream.ReadVarint32(&length)) {
            return errors::InvalidArgument(
                "Invalid protocol message input, example id: ", example_name);
          }
          auto limit = stream.PushLimit(length);
          if (!stream.ExpectTag(kDelimitedTag(1)) ||
              !ParseString(&stream, &key) ||
              !stream.ExpectTag(kDelimitedTag(2)) ||
              !ParseString(&stream, &value) || !stream.ExpectAtEnd()) {
            return errors::InvalidArgument(
                "Invalid protocol message input, example id: ", example_name);
          }
          stream.PopLimit(limit);
          // Only save if this feature was requested.
          if (config->count(key) > 0) {
            (*features)[key] = value;
          }
        }
        stream.PopLimit(limit);
      }
    }

    for (const auto& c : *context_features) {
      size_t num_elements = 0;
      if (!c.second.empty()) {
        protobuf::io::CodedInputStream stream(
            reinterpret_cast<const uint8*>(c.second.data()), c.second.size());
        EnableAliasing(&stream);
        DataType dtype = context_feature_type_and_lengths[c.first].first;
        int64 num;
        switch (dtype) {
          case DT_STRING:
            num = ParseBytesFeature(&stream, nullptr);
            break;
          case DT_FLOAT:
            num = ParseFloatFeature(&stream, nullptr);
            break;
          case DT_INT64:
            num = ParseInt64Feature(&stream, nullptr);
            break;
          default:
            num = -1;
            break;
        }
        if (num == -1) {
          return errors::InvalidArgument("Error in context feature ", c.first,
                                         " in example ", example_name);
        }
        num_elements += num;
      }
      if (context_is_sparse[c.first]) {
        context_feature_type_and_lengths[c.first].second += num_elements;
      } else {
        size_t current_max = context_feature_type_and_lengths[c.first].second;
        context_feature_type_and_lengths[c.first].second =
            std::max(current_max, num_elements);
      }
    }
    for (const auto& c : *sequence_features) {
      size_t num_elements = 0;
      if (!c.second.empty()) {
        protobuf::io::CodedInputStream stream(
            reinterpret_cast<const uint8*>(c.second.data()), c.second.size());
        EnableAliasing(&stream);
        DataType dtype = sequence_feature_type_and_lengths[c.first].first;
        while (!stream.ExpectAtEnd()) {
          uint32 feature_length;
          if (!stream.ExpectTag(kDelimitedTag(1)) ||
              !stream.ReadVarint32(&feature_length)) {
            return errors::InvalidArgument("Error in sequence feature ",
                                           c.first, " in example ",
                                           example_name);
          }
          if (feature_length > 2) {
            auto limit = stream.PushLimit(feature_length);
            int64 num;
            switch (dtype) {
              case DT_STRING:
                num = ParseBytesFeature(&stream, nullptr);
                break;
              case DT_FLOAT:
                num = ParseFloatFeature(&stream, nullptr);
                break;
              case DT_INT64:
                num = ParseInt64Feature(&stream, nullptr);
                break;
              default:
                num = -1;
                break;
            }
            if (num == -1) {
              return errors::InvalidArgument("Error in sequence feature ",
                                             c.first, " in example ",
                                             example_name);
            }
            num_elements += num;
            stream.PopLimit(limit);
          } else if (feature_length == 2) {
            if (!SkipEmptyFeature(&stream, dtype)) {
              return errors::InvalidArgument("Error in sequence feature ",
                                             c.first, " in example ",
                                             example_name);
            }
          } else if (feature_length != 0) {
            return errors::InvalidArgument("Error in sequence feature ",
                                           c.first, " in example ",
                                           example_name);
          }
        }
      }
      if (sequence_is_sparse[c.first]) {
        sequence_feature_type_and_lengths[c.first].second += num_elements;
      } else {
        size_t current_max = sequence_feature_type_and_lengths[c.first].second;
        sequence_feature_type_and_lengths[c.first].second =
            std::max(current_max, num_elements);
      }
    }
  }

  // Allocate memory.
  context_result->sparse_values.resize(context_config.sparse.size());
  context_result->sparse_indices.resize(context_config.sparse.size());
  context_result->sparse_shapes.resize(context_config.sparse.size());
  context_result->dense_values.resize(context_config.dense.size());
  feature_list_result->sparse_values.resize(feature_list_config.sparse.size());
  feature_list_result->sparse_indices.resize(feature_list_config.sparse.size());
  feature_list_result->sparse_shapes.resize(feature_list_config.sparse.size());
  feature_list_result->dense_values.resize(feature_list_config.dense.size());
  dense_feature_lengths->resize(feature_list_config.dense.size());

  int t = 0;
  for (const auto& c : context_config.dense) {
    TensorShape dense_shape, example_shape;
    DataType dtype = c.dtype;
    const size_t expected_max_elements =
        context_feature_type_and_lengths[c.feature_name].second;
    if (!c.shape.AsTensorShape(&example_shape) ||
        expected_max_elements != example_shape.num_elements()) {
      return errors::InvalidArgument(
          "Inconsistent number of elements for feature ", c.feature_name, ": ",
          expected_max_elements, " vs ", dense_shape.num_elements());
    }
    dense_shape.AddDim(num_examples);
    for (const int dim : c.shape.dim_sizes()) {
      dense_shape.AddDim(dim);
    }
    context_result->dense_values[t] = Tensor(dtype, dense_shape);

    // TODO(sundberg): Refactor to reduce code duplication, and add bounds
    // checking for the outputs.
    string* out_bytes = nullptr;
    float* out_float = nullptr;
    int64* out_int64 = nullptr;
    switch (dtype) {
      case DT_STRING:
        out_bytes = context_result->dense_values[t].flat<string>().data();
        break;
      case DT_FLOAT:
        out_float = context_result->dense_values[t].flat<float>().data();
        break;
      case DT_INT64:
        out_int64 = context_result->dense_values[t].flat<int64>().data();
        break;
      default:
        return errors::InvalidArgument("Unexpected dtype ", dtype,
                                       " in feature ", c.feature_name);
    }
    t++;

    // Fill in the values.
    for (int e = 0; e < num_examples; e++) {
      size_t num_elements = 0;
      const auto feature_iter = all_context_features[e].find(c.feature_name);
      const string& example_name =
          example_names.empty() ? kUnknown : example_names[e];
      if (feature_iter == all_context_features[e].end()) {
        // Copy the default value, if present. If not, return an error.
        if (c.default_value.NumElements() == 0) {
          return errors::InvalidArgument(
              "Feature: ", c.feature_name,
              " (data type: ", DataTypeString(c.dtype), ")",
              " is required but could not be found.");
        }
        const string* in_bytes = nullptr;
        const float* in_float = nullptr;
        const int64* in_int64 = nullptr;
        size_t num = 0;
        switch (dtype) {
          case DT_STRING:
            in_bytes = c.default_value.flat<string>().data();
            num = c.default_value.NumElements();
            for (int p = 0; p < num; p++) {
              *out_bytes++ = *in_bytes++;
            }
            break;
          case DT_FLOAT:
            in_float = c.default_value.flat<float>().data();
            num = c.default_value.NumElements();
            for (int p = 0; p < num; p++) {
              *out_float++ = *in_float++;
            }
            break;
          case DT_INT64:
            in_int64 = c.default_value.flat<int64>().data();
            num = c.default_value.NumElements();
            for (int p = 0; p < num; p++) {
              *out_int64++ = *in_int64++;
            }
            break;
          default:
            return errors::InvalidArgument("Unexpected dtype ", dtype,
                                           " in example ", example_name);
        }
        num_elements += num;
      } else if (!feature_iter->second.empty()) {
        const auto& feature = feature_iter->second;
        protobuf::io::CodedInputStream stream(
            reinterpret_cast<const uint8*>(feature.data()), feature.size());
        EnableAliasing(&stream);
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
            return errors::InvalidArgument("Unexpected dtype ", dtype,
                                           " in example ", example_name);
        }
        num_elements += num_added;
      }
      if (num_elements != expected_max_elements) {
        return errors::InvalidArgument(
            "Unexpected number of elements in example ", example_name);
      }
    }
  }
  t = 0;
  for (const auto& c : context_config.sparse) {
    TensorShape indices_shape, values_shape;
    DataType dtype = c.dtype;
    size_t expected_num_elements =
        context_feature_type_and_lengths[c.feature_name].second;
    indices_shape.AddDim(expected_num_elements);
    indices_shape.AddDim(2);
    values_shape.AddDim(expected_num_elements);
    context_result->sparse_indices[t] = Tensor(DT_INT64, indices_shape);
    context_result->sparse_values[t] = Tensor(dtype, values_shape);
    context_result->sparse_shapes[t] = Tensor(DT_INT64, TensorShape({2}));
    // TODO(sundberg): Refactor to reduce code duplication, and add bounds
    // checking for the outputs.
    string* out_bytes = nullptr;
    float* out_float = nullptr;
    int64* out_int64 = nullptr;
    switch (dtype) {
      case DT_STRING:
        out_bytes = context_result->sparse_values[t].flat<string>().data();
        break;
      case DT_FLOAT:
        out_float = context_result->sparse_values[t].flat<float>().data();
        break;
      case DT_INT64:
        out_int64 = context_result->sparse_values[t].flat<int64>().data();
        break;
      default:
        return errors::InvalidArgument("Unexpected dtype ", dtype,
                                       " in feature ", c.feature_name);
    }
    int64* out_indices = context_result->sparse_indices[t].flat<int64>().data();
    auto out_shape = context_result->sparse_shapes[t].vec<int64>();
    t++;

    // Fill in the values.
    size_t num_elements = 0;
    size_t max_num_cols = 0;
    for (int e = 0; e < num_examples; e++) {
      const auto& feature = all_context_features[e][c.feature_name];
      const string& example_name =
          example_names.empty() ? kUnknown : example_names[e];
      if (!feature.empty()) {
        protobuf::io::CodedInputStream stream(
            reinterpret_cast<const uint8*>(feature.data()), feature.size());
        EnableAliasing(&stream);
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
            return errors::InvalidArgument("Unexpected dtype ", dtype,
                                           " in example ", example_name);
        }
        num_elements += num_added;
        max_num_cols = std::max(max_num_cols, num_added);
        for (int i = 0; i < num_added; i++) {
          *out_indices++ = e;
          *out_indices++ = i;
        }
      }
    }
    if (num_elements != expected_num_elements) {
      return errors::InvalidArgument(
          "Unexpected total number of elements in feature ", c.feature_name);
    }
    out_shape(0) = num_examples;
    out_shape(1) = max_num_cols;
  }
  t = 0;
  TensorShape dense_length_shape({num_examples});
  for (const auto& c : feature_list_config.dense) {
    TensorShape dense_shape, row_shape;
    DataType dtype = c.dtype;
    const size_t expected_max_elements =
        sequence_feature_type_and_lengths[c.feature_name].second;
    if (!c.shape.AsTensorShape(&row_shape) ||
        expected_max_elements !=
            (expected_max_elements / row_shape.num_elements()) *
                row_shape.num_elements()) {
      return errors::InvalidArgument("Unexpected shape error in feature ",
                                     c.feature_name);
    }
    int64 expected_max_rows = expected_max_elements / row_shape.num_elements();
    dense_shape.AddDim(num_examples);
    dense_shape.AddDim(expected_max_rows);
    for (const int dim : feature_list_config.dense[t].shape.dim_sizes()) {
      dense_shape.AddDim(dim);
    }
    feature_list_result->dense_values[t] = Tensor(dtype, dense_shape);
    (*dense_feature_lengths)[t] = Tensor(DT_INT64, dense_length_shape);
    int64* out_lengths = (*dense_feature_lengths)[t].flat<int64>().data();

    string* out_bytes = nullptr;
    float* out_float = nullptr;
    int64* out_int64 = nullptr;
    switch (dtype) {
      case DT_STRING:
        out_bytes = feature_list_result->dense_values[t].flat<string>().data();
        break;
      case DT_FLOAT:
        out_float = feature_list_result->dense_values[t].flat<float>().data();
        break;
      case DT_INT64:
        out_int64 = feature_list_result->dense_values[t].flat<int64>().data();
        break;
      default:
        return errors::InvalidArgument("Unexpected dtype ", dtype,
                                       " in feature ", c.feature_name);
    }
    t++;

    // Fill in the values.
    for (int e = 0; e < num_examples; e++) {
      size_t num_elements = 0, num_rows = 0;
      const auto feature_iter = all_sequence_features[e].find(c.feature_name);
      const string& example_name =
          example_names.empty() ? kUnknown : example_names[e];
      if (feature_iter == all_sequence_features[e].end()) {
        // Return an error if this feature was not allowed to be missing.
        // Otherwise, we'll pad as needed below.
        if (!c.variable_length) {
          return errors::InvalidArgument("Missing feature ", c.feature_name,
                                         " in example ", example_name);
        }
      } else if (!feature_iter->second.empty()) {
        const auto& feature = feature_iter->second;
        protobuf::io::CodedInputStream stream(
            reinterpret_cast<const uint8*>(feature.data()), feature.size());
        EnableAliasing(&stream);
        while (!stream.ExpectAtEnd()) {
          uint32 feature_length;
          if (!stream.ExpectTag(kDelimitedTag(1)) ||
              !stream.ReadVarint32(&feature_length)) {
            return errors::InvalidArgument("Error in sequence feature ",
                                           c.feature_name, " in example ",
                                           example_name);
          }
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
              return errors::InvalidArgument("Unexpected dtype ", dtype,
                                             " in example ", example_name);
          }
          num_elements += num_added;
          num_rows++;
          if (num_added != row_shape.num_elements()) {
            return errors::InvalidArgument(
                "Unexpected number of elements in feature ", c.feature_name,
                ", example ", example_name);
          }
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
          return errors::InvalidArgument("Unexpected dtype ", dtype,
                                         " in example ", example_name);
      }
    }
  }
  t = 0;
  for (const auto& c : feature_list_config.sparse) {
    TensorShape indices_shape, values_shape;
    DataType dtype = c.dtype;
    size_t expected_num_elements =
        sequence_feature_type_and_lengths[c.feature_name].second;
    indices_shape.AddDim(expected_num_elements);
    indices_shape.AddDim(3);
    values_shape.AddDim(expected_num_elements);
    feature_list_result->sparse_indices[t] = Tensor(DT_INT64, indices_shape);
    feature_list_result->sparse_values[t] = Tensor(dtype, values_shape);
    feature_list_result->sparse_shapes[t] = Tensor(DT_INT64, TensorShape({3}));

    string* out_bytes = nullptr;
    float* out_float = nullptr;
    int64* out_int64 = nullptr;
    switch (dtype) {
      case DT_STRING:
        out_bytes = feature_list_result->sparse_values[t].flat<string>().data();
        break;
      case DT_FLOAT:
        out_float = feature_list_result->sparse_values[t].flat<float>().data();
        break;
      case DT_INT64:
        out_int64 = feature_list_result->sparse_values[t].flat<int64>().data();
        break;
      default:
        return errors::InvalidArgument("Unexpected dtype ", dtype,
                                       " in feature ", c.feature_name);
    }
    int64* out_indices =
        feature_list_result->sparse_indices[t].flat<int64>().data();
    auto out_shape = feature_list_result->sparse_shapes[t].vec<int64>();
    t++;

    // Fill in the values.
    size_t num_elements = 0;
    size_t max_num_rows = 0;
    size_t max_num_cols = 0;
    for (int e = 0; e < num_examples; e++) {
      const auto& feature = all_sequence_features[e][c.feature_name];
      const string& example_name =
          example_names.empty() ? kUnknown : example_names[e];
      if (!feature.empty()) {
        protobuf::io::CodedInputStream stream(
            reinterpret_cast<const uint8*>(feature.data()), feature.size());
        EnableAliasing(&stream);
        size_t num_rows = 0;
        while (!stream.ExpectAtEnd()) {
          uint32 feature_length;
          if (!stream.ExpectTag(kDelimitedTag(1)) ||
              !stream.ReadVarint32(&feature_length)) {
            return errors::InvalidArgument("Error in sequence feature ",
                                           c.feature_name, " in example ",
                                           example_name);
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
                return errors::InvalidArgument("Unexpected dtype ", dtype,
                                               " in example ", example_name);
            }
            num_elements += num_added;
            max_num_cols = std::max(max_num_cols, num_added);
            for (int i = 0; i < num_added; i++) {
              *out_indices++ = e;
              *out_indices++ = num_rows;
              *out_indices++ = i;
            }
            stream.PopLimit(limit);
          } else if (feature_length == 2) {
            if (!SkipEmptyFeature(&stream, dtype)) {
              return errors::InvalidArgument("Error in sequence feature ",
                                             c.feature_name, " in example ",
                                             example_name);
            }
          } else if (feature_length != 0) {
            return errors::InvalidArgument("Error in sequence feature ",
                                           c.feature_name, " in example ",
                                           example_name);
          }
          num_rows++;
        }
        max_num_rows = std::max(max_num_rows, num_rows);
      }
    }
    if (num_elements != expected_num_elements) {
      return errors::InvalidArgument(
          "Unexpected number of elements in feature ", c.feature_name);
    }
    out_shape(0) = num_examples;
    out_shape(1) = max_num_rows;
    out_shape(2) = max_num_cols;
  }

  return Status::OK();
}

}  // namespace example
}  // namespace tensorflow
