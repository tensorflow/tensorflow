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

#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb_text.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/casts.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
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
  Feature(StringPiece serialized) : serialized_(serialized) {}

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
        return errors::InvalidArgument("Unsuported datatype.");
    }
    return Status::OK();
  }

  bool ParseBytesList(SmallVector<string>* bytes_list) {
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

  bool ParseFloatList(SmallVector<float>* float_list) {
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

        while (!stream.ExpectAtEnd()) {
          uint32 buffer32;
          if (!stream.ReadLittleEndian32(&buffer32)) return false;
          float_list->push_back(bit_cast<float>(buffer32));
        }

        stream.PopLimit(packed_limit);
      } else {  // non-packed
        while (!stream.ExpectAtEnd()) {
          if (!stream.ExpectTag(kFixed32Tag(1))) return false;
          uint32 buffer32;
          if (!stream.ReadLittleEndian32(&buffer32)) return false;
          float_list->push_back(bit_cast<float>(buffer32));
        }
      }
    }

    stream.PopLimit(limit);
    return true;
  }

  bool ParseInt64List(SmallVector<int64>* int64_list) {
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
          int64_list->push_back(n);
        }

        stream.PopLimit(packed_limit);
      } else {  // non-packed
        while (!stream.ExpectAtEnd()) {
          if (!stream.ExpectTag(kVarintTag(1))) return false;
          protobuf_uint64 n;  // There is no API for int64
          if (!stream.ReadVarint64(&n)) return false;
          int64_list->push_back(n);
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
    if (stream->ExpectTag(kDelimitedTag(1))) {
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
  for (parsed::FeatureMapEntry& entry : parsed_example) {
    auto& value = (*features.mutable_feature())[entry.first.ToString()];
    DataType dtype;
    if (!entry.second.ParseDataType(&dtype).ok()) return false;
    switch (dtype) {
      case DT_INVALID:
        break;
      case DT_STRING: {
        SmallVector<string> list;
        if (!entry.second.ParseBytesList(&list)) return false;
        auto* result_list = value.mutable_bytes_list();
        for (auto& bytes : list) {
          result_list->add_value(std::move(bytes));
        }
        break;
      }
      case DT_FLOAT: {
        SmallVector<float> list;
        if (!entry.second.ParseFloatList(&list)) return false;
        auto* result_list = value.mutable_float_list();
        for (float f : list) {
          result_list->add_value(f);
        }
        break;
      }
      case DT_INT64: {
        SmallVector<int64> list;
        if (!entry.second.ParseInt64List(&list)) return false;
        auto* result_list = value.mutable_int64_list();
        for (int64 i : list) {
          result_list->add_value(i);
        }
        break;
      }
      default:
        CHECK(false) << "Should not happen.";
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

Status FastParseSerializedExample(
    const string& serialized_example, const string& example_name,
    const size_t example_index, const Config& config,
    const PresizedCuckooMap<std::pair<size_t, Type>>& config_index,
    SeededHasher hasher, std::vector<Tensor>* output_dense,
    std::vector<SparseBuffer>* output_sparse) {
  DCHECK(output_dense != nullptr);
  DCHECK(output_sparse != nullptr);
  parsed::Example parsed_example;
  if (!ParseExample(serialized_example, &parsed_example)) {
    return errors::InvalidArgument("Could not parse example input, value: '",
                                   serialized_example, "'");
  }
  constexpr size_t kMax = std::numeric_limits<size_t>::max();
  std::vector<size_t> sparse_features_found(config.sparse.size(), kMax);
  std::vector<size_t> dense_features_found(config.dense.size(), kMax);

  // Handle features present in the example.
  for (parsed::FeatureMapEntry& name_and_feature : parsed_example) {
    parsed::Feature& feature = name_and_feature.second;
    std::pair<size_t, Type> d_and_type;
    uint64 h = hasher(name_and_feature.first);
    if (!config_index.Find(h, &d_and_type)) continue;
    size_t d = d_and_type.first;

    auto parse_error = [&](StringPiece feature_name) {
      return errors::InvalidArgument("Name: ", example_name, ", Key: ",
                                     feature_name, ", Index: ", example_index,
                                     ". Can't parse serialized Example.");
    };

    if (d_and_type.second == Type::Dense) {
      DataType example_dtype;
      TF_RETURN_IF_ERROR(feature.ParseDataType(&example_dtype));
      if (example_dtype == DT_INVALID) continue;

      dense_features_found[d] = example_index;
      if (example_dtype != config.dense[d].dtype) {
        return errors::InvalidArgument(
            "Name: ", example_name, ", Feature: ", config.dense[d].feature_name,
            ".  Data types don't match. ", "Data type: ",
            DataTypeString(example_dtype), "Expected type: ",
            DataTypeString(config.dense[d].dtype));
      }
      const string& feature_name = config.dense[d].feature_name;
      const TensorShape& shape = config.dense[d].shape;
      Tensor& out = (*output_dense)[d];

      const std::size_t num_elements = shape.num_elements();
      const std::size_t offset = example_index * num_elements;

      auto shape_error = [&](size_t size, StringPiece type_str) {
        return errors::InvalidArgument(
            "Name: ", example_name, ", Key: ", feature_name, ", Index: ",
            example_index, ".  Number of ", type_str,
            " values != expected.  "
            "Values size: ",
            size, " but output shape: ", shape.DebugString());
      };

      // TODO(b/31499934): Make sure concatented serialized tf.Example protos
      // get parsed correctly when they contain dense features and add tests.
      switch (config.dense[d].dtype) {
        case DT_INT64: {
          SmallVector<int64> list;
          if (!feature.ParseInt64List(&list)) return parse_error(feature_name);
          if (list.size() != num_elements) {
            return shape_error(list.size(), "int64");
          }
          auto out_p = out.flat<int64>().data() + offset;
          std::copy_n(list.begin(), list.size(), out_p);
          break;
        }
        case DT_FLOAT: {
          SmallVector<float> list;
          if (!feature.ParseFloatList(&list)) return parse_error(feature_name);
          if (list.size() != num_elements) {
            return shape_error(list.size(), "float");
          }
          auto out_p = out.flat<float>().data() + offset;
          std::copy_n(list.begin(), list.size(), out_p);
          break;
        }
        case DT_STRING: {
          SmallVector<string> list;
          if (!feature.ParseBytesList(&list)) return parse_error(feature_name);
          if (list.size() != num_elements) {
            return shape_error(list.size(), "bytes");
          }
          auto out_p = out.flat<string>().data() + offset;
          for (size_t i = 0; i < list.size(); ++i) {
            out_p[i] = std::move(list[i]);
          }
          break;
        }
        default:
          CHECK(false) << "Should not happen.";
      }
    } else {
      // Handle sparse features.
      sparse_features_found[d] = example_index;
      const string& feature_name = config.sparse[d].feature_name;
      SparseBuffer& out = (*output_sparse)[d];
      DataType example_dtype;
      TF_RETURN_IF_ERROR(feature.ParseDataType(&example_dtype));
      if (example_dtype != DT_INVALID &&
          example_dtype != config.sparse[d].dtype) {
        return errors::InvalidArgument(
            "Name: ", example_name, ", Feature: ",
            config.sparse[d].feature_name, ".  Data types don't match. ",
            "Expected type: ", DataTypeString(config.sparse[d].dtype));
      }

      switch (config.sparse[d].dtype) {
        case DT_INT64: {
          if (example_dtype != DT_INVALID) {
            if (!feature.ParseInt64List(&out.int64_list)) {
              return parse_error(feature_name);
            }
          }
          out.example_end_indices.push_back(out.int64_list.size());
          break;
        }
        case DT_FLOAT: {
          if (example_dtype != DT_INVALID) {
            if (!feature.ParseFloatList(&out.float_list)) {
              return parse_error(feature_name);
            }
          }
          out.example_end_indices.push_back(out.float_list.size());
          break;
        }
        case DT_STRING: {
          if (example_dtype != DT_INVALID) {
            if (!feature.ParseBytesList(&out.bytes_list)) {
              return parse_error(feature_name);
            }
          }
          out.example_end_indices.push_back(out.bytes_list.size());
          break;
        }
        default:
          CHECK(false) << "Should not happen.";
      }
    }
  }

  // Handle missing dense features.
  for (size_t d = 0; d < config.dense.size(); ++d) {
    if (dense_features_found[d] == example_index) continue;
    if (config.dense[d].default_value.NumElements() == 0) {
      return errors::InvalidArgument("Name: ", example_name, ", Feature: ",
                                     config.dense[d].feature_name,
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
        CHECK(false) << "Should not happen.";
    }
  }

  // Handle missing sparse features.
  for (size_t d = 0; d < config.sparse.size(); ++d) {
    if (sparse_features_found[d] == example_index) continue;
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

  // Allocate dense output (sparse have to be buffered).
  for (size_t d = 0; d < config.dense.size(); ++d) {
    TensorShape out_shape;
    out_shape.AddDim(serialized.size());
    for (const int64 dim : config.dense[d].shape.dim_sizes()) {
      out_shape.AddDim(dim);
    }
    result->dense_values.emplace_back(config.dense[d].dtype, out_shape);
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
  std::vector<Status> status_of_minibatch(num_minibatches);
  auto ProcessMiniBatch = [&](size_t minibatch) {
    sparse_buffers[minibatch].resize(config.sparse.size());
    size_t start = first_example_of_minibatch(minibatch);
    size_t end = first_example_of_minibatch(minibatch + 1);
    for (size_t e = start; e < end; ++e) {
      status_of_minibatch[minibatch] = FastParseSerializedExample(
          serialized[e],
          (example_names.size() > 0 ? example_names[e] : "<unknown>"), e,
          config, config_index, hasher, &result->dense_values,
          &sparse_buffers[minibatch]);
      if (!status_of_minibatch[minibatch].ok()) break;
    }
  };

  ParallelFor(ProcessMiniBatch, num_minibatches, thread_pool);

  for (Status& status : status_of_minibatch) {
    TF_RETURN_IF_ERROR(status);
  }

  // Merge SparseBuffers from all minibatches for every config.sparse.
  auto MergeMinibatches = [&](size_t d) {
    // Loop over minibatches
    size_t total_num_features = 0;
    size_t max_num_features = 0;
    for (auto& sparse_values_tmp : sparse_buffers) {
      std::vector<size_t>& end_indices =
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
          CHECK(false) << "Should not happen.";
      }

      offset += delta;
    }
  };

  for (size_t d = 0; d < config.sparse.size(); ++d) {
    MergeMinibatches(d);
  }

  return Status::OK();
}

}  // namespace example
}  // namespace tensorflow
