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
#include "tensorflow/lite/kernels/parse_example/example_proto_fast_parsing.h"

#include <algorithm>
#include <utility>

namespace tensorflow {
namespace example {

string ExampleName(const absl::Span<const tstring> example_names, int n) {
  return example_names.empty() ? "<unknown>" : example_names[n];
}

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

uint8 PeekTag(protobuf::io::CodedInputStream* stream) {
  DCHECK(stream != nullptr);
  const void* ptr;
  int size;
  if (!stream->GetDirectBufferPointer(&ptr, &size)) return 0;
  return *static_cast<const uint8*>(ptr);
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

template <>
void CopyOrMoveBlock(const tstring* b, const tstring* e, tstring* t) {
  std::move(b, e, t);
}

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

}  // namespace example
}  // namespace tensorflow
