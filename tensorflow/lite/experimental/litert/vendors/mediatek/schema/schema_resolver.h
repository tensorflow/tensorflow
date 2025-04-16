// Copyright (c) 2025 MediaTek Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_SCHEMA_SCHEMA_RESOLVER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_SCHEMA_SCHEMA_RESOLVER_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/schema/neuron_schema_generated.h"

namespace neuron {

inline bool IsNeuronSchema(const uint8_t* buffer, size_t size) {
  if (buffer == nullptr) {
    return false;
  }
  flatbuffers::Verifier verifier(buffer, size);
  return NeuronSchema::VerifyGraphsBuffer(verifier);
}

class CompiledGraph {
 public:
  CompiledGraph(const NeuronSchema::Graphs& g, const NeuronSchema::Subgraph& s)
      : graph_(g), subgraph_(s) {};

  litert::Expected<std::pair<const void*, size_t>> GetCompiledNetwork() {
    // Neuron Adapter doesn't support DLB for now.
    assert(GetCompiledType() != NeuronSchema::CompiledType_DLB);
    // TODO: Support the external buffer.
    assert(subgraph_.compiled_index_type() ==
           NeuronSchema::BufferIndicate_Index);
    auto index = subgraph_.compiled_index_as_Index();
    return GetBuffer(index->value());
  }

  NeuronSchema::CompiledType GetCompiledType() { return subgraph_.type(); }

  litert::Expected<std::pair<const void*, size_t>> GetBuffer(int32_t i) {
    auto array_size = graph_.data()->size();
    if (i >= array_size) {
      return litert::Error(
          kLiteRtStatusErrorIndexOOB,
          absl::StrFormat("Buffer array index %d is OOB, the array size : %d",
                          i, array_size));
    }
    auto buffer = graph_.data()->Get(i);
    return std::pair<const void*, size_t>(buffer->data()->data(),
                                          buffer->data()->size());
  }

 private:
  const NeuronSchema::Graphs& graph_;
  const NeuronSchema::Subgraph& subgraph_;
};

class SchemaResolver {
 public:
  SchemaResolver() = default;

  litert::Expected<bool> Initialize(const uint8_t* buffer, size_t size) {
    if (!IsNeuronSchema(buffer, size)) {
      return litert::Error(kLiteRtStatusErrorInvalidFlatbuffer,
                           "buffer is not a valid NeuronSchema");
    }
    graph_ = NeuronSchema::GetGraphs(buffer);

    auto subgraphs = graph_->subgraphs();
    for (const auto& subgraph : *subgraphs) {
      auto graph_name = subgraph->entry_point()->str();
      if (entry_points_.count(graph_name)) {
        // shouldn't have the same name between graphs.
        return false;
      } else {
        LITERT_LOG(LITERT_INFO, "Found graph: %s", graph_name.c_str());
        entry_points_[graph_name] = subgraph;
      }
    }
    LITERT_LOG(LITERT_INFO, "There are %u subgraphs in the bytecode",
               entry_points_.size());
    return true;
  }

  std::optional<CompiledGraph> GetCompiledGraph(std::string& name) {
    if (entry_points_.count(name) == 0) {
      return std::nullopt;
    }
    return CompiledGraph(*graph_, *entry_points_[name]);
  };

 private:
  const NeuronSchema::Graphs* graph_ = nullptr;

  std::unordered_map<std::string, NeuronSchema::Subgraph const*> entry_points_;
};

class BytecodeBuilder {
 public:
  BytecodeBuilder() = default;

  int32_t AddCompiledNetwork(std::string& entry_point,
                             NeuronSchema::CompiledType type,
                             int32_t buffer_index) {
    auto index = NeuronSchema::CreateIndex(fb_, buffer_index);
    auto subgraph = NeuronSchema::CreateSubgraph(
        fb_, fb_.CreateString(entry_point), type,
        NeuronSchema::BufferIndicate_Index, index.Union());

    subgraphs_.push_back(subgraph);
    return subgraphs_count_++;
  };

  int32_t AddBuffer(std::string& identifier, const std::vector<int8_t>& data) {
    auto buffer =
        NeuronSchema::CreateBufferDirect(fb_, identifier.c_str(), &data);
    graph_data_.push_back(buffer);
    return buffer_count_++;
  }

  int32_t AddBuffer(std::string& identifier, const int8_t* data,
                    size_t length) {
    auto data_offset = fb_.CreateVector(data, length);
    auto identifier_offset = fb_.CreateString(identifier);
    auto buffer =
        NeuronSchema::CreateBuffer(fb_, identifier_offset, data_offset);
    graph_data_.push_back(buffer);
    return buffer_count_++;
  }

  bool Finish() {
    auto graphs =
        NeuronSchema::CreateGraphsDirect(fb_, 1, &subgraphs_, &graph_data_);
    fb_.Finish(graphs);
    raw_buffer_ = {fb_.GetBufferPointer(), fb_.GetSize()};
    return true;
  }

  std::pair<uint8_t*, size_t> GetBytecode() {
    if (!raw_buffer_.has_value()) {
      return {nullptr, 0};
    }
    return raw_buffer_.value();
  }

 private:
  ::flatbuffers::FlatBufferBuilder fb_;

  std::optional<std::pair<uint8_t*, size_t>> raw_buffer_;

  std::vector<::flatbuffers::Offset<NeuronSchema::Subgraph>> subgraphs_;

  std::vector<::flatbuffers::Offset<NeuronSchema::Buffer>> graph_data_;

  int32_t subgraphs_count_ = 0;
  int32_t buffer_count_ = 0;
};

};  // namespace neuron

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_SCHEMA_SCHEMA_RESOLVER_H_
