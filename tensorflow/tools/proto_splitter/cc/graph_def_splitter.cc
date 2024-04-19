/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/tools/proto_splitter/cc/graph_def_splitter.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/tools/proto_splitter/cc/large_node_splitter.h"
#include "tensorflow/tools/proto_splitter/cc/max_size.h"
#include "tensorflow/tools/proto_splitter/cc/repeated_field_splitter.h"
#include "tensorflow/tools/proto_splitter/cc/size_splitter.h"
#include "tensorflow/tools/proto_splitter/cc/split.h"
#include "tensorflow/tools/proto_splitter/cc/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace tools::proto_splitter {

namespace {

// Required in OSS to prevent string to bool conversion in FieldType variant.
using namespace std::string_literals;  // NOLINT

// Extracts the value of large constants into a new chunk.
// Only works on NodeDef messages.
class ConstantSplitter : public SizeSplitter {
 public:
  using SizeSplitter::SizeSplitter;
  absl::StatusOr<int> BuildChunksReturnSize() override {
    NodeDef* node = tsl::protobuf::DynamicCastToGenerated<NodeDef>(message());

    std::vector<FieldType> tensor_field = {"attr"s, "value"s, "tensor"s};
    std::vector<FieldType> content_field = {"attr"s, "value"s, "tensor"s,
                                            "tensor_content"s};

    TF_ASSIGN_OR_RETURN(auto ret, GetMutableField(node, tensor_field));
    auto tensor_msg =
        ret.parent->GetReflection()->MutableMessage(ret.parent, ret.field);
    TensorProto* tensor_proto =
        tsl::protobuf::DynamicCastToGenerated<TensorProto>(tensor_msg);

    int size_diff;

    if (tensor_proto->tensor_content().empty()) {
      Tensor t;
      if (!t.FromProto(*tensor_proto)) {
        return absl::InvalidArgumentError(
            "Invalid Const NodeDef.attr[\"value\"].tensor value.");
      }

      TensorProto container;
      t.AsProtoTensorContent(&container);
      size_diff = container.tensor_content().size();
      auto x = std::make_unique<std::string>(
          std::move(*container.mutable_tensor_content()));
      auto y = std::make_unique<MessageBytes>(std::move(*x));
      TF_RETURN_IF_ERROR(AddChunk(std::move(y), &content_field));
    } else {
      size_diff = tensor_proto->tensor_content().size();
      auto x = std::make_unique<std::string>(
          std::move(*tensor_proto->mutable_tensor_content()));
      auto y = std::make_unique<MessageBytes>(std::move(*x));
      TF_RETURN_IF_ERROR(AddChunk(std::move(y), &content_field));
    }

    // Keep the TensorProto's dtype, tensor_shape, and version_number fields,
    // but clear the raw tensor content / "xxx_val" attributes.
    auto dtype = tensor_proto->dtype();
    auto tensor_shape = tensor_proto->tensor_shape();
    auto version_number = tensor_proto->version_number();
    tensor_proto->Clear();
    tensor_proto->set_dtype(dtype);
    *tensor_proto->mutable_tensor_shape() = tensor_shape;
    tensor_proto->set_version_number(version_number);

    return size_diff;
  }
};

class ConstantSplitterFactory : public SizeSplitterFactory {
 public:
  using SizeSplitterFactory::SizeSplitterFactory;

  absl::StatusOr<std::unique_ptr<SizeSplitter>> CreateSplitter(
      tsl::protobuf::Message* message, ComposableSplitterBase* parent_splitter,
      std::vector<FieldType>* fields_in_parent, int size) override {
    if (size < GetMaxSize()) return nullptr;
    NodeDef* node = tsl::protobuf::DynamicCastToGenerated<NodeDef>(message);
    if (node->op() != "Const")
      return absl::UnimplementedError(absl::StrCat(
          "Currently only able to split 'Const' nodes that are larger than the "
          "2GB maximum proto size. Got node of type '",
          node->op(), "' with size: ", size, "."));
    ConstantSplitter* splitter =
        new ConstantSplitter(message, parent_splitter, fields_in_parent);
    return absl::WrapUnique(splitter);
  }
};

class FunctionDefSplitter : public SizeSplitter {
 public:
  using SizeSplitter::SizeSplitter;
  absl::StatusOr<int> BuildChunksReturnSize() override {
    size_t current_size = GetInitialSize();
    uint64_t max_size = GetMaxSize();
    std::vector<FieldType> fields = {};
    // First check if the entire FunctionDef can be split into a separate chunk.
    // We do this before the `RepeatedMessageSplitter`, which is costly because
    // it iterates through every `node_def`.
    if (LARGE_SIZE_CHECK(current_size, max_size) && current_size < max_size) {
      auto splitter = LargeNodeSplitter<FunctionDef>(message(), this, &fields);
      splitter.SetInitialSize(current_size);
      return splitter.BuildChunksReturnSize();
    } else if (current_size > max_size) {
      ConstantSplitterFactory constant_splitter_factory;
      LargeNodeSplitterFactory<NodeDef> large_node_splitter_factory;
      std::vector<SizeSplitterFactory*> factories = {
          &constant_splitter_factory, &large_node_splitter_factory};
      auto ret = RepeatedFieldSplitters<FunctionDef, NodeDef>::Create(
          message(), this, &fields, "node_def"s, &factories);
      if (!ret.ok()) return ret.status();
      auto splitter = ret.value();
      return splitter.BuildChunksReturnSize();
    }

    return 0;
  }
};

class FunctionDefSplitterFactory : public SizeSplitterFactory {
 public:
  using SizeSplitterFactory::SizeSplitterFactory;

  absl::StatusOr<std::unique_ptr<SizeSplitter>> CreateSplitter(
      tsl::protobuf::Message* message, ComposableSplitterBase* parent_splitter,
      std::vector<FieldType>* fields_in_parent, int size) override {
    FunctionDefSplitter* splitter =
        new FunctionDefSplitter(message, parent_splitter, fields_in_parent);
    return absl::WrapUnique(splitter);
  }
};

}  // namespace

absl::Status GraphDefSplitter::BuildChunks() {
  TF_RETURN_IF_ERROR(SetMessageAsBaseChunk());
  GraphDef* g = tsl::protobuf::DynamicCastToGenerated<GraphDef>(message());
  uint64_t max_size = GetMaxSize();
  size_t graph_size = GetInitialSize();

  if (graph_size < max_size) return absl::OkStatus();

  // Set up GraphDef.node and GraphDef.library.function splitters.
  std::vector<FieldType> field_in_parent = {};
  ConstantSplitterFactory constant_splitter_factory;
  LargeNodeSplitterFactory<NodeDef> large_node_splitter_factory;
  std::vector<SizeSplitterFactory*> factories = {&constant_splitter_factory,
                                                 &large_node_splitter_factory};
  auto node_splitter_ret = RepeatedFieldSplitters<GraphDef, NodeDef>::Create(
      g, this, &field_in_parent, "node"s, &factories);
  if (!node_splitter_ret.ok()) return node_splitter_ret.status();
  auto node_splitter = node_splitter_ret.value();

  FunctionDefSplitterFactory function_splitter_factory;
  std::vector<FieldType> library_field = {"library"s};
  std::vector<SizeSplitterFactory*> fn_factories = {&function_splitter_factory};
  auto library_splitter_ret =
      RepeatedFieldSplitters<FunctionDefLibrary, FunctionDef>::Create(
          g->mutable_library(), this, &library_field, "function"s,
          &fn_factories);
  if (!library_splitter_ret.ok()) return library_splitter_ret.status();
  auto library_splitter = library_splitter_ret.value();
  size_t library_size = g->library().ByteSizeLong();
  library_splitter.SetInitialSize(library_size);

  size_t approx_node_size = graph_size - library_size;
  node_splitter.SetInitialSize(approx_node_size);

  // Call node and library splitters.
  if (library_size > approx_node_size) {
    TF_ASSIGN_OR_RETURN(int size_diff,
                        library_splitter.BuildChunksReturnSize());
    library_size -= size_diff;
    if (approx_node_size + library_size > max_size) {
      TF_ASSIGN_OR_RETURN(int size_diff, node_splitter.BuildChunksReturnSize());
      approx_node_size -= size_diff;
    }
  } else {
    TF_ASSIGN_OR_RETURN(int size_diff, node_splitter.BuildChunksReturnSize());
    approx_node_size -= size_diff;
    if (approx_node_size + library_size > max_size) {
      TF_ASSIGN_OR_RETURN(int size_diff,
                          library_splitter.BuildChunksReturnSize());
      library_size -= size_diff;
    }
  }

  // Recompute graph size. If the graph size is still greater than the max size,
  // separate the entire library into a separate chunk.
  if (g->ByteSizeLong() > max_size) {
    LargeNodeSplitter<FunctionDefLibrary> entire_library_splitter(
        g->mutable_library(), this, &library_field);
    // The library chunk must be inserted before the function chunks generated
    // by `library_splitter` which may have the same field tags.
    int index = 1;
    entire_library_splitter.SetChunkIndex(&index);
    TF_RETURN_IF_ERROR(entire_library_splitter.BuildChunks());
  }

  return absl::OkStatus();
}

}  // namespace tools::proto_splitter
}  // namespace tensorflow
