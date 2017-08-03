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
#include "tensorflow/contrib/util/convert_graphdef_memmapped_format_lib.h"

#include <unordered_set>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/immutable_constant_op.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/memmapped_file_system_writer.h"

namespace tensorflow {
namespace {
class NodeConverter {
 public:
  // Converts one node. In-place updates node_def, writes the tensor in
  // memmapped
  // format, using writer. If the conversion has been done, convert_counter is
  // increased.
  Status ConvertConstantsToImmutable(NodeDef* node_def,
                                     MemmappedFileSystemWriter* writer,
                                     int* convert_counter,
                                     int min_conversion_size_bytes) {
    // Check the size.
    const AttrValue& value = node_def->attr().at("value");
    const TensorProto& tensor_proto = value.tensor();

    // Create copies of tensor datatype and shape, to put into the operator
    // after
    // the tensor is destroyed.
    const DataType tensor_data_type = tensor_proto.dtype();
    const TensorShapeProto tensor_shape = tensor_proto.tensor_shape();

    // Check that the tensor type is POD, only these types are supported for
    // memmapping.
    // DataType enum is explicitly converted to int to avoid errors with passing
    // enum type are a parameter type to std::unordered_set.
    static std::unordered_set<int> supported_types{
#define TYPE_FOR_SET(type) static_cast<int>(DataTypeToEnum<type>::value),
        TF_CALL_POD_TYPES(TYPE_FOR_SET)
#undef ADD_TYPE
    };

    if (supported_types.count(static_cast<int>(tensor_data_type)) == 0) {
      return Status::OK();
    }

    // Create Tensor from value and write it in memmapped format.
    Tensor parsed(tensor_proto.dtype());
    if (!parsed.FromProto(cpu_allocator(), tensor_proto)) {
      return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                     tensor_proto.DebugString());
    }
    if (parsed.TotalBytes() < static_cast<size_t>(min_conversion_size_bytes)) {
      return Status::OK();
    }

    const string memmapped_region_name =
        MemmappedFileSystem::kMemmappedPackagePrefix +
        ConvertVariableNameToUniqueRegionName(node_def->name());

    TF_RETURN_IF_ERROR(writer->SaveTensor(parsed, memmapped_region_name));

    node_def->set_op("ImmutableConst");

    // Erase all attributes and leave only attributes that can be understood by
    // ImmutableConst.
    auto* mutable_attr = node_def->mutable_attr();
    mutable_attr->clear();

    {
      AttrValue attr_value;
      attr_value.set_type(tensor_data_type);
      mutable_attr->insert({ImmutableConstantOp::kDTypeAttr, attr_value});
    }
    {
      AttrValue attr_value;
      *(attr_value.mutable_shape()) = tensor_shape;
      mutable_attr->insert({ImmutableConstantOp::kShapeAttr, attr_value});
    }
    {
      AttrValue attr_value;
      attr_value.set_s(memmapped_region_name);
      mutable_attr->insert(
          {ImmutableConstantOp::kMemoryRegionNameAttr, attr_value});
    }
    ++*convert_counter;
    return Status::OK();
  }

 private:
  string ConvertVariableNameToUniqueRegionName(const string& variable_name) {
    string region_name = SanitizeVariableName(variable_name);
    while (!used_names_.insert(region_name).second) {
      region_name += '_';
    }
    return region_name;
  }

  static string SanitizeVariableName(const string& variable_name) {
    string result;
    for (char c : variable_name) {
      if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
          (c >= '0' && c <= '9') || c == '_' || c == '.') {
        result += c;
      } else {
        result += '_';
      }
    }
    return result;
  }
  std::unordered_set<string> used_names_;
};

}  // namespace

// Loads the graph, replaces operators, and writes it out.
Status ConvertConstantsToImmutable(const string& in_graph_filename,
                                   const string& out_graph_filename,
                                   int min_conversion_size_bytes) {
  Env* default_env = Env::Default();
  GraphDef graph_def;
  const auto load_graph_status =
      ReadBinaryProto(default_env, in_graph_filename, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load graph at '",
                                        in_graph_filename, "' : ",
                                        load_graph_status.error_message());
  }

  NodeConverter node_converter;

  // Create output writer.
  MemmappedFileSystemWriter writer;
  TF_RETURN_IF_ERROR(writer.InitializeToFile(default_env, out_graph_filename));

  // Iterate over graph nodes, looking for Const and replacing it with
  // ImmutableConst.
  int convert_counter = 0;
  for (int i = 0; i < graph_def.node_size(); ++i) {
    const NodeDef& node = graph_def.node(i);
    if (node.op() == "Const") {
      // Try to convert to ImmutableConst
      TF_RETURN_IF_ERROR(node_converter.ConvertConstantsToImmutable(
          graph_def.mutable_node(i), &writer, &convert_counter,
          min_conversion_size_bytes));
    }
  }
  TF_RETURN_IF_ERROR(writer.SaveProtobuf(
      graph_def, MemmappedFileSystem::kMemmappedPackageDefaultGraphDef));
  TF_RETURN_IF_ERROR(writer.FlushAndClose());
  LOG(INFO) << "Converted " << convert_counter << " nodes";
  return Status::OK();
}

}  // namespace tensorflow
