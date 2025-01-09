/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <utility>

#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/service_config.pb.h"
#include "tensorflow/dtensor/proto/layout.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace python {
void protobuf_inline_symbols_enforcer() {
  tensorflow::NamedDevice named_device;
  named_device.mutable_properties();
  named_device.properties();

  tensorflow::NamedDevice named_device_move(std::move(named_device));
  named_device_move.mutable_properties();

  tensorflow::quantization::ExportedModel exported_model;
  exported_model.function_aliases();

  tensorflow::profiler::XSpace x_space;
  x_space.mutable_hostnames();
  x_space.mutable_hostnames(0);

  tensorflow::dtensor::LayoutProto layout_proto;
  layout_proto.GetDescriptor();
  layout_proto.GetReflection();
  layout_proto.default_instance();

  tensorflow::dtensor::MeshProto mesh_proto;
  mesh_proto.GetDescriptor();
  mesh_proto.GetReflection();
  mesh_proto.default_instance();

  tensorflow::FunctionDef function_def;
  function_def.descriptor();
  function_def.GetDescriptor();
  function_def.GetReflection();
  function_def.default_instance();

  tensorflow::FunctionDefLibrary function_def_library;
  function_def_library.descriptor();

  tensorflow::GraphDef graph_def;
  graph_def.descriptor();
  graph_def.GetDescriptor();
  graph_def.GetReflection();
  graph_def.default_instance();

  tensorflow::MetaGraphDef meta_graph_def;
  meta_graph_def.GetDescriptor();
  meta_graph_def.GetReflection();
  meta_graph_def.default_instance();

  tensorflow::AttrValue attr_value;
  attr_value.default_instance();

  tensorflow::ConfigProto config_proto;
  config_proto.default_instance();

  tensorflow::data::experimental::DispatcherConfig dispatcher_config;
  dispatcher_config.default_instance();

  tensorflow::data::experimental::WorkerConfig worker_config;
  worker_config.default_instance();

  tensorflow::data::DataServiceMetadata data_service_metadata;
}
}  // namespace python
}  // namespace tensorflow
