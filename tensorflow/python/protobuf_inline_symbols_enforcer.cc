#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/service_config.pb.h"
#include "tensorflow/dtensor/proto/layout.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include <utility>

namespace tensorflow {
namespace python {
void protobuf_inline_symbols_enforcer() {
  tensorflow::NamedDevice named_device;
  tensorflow::NamedDevice named_device_move(std::move(named_device));
  named_device.mutable_properties();
  named_device.properties();

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
}
}


