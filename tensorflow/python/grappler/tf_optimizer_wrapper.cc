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

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "pybind11/pybind11.h"  // from @pybind11
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/clusters/utils.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace py = pybind11;

void DetectDevices(
    std::unordered_map<std::string, tensorflow::DeviceProperties>* device_map) {
  tensorflow::SessionOptions options;
  std::vector<std::unique_ptr<tensorflow::Device>> devices;
  if (!tensorflow::DeviceFactory::AddDevices(options, "", &devices).ok()) {
    return;
  }

  for (const std::unique_ptr<tensorflow::Device>& device : devices) {
    tensorflow::DeviceProperties& prop = (*device_map)[device->name()];
    prop = tensorflow::grappler::GetDeviceInfo(device->parsed_name());

    // Overwrite the memory limit since users might have requested to use only a
    // fraction of the available device memory.
    const tensorflow::DeviceAttributes& attr = device->attributes();
    prop.set_memory_size(attr.memory_limit());
  }
}

PYBIND11_MODULE(_pywrap_tf_optimizer, m) {
  m.def("TF_OptimizeGraph",
        [](tensorflow::grappler::Cluster* cluster,
           const std::string& serialized_config_proto,
           const std::string& serialized_metagraph, bool verbose,
           const std::string& graph_id,
           bool strip_default_attributes) -> py::bytes {
          std::string out_graph_bytes;
          {
            py::gil_scoped_release gil_release;
            tensorflow::ConfigProto config_proto;
            if (!config_proto.ParseFromString(serialized_config_proto)) {
              throw std::invalid_argument(
                  "The ConfigProto could not be parsed as a valid protocol "
                  "buffer");
            }
            tensorflow::MetaGraphDef metagraph;
            if (!metagraph.ParseFromString(serialized_metagraph)) {
              throw std::invalid_argument(
                  "The MetaGraphDef could not be parsed as a valid protocol "
                  "buffer");
            }

            tensorflow::grappler::ItemConfig item_config;
            // This disables graph optimizations in the older graph optimizer,
            // which tend to overlap / be redundant with those in Grappler.
            item_config.apply_optimizations = false;
            item_config.ignore_user_placement = false;
            std::unique_ptr<tensorflow::grappler::GrapplerItem> grappler_item =
                tensorflow::grappler::GrapplerItemFromMetaGraphDef(
                    graph_id, metagraph, item_config);
            if (!grappler_item) {
              throw std::invalid_argument(
                  "Failed to import metagraph, check error log for more info.");
            }

            tensorflow::DeviceBase* cpu_device = nullptr;
            tensorflow::GraphDef out_graph;
            tensorflow::grappler::MetaOptimizer optimizer(cpu_device,
                                                          config_proto);

            MaybeRaiseRegisteredFromStatusWithGIL(
                optimizer.Optimize(cluster, *grappler_item, &out_graph));
            if (strip_default_attributes) {
              tensorflow::StripDefaultAttributes(
                  *tensorflow::OpRegistry::Global(), out_graph.mutable_node());
            }
            if (verbose) {
              optimizer.PrintResult();
            }
            out_graph_bytes = out_graph.SerializeAsString();
          }
          return py::bytes(std::move(out_graph_bytes));
        });
}
