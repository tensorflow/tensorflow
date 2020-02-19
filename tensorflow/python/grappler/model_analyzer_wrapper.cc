/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include <sstream>

#include "include/pybind11/pybind11.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/python/grappler/model_analyzer.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_model_analyzer, m) {
  m.def("GenerateModelReport",
        [](const py::bytes& serialized_metagraph, bool assume_valid_feeds,
           bool debug) -> py::bytes {
          tensorflow::MetaGraphDef metagraph;
          if (!metagraph.ParseFromString(serialized_metagraph)) {
            return "The MetaGraphDef could not be parsed as a valid protocol "
                   "buffer";
          }

          tensorflow::grappler::ItemConfig cfg;
          cfg.apply_optimizations = false;
          std::unique_ptr<tensorflow::grappler::GrapplerItem> item =
              tensorflow::grappler::GrapplerItemFromMetaGraphDef(
                  "metagraph", metagraph, cfg);
          if (item == nullptr) {
            return "Error: failed to preprocess metagraph: check your log file "
                   "for errors";
          }

          tensorflow::grappler::ModelAnalyzer analyzer(*item);

          std::ostringstream os;
          tensorflow::MaybeRaiseFromStatus(
              analyzer.GenerateReport(debug, assume_valid_feeds, os));
          return py::bytes(os.str());
        });
}
