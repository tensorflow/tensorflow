/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <string>

#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "xla/tsl/util/stat_summarizer_options.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/util/stat_summarizer.h"

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_stat_summarizer, m) {
  py::class_<tensorflow::StatSummarizer> stat_summ_class(m, "StatSummarizer",
                                                         py::dynamic_attr());
  stat_summ_class
      .def(py::init([](std::string graph_def_serialized) {
        tensorflow::GraphDef proto;
        proto.ParseFromString(graph_def_serialized);
        return new tensorflow::StatSummarizer(proto);
      }))
      .def(py::init([]() {
        return new tensorflow::StatSummarizer(tsl::StatSummarizerOptions());
      }))
      .def("ProcessStepStats", &tensorflow::StatSummarizer::ProcessStepStats)
      .def("GetOutputString", &tensorflow::StatSummarizer::GetOutputString)
      .def("PrintStepStats", &tensorflow::StatSummarizer::PrintStepStats)
      .def("ProcessStepStatsStr", [](tensorflow::StatSummarizer& self,
                                     const std::string& step_stats_str) {
        tensorflow::StepStats step_stats;
        step_stats.ParseFromString(step_stats_str);
        self.ProcessStepStats(step_stats);
      });
};
