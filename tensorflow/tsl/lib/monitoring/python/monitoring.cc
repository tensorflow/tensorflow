/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <string>

#include "absl/memory/memory.h"
#include "pybind11/pybind11.h"
#include "tensorflow/tsl/lib/monitoring/sampler.h"
#include "tensorflow/tsl/protobuf/histogram.pb.h"

namespace tsl {
namespace monitoring {
namespace {
namespace py = pybind11;
}
PYBIND11_MODULE(monitoring, m) {
  py::class_<Sampler<0>>(m, "Sampler")
      .def(
          "add",
          [](Sampler<0>& sampler, double value) {
            sampler.GetCell()->Add(value);
          },
          py::doc("Records a value in the sampler metric."), py::arg("value"))
      .def(
          "num_values",
          [](Sampler<0>& sampler) { return sampler.GetCell()->value().num(); },
          py::doc("Check number of values recorded in the metric."))
      .def(
          "sum",
          [](Sampler<0>& sampler) { return sampler.GetCell()->value().sum(); },
          py::doc("Check sum of values recorded in the metric."));
  m.def(
      "new_sampler_metric_with_exponential_buckets",
      [](std::string name, std::string description, double scale,
         double growth_factor,
         int bucket_count) -> std::unique_ptr<Sampler<0>> {
        return absl::WrapUnique(Sampler<0>::New(
            {name, description},
            Buckets::Exponential(scale, growth_factor, bucket_count)));
      },
      py::doc("Creates a new sampler metric that uses exponential buckets: "
              "[-DBL_MAX, "
              "..., scale * growth^i, scale * growth_factor^(i + 1), ..., "
              "DBL_MAX]."),
      py::arg("name"), py::arg("description"), py::arg("scale"),
      py::arg("growth_factor"), py::arg("bucket_count"));
}
}  // namespace monitoring
}  // namespace tsl
