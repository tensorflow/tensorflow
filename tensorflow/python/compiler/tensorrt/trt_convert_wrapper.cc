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

#include <cstddef>
#include <iterator>

#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/compiler/tf2tensorrt/convert/trt_parameters.h"
#include "tensorflow/compiler/tf2tensorrt/experimental/trt_convert_api.h"
#include "tensorflow/python/lib/core/ndarray_tensor.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace py = pybind11;

using tensorflow::MaybeRaiseRegisteredFromStatus;
using tensorflow::tensorrt::ProfileStrategy;
using tensorflow::tensorrt::TrtPrecisionMode;

namespace tensorflow {

// Wrapper for `py::iterator` that converts the yielded results
// into instances of `std::vector<Tensor>`.
class PyTensorInputsIterator {
 public:
  using iterator_category = std::input_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = std::vector<Tensor>;
  using pointer = value_type*;
  using reference = value_type&;

  PyTensorInputsIterator(py::iterator it) : it_(it) {}

  PyTensorInputsIterator& operator++() {
    it_++;
    return *this;
  }

  PyTensorInputsIterator operator++(int) {
    auto rv = *this;
    it_++;
    return rv;
  }

  reference operator*() {
    py::handle handle = *it_;
    auto vec = handle.cast<std::vector<py::object>>();
    value_.clear();
    for (const auto& input : vec) {
      Tensor tensor;
      MaybeRaiseRegisteredFromStatus(NdarrayToTensor(input.ptr(), &tensor));
      value_.push_back(tensor);
    }
    return value_;
  }

  pointer operator->() { return &(operator*()); }

  friend bool operator==(const PyTensorInputsIterator& a,
                         const PyTensorInputsIterator& b) {
    return a.it_ == b.it_;
  }

  friend bool operator!=(const PyTensorInputsIterator& a,
                         const PyTensorInputsIterator& b) {
    return a.it_ != b.it_;
  }

 private:
  py::iterator it_;
  value_type value_;
};

// Update bound attributes with the given values.
template <class T>
T Replace(T& t, py::kwargs& kwargs) {
  py::object obj = py::cast(t);
  for (auto item : kwargs) {
    if (py::hasattr(obj, item.first)) {
      py::setattr(obj, item.first, item.second);
    }
  }
  return py::cast<T>(obj);
}

}  // namespace tensorflow

namespace pybind11 {
namespace detail {

template <>
struct type_caster<TrtPrecisionMode> {
 public:
  PYBIND11_TYPE_CASTER(TrtPrecisionMode, _("TrtPrecisionMode"));

  bool load(handle src, bool convert) {
    if (py::isinstance(src, py::module_::import("builtins").attr("str"))) {
      auto status =
          TrtPrecisionModeFromName(py::cast<std::string>(src), &value);
      return status.ok();
    }
    return false;
  }

  static handle cast(TrtPrecisionMode src, return_value_policy policy,
                     handle parent) {
    std::string name;
    MaybeRaiseRegisteredFromStatus(TrtPrecisionModeToName(src, &name));
    py::object obj = py::cast(name);
    return obj.release();
  }
};

template <>
struct type_caster<ProfileStrategy> {
 public:
  PYBIND11_TYPE_CASTER(ProfileStrategy, _("ProfileStrategy"));

  bool load(handle src, bool convert) {
    if (py::isinstance(src, py::module_::import("builtins").attr("str"))) {
      auto status = ProfileStrategyFromName(py::cast<std::string>(src), &value);
      return status.ok();
    }
    return false;
  }

  static handle cast(ProfileStrategy src, return_value_policy policy,
                     handle parent) {
    std::string name = ProfileStrategyToName(src);
    py::object obj = py::cast(name);
    return obj.release();
  }
};

}  // namespace detail
}  // namespace pybind11

PYBIND11_MODULE(_pywrap_trt_convert, m) {
  py::enum_<TrtPrecisionMode>(m, "TrtPrecisionMode")
      .value("FP32", TrtPrecisionMode::FP32)
      .value("FP16", TrtPrecisionMode::FP16)
      .value("INT8", TrtPrecisionMode::INT8);

  py::enum_<ProfileStrategy>(m, "TrtProfileStrategy")
      .value("kRange", ProfileStrategy::kRange)
      .value("kOptimal", ProfileStrategy::kOptimal)
      .value("kRangeOptimal", ProfileStrategy::kRangeOptimal)
      .value("kImplicitBatchModeCompatible",
             ProfileStrategy::kImplicitBatchModeCompatible);

#if GOOGLE_CUDA && GOOGLE_TENSORRT

  using tensorflow::tensorrt::TrtConversionParams;
  using tensorflow::tensorrt::TrtGraphConverter;

  py::class_<TrtConversionParams>(m, "TrtConversionParams")
      .def(py::init<>())
      .def(py::init([](py::kwargs& kwargs) {
        auto params = TrtConversionParams();
        return tensorflow::Replace(params, kwargs);
      }))
      .def_readwrite("max_workspace_size_bytes",
                     &TrtConversionParams::max_workspace_size_bytes)
      .def_readwrite("precision_mode", &TrtConversionParams::precision_mode)
      .def_readwrite("minimum_segment_size",
                     &TrtConversionParams::minimum_segment_size)
      .def_readwrite("maximum_cached_engines",
                     &TrtConversionParams::maximum_cached_engines)
      .def_readwrite("use_calibration", &TrtConversionParams::use_calibration)
      .def_readwrite("use_dynamic_shape",
                     &TrtConversionParams::use_dynamic_shape)
      .def_readwrite("dynamic_shape_profile_strategy",
                     &TrtConversionParams::dynamic_shape_profile_strategy)
      .def_readwrite("allow_build_at_runtime",
                     &TrtConversionParams::allow_build_at_runtime);

  py::class_<TrtGraphConverter>(m, "TrtGraphConverter")
      .def(py::init([](const std::string& frozen_graph_def_str,
                       const std::vector<std::string>& input_names,
                       const std::vector<std::string>& output_names,
                       const TrtConversionParams& conversion_params) {
        tensorflow::GraphDef graph_def;
        graph_def.ParseFromString(frozen_graph_def_str);
        auto status_or_converter = TrtGraphConverter::Create(
            graph_def, input_names, output_names, conversion_params);
        MaybeRaiseRegisteredFromStatus(status_or_converter.status());
        return std::move(status_or_converter.value());
      }))
      .def(py::init([](const std::string& saved_model_dir,
                       const std::string& signature_key,
                       const std::unordered_set<std::string>& tags,
                       const TrtConversionParams& conversion_params) {
        auto status_or_converter = TrtGraphConverter::Create(
            saved_model_dir, signature_key, tags, conversion_params);
        MaybeRaiseRegisteredFromStatus(status_or_converter.status());
        return std::move(status_or_converter.value());
      }))
      // TODO: Figure out how much the overhead is for serializing /
      // deserializing GraphDef arguments and return values (mem and time).
      // TODO: For `convert` and `build`, change C++ API to take an iterator for
      // inputs.
      .def("convert",
           [](TrtGraphConverter& self, std::optional<py::iterator> inputs,
              bool disable_non_trt_optimizers,
              const std::string& device_requested) {
             tensorflow::StatusOr<tensorflow::GraphDef> status_or_graph_def;
             std::vector<std::vector<tensorflow::Tensor>> calibration_inputs =
                 {};
             if (inputs.has_value()) {
               tensorflow::PyTensorInputsIterator start(inputs.value());
               tensorflow::PyTensorInputsIterator end(inputs.value().end());
               calibration_inputs = std::vector(start, end);
             }
             status_or_graph_def =
                 self.Convert(calibration_inputs, disable_non_trt_optimizers,
                              device_requested);
             MaybeRaiseRegisteredFromStatus(status_or_graph_def.status());
             tensorflow::GraphDef graph_def = status_or_graph_def.value();
             std::string graph_def_str;
             graph_def.SerializeToString(&graph_def_str);
             return py::bytes(graph_def_str);
           })
      .def("build",
           [](TrtGraphConverter& self, py::iterator inputs) {
             tensorflow::PyTensorInputsIterator start(inputs);
             tensorflow::PyTensorInputsIterator end(inputs.end());
             std::vector<std::vector<tensorflow::Tensor>> infer_inputs(start,
                                                                       end);
             auto status_or_graph_def = self.Build(infer_inputs);
             MaybeRaiseRegisteredFromStatus(status_or_graph_def.status());
             tensorflow::GraphDef graph_def = status_or_graph_def.value();
             std::string graph_def_str;
             graph_def.SerializeToString(&graph_def_str);
             return py::bytes(graph_def_str);
           })
      .def("_serialize_engines",
           [](TrtGraphConverter& self, const std::string& out_dir,
              bool save_gpu_specific_engines) {
             auto status_or_map =
                 self.SerializeEngines(out_dir, save_gpu_specific_engines);
             MaybeRaiseRegisteredFromStatus(status_or_map.status());
             return status_or_map.value();
           })
      .def("_get_input_graph_def", [](TrtGraphConverter& self) {
        std::string input_graph_def_str;
        self.input_graph_def.SerializeToString(&input_graph_def_str);
        return py::bytes(input_graph_def_str);
      });

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
}