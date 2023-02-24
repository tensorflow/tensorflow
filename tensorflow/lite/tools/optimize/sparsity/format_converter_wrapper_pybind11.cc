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

#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/utils/sparsity_format_converter.h"

namespace py = pybind11;

using FormatConverterFp32 = tflite::internal::sparsity::FormatConverter<float>;

PYBIND11_MODULE(format_converter_wrapper_pybind11, m) {
  m.doc() = "Python wrapper for the tflite sparse tensor converter.";

  py::enum_<TfLiteStatus>(m, "TfLiteStatus")
      .value("TF_LITE_OK", TfLiteStatus::kTfLiteOk)
      .value("TF_LITE_ERROR", TfLiteStatus::kTfLiteError)
      .export_values();

  py::enum_<TfLiteDimensionType>(m, "TfLiteDimensionType")
      .value("TF_LITE_DIM_DENSE", TfLiteDimensionType::kTfLiteDimDense)
      .value("TF_LITE_DIM_SPARSE_CSR", TfLiteDimensionType::kTfLiteDimSparseCSR)
      .export_values();

  py::class_<FormatConverterFp32>(m, "FormatConverterFp32")
      .def(py::init</*shape=*/const std::vector<int>&,
                    /*traversal_order=*/const std::vector<int>&,
                    /*format=*/const std::vector<TfLiteDimensionType>&,
                    /*block_size=*/const std::vector<int>&,
                    /*block_map=*/const std::vector<int>&>())
      .def(py::init</*shape=*/const std::vector<int>&,
                    /*sparsity=*/const TfLiteSparsity&>())
      .def("GetData", &FormatConverterFp32::GetData)
      .def("GetDimMetadata", &FormatConverterFp32::GetDimMetadata)
      .def("DenseToSparse",
           [](FormatConverterFp32& converter, py::buffer buf) {
             py::buffer_info buffer_info = buf.request();
             return converter.DenseToSparse(
                 static_cast<float*>(buffer_info.ptr));
           })
      .def("SparseToDense", [](FormatConverterFp32& converter, py::buffer buf) {
        py::buffer_info buffer_info = buf.request();
        return converter.SparseToDense(static_cast<float*>(buffer_info.ptr));
      });
}
