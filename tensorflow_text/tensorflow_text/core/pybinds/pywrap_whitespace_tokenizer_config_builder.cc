// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdexcept>
#include <iostream>
#include "include/pybind11/pybind11.h"
#include "include/pybind11/stl.h"
#include "tensorflow_text/core/kernels/whitespace_tokenizer_config_builder.h"

namespace tensorflow {
namespace text {

namespace py = pybind11;

PYBIND11_MODULE(pywrap_whitespace_tokenizer_config_builder, m) {
  m.def("build_whitespace_tokenizer_config",
        []() {
          const auto result = BuildWhitespaceTokenizerConfig();
          return py::bytes(result);
        });
}

}  // namespace text
}  // namespace tensorflow
