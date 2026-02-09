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

#include "include/pybind11/pybind11.h"
#include "include/pybind11/stl.h"
#include "tensorflow_text/core/kernels/fast_wordpiece_tokenizer_model_builder.h"

namespace tensorflow {
namespace text {

namespace py = pybind11;

PYBIND11_MODULE(pywrap_fast_wordpiece_tokenizer_model_builder, m) {
  m.def("build_fast_wordpiece_model",
        [](const std::vector<std::string>& vocab, int max_bytes_per_token,
           const std::string& suffix_indicator, const std::string& unk_token,
           bool no_pretokenization, bool support_detokenization) {
          const auto result = BuildModelAndExportToFlatBuffer(
              vocab, max_bytes_per_token, suffix_indicator, unk_token,
              no_pretokenization, support_detokenization);
          if (!result.status().ok()) {
            // Propagate the error to the Python code.
            throw std::runtime_error(std::string(result.status().message()));
          }
          return py::bytes(*result);
        });
}

}  // namespace text
}  // namespace tensorflow
