/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include <utility>
#include <vector>

#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/tool_params.h"

#if defined(TFLITE_ENABLE_HEXAGON_YNN)
#include <cstdint>

#include "tensorflow/lite/delegates/hexagon_ynn/hexagon_ynn_delegate.h"
#endif

namespace tflite {
namespace tools {

class HexagonYnnDelegateProvider : public DelegateProvider {
 public:
  HexagonYnnDelegateProvider() {
#if defined(TFLITE_ENABLE_HEXAGON_YNN)
    default_params_.AddParam("use_hexagon_ynn", ToolParam::Create<bool>(false));
    default_params_.AddParam("hexagon_ynn_lib_path",
                             ToolParam::Create<std::string>("/data/local/tmp"));
    default_params_.AddParam("hexagon_ynn_backend",
                             ToolParam::Create<std::string>(""));
    default_params_.AddParam("hexagon_ynn_num_threads",
                             ToolParam::Create<int32_t>(0));
#endif
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "HexagonYnn"; }
};
REGISTER_DELEGATE_PROVIDER(HexagonYnnDelegateProvider);

std::vector<Flag> HexagonYnnDelegateProvider::CreateFlags(
    ToolParams* params) const {
#if defined(TFLITE_ENABLE_HEXAGON_YNN)
  std::vector<Flag> flags = {
      CreateFlag<bool>("use_hexagon_ynn", params,
                       "Use the Hexagon DSP delegate"),
      CreateFlag<std::string>(
          "hexagon_ynn_lib_path", params,
          "Path to the directory containing libdsp_graph.so (host stub) "
          "and the dsp/ subdirectory with libdsp_graph_skel.so."),
      CreateFlag<std::string>(
          "hexagon_ynn_backend", params,
          "Backend to run on the Hexagon DSP: 'ynnpack' (default) or "
          "'xnnpack'. When empty, respects --use_ynnpack / XNNPACK_USE_YNNPACK "
          "and defaults to 'ynnpack'."),
      CreateFlag<int32_t>(
          "hexagon_ynn_num_threads", params,
          "Number of Hexagon DSP worker threads (0 inherits --num_threads when "
          "set, or defaults to 4).")};
  return flags;
#else
  return {};
#endif
}

void HexagonYnnDelegateProvider::LogParams(const ToolParams& params,
                                           bool verbose) const {
#if defined(TFLITE_ENABLE_HEXAGON_YNN)
  LOG_TOOL_PARAM(params, bool, "use_hexagon_ynn", "Use HexagonYnn", verbose);
  LOG_TOOL_PARAM(params, std::string, "hexagon_ynn_lib_path",
                 "HexagonYnn lib path", verbose);
  LOG_TOOL_PARAM(params, std::string, "hexagon_ynn_backend",
                 "HexagonYnn DSP backend", verbose);
  LOG_TOOL_PARAM(params, int32_t, "hexagon_ynn_num_threads",
                 "HexagonYnn DSP threads", verbose);
#endif
}

TfLiteDelegatePtr HexagonYnnDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  TfLiteDelegatePtr delegate = CreateNullDelegate();
#if defined(TFLITE_ENABLE_HEXAGON_YNN)
  if (params.Get<bool>("use_hexagon_ynn")) {
    TfLiteHexagonYnnDelegateOptions options =
        TfLiteHexagonYnnDelegateOptionsDefault();
    options.max_delegated_partitions =
        params.Get<int>("max_delegated_partitions");
    options.min_nodes_per_partition =
        params.Get<int>("min_nodes_per_partition");

    const std::string backend = params.Get<std::string>("hexagon_ynn_backend");
    if (backend == "xnnpack" || backend == "XNNPACK") {
      options.backend = kTfLiteHexagonYnnBackendXnnpack;
    } else if (backend == "ynnpack" || backend == "YNNPACK") {
      options.backend = kTfLiteHexagonYnnBackendYnnpack;
    } else if (params.HasParam("use_ynnpack") &&
               params.HasValueSet<bool>("use_ynnpack")) {
      options.backend = params.Get<bool>("use_ynnpack")
                            ? kTfLiteHexagonYnnBackendYnnpack
                            : kTfLiteHexagonYnnBackendXnnpack;
    }

    int32_t dsp_threads = params.Get<int32_t>("hexagon_ynn_num_threads");
    if (dsp_threads <= 0 && params.HasParam("num_threads")) {
      const int32_t host_threads = params.Get<int32_t>("num_threads");
      if (host_threads > 0) {
        dsp_threads = host_threads;
      }
    }
    if (dsp_threads > 0) {
      options.num_threads = dsp_threads;
    }

    TfLiteHexagonYnnInit(
        params.Get<std::string>("hexagon_ynn_lib_path").c_str());
    delegate = TfLiteDelegatePtr(TfLiteHexagonYnnDelegateCreate(&options),
                                 TfLiteHexagonYnnDelegateDelete);

    if (!delegate.get()) {
      TFLITE_LOG(WARN)
          << "Could not create HexagonYnn delegate: platform may not "
             "support delegate or required libraries are missing";
    }
  }
#endif
  return delegate;
}

std::pair<TfLiteDelegatePtr, int>
HexagonYnnDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  int rank = 0;
#if defined(TFLITE_ENABLE_HEXAGON_YNN)
  rank = params.GetPosition<bool>("use_hexagon_ynn");
#endif
  return std::make_pair(std::move(ptr), rank);
}

}  // namespace tools
}  // namespace tflite
