/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {
namespace grappler {

int GetNumAvailableGPUs() {
  int num_eligible_gpus = 0;
  if (ValidateGPUMachineManager().ok()) {
    perftools::gputools::Platform* gpu_manager = GPUMachineManager();
    if (gpu_manager != nullptr) {
      int num_gpus = gpu_manager->VisibleDeviceCount();
      for (int i = 0; i < num_gpus; i++) {
        auto exec_status = gpu_manager->ExecutorForDevice(i);
        if (exec_status.ok()) {
          perftools::gputools::StreamExecutor* se = exec_status.ValueOrDie();
          const perftools::gputools::DeviceDescription& desc =
              se->GetDeviceDescription();
          int min_gpu_core_count = 8;
          if (desc.core_count() >= min_gpu_core_count) {
            num_eligible_gpus++;
          }
        }
      }
    }
  }
  LOG(INFO) << "Number of eligible GPUs (core count >= 8): "
            << num_eligible_gpus;
  return num_eligible_gpus;
}

int GetNumAvailableLogicalCPUCores() { return port::NumSchedulableCPUs(); }

string ParseNodeName(const string& name, int* position) {
  // Strip the prefix '^' (if any), and strip the trailing ":{digits} (if any)
  // to get a node name.
  strings::Scanner scan(name);
  scan.ZeroOrOneLiteral("^")
      .RestartCapture()
      .One(strings::Scanner::LETTER_DIGIT_DOT)
      .Any(strings::Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE);
  StringPiece capture;
  StringPiece remaining;
  if (scan.Peek(':') != ':' || !scan.GetResult(&remaining, &capture)) {
    *position = 0;
    return "";
  } else {
    if (name[0] == '^') {
      *position = -1;
    } else if (remaining.empty()) {
      *position = 0;
    } else {
      // Skip the first ':' character.
      *position = std::stoi(remaining.substr(1).ToString());
    }
    return capture.ToString();
  }
}

string NodeName(const string& name) {
  int position;
  return ParseNodeName(name, &position);
}

int NodePosition(const string& name) {
  int position;
  ParseNodeName(name, &position);
  return position;
}

string AddPrefixToNodeName(const string& name, const string& prefix) {
  if (!name.empty()) {
    if (name[0] == '^') {
      return strings::StrCat("^", prefix, "-", name.substr(1));
    }
  }
  return strings::StrCat(prefix, "-", name);
}

}  // end namespace grappler
}  // end namespace tensorflow
