// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_compiled_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_environment.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"

ABSL_FLAG(std::string, graph, "", "Model filename to use for testing.");
ABSL_FLAG(std::string, dispatch_library_dir, "",
          "Path to the dispatch library.");
ABSL_FLAG(bool, use_gpu, false, "Use GPU Accelerator.");

namespace litert {
namespace {

Expected<void> RunModel() {
  if (absl::GetFlag(FLAGS_graph).empty()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Model filename is empty. Use --graph to provide it.");
  }

  ABSL_LOG(INFO) << "Model: " << absl::GetFlag(FLAGS_graph);
  LITERT_ASSIGN_OR_RETURN(auto model,
                          Model::CreateFromFile(absl::GetFlag(FLAGS_graph)));

  const std::string dispatch_library_dir =
      absl::GetFlag(FLAGS_dispatch_library_dir);

  std::vector<litert::Environment::Option> environment_options = {};
  if (!dispatch_library_dir.empty()) {
    environment_options.push_back(litert::Environment::Option{
        litert::Environment::OptionTag::DispatchLibraryDir,
        absl::string_view(dispatch_library_dir)});
  };

  LITERT_ASSIGN_OR_RETURN(
      auto env,
      litert::Environment::Create(absl::MakeConstSpan(environment_options)));

  ABSL_LOG(INFO) << "Create CompiledModel";
  auto accelerator = absl::GetFlag(FLAGS_use_gpu) ? kLiteRtHwAcceleratorGpu
                                                  : kLiteRtHwAcceleratorNone;
  if (accelerator == kLiteRtHwAcceleratorGpu) {
    ABSL_LOG(INFO) << "Using GPU Accelerator";
  }
  LITERT_ASSIGN_OR_RETURN(auto compiled_model,
                          CompiledModel::Create(env, model, accelerator));

  LITERT_ASSIGN_OR_RETURN(auto signatures, model.GetSignatures());
  size_t signature_index = 0;

  ABSL_LOG(INFO) << "Prepare input buffers";

  LITERT_ASSIGN_OR_RETURN(auto input_buffers,
                          compiled_model.CreateInputBuffers(signature_index));

  ABSL_LOG(INFO) << "Prepare output buffers";

  LITERT_ASSIGN_OR_RETURN(auto output_buffers,
                          compiled_model.CreateOutputBuffers(signature_index));

  ABSL_LOG(INFO) << "Run model";
  auto status =
      compiled_model.Run(signature_index, input_buffers, output_buffers);

  ABSL_LOG(INFO) << "Model run completed";

  return status;
}

}  // namespace
}  // namespace litert

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  auto res = litert::RunModel();
  if (!res) {
    LITERT_LOG(LITERT_ERROR, "%s", res.Error().Message().c_str());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
