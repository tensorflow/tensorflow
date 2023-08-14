/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_UTIL_H_
#define TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_UTIL_H_

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/graph_executor/graph_executor.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/tsl/platform/protobuf.h"
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/request_deadline_tracker.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {

// Filename for serialized BEF Buffer.
inline constexpr char kBefBufferFilenameMLIRBEF[] = "serialized_bef.mlir.bef";

// Filename for serialized MLIR_MODULE.
inline constexpr char kMLIRModuleFilename[] = "serialized_mlir.mlir";

// Subdirectory where AoT Packages are saved
inline constexpr char kAoTPackagesDirectory[] = "aot_packages";

// TODO(tfrt-dev): Replace tfrt::TensorSpec with tensorflow::TensorSpec once the
// latter is checked in.
struct TensorSpec {
  tensorflow::DataType dtype;
  tensorflow::PartialTensorShape shape;

  explicit TensorSpec(tensorflow::DataType dtype) : dtype(dtype) {}
  TensorSpec(tensorflow::DataType dtype, tensorflow::PartialTensorShape shape)
      : dtype(dtype), shape(std::move(shape)) {}
};

inline bool operator==(const TensorSpec& a, const TensorSpec& b) {
  return a.dtype == b.dtype && a.shape.IsIdenticalTo(b.shape);
}

namespace internal {

struct Signature {
  // The following three fields should have the same size.
  std::vector<std::string> input_names;
  std::vector<TensorSpec> input_specs;
  std::vector<std::string> input_devices;

  // The following two fields should have the same size.
  std::vector<std::string> output_names;
  std::vector<TensorSpec> output_specs;
  protobuf::Map<std::string, TensorProto> default_inputs;
};

}  // namespace internal

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportSavedModel(
    mlir::MLIRContext* context, const tensorflow::MetaGraphDef& meta_graph_def,
    const FallbackState& fallback_state, std::string saved_model_dir,
    bool import_user_signatures, bool run_placer_grappler_on_functions);

StatusOr<tensorflow::MetaGraphDef> ReadSavedModel(
    absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& tags);

using SignatureMap = absl::flat_hash_map<std::string, internal::Signature>;
using ::tensorflow::StatusOr;

struct Initializer {
  std::string name;
};

struct InitializersAndSignatures {
  // Initializers are kept in a certain order as they need to be executed in
  // that order.
  std::vector<Initializer> initializers;
  SignatureMap signature_map;
};

StatusOr<InitializersAndSignatures> GetInitializersAndSignatures(
    mlir::ModuleOp module);

std::string GetAotPackagePath(absl::string_view saved_model_dir);

std::string GetBEFFilePath(std::string aot_package_directory);

// TODO(b/295241000): Implement MLIR deserialization to skip it AoT and remove
// redundant steps
absl::StatusOr<tfrt::BefBuffer> LoadAotPackages(
    const TfrtCompileOptions& options, mlir::ModuleOp mlir_module,
    const std::string& saved_model_dir,
    tfrt_stub::FallbackState* fallback_state);

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_UTIL_H_
