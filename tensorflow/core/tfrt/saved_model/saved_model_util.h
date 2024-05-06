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

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tsl/platform/protobuf.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {

// Filename for serialized BEF Buffer.
inline constexpr char kBefBufferFileName[] = "serialized_bef.mlir.bef";

// Filename for serialized MLRT bytecode Buffer.
inline constexpr char kMlrtBufferFileName[] = "serialized_mlrt.mlir.mlrt";

// Filename for serialized MLIR_MODULE.
inline constexpr char kMlirModuleFilename[] = "serialized_mlir.mlir";

// Subdirectory where AoT Packages are saved
inline constexpr char kAotPackagesDirectory[] = "aot_packages";

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

// If `import_signature_names` is non-empty, this function only imports the
// graph that corresponds to this list.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportSavedModel(
    mlir::MLIRContext* context, const tensorflow::MetaGraphDef& meta_graph_def,
    const FallbackState& fallback_state, std::string saved_model_dir,
    bool import_user_signatures, bool run_placer_grappler_on_functions,
    const std::vector<std::string>& import_signature_names = {});

absl::StatusOr<tensorflow::MetaGraphDef> ReadSavedModel(
    absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& tags);

using SignatureMap = absl::flat_hash_map<std::string, internal::Signature>;
using ::tensorflow::StatusOr;

struct Initializer {
  std::string name;
  std::vector<tensorflow::Tensor> inputs;
};

struct InitializersAndSignatures {
  // Initializers are kept in a certain order as they need to be executed in
  // that order.
  std::vector<Initializer> initializers;
  SignatureMap signature_map;
};

// If `saved_model_dir` is non-empty, this function fills in the Initializer's
// inputs in the returned result.
absl::StatusOr<InitializersAndSignatures> GetInitializersAndSignatures(
    mlir::ModuleOp module, absl::string_view saved_model_dir = "");

std::string GetAotPackagePath(absl::string_view saved_model_dir);

std::string GetBefFilePath(std::string aot_package_directory);

std::string GetMlirFilePath(const std::string& aot_package_directory);

// TODO(b/295241000): Implement MLIR deserialization to skip it AoT and remove
// redundant steps
absl::StatusOr<tfrt::BefBuffer> LoadBefAndMlir(
    const TfrtCompileOptions& options, mlir::ModuleOp mlir_module,
    const std::string& saved_model_dir,
    tfrt_stub::FallbackState* fallback_state);

absl::StatusOr<mlrt::bc::Buffer> LoadMlrtAndMlir(
    const TfrtCompileOptions& options, mlir::ModuleOp mlir_module,
    const std::string& saved_model_dir,
    tfrt_stub::FallbackState* fallback_state);

absl::Status DeserializeAoTMlirModule(
    absl::string_view saved_model_dir, mlir::MLIRContext* context,
    mlir::OwningOpRef<mlir::ModuleOp>* mlir_module);

CallableOptions CombineSignatureDefs(
    const google::protobuf::Map<std::string, SignatureDef>& signature_defs);

void RegisterTfrtDialectsForAot(mlir::DialectRegistry& registry);

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_UTIL_H_
