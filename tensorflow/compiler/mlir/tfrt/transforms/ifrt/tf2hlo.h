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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_TF2HLO_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_TF2HLO_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_compilation.pb.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/topology.h"
#include "xla/service/hlo.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"

namespace tensorflow {
namespace ifrt_serving {

struct Tf2HloArg {
  mlir::ModuleOp module;
  // `input_dtypes_and_shapes` can be mutable during Tf2HLO compilation.
  std::vector<DtypeAndShape> input_dtypes_and_shapes;
  absl::Span<const int> variable_arg_indices;
  absl::string_view entry_function_name;
  // `compile_metadata` can be mutable during Tf2HLO compilation.
  tensorflow::tpu::TPUCompileMetadataProto compile_metadata;
  tensorflow::XlaHelpers::ShapeRepresentationFn shape_representation_fn;
  std::shared_ptr<xla::ifrt::Topology> topology;
  absl::string_view platform_name;

  absl::StatusOr<uint64_t> Fingerprint() const;
};

struct Tf2HloResult {
  xla::HloModuleProto hlo_module_proto;
  tensorflow::tpu::TPUCompileMetadataProto compile_metadata;
  tf2xla::HostComputeMetadata host_compute_metadata;
  Tf2HLOResultProto ToProto() const;
};

absl::Status UpdateCompileMetadata(
    tensorflow::tpu::TPUCompileMetadataProto& metadata,
    absl::Span<const DtypeAndShape> inputs);

absl::StatusOr<tensorflow::tpu::TPUCompileMetadataProto> GetCompileMetadata(
    mlir::ModuleOp module, const xla::ifrt::Client& ifrt_client);

class TfToHloCompiler {
 public:
  TfToHloCompiler() = default;
  virtual ~TfToHloCompiler() = default;

  // Returns a cache key that can be used to identify the result of
  // CompileTfToHlo.
  virtual absl::StatusOr<std::string> Key(const Tf2HloArg& arg);

  virtual absl::StatusOr<Tf2HloResult> CompileTfToHlo(Tf2HloArg& arg);
};

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_TF2HLO_H_
