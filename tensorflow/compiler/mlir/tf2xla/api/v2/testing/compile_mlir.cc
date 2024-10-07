/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tf2xla/api/v2/testing/compile_mlir.h"

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tf2xla/api/v2/legalize_tf.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/utils/test_metadata_config.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "xla/client/client_library.h"
#include "xla/shape.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace tf2xla {
namespace v2 {
namespace testing {

using tpu::MlirToHloArgs;
using tpu::ShardingAndIndex;
using tpu::TPUCompileMetadataProto;

absl::StatusOr<XlaCompiler::CompilationResult> CompileMlirModule(
    const char* mlir_module_str,
    ConfigProto::Experimental::MlirBridgeRollout rollout_state,
    absl::string_view device_type) {
  MlirToHloArgs mlir_to_hlo_args;
  mlir_to_hlo_args.rollout_state = rollout_state;
  mlir_to_hlo_args.mlir_module = mlir_module_str;

  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      se::PlatformManager::PlatformWithName("Host"));
  TF_ASSIGN_OR_RETURN(
      auto client, xla::ClientLibrary::GetOrCreateCompileOnlyClient(platform));

  std::vector<TensorShape> arg_shapes;
  TPUCompileMetadataProto metadata_proto;
  // Configure metadata requires parsing the module and if we are testing a
  // failure, we ignore this particular set up error assuming we'll not get
  // far enough to need valid metadata.
  tensorflow::tf2xla::internal::ConfigureMetadata(mlir_module_str, arg_shapes,
                                                  metadata_proto)
      .IgnoreError();
  bool use_tuple_args = true;
  std::vector<ShardingAndIndex> arg_core_mapping;
  std::vector<std::vector<xla::Shape>> per_core_arg_shapes;
  std::vector<std::unique_ptr<mlir::Pass>> custom_legalization_passes;

  return LegalizeMlirToHlo(mlir_to_hlo_args, metadata_proto, use_tuple_args,
                           device_type, custom_legalization_passes,
                           /*shape_determination_fns=*/{}, arg_shapes,
                           &arg_core_mapping, &per_core_arg_shapes, client);
}

}  // namespace testing
}  // namespace v2
}  // namespace tf2xla
}  // namespace tensorflow
