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

#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/tf2hlo.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v2/legalize_tf.h"
#include "tensorflow/compiler/tf2xla/layout_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/client/client_library.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/shape.h"
#include "xla/stream_executor/multi_platform_manager.h"
#include "xla/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace ifrt_serving {

static constexpr absl::string_view kEntryFuncName = "main";

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CompileTfToHlo(
    mlir::ModuleOp module, absl::Span<const tensorflow::Tensor> inputs,
    absl::string_view entry_function_name, xla::ifrt::Compiler* ifrt_compiler,
    tensorflow::XlaHelpers::ShapeRepresentationFn shape_representation_fn) {
  if (VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("ifrt_before_bridge_phase2", module);
  }

  tpu::MlirToHloArgs mlir_to_hlo_args;
  std::string module_str = tensorflow::SerializeMlirModule(module);
  mlir_to_hlo_args.mlir_module = module_str;

  TF_ASSIGN_OR_RETURN(
      auto* platform,
      stream_executor::MultiPlatformManager::PlatformWithName("Host"));
  TF_ASSIGN_OR_RETURN(
      auto* client, xla::ClientLibrary::GetOrCreateCompileOnlyClient(platform));

  std::vector<TensorShape> arg_shapes;
  tpu::TPUCompileMetadataProto metadata;
  metadata.set_num_cores_per_replica(1);
  metadata.set_num_replicas(1);

  for (const auto& input : inputs) {
    arg_shapes.push_back(input.shape());

    auto* metadata_arg1 = metadata.add_args();
    metadata_arg1->set_dtype(input.dtype());
    *metadata_arg1->mutable_shape() = input.shape().AsProto();
    // metadata_arg1->set_name(input.name());

    // TODO(b/305734600): populate right kind once variable loading is
    // supported.
    metadata_arg1->set_kind(tpu::TPUCompileMetadataProto::Arg::PARAMETER);
  }

  auto entry_fn = module.lookupSymbol<mlir::func::FuncOp>(kEntryFuncName);
  if (!entry_fn) {
    return absl::InternalError("Could not find entry function in MLIR Module.");
  }

  if (inputs.size() != entry_fn.getNumArguments()) {
    return absl::InternalError(
        absl::StrCat("Number of inputs mismatched! Expect",
                     entry_fn.getNumArguments(), " got", inputs.size()));
  }

  for (int i = 0; i < entry_fn.getNumResults(); i++) {
    metadata.add_retvals();
  }

  bool use_tuple_args = false;
  std::vector<tpu::ShardingAndIndex> arg_core_mapping;
  std::vector<std::vector<xla::Shape>> per_core_arg_shapes;
  std::vector<std::unique_ptr<mlir::Pass>> custom_legalization_passes;

  TF_ASSIGN_OR_RETURN(
      tensorflow::XlaCompiler::CompilationResult compilation_result,
      tensorflow::tf2xla::v2::LegalizeMlirToHlo(
          mlir_to_hlo_args, metadata, use_tuple_args,
          /*device_type=*/"XLA_TPU_JIT", custom_legalization_passes,
          /*shape_determination_fns=*/
          tensorflow::XlaShapeLayoutHelpers::ShapeDeterminationFns(
              tensorflow::UseNoPreferenceLayoutFn(), shape_representation_fn),
          arg_shapes, &arg_core_mapping, &per_core_arg_shapes, client));

  mlir::OwningOpRef<mlir::ModuleOp> mlir_hlo_module =
      mlir::ModuleOp::create(module->getLoc());

  TF_RETURN_IF_ERROR(xla::ConvertHloToMlirHlo(
      *mlir_hlo_module, &compilation_result.computation->proto()));

  if (VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("ifrt_after_bridge_phase2",
                                 mlir_hlo_module.get());
  }

  return mlir_hlo_module;
}

}  // namespace ifrt_serving
}  // namespace tensorflow
