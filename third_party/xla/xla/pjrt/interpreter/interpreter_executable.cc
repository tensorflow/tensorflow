/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/pjrt/interpreter/interpreter_executable.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/expanders/cholesky_expander.h"
#include "xla/hlo/transforms/expanders/convolution_type_canonicalizer.h"
#include "xla/hlo/transforms/expanders/dynamic_index_splitter.h"
#include "xla/hlo/transforms/expanders/eigh_expander.h"
#include "xla/hlo/transforms/expanders/qr_expander.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/interpreter/interpreter_topology_description.h"
#include "xla/pjrt/layout_mode.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/proto/topology_description.pb.h"
#include "xla/pjrt/utils.h"
#include "xla/service/batchnorm_expander.h"
#include "xla/service/dynamic_dimension_inference.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_module_util.h"
#include "xla/service/layout_assignment.h"
#include "xla/service/topk_rewriter.h"
#include "xla/service/triangular_solve_expander.h"
#include "xla/shape.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/fingerprint.h"

namespace xla {

InterpreterExecutable::InterpreterExecutable(
    std::shared_ptr<HloModule> hlo_module,
    std::optional<DynamicDimensionInference> dynamic_dimension_inference,
    CompileOptions compile_options, InterpreterTopologyDescription topology)
    : hlo_module_(std::move(hlo_module)),
      dynamic_dimension_inference_(std::move(dynamic_dimension_inference)),
      compile_options_(std::move(compile_options)),
      topology_(std::move(topology)) {}

int InterpreterExecutable::num_replicas() const {
  return hlo_module_ ? hlo_module_->config().replica_count() : 1;
}

int InterpreterExecutable::num_partitions() const {
  return hlo_module_ ? hlo_module_->config().num_partitions() : 1;
}

absl::string_view InterpreterExecutable::name() const {
  if (hlo_module_ == nullptr) {
    return "<unknown>";
  }
  return hlo_module_->name();
}

absl::StatusOr<std::vector<std::shared_ptr<HloModule>>>
InterpreterExecutable::GetHloModules() const {
  if (hlo_module_ == nullptr) {
    return std::vector<std::shared_ptr<HloModule>>{};
  }
  return std::vector<std::shared_ptr<HloModule>>{hlo_module_};
}

absl::StatusOr<std::vector<std::vector<absl::string_view>>>
InterpreterExecutable::GetParameterMemoryKinds() const {
  if (hlo_module_ == nullptr) {
    return std::vector<std::vector<absl::string_view>>{};
  }
  std::vector<std::vector<absl::string_view>> out;
  std::vector<absl::string_view>& memory_kinds = out.emplace_back();
  int num_params = hlo_module_->entry_computation()->num_parameters();
  memory_kinds.resize(num_params, "interpreter");
  return out;
}

absl::StatusOr<std::vector<std::vector<absl::string_view>>>
InterpreterExecutable::GetOutputMemoryKinds() const {
  if (hlo_module_ == nullptr) {
    return std::vector<std::vector<absl::string_view>>{};
  }
  std::vector<std::vector<absl::string_view>> out;
  std::vector<absl::string_view>& memory_kinds = out.emplace_back();
  const Shape& result_shape =
      hlo_module_->entry_computation()->root_instruction()->shape();
  int num_outputs =
      result_shape.IsTuple() ? result_shape.tuple_shapes().size() : 1;
  memory_kinds.resize(num_outputs, "interpreter");
  return out;
}

absl::StatusOr<std::string> InterpreterExecutable::FingerprintExecutable()
    const {
  if (hlo_module_ == nullptr) {
    return absl::StrCat(tsl::Fingerprint64("empty"));
  }
  std::string result;
  if (!tsl::SerializeToStringDeterministic(hlo_module_->ToProto(), &result)) {
    return absl::InternalError("Failed to serialize HloModule");
  }
  return absl::StrCat(tsl::Fingerprint64(result));
}

absl::StatusOr<std::string> InterpreterExecutable::SerializeExecutable() const {
  ExecutableAndOptionsProto proto;
  if (hlo_module_ != nullptr) {
    std::string serialized_module;
    if (!tsl::SerializeToStringDeterministic(hlo_module_->ToProto(),
                                             &serialized_module)) {
      return absl::InternalError("Failed to serialize HloModule");
    }
    *proto.mutable_serialized_executable() = std::move(serialized_module);
  }
  ABSL_ASSIGN_OR_RETURN(*proto.mutable_compile_options(),
                   compile_options_.ToProto());
  proto.set_pjrt_client_name(xla::InterpreterName());
  std::string serialized_proto;
  if (!tsl::SerializeToStringDeterministic(proto, &serialized_proto)) {
    return absl::InternalError("Failed to serialize ExecutableAndOptionsProto");
  }
  return serialized_proto;
}

/*static*/ absl::StatusOr<std::unique_ptr<InterpreterExecutable>>
InterpreterExecutable::Deserialize(
    absl::string_view serialized,
    std::optional<InterpreterTopologyDescription> topology,
    std::optional<CompileOptions> options) {
  ExecutableAndOptionsProto proto;
  if (!proto.ParseFromArray(serialized.data(), serialized.size())) {
    return absl::InvalidArgumentError(
        "Failed to parse ExecutableAndOptionsProto from serialized executable");
  }
  HloModuleProto hlo_proto;
  if (!hlo_proto.ParseFromString(proto.serialized_executable())) {
    return absl::InvalidArgumentError(
        "Failed to parse HloModuleProto from serialized executable");
  }
  CompileOptions compile_options;
  if (options.has_value()) {
    compile_options = *std::move(options);
  } else if (proto.has_compile_options()) {
    ABSL_ASSIGN_OR_RETURN(compile_options,
                     CompileOptions::FromProto(proto.compile_options()));
  }
  DebugOptions debug_options =
      compile_options.executable_build_options.has_debug_options()
          ? compile_options.executable_build_options.debug_options()
          : DebugOptions();
  ABSL_ASSIGN_OR_RETURN(
      auto hlo_module_config,
      HloModule::CreateModuleConfigFromProto(hlo_proto, debug_options));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                   HloModule::CreateFromProto(hlo_proto, hlo_module_config));

  ABSL_ASSIGN_OR_RETURN(
      DynamicDimensionInference dynamic_dimension_inference,
      DynamicDimensionInference::Run(
          hlo_module.get(),
          /*op_supports_dynamism_handler=*/[&](HloInstruction* hlo) {
            return OpDynamismSupport::kOptional;
          }));

  return std::make_unique<InterpreterExecutable>(
      std::move(hlo_module), std::move(dynamic_dimension_inference),
      std::move(compile_options),
      topology.has_value() ? *std::move(topology)
                           : InterpreterTopologyDescription());
}

absl::StatusOr<std::unique_ptr<HloModule>> RunInterpreterHloPasses(
    std::unique_ptr<HloModule> hlo_module) {
  HloPassPipeline pipeline("Interpreter");

  // The TopkDecomposer generates a compare op with type=TOTALORDER and must
  // run before the ComparisonExpander which rewrites such comparisons.
  pipeline.AddPass<TopkDecomposer>();
  pipeline.AddPass<DynamicIndexSplitter>();
  pipeline.AddPass<CholeskyExpander>();
  pipeline.AddPass<QrExpander>();
  pipeline.AddPass<EighExpander>();
  pipeline.AddPass<TriangularSolveExpander>();
  pipeline.AddPass<BatchNormExpander>(
      /*rewrite_training_op=*/true,
      /*rewrite_inference_op=*/true,
      /*rewrite_grad_op=*/true);
  pipeline.AddPass<LayoutAssignment>(
      hlo_module->mutable_entry_computation_layout());
  pipeline.AddPass<ConvolutionTypeCanonicalizer>();

  ABSL_RETURN_IF_ERROR(pipeline.Run(hlo_module.get()).status());
  return hlo_module;
}

namespace {

absl::StatusOr<Shape> ChooseCompactLayoutForShape(const Shape& shape) {
  return LayoutUtil::GetWithDefaultLayout(shape);
}

absl::StatusOr<std::unique_ptr<InterpreterExecutable>>
CompileInterpreterExecutableInternal(
    const XlaComputation& computation,
    const std::vector<const Shape*>& argument_shapes,
    LayoutCanonicalizationCallback layout_canonicalization_callback,
    CompileOptions options, const InterpreterTopologyDescription& topology) {
  ABSL_RETURN_IF_ERROR(options.ApplyAllOptionOverrides());
  if (layout_canonicalization_callback != nullptr) {
    options.executable_build_options.set_layout_canonicalization_callback(
        layout_canonicalization_callback);
  }

  ABSL_ASSIGN_OR_RETURN(ProgramShape program_shape, computation.GetProgramShape());

  const ExecutableBuildOptions& build_options =
      options.executable_build_options;
  ExecutionOptions execution_options =
      CreateExecutionOptions(build_options, &program_shape);

  // Unoptimized HloModuleConfig.
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModuleConfig> hlo_module_config,
      CreateModuleConfig(program_shape, argument_shapes, &execution_options,
                         execution_options.num_replicas(),
                         /*num_threads=*/std::nullopt,
                         /*aot_options=*/nullptr));
  // Unoptimized HloModule.
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      HloModule::CreateFromProto(computation.proto(), *hlo_module_config));

  if (build_options.num_partitions() != 1) {
    return absl::UnimplementedError(
        "For the time being, only num_partitions=1 is supported.");
  }

  if (!build_options.run_backend_only()) {
    ABSL_ASSIGN_OR_RETURN(hlo_module,
                     RunInterpreterHloPasses(std::move(hlo_module)));
  }

  ABSL_ASSIGN_OR_RETURN(
      DynamicDimensionInference dynamic_dimension_inference,
      DynamicDimensionInference::Run(
          hlo_module.get(),
          /*op_supports_dynamism_handler=*/[&](HloInstruction* hlo) {
            return OpDynamismSupport::kOptional;
          }));

  return std::make_unique<InterpreterExecutable>(
      std::move(hlo_module), std::move(dynamic_dimension_inference),
      std::move(options), topology);
}

}  // namespace

absl::StatusOr<std::unique_ptr<InterpreterExecutable>>
CompileInterpreterExecutable(const XlaComputation& computation,
                             CompileOptions options,
                             const InterpreterTopologyDescription& topology) {
  std::vector<const Shape*> argument_layout_pointers;
  const ExecutableBuildOptions& build_options =
      options.executable_build_options;
  const bool allow_auto_layout =
      build_options.has_debug_options() &&
      build_options.debug_options().xla_pjrt_allow_auto_layout_in_hlo();
  ABSL_RETURN_IF_ERROR(DetermineArgumentLayoutsFromCompileOptions(
      computation,
      [allow_auto_layout](Shape shape) -> absl::StatusOr<Shape> {
        if (allow_auto_layout && !shape.has_layout()) {
          return shape;
        }
        return ChooseCompactLayoutForShape(shape);
      },
      options.argument_layouts, &options.executable_build_options,
      &argument_layout_pointers));
  return CompileInterpreterExecutableInternal(
      computation, argument_layout_pointers,
      /*layout_canonicalization_callback=*/nullptr, std::move(options),
      topology);
}

absl::StatusOr<std::unique_ptr<InterpreterExecutable>>
CompileInterpreterExecutable(MaybeOwningMlirModule module,
                             CompileOptions options,
                             const InterpreterTopologyDescription& topology) {
  XlaComputation xla_computation;
  ExecutableBuildOptions& exec_build_options = options.executable_build_options;
  ABSL_RETURN_IF_ERROR(MlirToXlaComputation(
      module.mlir_module(), xla_computation,
      /*use_tuple_args=*/options.parameter_is_tupled_arguments,
      /*return_tuple=*/false, &exec_build_options));

  // If the compile options specify argument layout, then let's
  // fall back to using the options to determine layouts.
  if (options.argument_layouts) {
    return CompileInterpreterExecutable(xla_computation, std::move(options),
                                        topology);
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<LayoutMode> arg_layout_modes,
                   GetArgLayoutModes(module.mlir_module()));
  ABSL_ASSIGN_OR_RETURN(std::vector<LayoutMode> out_layout_modes,
                   GetOutputLayoutModes(module.mlir_module()));
  ABSL_ASSIGN_OR_RETURN(std::vector<MemorySpaceColor> arg_memory_spaces,
                   GetArgMemoryKinds(module.mlir_module()));
  ABSL_ASSIGN_OR_RETURN(std::vector<MemorySpaceColor> out_memory_spaces,
                   GetOutputMemoryKinds(module.mlir_module()));

  // MLIR module no longer required - release any memory if owned.
  module = MaybeOwningMlirModule();

  // If auto-sharding modifies shapes of arguments and/or result,
  // we get a callback to restore the layouts. Let us restore the layouts
  // according to the attributes we parsed from MLIR.
  auto layout_callback = [arg_layout_modes, out_layout_modes, arg_memory_spaces,
                          out_memory_spaces](const HloModule& module)
      -> absl::StatusOr<std::pair<std::vector<Shape>, Shape>> {
    XlaComputation xla_computation(XlaComputation(module.ToProto()));
    return LayoutModesToXlaShapes(
        xla_computation, arg_layout_modes, out_layout_modes, arg_memory_spaces,
        out_memory_spaces, ChooseCompactLayoutForShape);
  };

  // This call will update result_layout in options.executable_build_options.
  ABSL_ASSIGN_OR_RETURN(
      auto arg_layouts_and_pointers,
      LayoutModesToXla(xla_computation, arg_layout_modes, out_layout_modes,
                       arg_memory_spaces, out_memory_spaces,
                       ChooseCompactLayoutForShape,
                       options.executable_build_options));
  options.argument_layouts = std::move(arg_layouts_and_pointers.first);
  return CompileInterpreterExecutableInternal(
      xla_computation, arg_layouts_and_pointers.second,
      std::move(layout_callback), std::move(options), topology);
}

}  // namespace xla
