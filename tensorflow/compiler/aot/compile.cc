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

#include "tensorflow/compiler/aot/compile.h"

#include <cstddef>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm-c/Target.h"
#include "llvm/Support/ManagedStatic.h"
#include "tensorflow/compiler/aot/codegen.h"
#include "tensorflow/compiler/aot/embedded_constant_buffers.h"
#include "tensorflow/compiler/aot/flags.h"
#include "tensorflow/compiler/aot/quantize.h"
#include "tensorflow/compiler/tf2xla/tf2xla.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "xla/backends/cpu/codegen/symbol_name_util.h"
#include "xla/client/client_library.h"
#include "xla/client/compile_only_client.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/regexp.h"  // IWYU pragma: keep
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace tfcompile {

static llvm::ManagedStatic<QuantizeXlaFn> quantize_xla;

bool RegisterQuantizeFn(const QuantizeXlaFn& fn) {
  if (*quantize_xla) return false;
  *quantize_xla = fn;
  return true;
}

namespace {

// Compiles the XLA computation into executable code.
absl::Status CompileXla(xla::CompileOnlyClient* client,
                        const xla::XlaComputation& computation,
                        const xla::cpu::CpuAotCompilationOptions& aot_opts,
                        CompileResult* compile_result) {
  // Retrieves arg and result layouts from the computation.
  // TODO(toddw): Should we let the user choose the major/minor ordering?
  absl::StatusOr<std::unique_ptr<xla::ProgramShape>> pshape_or =
      client->GetComputationShape(computation);
  if (!pshape_or.ok()) {
    return errors::Unknown("Couldn't get XLA program shape: ",
                           pshape_or.status().message());
  }
  compile_result->program_shape = pshape_or.value()->ToProto();
  xla::ProgramShapeProto* pshape = &compile_result->program_shape;

  // AotXlaComputationInstance::argument_layouts is a vector of Shape
  // pointers. Accumulate the Shape objects themselves in a separate vector
  // while building the vector of pointers.
  std::vector<const xla::Shape*> arg_layout_ptrs(pshape->parameters_size());
  std::vector<xla::Shape> arg_layouts(pshape->parameters_size());
  for (int i = 0; i < pshape->parameters_size(); ++i) {
    TF_ASSIGN_OR_RETURN(arg_layouts[i],
                        xla::Shape::FromProto(pshape->parameters(i)));
    arg_layout_ptrs[i] = &arg_layouts[i];
  }
  xla::CompileOnlyClient::AotXlaComputationInstance instance;
  instance.computation = &computation;
  instance.argument_layouts = std::move(arg_layout_ptrs);
  TF_ASSIGN_OR_RETURN(xla::Shape result_shape,
                      xla::Shape::FromProto(pshape->result()));
  instance.result_layout = &result_shape;
  absl::StatusOr<std::vector<std::unique_ptr<xla::AotCompilationResult>>>
      aot_or = client->CompileAheadOfTime({instance}, aot_opts);
  if (!aot_or.ok()) {
    return errors::Unknown("XLA compilation failed: ",
                           aot_or.status().message());
  }
  compile_result->set_aot(
      xla::unique_ptr_down_cast<xla::cpu::CpuAotCompilationResult>(
          std::move(aot_or.value().back())));
  compile_result->entry_point = aot_opts.entry_point_name();
  compile_result->pointer_size =
      xla::CompileOnlyClient::PointerSizeForTriple(aot_opts.triple());
  return absl::OkStatus();
}

// Renames the computation proto to ensure unique symbol names to avoid linking
// errors when linking multiple tf_library targets.
absl::Status ConfigureKernelNamingConvention(
    xla::cpu::CpuAotCompilationOptions& aot_opts,
    xla::XlaComputation& computation, const std::string& cpp_class) {
  aot_opts.mutable_debug_options()
      ->set_xla_cpu_generate_unique_c_style_kernel_entry_points(true);

  TF_ASSIGN_OR_RETURN(std::string class_name_as_valid_c_name,
                      xla::cpu::ConvertToCName(cpp_class));
  // Rename proto to ensure unique symbol names.
  *computation.mutable_proto()->mutable_name() =
      absl::StrCat(computation.proto().name(), "_", class_name_as_valid_c_name);

  return absl::OkStatus();
}

}  // namespace

absl::Status CompileGraph(GraphDef graph_def, const tf2xla::Config& config,
                          const MainFlags& flags,
                          CompileResult* compile_result) {
  // Converts the graph into an XLA computation, and compiles the
  // computation.
  // TODO(toddw): Should we let the user pick the XLA cpu vs. gpu client?
  se::Platform* cpu_platform =
      se::PlatformManager::PlatformWithName("Host").value();
  xla::CompileOnlyClient* client =
      xla::ClientLibrary::GetOrCreateCompileOnlyClient(cpu_platform).value();
  xla::XlaComputation computation;

  bool use_mlir_bridge = false;
  if (!flags.mlir_components.empty() && flags.mlir_components != "None") {
    for (auto component : absl::StrSplit(flags.mlir_components, ',')) {
      if (component == "Bridge") {
        use_mlir_bridge = true;
      } else {
        return errors::Unknown("Unknown mlir_component ", component);
      }
    }
  }
  if (use_mlir_bridge) {
    TF_RETURN_IF_ERROR(ConvertGraphDefToXlaViaMlir(
        graph_def, config, &computation, flags.debug_info,
        flags.debug_info_path_begin_marker));
  } else {
    TF_RETURN_IF_ERROR(ConvertGraphDefToXla(std::move(graph_def), config,
                                            client, &computation));
  }

  if (flags.experimental_quantize && *quantize_xla) {
    TF_RETURN_IF_ERROR((*quantize_xla)(config, &computation));
  }

  if (!flags.out_session_module.empty()) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::HloSnapshot> module,
                        computation.Snapshot());
    // Serialize the HloSnapshot deterministically so that all the outputs of a
    // tf_library genrule are deterministic.
    const size_t size = module->ByteSizeLong();
    auto serialized = std::make_unique<char[]>(size);
    TF_RET_CHECK(
        SerializeToBufferDeterministic(*module, serialized.get(), size));
    TF_RETURN_IF_ERROR(
        WriteStringToFile(Env::Default(), flags.out_session_module,
                          absl::string_view(serialized.get(), size)));
  }
  xla::cpu::CpuAotCompilationOptions aot_opts(
      flags.target_triple, flags.target_cpu, flags.target_features,
      flags.entry_point,
      xla::cpu::CpuAotCompilationOptions::RelocationModel::BigPic,
      /*compile_copy_as_llvm_kernel=*/true);

  if (flags.sanitize_dataflow) {
    aot_opts.set_sanitize_dataflow(flags.sanitize_dataflow);
    aot_opts.set_sanitize_abilists_dataflow(absl::StrSplit(
        flags.sanitize_abilists_dataflow, ',', absl::SkipEmpty()));
  }

  TF_RETURN_IF_ERROR(
      ConfigureKernelNamingConvention(aot_opts, computation, flags.cpp_class));

  return CompileXla(client, computation, aot_opts, compile_result);
}

static absl::Status ReadProtoFile(const string& fname,
                                  protobuf::Message* proto) {
  if (absl::EndsWith(fname, ".pbtxt")) {
    return ReadTextProto(Env::Default(), fname, proto);
  } else {
    return ReadBinaryProto(Env::Default(), fname, proto);
  }
}

static absl::once_flag targets_init;

static void InitializeTargets() {
  // Initialize all LLVM targets so we can cross compile.
#if TF_LLVM_AARCH32_AVAILABLE
  LLVMInitializeARMTarget();
  LLVMInitializeARMTargetInfo();
  LLVMInitializeARMTargetMC();
  LLVMInitializeARMAsmParser();
  LLVMInitializeARMAsmPrinter();
#endif
#if TF_LLVM_AARCH64_AVAILABLE
  LLVMInitializeAArch64Target();
  LLVMInitializeAArch64TargetInfo();
  LLVMInitializeAArch64TargetMC();
  LLVMInitializeAArch64AsmParser();
  LLVMInitializeAArch64AsmPrinter();
#endif
#if TF_LLVM_HEXAGON_AVAILABLE
  LLVMInitializeHexagonTarget();
  LLVMInitializeHexagonTargetInfo();
  LLVMInitializeHexagonTargetMC();
  LLVMInitializeHexagonAsmParser();
  LLVMInitializeHexagonAsmPrinter();
#endif
#if TF_LLVM_POWERPC_AVAILABLE
  LLVMInitializePowerPCTarget();
  LLVMInitializePowerPCTargetInfo();
  LLVMInitializePowerPCTargetMC();
  LLVMInitializePowerPCAsmParser();
  LLVMInitializePowerPCAsmPrinter();
#endif
#if TF_LLVM_S390X_AVAILABLE
  LLVMInitializeSystemZTarget();
  LLVMInitializeSystemZTargetInfo();
  LLVMInitializeSystemZTargetMC();
  LLVMInitializeSystemZAsmParser();
  LLVMInitializeSystemZAsmPrinter();
#endif
#if TF_LLVM_X86_AVAILABLE
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86AsmParser();
  LLVMInitializeX86AsmPrinter();
#endif
}

// Replaces {{tag.type tag.name}} in the error message with tag_name.
// TODO(bixia): We currently only handlge tag.type == "node".
//
// In the error message, a graph node is represented as {{tag.type, tag.name}},
// to allow a Python debugger to insert source information about the graph node.
// For example, a Python add expression may be represented as
// {{node, x_y_sum}} = Add(x, y) in the error message. See routine interpolate
// in tensorflow/python/framework/error_interpolation.py for more detail.
static std::string InterpolateErrorMessage(std::string message) {
  // See _NAME_REGEX in tensorflow/python/framework/error_interpolation.py
  // Change "prefix {{node tag.name}} suffix" to "prefix tag.name suffix".
  static LazyRE2 pattern{"(.*){{node (.*)}}(.*)"};
  RE2::GlobalReplace(&message, *pattern, "\\1\\2\\3");

  return message;
}

absl::Status Main(const MainFlags& flags) {
  absl::call_once(targets_init, &InitializeTargets);

  // Process config.
  tf2xla::Config config;
  if (flags.config.empty()) {
    return errors::InvalidArgument("Must specify --config");
  }
  TF_RETURN_IF_ERROR(ReadProtoFile(flags.config, &config));
  TF_RETURN_IF_ERROR(ValidateConfig(config));
  if (flags.dump_fetch_nodes) {
    std::set<string> nodes;
    for (const tf2xla::Fetch& fetch : config.fetch()) {
      nodes.insert(fetch.id().node_name());
    }
    std::cout << absl::StrJoin(nodes, ",");
    return absl::OkStatus();
  }

  // Read and initialize the graph.
  if (flags.graph.empty()) {
    return errors::InvalidArgument("Must specify --graph");
  }
  GraphDef graph_def;
  TF_RETURN_IF_ERROR(ReadProtoFile(flags.graph, &graph_def));
  CompileResult compile_result;

  absl::Status status =
      CompileGraph(std::move(graph_def), config, flags, &compile_result);
  if (!status.ok()) {
    return errors::CreateWithUpdatedMessage(
        status, InterpolateErrorMessage(std::string(status.message())));
  }

  // Write output files.
  Env* env = Env::Default();

  if (compile_result.is_aot_thunks()) {
    const auto obj_files = compile_result.get_aot_thunks().value()->obj_files();
    DCHECK_EQ(obj_files.size(), 1);
    const absl::string_view obj_file = obj_files[0];
    TF_RETURN_IF_ERROR(
        WriteStringToFile(env, flags.out_function_object, obj_file));
  } else {
    const std::vector<char>& obj_file =
        compile_result.get_aot_legacy().value()->object_file_data();
    TF_RETURN_IF_ERROR(
        WriteStringToFile(env, flags.out_function_object,
                          absl::string_view(obj_file.data(), obj_file.size())));
  }

  CodegenOpts codegen_opts;
  codegen_opts.gen_name_to_index = flags.gen_name_to_index;
  codegen_opts.gen_program_shape = flags.gen_program_shape;
  codegen_opts.target_triple = flags.target_triple;
  codegen_opts.use_xla_nanort_runtime = flags.use_xla_nanort_runtime;
  // Set the XLA Runtime bit if this is an HloLowering.
  if (!flags.mlir_components.empty() && flags.mlir_components != "None") {
    for (auto component : absl::StrSplit(flags.mlir_components, ',')) {
      if (component == "HloLowering") {
        codegen_opts.use_xla_runtime = true;
      }
    }
  }

  if (flags.cpp_class.empty()) {
    return errors::InvalidArgument("Must specify --cpp_class");
  }
  codegen_opts.gen_hlo_profile_printer_data =
      xla::GetDebugOptionsFromFlags().xla_hlo_profile();
  TF_RETURN_IF_ERROR(ParseCppClass(flags.cpp_class, &codegen_opts.class_name,
                                   &codegen_opts.namespaces));

  EmbeddedConstantBuffers embedded_constant_buffers;
  if (compile_result.is_aot_thunks()) {
    if (flags.out_constant_buffers_object.empty()) {
      return absl::InvalidArgumentError(
          "Must specify --out_constant_buffers_object when using AOT thunks");
    }
    TF_ASSIGN_OR_RETURN(
        embedded_constant_buffers,
        GenerateConstantBuffersData(codegen_opts, compile_result));
    TF_RETURN_IF_ERROR(
        WriteStringToFile(env, flags.out_constant_buffers_object,
                          embedded_constant_buffers.object_file_data));
  }

  MetadataResult metadata_result;
  TF_RETURN_IF_ERROR(
      GenerateMetadata(codegen_opts, compile_result, &metadata_result));
  TF_RETURN_IF_ERROR(WriteStringToFile(env, flags.out_metadata_object,
                                       metadata_result.object_file_data));
  string header;
  TF_RETURN_IF_ERROR(GenerateHeader(codegen_opts, config, compile_result,
                                    metadata_result, embedded_constant_buffers,
                                    &header));
  TF_RETURN_IF_ERROR(WriteStringToFile(env, flags.out_header, header));
  return absl::OkStatus();
}

}  // namespace tfcompile
}  // namespace tensorflow
