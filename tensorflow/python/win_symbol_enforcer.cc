/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/TypeID.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "llvm/ADT/ilist.h"
#if __has_include("third_party/protobuf/message_lite.h")
#include "third_party/protobuf/message_lite.h"
#else
#include "google/protobuf/message_lite.h"
#endif
#include "absl/strings/cord.h"
#if __has_include("third_party/protobuf/util/message_differencer.h")
#include "third_party/protobuf/util/field_comparator.h"
#include "third_party/protobuf/util/message_differencer.h"
#else
#include "google/protobuf/util/field_comparator.h"
#include "google/protobuf/util/message_differencer.h"
#endif
#include "absl/log/absl_log.h"
#include "llvm/FileCheck/FileCheck.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "pybind11_abseil/import_status_module.h"  // from @pybind11_abseil
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/experimental/gradients/math_grad.h"
#include "tensorflow/c/experimental/gradients/nn_grad.h"
#include "tensorflow/c/experimental/gradients/tape/tape_context.h"
#include "tensorflow/c/safe_ptr.h"
#include "tensorflow/c/tf_buffer.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/cc/saved_model/metrics.h"
#include "tensorflow/compiler/aot/compile.h"
#include "tensorflow/compiler/jit/get_compiler_ir.h"
#include "tensorflow/compiler/mlir/lite/python/converter_python_api.h"
#include "tensorflow/compiler/mlir/python/mlir.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/quantize_model.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow_to_stablehlo/python/pywrap_tensorflow_to_stablehlo_lib.h"
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/py_utils.h"
#include "tensorflow/compiler/tf2xla/tf2xla_opset.h"
#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/status_macros.h"
#include "xla/tsl/c/tsl_status.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/stack_frame.h"
#include "xla/tsl/profiler/rpc/profiler_server.h"
#include "xla/tsl/util/determinism.h"
#include "xla/tsl/util/device_name_utils.h"
#include "tensorflow/core/config/flag_defs.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/py_utils.h"
#include "tensorflow/core/data/service/server_lib.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/framework/resource_base.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_debug_info_builder.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/costs/graph_memory.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/graph_analyzer/sig_node.h"
#include "tensorflow/core/grappler/graph_analyzer/subgraph.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/platform/enable_tf2_utils.h"
#include "tensorflow/core/profiler/internal/print_model_analysis.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"
#include "tensorflow/core/protobuf/fingerprint.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/tpu/kernels/sparse_core_layout.pb.h"
#include "tensorflow/core/tpu/kernels/sparse_core_ops_utils.h"
#include "tensorflow/core/util/debug_events_writer.h"
#include "tensorflow/core/util/events_writer.h"
#include "tensorflow/core/util/managed_stack_trace.h"
#include "tensorflow/core/util/stat_summarizer.h"
#include "tensorflow/dtensor/cc/dtensor_device.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/proto/layout.pb.h"
#include "tensorflow/lite/toco/python/toco_python_api.h"
#include "tensorflow/python/eager/pywrap_tensor.h"
#include "tensorflow/python/eager/pywrap_tensor_conversion.h"
#include "tensorflow/python/eager/pywrap_tfe.h"
#include "tensorflow/python/framework/op_def_util.h"
#include "tensorflow/python/framework/python_api_dispatcher.h"
#include "tensorflow/python/framework/python_api_info.h"
#include "tensorflow/python/framework/python_api_parameter_converter.h"
#include "tensorflow/python/lib/core/py_exception_registry.h"
#include "tensorflow/python/lib/core/py_func.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"
#include "tensorflow/python/profiler/internal/profiler_pywrap_impl.h"
#include "tensorflow/python/pywrap_library_dependency_enforcer.h"
#include "tensorflow/python/util/function_parameter_canonicalizer.h"
#include "tensorflow/python/util/kernel_registry.h"
#include "tensorflow/tools/graph_transforms/transform_graph.h"

// Forward declarations for proto default instances
namespace tensorflow {
struct AttrValue_ListValueDefaultTypeInternal;
extern AttrValue_ListValueDefaultTypeInternal
    _AttrValue_ListValue_default_instance_;

struct FunctionDefLibraryDefaultTypeInternal;
extern FunctionDefLibraryDefaultTypeInternal
    _FunctionDefLibrary_default_instance_;

struct FunctionDefDefaultTypeInternal;
extern FunctionDefDefaultTypeInternal _FunctionDef_default_instance_;

struct GraphDefDefaultTypeInternal;
extern GraphDefDefaultTypeInternal _GraphDef_default_instance_;

struct MetaGraphDefDefaultTypeInternal;
extern MetaGraphDefDefaultTypeInternal _MetaGraphDef_default_instance_;
}  // namespace tensorflow

// Forward declarations for saved_model metrics (to bypass visibility issues)
namespace tensorflow {
namespace metrics {
absl::StatusOr<std::string> MakeSavedModelPathAndSingleprint(
    std::string path, std::string singleprint);
}  // namespace metrics
}  // namespace tensorflow

// Forward declare TFR dialect since its target visibility is restricted
namespace mlir {
class MLIRContext;
namespace TFR {
class TFRDialect {
 public:
  explicit TFRDialect(MLIRContext* context);
};
}  // namespace TFR
}  // namespace mlir

#include "tensorflow/python/framework/py_context_manager.h"

// Forward declare tf_session_helper functions without including
// tf_session_helper.h
namespace tensorflow {
struct TF_ImportGraphDefResults;

std::vector<std::string>
TF_ImportGraphDefResultsMissingUnusedInputMappings_wrapper(
    TF_ImportGraphDefResults* results);

TF_Function* TF_GraphToFunction_wrapper(
    const TF_Graph* graph, const char* name, bool append_to_graph,
    const std::vector<TF_Operation*>* op_list,
    const std::vector<TF_Output>& inputs, const std::vector<TF_Output>& outputs,
    const std::vector<std::string>& output_names,
    const std::vector<TF_Operation*>* control_outputs,
    const std::vector<TF_Output>& control_output_datas,
    bool add_control_dependencies, const TF_FunctionOptions* opts,
    const char* description, TSL_Status* status);

void TF_GraphSetOutputHandleShapesAndTypes_wrapper(
    TF_Graph* graph, TF_Output output,
    const std::vector<std::vector<int64_t>>& shapes,
    const std::vector<int>& ranks, const std::vector<TF_DataType>& types,
    TSL_Status* status);

void TF_SessionRun_wrapper(TF_Session* session, const TF_Buffer* run_options,
                           const std::vector<TF_Output>& inputs,
                           const std::vector<PyObject*>& input_ndarrays,
                           const std::vector<TF_Output>& outputs,
                           const std::vector<TF_Operation*>& targets,
                           TF_Buffer* run_metadata, TF_Status* status,
                           std::vector<PyObject*>* py_outputs);

void TF_SessionPRun_wrapper(TF_Session* session, const char* handle,
                            const std::vector<TF_Output>& inputs,
                            const std::vector<PyObject*>& input_ndarrays,
                            const std::vector<TF_Output>& outputs,
                            TF_Status* status,
                            std::vector<PyObject*>* py_outputs);

void TF_SessionPRunSetup_wrapper(TF_Session* session,
                                 const std::vector<TF_Output>& inputs,
                                 const std::vector<TF_Output>& outputs,
                                 const std::vector<TF_Operation*>& targets,
                                 const char** out_handle, TF_Status* status);
}  // namespace tensorflow
#include "pybind11_abseil/import_status_module.h"  // from @pybind11_abseil
#include "pybind11_protobuf/native_proto_caster.h"  // from @pybind11_protobuf
#include "tensorflow/compiler/mlir/quantization/stablehlo/python/pywrap_quantization_lib.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/service_config.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace python {
void protobuf_inline_symbols_enforcer() {
  tensorflow::NamedDevice named_device;
  named_device.mutable_properties();
  (void)named_device.properties();

  tensorflow::NamedDevice named_device_move(std::move(named_device));
  named_device_move.mutable_properties();

  tensorflow::quantization::ExportedModel exported_model;
  (void)exported_model.function_aliases();

  tensorflow::profiler::XSpace x_space;
  (void)x_space.mutable_hostnames();
  x_space.mutable_hostnames(0);

  tensorflow::dtensor::LayoutProto layout_proto;
  layout_proto.GetDescriptor();
  layout_proto.GetReflection();
  layout_proto.default_instance();

  tensorflow::dtensor::MeshProto mesh_proto;
  mesh_proto.GetDescriptor();
  mesh_proto.GetReflection();
  mesh_proto.default_instance();

  tensorflow::FunctionDef function_def;
  function_def.descriptor();
  function_def.GetDescriptor();
  function_def.GetReflection();
  function_def.default_instance();

  tensorflow::FunctionDefLibrary function_def_library;
  function_def_library.descriptor();

  tensorflow::GraphDef graph_def;
  graph_def.descriptor();
  graph_def.GetDescriptor();
  graph_def.GetReflection();
  graph_def.default_instance();

  tensorflow::MetaGraphDef meta_graph_def;
  meta_graph_def.GetDescriptor();
  meta_graph_def.GetReflection();
  meta_graph_def.default_instance();

  tensorflow::AttrValue attr_value;
  attr_value.default_instance();
  tensorflow::AttrValue_ListValue list_value;
  list_value.add_b(false);

  OpPerformanceList performance_list;

  tensorflow::ConfigProto config_proto;
  config_proto.default_instance();
  config_proto.CopyFrom(config_proto);

  tensorflow::SessionOptions session_options;

  tensorflow::data::experimental::DispatcherConfig dispatcher_config;
  dispatcher_config.default_instance();
  dispatcher_config.CopyFrom(dispatcher_config);

  tensorflow::data::experimental::WorkerConfig worker_config;
  worker_config.default_instance();
  worker_config.CopyFrom(worker_config);

  tensorflow::data::DataServiceMetadata data_service_metadata;
  data_service_metadata.CopyFrom(data_service_metadata);
  tensorflow::quantization::QuantizationOptions quantization_options;
  tensorflow::CoordinatedTask coordinated_task;
  tensorflow::DeviceAttributes device_attributes;

  // Force link/instantiation of TFLite converter and Abseil FormatPack APIs
  (void)&tflite::RetrieveCollectedErrors;
  (void)&tflite::Convert;
  (void)&tflite::MlirQuantizeModel;
  (void)&tflite::MlirSparsifyModel;
  (void)&tflite::RegisterCustomOpdefs;
  (void)&tflite::FlatBufferFileToMlir;
  (void)&toco::TocoConvert;

  // Force link/instantiation of tfcompile::Main
  (void)&tensorflow::tfcompile::Main;

  // Force link/instantiation of xla::HloComputation and xla::HloInstruction
  // symbols
  typedef std::string (xla::HloInstruction::*ToStringPtr)() const;
  (void)(ToStringPtr)&xla::HloInstruction::ToString;

  // Force template instantiations of Accept, Visit, and AcceptOrdered
  typedef absl::Status (xla::HloInstruction::*AcceptPtr)(
      xla::DfsHloVisitorBase<xla::HloInstruction*>*, bool, bool, bool);
  (void)(AcceptPtr)&xla::HloInstruction::Accept<xla::HloInstruction*>;

  typedef absl::Status (xla::HloInstruction::*AcceptConstPtr)(
      xla::DfsHloVisitorBase<const xla::HloInstruction*>*, bool, bool, bool);
  (void)(AcceptConstPtr)&xla::HloInstruction::Accept<
      const xla::HloInstruction*>;

  typedef absl::Status (xla::HloInstruction::*VisitPtr)(
      xla::DfsHloVisitorBase<xla::HloInstruction*>*);
  (void)(VisitPtr)&xla::HloInstruction::Visit<xla::HloInstruction*>;

  typedef absl::Status (xla::HloInstruction::*VisitConstPtr)(
      xla::DfsHloVisitorBase<const xla::HloInstruction*>*);
  (void)(VisitConstPtr)&xla::HloInstruction::Visit<const xla::HloInstruction*>;

  typedef absl::Status (xla::HloComputation::*AcceptOrderedPtr)(
      xla::DfsHloVisitorBase<xla::HloInstruction*>*,
      absl::Span<xla::HloInstruction* const>) const;
  (void)(AcceptOrderedPtr)&xla::HloComputation::AcceptOrdered<
      xla::HloInstruction*>;

  typedef absl::Status (xla::HloComputation::*AcceptOrderedConstPtr)(
      xla::DfsHloVisitorBase<const xla::HloInstruction*>*,
      absl::Span<const xla::HloInstruction* const>) const;
  (void)(AcceptOrderedConstPtr)&xla::HloComputation::AcceptOrdered<
      const xla::HloInstruction*>;

  typedef void (xla::DfsHloVisitorBase<xla::HloInstruction*>::*SetVisitedPtr)(
      const xla::HloInstruction&);
  (void)(SetVisitedPtr)&xla::DfsHloVisitorBase<
      xla::HloInstruction*>::SetVisited;

  typedef void (
      xla::DfsHloVisitorBase<const xla::HloInstruction*>::*SetVisitedConstPtr)(
      const xla::HloInstruction&);
  (void)(SetVisitedConstPtr)&xla::DfsHloVisitorBase<
      const xla::HloInstruction*>::SetVisited;

  // Force link/instantiation of status_macros::MakeErrorStream
  {
    xla::status_macros::MakeErrorStream error_stream(
        "", 0, absl::StatusCode::kInternal);
    (void)error_stream.add_ret_check_failure("");
  }

  // Force link/instantiation of absl::str_format_internal::FormatPack
  (void)absl::StrFormat("%d", 0);
  (void)&tensorflow::metrics::TestCounter;

  // Force link/instantiation of tf_session_helper wrapper symbols
  (void)&tensorflow::TF_ImportGraphDefResultsMissingUnusedInputMappings_wrapper;
  (void)&tensorflow::TF_GraphToFunction_wrapper;
  (void)&tensorflow::TF_GraphSetOutputHandleShapesAndTypes_wrapper;
  (void)&tensorflow::TF_SessionRun_wrapper;
  (void)&tensorflow::TF_SessionPRun_wrapper;
  (void)&tensorflow::TF_SessionPRunSetup_wrapper;

  // Force link/instantiation of pybind11_protobuf wrapper symbols
  (void)&pybind11_protobuf::PyProtoGetCppMessagePointer;
  (void)&pybind11_protobuf::PyProtoHasMatchingFullName;
  (void)&pybind11_protobuf::PyProtoSerializePartialToString;
  (void)&pybind11_protobuf::PyBytesAsStringView;
  (void)&pybind11_protobuf::GenericProtoCast;

  // Force link/instantiation of CatPieces and AppendPieces (Abseil)
  std::string dummy_str;
  absl::StrAppend(&dummy_str, "a", "b");
  (void)absl::StrCat("a", "b");

  // Force link/instantiation of DeviceNameUtils
  (void)(bool (*)(absl::string_view,
                  absl::string_view))&tsl::DeviceNameUtils::IsSameAddressSpace;

  // Force link/instantiation of FunctionRecord constructor
  tensorflow::FunctionDef dummy_fd;
  tensorflow::StackTracesMap dummy_map;
  auto* dummy_record = new tensorflow::FunctionRecord(
      std::move(dummy_fd), std::move(dummy_map), false);
  dummy_record->Unref();

  // Force link/instantiation of stablehlo quantization and status module
  (void)&pybind11::google::ImportStatusModule;
  (void)&stablehlo::quantization::pywrap::PywrapQuantizeStaticRangePtq;
  (void)&stablehlo::quantization::pywrap::PywrapQuantizeWeightOnlyPtq;
  (void)&stablehlo::quantization::pywrap::PywrapPopulateDefaults;
  (void)&stablehlo::quantization::pywrap::PywrapExpandPresets;

  // Force link/instantiation of GetRegisteredXlaOpsForDevice
  (void)&tensorflow::GetRegisteredXlaOpsForDevice;

  // Force link/instantiation of TensorRT helper symbols
  (void)&tensorflow::tensorrt::GetLinkedTensorRTVersion;
  (void)&tensorflow::tensorrt::GetLoadedTensorRTVersion;
  (void)&tensorflow::tensorrt::IsGoogleTensorRTEnabled;
  (void)&tensorflow::tensorrt::GetRegisteredOpConverters;

  // Force link/instantiation of PyContextManager
  (void)&tensorflow::PyContextManager::Enter;

  // Force link/instantiation of GetTableStacks
  (void)&tensorflow::GetTableStacks;

  // Force link/instantiation of pywrap_library_dependency_symbol
  (void)&tensorflow::python::pywrap_library_dependency_symbol;

  // Force link/instantiation of DataType_internal_data_
  (void)tensorflow::DataType_internal_data_[0];

  // Force link/instantiation of tensorflow quantization models
  (void)&tensorflow::quantization::QuantizeQatModel;
  (void)&tensorflow::quantization::QuantizeDynamicRangePtq;
  (void)&tensorflow::quantization::QuantizeStaticRangePtq;

  // Force link tf.data service server & client classes
  {
    (void)&tensorflow::data::NewDispatchServer;
    (void)&tensorflow::data::NewWorkerServer;

    // GrpcDataServerBase methods
    typedef absl::Status (
        tensorflow::data::GrpcDataServerBase::*ServerStartPtr)();
    (void)(ServerStartPtr)&tensorflow::data::GrpcDataServerBase::Start;
    (void)&tensorflow::data::GrpcDataServerBase::Stop;
    (void)&tensorflow::data::GrpcDataServerBase::Join;
    (void)&tensorflow::data::GrpcDataServerBase::BoundPort;

    // DispatchGrpcDataServer methods
    (void)&tensorflow::data::DispatchGrpcDataServer::NumWorkers;
    (void)&tensorflow::data::DispatchGrpcDataServer::SnapshotStreams;

    // WorkerGrpcDataServer methods
    (void)&tensorflow::data::WorkerGrpcDataServer::NumTasks;
    (void)&tensorflow::data::WorkerGrpcDataServer::SnapshotTaskProgresses;

    // DataServiceDispatcherClient virtual or key methods
    typedef absl::Status (
        tensorflow::data::DataServiceDispatcherClient::*GetSnapshotSplitPtr)(
        const std::string&, const std::string&, int64_t, const std::string&,
        const std::string&, tensorflow::Tensor*, int64_t*, bool*);
    (void)(GetSnapshotSplitPtr)&tensorflow::data::DataServiceDispatcherClient::
        GetSnapshotSplit;

    (void)&tensorflow::data::DataServiceDispatcherClient::Initialize;

    // Force link grpc_util::Retry
    typedef absl::Status (*RetryFn)(const std::function<absl::Status()>&,
                                    const std::string&, int64_t);
    (void)(RetryFn)&tensorflow::data::grpc_util::Retry;
  }

  // Force link tf.data snapshot_utils symbols
  (void)&tensorflow::data::SnapshotDoneFilePath;
  (void)&tensorflow::data::SnapshotErrorFilePath;
  (void)&tensorflow::data::SnapshotMetadataFilePath;
  (void)&tensorflow::data::CommittedChunksDirectory;

  // Force link/instantiation of ProfilerSessionWrapper
  (void)&tensorflow::profiler::pywrap::ProfilerSessionWrapper::Start;
  (void)&tensorflow::profiler::pywrap::ProfilerSessionWrapper::Stop;
  (void)&tensorflow::profiler::pywrap::ProfilerSessionWrapper::
      ExportToTensorBoard;

  // Force link/instantiation of tensorflow_to_stablehlo wrapper symbols
  (void)&mlir::tensorflow_to_stablehlo::pywrap::PywrapSavedModelToStablehlo;
  (void)&mlir::tensorflow_to_stablehlo::pywrap::PywrapTfModuleToStablehlo;

  // Force link/instantiation of Grappler Clusters & costs
  using MemoryTypesForNodeType = absl::Status (*)(
      const OpRegistryInterface*, const DeviceType&, const NodeDef&,
      absl::InlinedVector<MemoryType, 4>*, absl::InlinedVector<MemoryType, 4>*);
  (void)(MemoryTypesForNodeType)&tensorflow::MemoryTypesForNode;
  (void)&tensorflow::grappler::GraphMemory::InferStatically;
  auto* dummy_vc = new tensorflow::grappler::VirtualCluster({});
  delete dummy_vc;

  // Force link/instantiation of DTensor Mesh & Layout
  (void)&tensorflow::dtensor::Mesh::CreateMesh;
  (void)&tensorflow::dtensor::Mesh::MeshDimNames;
  (void)&tensorflow::dtensor::Layout::sharding_spec_strs;
  using GetLayoutType = absl::StatusOr<tensorflow::dtensor::Layout> (*)(
      const std::vector<std::string>&, const tensorflow::dtensor::Mesh&);
  (void)(GetLayoutType)&tensorflow::dtensor::Layout::GetLayout;
  (void)&tensorflow::dtensor::Layout::GlobalShapeFromLocalShape;
  (void)&tensorflow::dtensor::SetTPUCoreIDs;
  (void)&tensorflow::dtensor::TPUCoreIDsToLocations;
  (void)&tensorflow::dtensor::TPUCoreLocationsToIDs;
  (void)&tensorflow::dtensor::GetStats;
  (void)&tensorflow::dtensor::SetIteratorElementLayouts;

  // Force link/instantiation of Python API Dispatcher
  auto* dummy_checker = new tensorflow::py_dispatch::PySignatureChecker({});
  (void)dummy_checker;
  auto* dummy_dispatcher =
      new tensorflow::py_dispatch::PythonAPIDispatcher("", {}, {});
  using DispatchType =
      std::unique_ptr<PyObject, tensorflow::detail::PyDecrefDeleter> (
          tensorflow::py_dispatch::PythonAPIDispatcher::*)(PyObject*,
                                                           PyObject*);
  (void)(DispatchType)&tensorflow::py_dispatch::PythonAPIDispatcher::Dispatch;
  (void)dummy_dispatcher;

  // Force link/instantiation of Stack Trace Builder & FrozenStackTrace
  (void)&tensorflow::GraphDebugInfoBuilder::AppendGraphDebugInfoStr;
  (void)&tensorflow::GraphDebugInfoBuilder::AccumulateStackTrace;
  (void)&tensorflow::LoadTracesFromDebugInfoStr;
  auto* dummy_f_trace = new tensorflow::FrozenStackTrace(
      absl::Span<const tsl::StackFrame>{}, absl::Span<const tsl::StackFrame>{});
  delete dummy_f_trace;

  // Force link/instantiation of Graph Transforms
  (void)&tensorflow::graph_transforms::ParseTransformParameters;
  (void)&tensorflow::graph_transforms::TransformGraph;

  // Force link/instantiation of Proto default instances
  (void)&tensorflow::_AttrValue_ListValue_default_instance_;
  (void)&tensorflow::_FunctionDefLibrary_default_instance_;
  (void)&tensorflow::_FunctionDef_default_instance_;
  (void)&tensorflow::_GraphDef_default_instance_;
  (void)&tensorflow::_MetaGraphDef_default_instance_;

  // Force link/instantiation of new metrics symbols
  using MakeSavedModelPathAndSingleprintType =
      absl::StatusOr<std::string> (*)(std::string, std::string);
  (void)(MakeSavedModelPathAndSingleprintType)&tensorflow::metrics::
      MakeSavedModelPathAndSingleprint;

  // Force link/instantiation of Google empty string init
  (void)&google::protobuf::internal::GetEmptyStringAlreadyInited;

  // Force link/instantiation of PyContextManager
  auto* dummy_manager = new tensorflow::PyContextManager();
  delete dummy_manager;

  // Force link/instantiation of Env methods
  using GetChildrenType =
      absl::Status (tsl::Env::*)(const std::string&, std::vector<std::string>*);
  (void)(GetChildrenType)&tsl::Env::GetChildren;
#ifdef CopyFile
#undef CopyFile
#endif
  using CopyFileType =
      absl::Status (tsl::Env::*)(const std::string&, const std::string&);
  (void)(CopyFileType)&tsl::Env::CopyFile;
  using RenameFileType =
      absl::Status (tsl::Env::*)(const std::string&, const std::string&);
  (void)(RenameFileType)&tsl::Env::RenameFile;

  // Force link/instantiation of ResourceHandle
  using MakeRefCountingHandleType = tensorflow::ResourceHandle (*)(
      tensorflow::ResourceBase*, const std::string&,
      const tensorflow::TypeIndex&,
      const std::vector<tensorflow::DtypeAndPartialTensorShape>&,
      const std::optional<tensorflow::ManagedStackTrace>&);
  (void)(MakeRefCountingHandleType)&tensorflow::ResourceHandle::
      MakeRefCountingHandle;

  // Force link/instantiation of tfprof profiler
  (void)&tensorflow::tfprof::NewProfiler;

  // Force link/instantiation of data service default protocol
  (void)&tensorflow::data::DefaultProtocol;

  // Force link/instantiation of device factory symbols
  (void)&tensorflow::DeviceFactory::AddDevices;
  (void)&tensorflow::DeviceFactory::ListAllPhysicalDevices;
  (void)&tensorflow::DeviceFactory::ListPluggablePhysicalDevices;
  (void)&tensorflow::DeviceFactory::GetAnyDeviceDetails;

  // Force link/instantiation of TFE Eager runtime
  (void)&::TFE_Py_ExecuteCancelable;
  (void)&::TFE_Py_TapeWatch;
  (void)&::TFE_Py_TapeWatchVariable;
  (void)&::TFE_Py_ForwardAccumulatorWatch;

  // Force link/instantiation of Tape and Gradients symbols
  tensorflow::gradients::GradientRegistry dummy_registry;
  auto* dummy_tape_context =
      new tensorflow::gradients::TapeContext(nullptr, nullptr, dummy_registry);
  delete dummy_tape_context;

  auto* dummy_tape_tensor = new tensorflow::gradients::TapeTensor(nullptr);
  delete dummy_tape_tensor;

  (void)&tensorflow::gradients::AddRegisterer;
  (void)&tensorflow::gradients::ExpRegisterer;
  (void)&tensorflow::gradients::MatMulRegisterer;
  (void)&tensorflow::gradients::NegRegisterer;
  (void)&tensorflow::gradients::SubRegisterer;
  (void)&tensorflow::gradients::MulRegisterer;
  (void)&tensorflow::gradients::Log1pRegisterer;
  (void)&tensorflow::gradients::DivNoNanRegisterer;
  (void)&tensorflow::gradients::ReluRegisterer;
  (void)&tensorflow::gradients::SparseSoftmaxCrossEntropyWithLogitsRegisterer;
  (void)&tensorflow::MakeEagerContextThreadLocalData;
  (void)&tensorflow::TFE_TensorHandleCache::Get;
  using GetCompilerIrType1 = absl::StatusOr<std::string> (*)(
      tensorflow::IrExportStage, tensorflow::ProcessFunctionLibraryRuntime*,
      absl::string_view, tensorflow::Device*, tensorflow::EagerContext*,
      absl::Span<const tensorflow::ArgShapeAndDType>,
      absl::Span<const tensorflow::TensorHandle* const>,
      tensorflow::CompilerArgSource);
  (void)(GetCompilerIrType1)&tensorflow::GetCompilerIr;
  using GetCompilerIrType2 = absl::StatusOr<std::string> (*)(
      tensorflow::IrExportStage, tensorflow::ProcessFunctionLibraryRuntime*,
      absl::string_view, absl::string_view, tensorflow::EagerContext*,
      absl::Span<const tensorflow::ArgShapeAndDType>,
      absl::Span<const tensorflow::TensorHandle* const>,
      tensorflow::CompilerArgSource);
  (void)(GetCompilerIrType2)&tensorflow::GetCompilerIr;

  // Force link/instantiation of DebugEventsWriter
  (void)&tensorflow::tfdbg::DebugEventsWriter::GetDebugEventsWriter;

  // Force link/instantiation of snapshot files utilities
  (void)&tensorflow::data::SnapshotDoneFilePath;
  (void)&tensorflow::data::SnapshotErrorFilePath;
  (void)&tensorflow::data::SnapshotMetadataFilePath;
  (void)&tensorflow::data::CommittedChunksDirectory;

  // Force link/instantiation of CheckpointReader
  (void)&tensorflow::checkpoint::CheckpointReader::GetVariableToShapeMap;
  (void)&tensorflow::checkpoint::CheckpointReader::GetVariableToDataTypeMap;
  (void)&tensorflow::checkpoint::CheckpointReader::GetTensor;

  // Force link/instantiation of grappler builders and properties
  (void)&tensorflow::grappler::GrapplerItemFromMetaGraphDef;
  (void)&tensorflow::grappler::GraphProperties::GetOutputProperties;
  // Force link/instantiation of grappler graph analyzer symbols
  (void)&tensorflow::grappler::graph_analyzer::Signature::Compute;
  (void)&tensorflow::grappler::graph_analyzer::Signature::ToString;
  (void)(&tensorflow::grappler::graph_analyzer::Signature::operator==);
  (void)&tensorflow::grappler::graph_analyzer::Subgraph::Identity::Hash;
  (void)&tensorflow::grappler::graph_analyzer::Subgraph::Identity::operator<;
  (void)(&tensorflow::grappler::graph_analyzer::Subgraph::Identity::operator==);
  using NameRangesForNodeType =
      absl::Status (*)(const tensorflow::AttrSlice&, const tensorflow::OpDef&,
                       tensorflow::NameRangeMap*, tensorflow::NameRangeMap*);
  (void)(NameRangesForNodeType)&tensorflow::NameRangesForNode;

  // Force link/instantiation of MLIR python utilities
  using ImportGraphDefMlirType = std::string (*)(
      const std::string&, const std::string&, bool, absl::string_view,
      absl::string_view, absl::string_view, absl::string_view, TSL_Status*);
  (void)(ImportGraphDefMlirType)&tensorflow::ImportGraphDef;
  (void)&tensorflow::ExperimentalWriteBytecode;

  // Force link/instantiation of PythonAPIInfo
  (void)&tensorflow::PythonAPIInfo::InitializeFromParamSpecs;

  // Force link/instantiation of MLIR dialects TypeIDResolver template variables
  (void)
      mlir::detail::TypeIDResolver<mlir::arith::ArithDialect>::resolveTypeID();
  (void)mlir::detail::TypeIDResolver<mlir::scf::SCFDialect>::resolveTypeID();
  (void)mlir::detail::TypeIDResolver<mlir::func::FuncDialect>::resolveTypeID();
  (void)
      mlir::detail::TypeIDResolver<mlir::shape::ShapeDialect>::resolveTypeID();
  (void)mlir::detail::TypeIDResolver<mlir::ModuleOp>::resolveTypeID();
  (void)mlir::detail::TypeIDResolver<mlir::TFR::TFRDialect>::resolveTypeID();

  // Force link/instantiation of llvm::ilist templates via mlir::Block
  // operations
  mlir::Block dummy_block;
  auto& ops_list = dummy_block.getOperations();
  (void)ops_list.begin();
  (void)ops_list.end();

  // Force link/instantiation of DTensor functions and variables
  (void)&tensorflow::dtensor::SetTPUCoreIDs;
  (void)&tensorflow::dtensor::TPUCoreIDsToLocations;
  (void)&tensorflow::dtensor::TPUCoreLocationsToIDs;
  (void)&tensorflow::dtensor::GetStats;
  (void)&tensorflow::dtensor::SetIteratorElementLayouts;
  (void)&tensorflow::dtensor::MeshProto::default_instance;
  (void)&tensorflow::dtensor::LayoutProto::default_instance;
  using GlobalShapeType = std::vector<int64_t> (tensorflow::dtensor::Layout::*)(
      absl::Span<const int64_t>, const std::vector<std::vector<int64_t>>*)
      const;
  (void)(GlobalShapeType)&tensorflow::dtensor::Layout::
      GlobalShapeFromLocalShape;

  // Force link/instantiation of python API registry, parameter converter,
  // and Abseil Cord/Status APIs
  (void)(PyObject * (*)(TF_Code)) & tensorflow::PyExceptionRegistry::Lookup;
  (void)&tensorflow::CopyPythonAPITensorLists;
  (void)&tensorflow::ConvertPythonAPIParameters;
  absl::Cord dummy_cord("abc");
  (void)std::string(dummy_cord);
  absl::Status dummy_status(absl::StatusCode::kCancelled, "msg");
  dummy_status.ForEachPayload([](absl::string_view, const absl::Cord&) {});

  // Force link toco::TocoConvert
  (void)&toco::TocoConvert;

  // Force link Abseil Mutex & CondVar
  absl::Mutex dummy_mutex;
  dummy_mutex.Lock();
  dummy_mutex.Unlock();
  (void)(void (absl::CondVar::*)(absl::Mutex*))&absl::CondVar::Wait;
  (void)&absl::CondVar::Signal;

  // Force link Abseil Numbers
  char dummy_num_buf[32];
  (void)absl::numbers_internal::FastIntToBuffer(0, dummy_num_buf);
  (void)absl::numbers_internal::FastIntToBuffer(0ULL, dummy_num_buf);

  // Force link Env & CancellationManager
  (void)&tsl::Env::Default;
  auto* dummy_cm = new tsl::CancellationManager();
  dummy_cm->StartCancel();
  delete dummy_cm;
  using IsSameAddrSpaceType = bool (*)(absl::string_view, absl::string_view);
  (void)(IsSameAddrSpaceType)&tsl::DeviceNameUtils::IsSameAddressSpace;

  // Force link Tensor & Shape helper templates
  tensorflow::Tensor dummy_t;
  (void)dummy_t.flat<float>();

  // Force link EventsWriter
  tensorflow::EventsWriter dummy_ew("");
  (void)dummy_ew.InitWithSuffix("");
  (void)dummy_ew.FileName();
  (void)dummy_ew.WriteSerializedEvent("");
  (void)dummy_ew.Flush();
  (void)dummy_ew.Close();

  // Force link absl::Mutex lock/unlock compatibility methods
  typedef void (absl::Mutex::*LockUnlockPtr)();
  (void)(LockUnlockPtr)&absl::Mutex::lock;
  (void)(LockUnlockPtr)&absl::Mutex::unlock;

  // Force link tensorflow::Graph & Node methods
  (void)&tensorflow::Graph::AddControlEdge;
  (void)&tensorflow::Graph::RemoveControlEdge;
  (void)&tensorflow::Node::set_requested_device;
  (void)&tensorflow::Node::op_def;
  (void)&tensorflow::Node::def;
  (void)&tensorflow::Node::type_string;
  (void)&tensorflow::Node::requested_device;

  // Force link ApiDefMap destructor
  {
    tensorflow::ApiDefMap* api_def_map = nullptr;
    delete api_def_map;
  }

  // Force link StatSummarizer
  tensorflow::StatSummarizer dummy_ss((tsl::StatSummarizerOptions()));
  (void)dummy_ss.PrintStepStats();

  // Force link Determinism, metrics & thread
  (void)&tsl::EnableOpDeterminism;
  (void)&tsl::OpDeterminismRequired;
  (void)&tensorflow::metrics::SavedModelWriteCount;
  (void)&tensorflow::metrics::SavedModelWriteApi;
  (void)&tensorflow::metrics::SavedModelReadCount;
  (void)&tensorflow::metrics::SavedModelReadApi;
  using ScheduleType = void (tsl::thread::ThreadPool::*)(std::function<void()>);
  (void)(ScheduleType)&tsl::thread::ThreadPool::Schedule;

  // Force link Global Flags
  (void)&tensorflow::flags::Global;

  // Force link HloInstruction & DfsHloVisitorBase SetVisited
  xla::HloInstruction* dummy_hlo = nullptr;
  if (dummy_hlo) {
    (void)dummy_hlo->ToString();
  }
  using VisitorPtrType1 = xla::DfsHloVisitorBase<xla::HloInstruction*>;
  using SetVisitedType1 = void (VisitorPtrType1::*)(const xla::HloInstruction&);
  (void)(SetVisitedType1)&VisitorPtrType1::SetVisited;

  using VisitorPtrType2 = xla::DfsHloVisitorBase<const xla::HloInstruction*>;
  using SetVisitedType2 = void (VisitorPtrType2::*)(const xla::HloInstruction&);
  (void)(SetVisitedType2)&VisitorPtrType2::SetVisited;

  // Force link ProfilerServer
  tsl::profiler::ProfilerServer* dummy_prof_srv = nullptr;
  if (dummy_prof_srv) {
    dummy_prof_srv->StartProfilerServer(0);
  }

  // Active volatile-guarded call enforcers to prevent dead-code optimization
  volatile bool force_link_enable = false;
  if (force_link_enable) {
    // Abseil Mutex & CondVar active calls
    absl::Mutex active_mutex;
    active_mutex.lock();
    active_mutex.unlock();
    absl::CondVar active_condvar;
    active_condvar.Signal();
    active_condvar.Wait(&active_mutex);

    // Abseil HashMap instantiation to pull in absl::container_internal
    absl::flat_hash_map<std::string, std::string> active_map;
    active_map["key"] = "val";
    active_map.erase("key");

    // Abseil Numbers & Strings
    char active_num_buf[32];
    absl::numbers_internal::FastIntToBuffer(0, active_num_buf);
    absl::numbers_internal::FastIntToBuffer(0ULL, active_num_buf);
    std::string active_dummy_str = "a";
    absl::StrAppend(&active_dummy_str, "b", "c");
    (void)absl::StrCat(active_dummy_str, "d");

    // TSL CancellationManager & DeviceNameUtils & status utils
    tsl::CancellationManager active_cm;
    active_cm.StartCancel();
    tsl::DeviceNameUtils::ParsedName active_parsed_name;
    tsl::DeviceNameUtils::ParseFullName("", &active_parsed_name);
    (void)tsl::DeviceNameUtils::IsSameAddressSpace("", "");
    absl::Status active_status = absl::InternalError("msg");
    (void)active_status.ToString();
    tsl::Set_TF_Status_from_Status(nullptr, active_status);

    // LLVM FileCheck active calls
    llvm::FileCheckRequest fcr;
    llvm::FileCheck fc(fcr);
    llvm::SourceMgr SM;
    (void)fc.readCheckFile(SM, llvm::StringRef("check"));
    (void)fc.checkInput(SM, llvm::StringRef("input"));
    (void)llvm::MemoryBuffer::getMemBuffer("buf");

    // Eager Python API active calls
    (void)TFE_GetPythonString(nullptr);
    (void)TFE_TensorHandleToNumpy(nullptr, nullptr);
    (void)EagerTensor_CheckExact(nullptr);
    (void)EagerTensor_Handle(nullptr);

    // Python API parameter converter active calls
    tensorflow::PythonAPIInfo py_api_info("");
    tensorflow::PythonTensorConverter py_tensor_conv(nullptr, nullptr, "");
    absl::Span<PyObject*> py_objs_span;
    tensorflow::PythonAPIInfo::InferredAttributes inf_attrs;
    (void)tensorflow::ConvertPythonAPIParameters(py_api_info, py_tensor_conv,
                                                 py_objs_span, &inf_attrs);
    (void)tensorflow::CopyPythonAPITensorLists(py_api_info, py_objs_span);

    // Exception registry active calls
    (void)tensorflow::PyExceptionRegistry::Lookup(TF_CANCELLED);

    // Function parameter canonicalizer active calls
    tensorflow::FunctionParameterCanonicalizer active_canon({}, {});
    (void)active_canon.Canonicalize(nullptr, nullptr, {});

    // Absl fatal logger active calls
    ABSL_LOG(FATAL) << "enforcer fatal";

    // PyContextManager active calls
    tensorflow::PyContextManager py_ctx_mgr;
    (void)py_ctx_mgr.Enter(nullptr);
    (void)py_ctx_mgr.var();

    // TensorFlow TensorShapeBase & PartialTensorShape active calls
    tensorflow::PartialTensorShape active_pts;
    (void)active_pts.dim_sizes();
    (void)active_pts.dim_size(0);
    (void)active_pts.SetDimWithStatus(0, 0);

    // Proto message active calls
    tensorflow::GraphDef active_graph_def;
    (void)active_graph_def.GetDescriptor();
    (void)active_graph_def.GetReflection();
    (void)active_graph_def.default_instance();

    tensorflow::MetaGraphDef active_meta_graph_def;
    (void)active_meta_graph_def.GetDescriptor();
    (void)active_meta_graph_def.GetReflection();
    (void)active_meta_graph_def.default_instance();

    tensorflow::FingerprintDef active_fingerprint_def;
    (void)active_fingerprint_def.GetDescriptor();
    (void)active_fingerprint_def.default_instance();

    tensorflow::AttrValue active_attr_val;
    (void)active_attr_val.GetDescriptor();
    (void)active_attr_val.default_instance();

    tensorflow::ConfigProto active_config;
    (void)active_config.GetDescriptor();
    (void)active_config.default_instance();

    tensorflow::DeviceAttributes active_dev_attr;
    (void)active_dev_attr.GetDescriptor();
    (void)active_dev_attr.default_instance();

    tensorflow::DeviceProperties active_dev_prop;
    (void)active_dev_prop.GetDescriptor();
    (void)active_dev_prop.default_instance();

    tensorflow::data::experimental::DispatcherConfig active_disp_conf;
    (void)active_disp_conf.GetDescriptor();
    (void)active_disp_conf.default_instance();

    tensorflow::data::experimental::WorkerConfig active_work_conf;
    (void)active_work_conf.GetDescriptor();
    (void)active_work_conf.default_instance();

    tensorflow::data::DataServiceMetadata active_ds_metadata;
    (void)active_ds_metadata.GetDescriptor();
    (void)active_ds_metadata.default_instance();

    tensorflow::CoordinatedTask active_coord_task;
    (void)active_coord_task.GetDescriptor();
    (void)active_coord_task.default_instance();

    tensorflow::NamedDevice active_named_dev;
    (void)active_named_dev.GetDescriptor();
    (void)active_named_dev.default_instance();

    tensorflow::RunMetadata active_run_metadata;
    (void)active_run_metadata.GetDescriptor();
    (void)active_run_metadata.default_instance();

    tensorflow::StepStats active_step_stats;
    (void)active_step_stats.GetDescriptor();
    (void)active_step_stats.default_instance();

    tensorflow::SignatureDef active_sig_def;
    (void)active_sig_def.GetDescriptor();
    (void)active_sig_def.default_instance();

    tensorflow::FunctionDef active_func_def;
    (void)active_func_def.GetDescriptor();
    (void)active_func_def.default_instance();

    tensorflow::FunctionDefLibrary active_func_def_lib;
    (void)active_func_def_lib.GetDescriptor();
    (void)active_func_def_lib.default_instance();

    tensorflow::dtensor::LayoutProto active_layout_proto;
    (void)active_layout_proto.GetDescriptor();
    (void)active_layout_proto.default_instance();

    tensorflow::dtensor::MeshProto active_mesh_proto;
    (void)active_mesh_proto.GetDescriptor();
    (void)active_mesh_proto.default_instance();

    stablehlo::quantization::CalibrationOptions active_cal_opt;
    (void)active_cal_opt.GetDescriptor();
    (void)active_cal_opt.default_instance();

    tensorflow::calibrator::CalibrationStatistics active_cal_stats;
    (void)active_cal_stats.GetDescriptor();
    (void)active_cal_stats.default_instance();

    stablehlo::quantization::QuantizationConfig active_quant_config;
    (void)active_quant_config.GetDescriptor();
    (void)active_quant_config.default_instance();

    tensorflow::quantization::QuantizationOptions active_quant_options;
    (void)active_quant_options.GetDescriptor();
    (void)active_quant_options.default_instance();

    tensorflow::quantization::RepresentativeDatasetFile active_rep_dataset;
    (void)active_rep_dataset.GetDescriptor();
    (void)active_rep_dataset.default_instance();

    tensorflow::tpu::SparseCoreTableLayout active_sc_layout;
    (void)active_sc_layout.GetDescriptor();
    (void)active_sc_layout.default_instance();

    tensorflow::tpu::SparseCoreTableLayouts active_sc_layouts;
    (void)active_sc_layouts.GetDescriptor();
    (void)active_sc_layouts.default_instance();

    tensorflow::HistogramProto active_hist_proto;
    (void)active_hist_proto.GetDescriptor();
    (void)active_hist_proto.default_instance();

    // MessageDifferencer active calls
    google::protobuf::util::MessageDifferencer active_diff;
    google::protobuf::util::DefaultFieldComparator active_comp;
    active_diff.set_field_comparator(&active_comp);
    active_diff.TreatAsSet(nullptr);
    (void)active_diff.Compare(active_graph_def, active_graph_def);

    // Custom Python framework C++ utilities
    (void)tensorflow::AttributeTypeFromName("");
    (void)tensorflow::ConvertPyObjectToAttributeType(
        nullptr, tensorflow::AttributeType::STRING);
    (void)tensorflow::AttrValueToPyObject(tensorflow::AttrValue());
    (void)tensorflow::InitializePyTrampoline(nullptr);
    (void)tensorflow::swig::TryFindKernelClass("");
    (void)pybind11::google::ImportStatusModule(false);

    // OpDefBuilder & KernelDefBuilder
    tensorflow::OpDefBuilder odb("dummy");
    odb.Input("").Output("").Attr("").SetShapeFn(nullptr);
    tensorflow::KernelDefBuilder kdb("dummy");
    kdb.Device("").HostMemory("");
    (void)kdb.Build();

    // ResourceHandle
    tensorflow::ResourceHandle rh;
    (void)rh.ValidateType<float>();

    // DebugEventsWriter
    tensorflow::tfdbg::DebugEventsWriter* dummy_dew_ptr = nullptr;
    if (dummy_dew_ptr) {
      (void)dummy_dew_ptr->RegisterDeviceAndGetId("");
      dummy_dew_ptr->WriteSerializedNonExecutionDebugEvent(
          "", tensorflow::tfdbg::DebugEventFileType::METADATA);
      dummy_dew_ptr->WriteSerializedExecutionDebugEvent(
          "", tensorflow::tfdbg::DebugEventFileType::METADATA);
      (void)dummy_dew_ptr->FlushNonExecutionFiles();
      (void)dummy_dew_ptr->FlushExecutionFiles();
      (void)dummy_dew_ptr->Close();
    }

    // ProfilerSessionWrapper
    tensorflow::profiler::pywrap::ProfilerSessionWrapper psw;
    (void)psw.Start(nullptr, {});
    (void)psw.Stop(nullptr);
    (void)psw.ExportToTensorBoard();

    // Tf2 execution utils
    tensorflow::set_tf2_execution(true);
    (void)tensorflow::tf2_execution_enabled();

    // TOCO Converter
    (void)toco::TocoConvert(nullptr, nullptr, nullptr, false);

    // PyExceptionRegistry
    tensorflow::PyExceptionRegistry::Init(nullptr);
    (void)tensorflow::PyExceptionRegistry::Lookup(TSL_Code::TSL_OK);

    // Safe_TF_StatusPtr
    TF_Status* status = nullptr;
    (void)tensorflow::make_safe(status);

    // Pybind11 Protobuf
    (void)pybind11_protobuf::InitializePybindProtoCastUtil();
    (void)pybind11_protobuf::GenericProtoCast(
        nullptr, pybind11::return_value_policy::automatic, pybind11::handle(),
        false);
    (void)pybind11_protobuf::PyProtoGetCppMessagePointer(pybind11::handle());
    (void)pybind11_protobuf::PyProtoHasMatchingFullName(pybind11::handle(),
                                                        nullptr);
    (void)pybind11_protobuf::PyProtoSerializePartialToString(pybind11::handle(),
                                                             false);
    (void)pybind11_protobuf::PyBytesAsStringView(pybind11::bytes());
  }
}

namespace tsl::internal::profiler {
extern std::atomic<int> g_trace_level;
extern std::atomic<uint64_t> g_trace_filter_bitmap;
}  // namespace tsl::internal::profiler
namespace xla::profiler {
extern std::atomic<bool> traceme_enabled;
}

void dummy_profiler_linker_references() {
  (void)&tsl::internal::profiler::g_trace_level;
  (void)&tsl::internal::profiler::g_trace_filter_bitmap;
  (void)&xla::profiler::traceme_enabled;
}
}  // namespace python
}  // namespace tensorflow
