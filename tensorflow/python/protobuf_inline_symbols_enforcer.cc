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
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/experimental/gradients/math_grad.h"
#include "tensorflow/c/experimental/gradients/nn_grad.h"
#include "tensorflow/c/experimental/gradients/tape/tape_context.h"
#include "tensorflow/c/tf_buffer.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/compiler/jit/get_compiler_ir.h"
#include "tensorflow/compiler/mlir/lite/python/converter_python_api.h"
#include "tensorflow/compiler/mlir/python/mlir.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/quantize_model.h"
#include "tensorflow/compiler/mlir/tensorflow_to_stablehlo/python/pywrap_tensorflow_to_stablehlo_lib.h"
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/py_utils.h"
#include "tensorflow/compiler/tf2xla/tf2xla_opset.h"
#include "xla/tsl/c/tsl_status.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/stack_frame.h"
#include "xla/tsl/util/device_name_utils.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/py_utils.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_base.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph_debug_info_builder.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/costs/graph_memory.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/profiler/internal/print_model_analysis.h"
#include "tensorflow/core/tpu/kernels/sparse_core_ops_utils.h"
#include "tensorflow/core/util/debug_events_writer.h"
#include "tensorflow/core/util/managed_stack_trace.h"
#include "tensorflow/dtensor/cc/dtensor_device.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/proto/layout.pb.h"
#include "tensorflow/python/eager/pywrap_tensor_conversion.h"
#include "tensorflow/python/eager/pywrap_tfe.h"
#include "tensorflow/python/framework/python_api_dispatcher.h"
#include "tensorflow/python/framework/python_api_info.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"
#include "tensorflow/python/pywrap_library_dependency_enforcer.h"
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

// Forward declare ProfilerSessionWrapper to avoid visibility issues
namespace tensorflow {
namespace profiler {
namespace pywrap {
class ProfilerSessionWrapper {
 public:
  absl::Status Start(
      const char* logdir,
      const absl::flat_hash_map<
          std::string, absl::variant<bool, int, std::string>>& options);
  absl::Status Stop(std::string* profile_proto);
  absl::Status ExportToTensorBoard();
};
}  // namespace pywrap
}  // namespace profiler
}  // namespace tensorflow
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

  tensorflow::data::experimental::DispatcherConfig dispatcher_config;
  dispatcher_config.default_instance();

  tensorflow::data::experimental::WorkerConfig worker_config;
  worker_config.default_instance();

  tensorflow::data::DataServiceMetadata data_service_metadata;
  tensorflow::quantization::QuantizationOptions quantization_options;
  tensorflow::CoordinatedTask coordinated_task;
  tensorflow::DeviceAttributes device_attributes;

  // Force link/instantiation of TFLite converter and Abseil FormatPack APIs
  (void)&tflite::RetrieveCollectedErrors;
  (void)&tflite::Convert;

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

  // Force link/instantiation of DataServiceDispatcherClient virtual methods
  (void)(absl::Status (tensorflow::data::DataServiceDispatcherClient::*)(
      const std::string&, const std::string&, int64_t, int64_t, int64_t,
      tensorflow::Tensor&, int64_t&,
      bool&))&tensorflow::data::DataServiceDispatcherClient::GetSnapshotSplit;

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
}
}  // namespace python
}  // namespace tensorflow
