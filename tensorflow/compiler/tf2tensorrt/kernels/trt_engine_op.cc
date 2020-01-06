/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <algorithm>
#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_shape_optimization_profiles.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_allocator.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/lib/statusor.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
static Logger logger;
using absl::StrAppend;
using absl::StrCat;
using ::nvinfer1::IRuntime;
using ::stream_executor::port::StatusOr;

// A helper class to call done() when destructed for asynchronous execution.
// Helps simultaneous execution of native and TRT engines.

class AsyncHelper : public core::RefCounted {
 public:
  AsyncHelper(AsyncOpKernel::DoneCallback done) : done_(done) {}

  ~AsyncHelper() override { this->operator()(); }

  void operator()() {
    if (!called_) {
      done_();
      called_ = true;
    }
  }

 private:
  AsyncOpKernel::DoneCallback done_;
  bool called_ = false;  // Has `done_` been called?
};

//  This OP can construct TRTEngine on the fly and if construction of engine
//  fails, executes equivalent subgraph as a TensorFlow function.
class TRTEngineOp : public AsyncOpKernel {
 public:
  explicit TRTEngineOp(OpKernelConstruction* context);

  void ComputeAsync(OpKernelContext* context,
                    AsyncOpKernel::DoneCallback done) override;

 private:
  using CacheType =
      LRUCache<std::vector<TensorShape>, std::unique_ptr<EngineContext>,
               VectorTensorShapeHasher>;

  // Execute calibration
  void ExecuteCalibration(OpKernelContext* ctx,
                          TRTEngineCacheResource* cache_res,
                          AsyncHelper* helper);

  // Construct a function handle for executing native funcdef graph
  // These are the exact same function.

  Status ConstructFunctionHandle(FunctionLibraryRuntime* lib,
                                 const string& device_name);

  // Execute replaced native segment as function Op.
  void ExecuteNativeSegment(OpKernelContext* ctx, AsyncHelper* helper);

  // Execute the tensorrt engine. Returns whether we need to retry by running
  // the native segment.
  bool ExecuteTrtEngine(OpKernelContext* ctx, EngineContext* engine_context,
                        int opt_profile_idx);

  // Allocate necessary resources for calibration
  Status AllocateCalibrationResources(OpKernelContext* ctx,
                                      TRTEngineCacheResource* cache_res);

  Status GetEngineCacheResource(OpKernelContext* ctx,
                                TRTEngineCacheResource** cache_res);

  // Get engine for the input shape
  StatusOr<EngineContext*> GetEngine(
      const std::vector<TensorShape>& input_shapes, OpKernelContext* ctx,
      TRTEngineCacheResource* cache_res);

  // Verify that the input shapes are consistent and can be handled by this op.
  Status VerifyInputShapes(const std::vector<TensorShape>& shapes);

  // Make actual_input_shapes compatible with an existing shape
  // in the cache and return it. Two shapes are compatibile if their
  // only different dimension (if any) is the batch dimension. The returned
  // shape (engine_input_shapes) must be ready to use for either cache
  // lookup or building an engine.
  Status GetEngineInputShapes(
      const CacheType& cache,
      const std::vector<TensorShape>& actual_input_shapes,
      std::vector<TensorShape>* engine_input_shapes);

  std::vector<string> input_nodes_;
  std::vector<string> output_nodes_;

  // serialized protobuf segment or trt engine depending on static_engine_ flag.
  string serialized_segment_;

  // The function for TF native execution of the segment.
  NameAttrList func_;

  // GraphDef representation of the segment.
  GraphDef segment_graph_def_;

  // Engine Precision mode.
  TrtPrecisionMode precision_mode_;

  // Whether engine is constructed during the conversion or needs to be
  // constructed from protobuf segment.
  bool static_engine_;

  // Whether to calibrate INT8 engine.
  bool calibration_mode_;

  // Whether to use implicit batch dimension for TensorRT
  bool use_implicit_batch_;

  // Whether to collect optimization profiles for TensorRT
  // Only used when use_implicit_batch_=false
  bool profile_generation_mode_;

  // Maximum number of cached engines
  int max_cached_engines_;

  int64 workspace_size_;
  mutex engine_mutex_;
  FunctionLibraryRuntime::Handle func_handle_;

  // The finalized calibrator for inference.
  std::unique_ptr<TRTInt8Calibrator> calibrator_;

  // If true, create calibration graph for INT8 mode. Otherwise, we are using
  // user-provided quantization ranges.
  bool use_calibration_;
};

#define TYPECASE(dt, X, Y)                                    \
  case dt: {                                                  \
    return (void*)X->flat<EnumToDataType<dt>::Type>().data(); \
  }

void* GetTensorAddress(const Tensor* tensor_ptr) {
  auto tensor_type = tensor_ptr->dtype();
  switch (tensor_type) {
    TYPECASE(DT_FLOAT, tensor_ptr, dest_ptr);
    TYPECASE(DT_HALF, tensor_ptr, dest_ptr);
    TYPECASE(DT_INT8, tensor_ptr, dest_ptr);
    TYPECASE(DT_INT32, tensor_ptr, dest_ptr);
    default: {
      LOG(ERROR) << "Unsupported Data type " << DataTypeString(tensor_type);
      return nullptr;
    }
  }
}

static Status FunctionDefToGraphDef(FunctionLibraryRuntime::Handle handle,
                                    FunctionLibraryRuntime* flib_runtime,
                                    GraphDef* graph_def) {
  const FunctionLibraryDefinition* flib_def =
      flib_runtime->GetFunctionLibraryDefinition();
  const FunctionBody* fbody;
  fbody = flib_runtime->GetFunctionBody(handle);
  if (!fbody) {
    return errors::Internal(
        "Function body is null when converting from FuncDef to GraphDef.");
  }
  std::unique_ptr<Graph> graph(new Graph(flib_def));
  CopyGraph(*fbody->graph, graph.get());

  auto replace_name = [](const char* const prefix, string* name) {
    if (absl::StartsWith(*name, absl::AsciiStrToLower(prefix))) {
      name->replace(0, strlen(prefix), prefix);
      return true;
    }
    return false;
  };
  graph->ToGraphDef(graph_def);
  // GraphToFunctionDef() will convert all the node names to lowercase.
  for (auto& node : *graph_def->mutable_node()) {
    if (!replace_name(IONamePrefixes::kInputPHName, node.mutable_name())) {
      if (replace_name(IONamePrefixes::kOutputPHName, node.mutable_name())) {
        // Instantiation of the function will append _RetVal to the node name,
        // need to remove it for backward compatibility.
        const char* const suffix_to_remove = "_RetVal";
        if (absl::EndsWith(node.name(), suffix_to_remove)) {
          node.mutable_name()->erase(node.name().size() -
                                     strlen(suffix_to_remove));
        }
      }
    }
    for (auto& input : *node.mutable_input()) {
      if (!replace_name(IONamePrefixes::kInputPHName, &input)) {
        replace_name(IONamePrefixes::kOutputPHName, &input);
      }
    }
  }
  return Status::OK();
}

Status TRTEngineOp::ConstructFunctionHandle(FunctionLibraryRuntime* lib,
                                            const string& device_name) {
  VLOG(1) << "Constructing function handle";
  if (lib == nullptr) {
    return errors::Internal("Context function library is null");
  }
  FunctionLibraryRuntime::InstantiateOptions inst_ops;
  inst_ops.state_handle = "";
  inst_ops.target = device_name;
  return lib->Instantiate(func_.name(), AttrSlice(&func_.attr()), inst_ops,
                          &func_handle_);
}

TRTEngineOp::TRTEngineOp(OpKernelConstruction* context)
    : AsyncOpKernel(context) {
  // read serialized_engine
  OP_REQUIRES_OK(context,
                 context->GetAttr("serialized_segment", &serialized_segment_));
  OP_REQUIRES_OK(context,
                 context->GetAttr("workspace_size_bytes", &workspace_size_));
  OP_REQUIRES_OK(context, context->GetAttr("static_engine", &static_engine_));

  VLOG(1) << "Constructing " << name();
  string precision_string;
  OP_REQUIRES_OK(context,
                 context->GetAttr("precision_mode", &precision_string));
  string calibration_data;
  OP_REQUIRES_OK(context,
                 context->GetAttr("calibration_data", &calibration_data));
  OP_REQUIRES_OK(context, context->GetAttr("segment_func", &func_));
  OP_REQUIRES(context, !func_.name().empty(),
              errors::InvalidArgument(
                  "The TF function for the TRT segment could not be empty"));
  OP_REQUIRES_OK(context,
                 TrtPrecisionModeFromName(precision_string, &precision_mode_));
  OP_REQUIRES_OK(context,
                 context->GetAttr("use_calibration", &use_calibration_));
  func_handle_ = kInvalidHandle;
  if (!static_engine_) {
    FunctionLibraryRuntime* lib = context->function_library();
    OP_REQUIRES_OK(context,
                   ConstructFunctionHandle(lib, context->device()->name()));
    OP_REQUIRES_OK(
        context, FunctionDefToGraphDef(func_handle_, lib, &segment_graph_def_));
  }
  // TODO(laigd): calibration_data is used in TF v1.x and we keep it only for
  // backward compatibility reasons. Remove it once all known users switch to
  // 2.0.
  calibration_mode_ =
      (use_calibration_ && precision_mode_ == TrtPrecisionMode::INT8 &&
       calibration_data.empty());
  if (!calibration_data.empty()) {
    calibrator_.reset(new TRTInt8Calibrator(calibration_data));
    calibration_data.resize(0);
  }
  OP_REQUIRES_OK(context, context->GetAttr("max_cached_engines_count",
                                           &max_cached_engines_));

  auto status = context->GetAttr("_use_implicit_batch", &use_implicit_batch_);
  if (status.code() == tensorflow::error::NOT_FOUND) {
    VLOG(2) << "Not found _use_implicit_batch in " << context->device()->name()
            << ", thus setting _use_implicit_batch=true";
    use_implicit_batch_ = true;
  }
  status = context->GetAttr("_profile_generation_mode", &profile_generation_mode_);
  if (status.code() == tensorflow::error::NOT_FOUND) {
    VLOG(2) << "Not found _profile_generation_mode in " << context->device()->name()
            << ", thus setting _profile_generation_mode=false";
    profile_generation_mode_ = false;
  }
  if (use_implicit_batch_) {
    OP_REQUIRES(context, !profile_generation_mode_,
                errors::InvalidArgument(
                    "profile_generation_mode_=true is only supported if "
                    "use_implicit_batch=false"));
  }
}

void TRTEngineOp::ExecuteNativeSegment(OpKernelContext* ctx,
                                       AsyncHelper* helper) {
  std::vector<Tensor> inputs;
  std::vector<Tensor>* outputs = new std::vector<Tensor>();
  if (func_handle_ == kInvalidHandle) {
    OP_REQUIRES_OK_ASYNC(
        ctx,
        ConstructFunctionHandle(ctx->function_library(), ctx->device()->name()),
        *helper);
  }
  auto lib = ctx->function_library();
  FunctionLibraryRuntime::Options opts;
  opts.rendezvous = ctx->rendezvous();
  opts.cancellation_manager = ctx->cancellation_manager();
  opts.runner = ctx->runner();
  inputs.reserve(ctx->num_inputs());
  for (int i = 0; i < ctx->num_inputs(); i++) {
    inputs.push_back(ctx->input(i));
  }
  helper->Ref();  // Increment count for calculating native graph
  VLOG(1) << "Executing native segment: " << name();
  lib->Run(opts, func_handle_, inputs, outputs,
           [this, ctx, outputs, helper](const Status& s) {
             core::ScopedUnref sc(helper);
             OP_REQUIRES_OK_ASYNC(ctx, s, *helper);
             VLOG(1) << "Native Segment completed";
             for (size_t t = 0; t < outputs->size(); ++t) {
               ctx->set_output(t, outputs->at(t));
             }
             delete outputs;
           });
}

void TRTEngineOp::ExecuteCalibration(OpKernelContext* ctx,
                                     TRTEngineCacheResource* cache_res,
                                     AsyncHelper* helper) {
  VLOG(1) << "Executing TRT calibration: " << name();
  helper->Ref();
  core::ScopedUnref sc(helper);

  CalibrationContext* calib_ctx = cache_res->calib_ctx_.get();
  const int num_inputs = ctx->num_inputs();
  // TODO(laigd): need to check that input shape matches.
  // Pass input data to calibrator
  std::unordered_map<string, void*> input_data;
  for (int i = 0; i < num_inputs; i++) {
    const Tensor& t = ctx->input(i);
    void* data_address = GetTensorAddress(&t);
    OP_REQUIRES_ASYNC(ctx, data_address,
                      errors::InvalidArgument(
                          "Unsupported data type encountered in input ", i),
                      *helper);
    // Check the allocated buffer is sufficient for input
    const auto device_tensor =
        calib_ctx->device_tensors_.at(i).AccessTensor(ctx);
    CHECK_EQ(t.TotalBytes(), device_tensor->TotalBytes());
    input_data.emplace(StrCat(IONamePrefixes::kInputPHName, i), data_address);
  }
  VLOG(2) << "Filled map for sending";
  // copied from cuda_kernel_helper since it seems only valid in *.cu.cc files
  const cudaStream_t* stream = CHECK_NOTNULL(
      reinterpret_cast<const cudaStream_t*>(ctx->op_device_context()
                                                ->stream()
                                                ->implementation()
                                                ->GpuStreamMemberHack()));
  // If calibrator is terminated before, it means an error has occurred.
  //
  // Note: setBatch() will wait until TRTInt8Calibrator::getBatch() is called
  // the first time before proceeding, so if buildCudaEngine() returns an error,
  // it means getBatch() is never called, and the setBatch() here will hang
  // until setDone() is called later by the calibration thread in
  // AllocateCalibrationResources(). In that case, this setBatch() will always
  // be able to detect the error and return false.
  OP_REQUIRES_ASYNC(ctx, calib_ctx->calibrator_->setBatch(input_data, *stream),
                    errors::Internal("Failed to feed calibration data"),
                    *helper);
  VLOG(2) << "Passed calibration data";
  ExecuteNativeSegment(ctx, helper);
}

Status TRTEngineOp::VerifyInputShapes(const std::vector<TensorShape>& shapes) {
  if (shapes.empty()) {
    return errors::InvalidArgument("Input shapes are empty, for ", name());
  }
  if (shapes[0].dims() < 1) {
    return errors::InvalidArgument("Input shapes contain scalar, for ", name(),
                                   ": ",
                                   TensorShapeUtils::ShapeListString(shapes));
  }

  const int batch_size = shapes[0].dim_size(0);
  for (const TensorShape& shape : shapes) {
    if (shape.dims() < 1 || batch_size != shape.dim_size(0)) {
      return errors::InvalidArgument(
          "Input shapes are inconsistent on the batch dimension, for ", name(),
          ": ", TensorShapeUtils::ShapeListString(shapes));
    }
  }
  return Status::OK();
}

bool AreShapesCompatible(const std::vector<TensorShape>& actual_shapes,
                         const std::vector<TensorShape>& cached_shapes) {
  auto match_shape = [](const TensorShape& actual_shape,
                        const TensorShape& cached_shape) {
    // Match the rank.
    if (actual_shape.dims() != cached_shape.dims()) return false;
    // Match the batch size.
    if (actual_shape.dim_size(0) > cached_shape.dim_size(0)) return false;
    // Match remaining dimensions.
    for (int i = 1; i < actual_shape.dims(); ++i) {
      if (actual_shape.dim_size(i) != cached_shape.dim_size(i)) return false;
    }
    return true;
  };
  for (int i = 0; i < actual_shapes.size(); ++i) {
    if (!match_shape(actual_shapes[i], cached_shapes[i])) {
      return false;
    }
  }
  return true;
}

Status TRTEngineOp::GetEngineInputShapes(
    const CacheType& cache, const std::vector<TensorShape>& actual_input_shapes,
    std::vector<TensorShape>* engine_input_shapes) {
  // VerifyInputShapes() already ensured that all input shapes have same
  // batch size, and are not scalars.
  *engine_input_shapes = actual_input_shapes;
  int64 min_matched_batch_size = kint64max;
  for (const auto& pair : cache) {
    const std::vector<TensorShape>& cached_input_shapes = pair.first;
    // This should not happen, but just for safety.
    if (actual_input_shapes.size() != cached_input_shapes.size()) {
      return errors::InvalidArgument(
          "Input shape list size mismatch for ", name(),
          ", cached size: ", cached_input_shapes.size(),
          " vs. actual size: ", actual_input_shapes.size());
    }
    if (AreShapesCompatible(actual_input_shapes, cached_input_shapes)) {
      const int cached_batch_size = cached_input_shapes[0].dim_size(0);
      if (min_matched_batch_size > cached_batch_size) {
        min_matched_batch_size = cached_batch_size;
        *engine_input_shapes = cached_input_shapes;
      }
    }
  }
  return Status::OK();
}

void TRTEngineOp::ComputeAsync(OpKernelContext* ctx,
                               AsyncOpKernel::DoneCallback done) {
  auto helper = new AsyncHelper(done);
  core::ScopedUnref sc(helper);

  // Get TRT resource.
  TRTEngineCacheResource* cache_res = nullptr;
  OP_REQUIRES_OK_ASYNC(ctx, GetEngineCacheResource(ctx, &cache_res), *helper);
  core::ScopedUnref unref_cache_res(cache_res);

  // Run calibration if in int8+calibration mode.
  // * Logic in TF 1.x:
  //   - During conversion: calibration_mode_ is true and cache size is 0, so it
  //     will run calibration.
  //   - During inference: calibration_data will be set, so calibration_mode_ is
  //     false and it won't trigger calibration.
  // * Logic in TF 2.0:
  //   - During conversion: similar to 1.x.
  //   - During inference: calibration_data will still be empty, but cache will
  //     contain the the calibrated engine, so it won't trigger calibration.
  //
  // TODO(laigd): consider the following alternatives:
  // 1. Serialize the state (calibration or inference) using
  //    TRTEngineInstance proto (or a new proto), so we know which mode we're
  //    in and don't run calibration during inference (which is invalid).
  // 2. Reuse the calibration_data attribute or use a new attribute in the
  //    NodeDef to indicate whether it's in calibration mode.
  if (calibration_mode_ && cache_res->cache_.size() == 0) {
    if (!cache_res->calib_ctx_) {
      // TODO(laigd): better encapsulation.
      mutex_lock lock(engine_mutex_);
      if (!cache_res->calib_ctx_) {
        OP_REQUIRES_OK_ASYNC(ctx, AllocateCalibrationResources(ctx, cache_res),
                             *helper);
      }
    }
    // TODO(laigd): check that the input shapes match the shapes of the
    // persistent tensor in the calibration resource.
    ExecuteCalibration(ctx, cache_res, helper);
    return;
  }

  // Get shapes of inputs to engine.
  std::vector<TensorShape> input_shapes;
  input_shapes.reserve(ctx->num_inputs());
  for (int i = 0; i < ctx->num_inputs(); ++i) {
    input_shapes.push_back(ctx->input(i).shape());
  }
  OP_REQUIRES_OK_ASYNC(ctx, VerifyInputShapes(input_shapes), *helper);

  if (profile_generation_mode_) {
    // Just collect the input shape info and return. The shapes are used to
    // generate optimization profiles during engine creation.
    cache_res->profiles_.addShape(input_shapes);
    ExecuteNativeSegment(ctx, helper);
    return;
  }
  StatusOr<EngineContext*> status = GetEngine(input_shapes, ctx, cache_res);
  OP_REQUIRES_OK_ASYNC(ctx, status.status(), *helper);

  EngineContext* engine_context = status.ValueOrDie();
  if (!engine_context->cuda_engine) {
    VLOG(1) << "Engine retrieval for input shapes: "
            << TensorShapeUtils::ShapeListString(input_shapes)
            << " failed. Running native segment for " << name();
    ExecuteNativeSegment(ctx, helper);
    return;
  }
  int opt_profile_idx = cache_res->profiles_.getProfileNumber(input_shapes);

  const bool retry = ExecuteTrtEngine(ctx, engine_context, opt_profile_idx);
  if (retry) {
    LOG(WARNING) << "Failed to execute engine, "
                 << "retrying with native segment for " << name();
    // Release any outputs that are allocated, ExecuteNativeSegment will
    // re-allocate them and fail if they are currently allocated.
    for (int i = 0; i < ctx->num_outputs(); i++) {
      ctx->release_output(i);
    }
    ExecuteNativeSegment(ctx, helper);
    return;
  }
}

bool TRTEngineOp::ExecuteTrtEngine(OpKernelContext* ctx,
                                   EngineContext* engine_context,
                                   int opt_profile_idx) {
  VLOG(1) << "Executing TRT engine: " << name();
  auto& cuda_engine = engine_context->cuda_engine;

  if (VLOG_IS_ON(2)) {
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
    VLOG(2) << "  Network name: " << cuda_engine->getName();
#endif  // #if IS_TRT_VERSION_GE(6, 0, 0, 0)
    VLOG(2) << "  Activation size: " << cuda_engine->getDeviceMemorySize()
            << " bytes";
    VLOG(2) << "  Workspace size: " << cuda_engine->getWorkspaceSize()
            << " bytes";
    VLOG(2) << "  Datatype of " << cuda_engine->getNbBindings()
            << " inputs/outputs";
    string binding_types = "";
    for (int i = 0; i < cuda_engine->getNbBindings(); i++) {
      binding_types += "    " + string(cuda_engine->getBindingName(i)) + ": " +
                       DebugString(cuda_engine->getBindingDataType(i)) + "\n";
    }
    VLOG(2) << binding_types;
  }

  const bool kRetry = true;
  if (opt_profile_idx >= engine_context->execution_context.size()) {
    LOG(ERROR) << "Requested engine context with index " << opt_profile_idx
               << ", but only " <<  engine_context->execution_context.size()
               << "contexts are present.";
    return kRetry;
  }
  auto& execution_context = engine_context->execution_context[opt_profile_idx];
  const int num_binding = ctx->num_inputs() + ctx->num_outputs();
  std::vector<void*> buffers(num_binding);

  // Setup engine inputs
  for (int i = 0; i < ctx->num_inputs(); i++) {
    const string input_name = StrCat(IONamePrefixes::kInputPHName, i);
    const int binding_index = cuda_engine->getBindingIndex(input_name.c_str());
    if (binding_index == -1) {
      const string msg =
          StrCat("Input node ", input_name, " not found, at ", name());
      LOG(ERROR) << msg;
      ctx->SetStatus(errors::NotFound(msg));
      return !kRetry;
    }

    const Tensor& input_tensor = ctx->input(i);
    const TensorShape& input_shape = input_tensor.shape();

    if (use_implicit_batch_) {
      // Ensure all inputs have the same batch size
      const int num_batch = ctx->input(0).shape().dim_size(0);
      if (num_batch != input_shape.dim_size(0)) {
        LOG(ERROR) << "Input data has inconsistent batch size: " << num_batch
                   << " vs " << input_shape.dim_size(0);
        return kRetry;
      }
    }
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
    // Set known input dimensions. This is necessary because TRT network
    // could be made with dynamic dimensions.
    if (!use_implicit_batch_) {
      nvinfer1::Dims trt_dims;
      trt_dims.nbDims = input_shape.dims();
      for (int k = 0; k < input_shape.dims(); k++) {
        trt_dims.d[k] = input_shape.dim_size(k);
      }
      execution_context->setBindingDimensions(binding_index, trt_dims);
    }
#endif
    // Setup input bindings.
    auto dtype = cuda_engine->getBindingDataType(binding_index);
    switch (dtype) {
      case nvinfer1::DataType::kFLOAT:
        buffers[binding_index] =
            const_cast<float*>(input_tensor.flat<float>().data());
        break;
      case nvinfer1::DataType::kHALF:
        buffers[binding_index] =
            const_cast<Eigen::half*>(input_tensor.flat<Eigen::half>().data());
        break;
      case nvinfer1::DataType::kINT8:
        LOG(ERROR) << "INT8 inputs are not supported yet!";
        return kRetry;
      case nvinfer1::DataType::kINT32:
        buffers[binding_index] =
            const_cast<int32*>(input_tensor.flat<int32>().data());
        break;
      default:
        LOG(ERROR) << "Unknown TRT data type: " << static_cast<int>(dtype);
        return kRetry;
    }
  }

#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  // Ensure all network dynamic dimensions (if any) are set in execution context.
  if (!execution_context->allInputDimensionsSpecified()) {
    LOG(WARNING) << "Failed to set dimensions for all dynamic input tensors.";
    return kRetry;
  }
  if (!execution_context->allInputShapesSpecified()) {
    LOG(WARNING) << "Failed to set dimensions for all shape input tensors.";
    return kRetry;
  }
#endif

  // Setup engine outputs
  for (int i = 0; i < ctx->num_outputs(); i++) {
    // Create an output tensor
    const string output_name = StrCat(IONamePrefixes::kOutputPHName, i);
    const int binding_index = cuda_engine->getBindingIndex(output_name.c_str());
    if (binding_index == -1) {
      const string msg =
          StrCat("Ouput node ", output_name, " not found, at ", name());
      LOG(ERROR) << msg;
      ctx->SetStatus(errors::NotFound(msg));
      return !kRetry;
    }
    // Get TRT output shapes for allocating output memory.
    std::vector<int> trt_shape;
    if (!use_implicit_batch_) {
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
      // Get dims from context instead of engine in explicit batch mode
      // because engine might have dynamic shapes.
      auto dims = execution_context->getBindingDimensions(binding_index);
      for (int j = 0; j < dims.nbDims; j++) {
        trt_shape.push_back(dims.d[j]);
      }
#else
      LOG(ERROR) << "Explicit batch mode is only supported with TensorRT 6 and above.";
      return kRetry;
#endif
    } else {
      auto dims = cuda_engine->getBindingDimensions(binding_index);
      // In implicit batch mode, it's assumed that first dimension of
      // all inputs and outputs is batch size.
      trt_shape.push_back(ctx->input(0).shape().dim_size(0));
      for (int j = 0; j < dims.nbDims; j++) {
        trt_shape.push_back(dims.d[j]);
      }
    }
    // Allocate output tensor of TRTEngineOp.
    Tensor* output_tensor = nullptr;
    TensorShape output_shape;
    auto status = TensorShapeUtils::MakeShape(
        trt_shape.data(), trt_shape.size(), &output_shape);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to get output shape: " << status;
      return kRetry;
    }
    status = ctx->allocate_output(i, output_shape, &output_tensor);
    if (!status.ok()) {
      LOG(ERROR) << "Allocating output failed with " << status;
      ctx->SetStatus(status);
      return kRetry;
    }
    // Setup output bindings.
    auto dtype = cuda_engine->getBindingDataType(binding_index);
    switch (dtype) {
      case nvinfer1::DataType::kFLOAT:
        buffers[binding_index] =
            const_cast<float*>(output_tensor->flat<float>().data());
        break;
      case nvinfer1::DataType::kHALF:
        buffers[binding_index] =
            const_cast<Eigen::half*>(output_tensor->flat<Eigen::half>().data());
        break;
      case nvinfer1::DataType::kINT8:
        LOG(WARNING) << "int8 is not supported yet!";
        return kRetry;
      case nvinfer1::DataType::kINT32:
        buffers[binding_index] =
            const_cast<int32*>(output_tensor->flat<int32>().data());
        break;
      default:
        LOG(WARNING) << "Unknown TRT data type: " << static_cast<int>(dtype);
        return kRetry;
    }
  }
  // Copied from cuda_kernel_helper since it seems only valid in *.cu.cc files
  const cudaStream_t* stream = CHECK_NOTNULL(
      reinterpret_cast<const cudaStream_t*>(ctx->op_device_context()
                                                ->stream()
                                                ->implementation()
                                                ->GpuStreamMemberHack()));

  // nvinfer1::IExecutionContext::enqueue* is not thread safe and we need a mutex
  // for it.
  mutex_lock lock(engine_context->mu);
  bool ret = false;
  if (use_implicit_batch_) {
    const int num_batch = ctx->input(0).shape().dim_size(0);
    ret = execution_context->enqueue(num_batch, &buffers[0], *stream, nullptr);
  } else {
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
    ret = execution_context->enqueueV2(&buffers[0], *stream, nullptr);
#else
    LOG(ERROR) << "Explicit batch mode is only supported with TensorRT 6 and above.";
    return kRetry;
#endif
  }
  if (!ret) {
    LOG(WARNING) << "Failed to enqueue batch for TRT engine: " << name();
    return kRetry;
  }
  // Synchronization will be done by TF.
  return !kRetry;
}

Status TRTEngineOp::GetEngineCacheResource(OpKernelContext* ctx,
                                           TRTEngineCacheResource** cache_res) {
  // Canonicalize the op name by removing the scopes if any. This is mainly
  // because in TFv2, the function graph can be instantiated in various ways and
  // it'll insert scope names to the name of the TRTEngineOps, which will result
  // in many different engine caches if we use the instantiated op name
  // directly, but we still want all of them share the same cache (if they were
  // representing the same subgraph).
  absl::string_view resource_name = name();
  size_t last_slash = resource_name.find_last_of('/');
  if (last_slash != absl::string_view::npos) {
    resource_name.remove_prefix(last_slash + 1);
  }

  // Get engine cache.
  return ctx->resource_manager()->LookupOrCreate(
      std::string(kTfTrtContainerName), std::string(resource_name), cache_res,
      {[this, ctx](TRTEngineCacheResource** cr) -> Status {
        *cr = new TRTEngineCacheResource(ctx, this->max_cached_engines_);
        return Status::OK();
      }});
}

StatusOr<EngineContext*> TRTEngineOp::GetEngine(
    const std::vector<TensorShape>& input_shapes, OpKernelContext* ctx,
    TRTEngineCacheResource* cache_res) {
  static EngineContext empty_context;

  mutex_lock lock(engine_mutex_);
  // Using first input to get batch size is reliable - VerifyInputShapes() has
  // verified that. Although this is not needed for explicit batch mode, but
  // we still have to pass it to ConvertGraphDefToEngine until the requirement
  // is removed from that function.
  const int batch_size = input_shapes[0].dim_size(0);
  auto& cache = cache_res->cache_;
  auto allocator = cache_res->allocator_.get();
  if (allocator == nullptr) {
    return &empty_context;
  }

  // Handle the static engine case. For static engines, the cache will have a
  // single element containing the only engine.
  if (static_engine_) {
    if (cache.size()) {
      // TODO(laigd): need a better shape compatibility check for the case where
      // implicit batch is disabled.
      if (!use_implicit_batch_ ||
          AreShapesCompatible(input_shapes, cache.begin()->first)) {
        return cache.begin()->second.get();
      }
      return &empty_context;
    }

    TrtUniquePtrType<IRuntime> infer(nvinfer1::createInferRuntime(logger));
    infer->setGpuAllocator(allocator);
    TrtUniquePtrType<nvinfer1::ICudaEngine> static_engine(
        infer->deserializeCudaEngine(serialized_segment_.c_str(),
                                     serialized_segment_.size(), nullptr));
    if (!static_engine) {
      return &empty_context;
    }
    auto raw_static_engine = static_engine.get();
    const auto max_batch_size = raw_static_engine->getMaxBatchSize();
    // Static engine will have max_batch_size for batch size so that all inputs
    // will map to this single engine.
    std::vector<TensorShape> engine_input_shapes(input_shapes);
    if (use_implicit_batch_) {
      for (int i = 0; i < engine_input_shapes.size(); i++) {
        engine_input_shapes[i].set_dim(0, max_batch_size);
      }
    }
    // TODO(laigd): here we assume engine_input_shapes matches the actual input
    // shapes of the engine, we should verify that.
    TrtUniquePtrType<nvinfer1::IExecutionContext> exec_ctx(
        raw_static_engine->createExecutionContext());
    std::vector<TrtUniquePtrType<nvinfer1::IExecutionContext>> ctx_vec;
    ctx_vec.push_back(std::move(exec_ctx));
    cache.emplace(engine_input_shapes,
                  absl::make_unique<EngineContext>(
                      std::move(static_engine), std::move(ctx_vec)));
    // Runtime is safe to delete after engine creation
    VLOG(1) << "Size of serialized TRT engine: "
            << serialized_segment_.capacity();
    string tmp;
    // Swap with temporary empty string to deallocate the CPU memory.
    serialized_segment_.swap(tmp);
    if (use_implicit_batch_ && (max_batch_size < batch_size)) {
      return &empty_context;
    }
    return cache.at(engine_input_shapes).get();
  }  // static_engine_

  // Find a shape from cache that is compatible with input_shapes
  // TODO(Tamas): if the input_shapes are the concrete shapes (knows shapes, without -1),
  // then we should look up the engine based on optimization profile ranges.
  std::vector<TensorShape> engine_input_shapes;
  TF_RETURN_IF_ERROR(
      GetEngineInputShapes(cache, input_shapes, &engine_input_shapes));

  // If cache does not have an engine with engine_input_shapes,
  // then create a new engine.
  if (!cache.count(engine_input_shapes)) {
    TrtUniquePtrType<nvinfer1::ICudaEngine> engine;
    bool convert_successfully = false;
    LOG(INFO) << "Building a new TensorRT engine for " << name()
              << " with input shapes: "
              << TensorShapeUtils::ShapeListString(engine_input_shapes);

    // In case we did not collect any shape information so far,
    // then we add the current input shape so that we can create an engine for that.
    cache_res->profiles_.addShapeIfEmpty(engine_input_shapes);

    // Convert to partial shapes.
    std::vector<PartialTensorShape> partial_shapes(engine_input_shapes.begin(),
                                                   engine_input_shapes.end());

    // Up to this point, calibrator_ can never be empty, since otherwise it
    // means calibration_mode_ is true and this path won't get executed.
    auto status = convert::ConvertGraphDefToEngine(
        segment_graph_def_, precision_mode_, batch_size, workspace_size_,
        partial_shapes, &logger, allocator, calibrator_.get(), &engine,
        use_calibration_, use_implicit_batch_, &convert_successfully,
        cache_res->profiles_);
    if (!status.ok()) {
      LOG(WARNING) << "Engine creation for " << name() << " failed. "
                   << "The native segment will be used instead. "
                   << "Reason: " << status;
      // Store an empty engine in the cache for these input shapes so we don't
      // try to build the same failing engine again.
      cache.emplace(engine_input_shapes, absl::make_unique<EngineContext>());
      return &empty_context;
    }
    std::vector<TrtUniquePtrType<nvinfer1::IExecutionContext>> exec_context;
    cache_res->profiles_.createExcecutionContexts(engine.get(), exec_context);
    cache.emplace(engine_input_shapes,
                  absl::make_unique<EngineContext>(std::move(engine),
                                                   std::move(exec_context)));
    VLOG(1) << "Added new engine to cache of " << name()
            << ". Cache size: " << cache.size();
  }
  return cache.at(engine_input_shapes).get();
}

// TODO(hinsu): Move this allocation to CalibrationContext constructor, if
// possible.
Status TRTEngineOp::AllocateCalibrationResources(
    OpKernelContext* ctx, TRTEngineCacheResource* cache_res) {
  cache_res->calib_ctx_ = absl::make_unique<CalibrationContext>();
  auto* cres = cache_res->calib_ctx_.get();

  // Get the input shapes.
  const int batch_size = ctx->input(0).dim_size(0);
  const int num_inputs = ctx->num_inputs();
  std::vector<TensorShape> shapes;
  cres->device_tensors_.resize(num_inputs);
  VLOG(1) << "Constructing calibrator";
  for (int i = 0; i < num_inputs; i++) {
    // allocate workspace on device for inputs
    const Tensor& t = ctx->input(i);
    shapes.emplace_back(t.shape());
    Tensor* device_tensor;
    TF_RETURN_IF_ERROR(ctx->allocate_persistent(
        t.dtype(), t.shape(), &cres->device_tensors_.at(i), &device_tensor));
    CHECK_EQ(t.TotalBytes(), device_tensor->TotalBytes());
    void* device_address = GetTensorAddress(device_tensor);
    if (device_address == nullptr) {
      return errors::InvalidArgument(
          "Unsupported data type encountered in input ", i);
    }
    cres->device_buffers_.emplace(
        StrCat(IONamePrefixes::kInputPHName, i),
        std::pair<void*, size_t>(device_address, device_tensor->TotalBytes()));
  }
  cres->calibrator_.reset(
      new TRTInt8Calibrator(cres->device_buffers_, batch_size, name()));
  const int platform_gpu_id =
      ctx->device()->tensorflow_gpu_device_info()->gpu_id;
  if (platform_gpu_id < 0) {
    LOG(ERROR) << "Can't get gpu_device_info from context->device()";
    return errors::InvalidArgument(
        "Context->device doesn't contain device info!");
  }

  cache_res->Ref();
  cres->thr_.reset(new std::thread([this, cres, shapes, platform_gpu_id,
                                    cache_res]() {
    core::ScopedUnref sc(cache_res);

    VLOG(1) << "Starting calibration thread on device " << platform_gpu_id
            << ", Calibration Resource @ " << cres;
    auto err = cudaSetDevice(platform_gpu_id);
    if (err != cudaSuccess) {
      // TODO(aaroey): should return error here.
      LOG(ERROR) << "Couldn't set cuda device to " << platform_gpu_id
                 << " in calibration thread";
    }
    std::vector<PartialTensorShape> partial_shapes(shapes.begin(),
                                                   shapes.end());
    // ConvertGraphDefToEngine() will try to build the engine. This thread
    // will loop inside buildCudaEngine() consuming the calibration data
    // that is set by the TF op, and drive the builder until calibrator
    // returns false. Engine is discarded after calibration table is
    // generated
    //
    // TODO(aaroey): maybe setting the max batch size using the python
    // calibration wrapper class.
    auto s = convert::ConvertGraphDefToEngine(
        this->segment_graph_def_, TrtPrecisionMode::INT8,
        cres->calibrator_->getBatchSize(), this->workspace_size_,
        partial_shapes, &cache_res->GetLogger(), cache_res->allocator_.get(),
        cres->calibrator_.get(), &cres->engine_,
        /*use_calibration=*/true,
        this->use_implicit_batch_,
        /*convert_successfully=*/nullptr, cache_res->profiles_);
    if (!s.ok()) {
      LOG(ERROR) << "Calibration failed: " << s;
      cres->calibrator_->setDone();  // Ignore further pushes
    } else {
      // Transfer the ownership of the engine to the engine cache, so we can
      // dump it out during conversion for TF 2.0.
      mutex_lock lock(this->engine_mutex_);
      this->calibrator_ = std::move(cres->calibrator_);
      TrtUniquePtrType<nvinfer1::IExecutionContext> exec_context(
          cres->engine_->createExecutionContext());
      cache_res->cache_.emplace(
          shapes, absl::make_unique<EngineContext>(std::move(cres->engine_),
                                                   std::move(exec_context)));
    }

    VLOG(1) << "Calibration loop terminated " << this->name();
  }));
  VLOG(1) << "initialized calibrator resource";
  return Status::OK();
}

REGISTER_KERNEL_BUILDER(Name("TRTEngineOp").Device(DEVICE_GPU), TRTEngineOp);

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
