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
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/plugin/trt_plugin_factory.h"
#include "tensorflow/compiler/tf2tensorrt/utils/test_utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_allocator.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_resource_manager.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_resources.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "cuda/include/cuda_runtime_api.h"
#include "tensorrt/include/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
static Logger logger;
using absl::StrAppend;
using absl::StrCat;
using ::nvinfer1::IRuntime;

// A helper class to call done() when destructed for asynchronous execution.
// Helps simultaneous execution of native and TRT engines.
class AsyncHelper : public tensorflow::core::RefCounted {
 public:
  AsyncHelper(AsyncOpKernel::DoneCallback done) { done_ = done; }
  ~AsyncHelper() override { done_(); }

 private:
  AsyncOpKernel::DoneCallback done_;
};

//  This OP can construct TRTEngine on the fly and if construction of engine
//  fails, executes equivalent subgraph as a TensorFlow function.
class TRTEngineOp : public AsyncOpKernel {
 public:
  explicit TRTEngineOp(OpKernelConstruction* context);

  void ComputeAsync(OpKernelContext* context,
                    AsyncOpKernel::DoneCallback done) override;

 private:
  // Execute calibration
  void ExecuteCalibration(OpKernelContext* ctx, AsyncHelper* helper);

  // Construct a function handle for executing native funcdef graph
  Status ConstructFunctionHandle(OpKernelContext* ctx);

  // Execute replaced native segment as function Op.
  void ExecuteNativeSegment(OpKernelContext* ctx, AsyncHelper* helper);

  // Execute the tensorrt engine. Returns whether we need to retry by running
  // the native segment.
  bool ExecuteTrtEngine(OpKernelContext* ctx, EngineContext* engine_context);

  // Allocate necessary resources for calibration
  Status AllocateCalibrationResources(OpKernelContext* ctx,
                                      SerializableResourceBase** cr);

  // Get engine for the input shape
  EngineContext* GetEngine(const std::vector<TensorShape>& input_shapes,
                           OpKernelContext* ctx);

  // Return engine batch in cached_engne_batch_sizes_ which is closest to input
  // batch.
  bool GetCompatibleCachedEngine(
      const std::vector<TensorShape>& actual_input_shapes,
      std::vector<TensorShape>* engine_input_shapes);

  std::vector<string> input_nodes_;
  std::vector<string> output_nodes_;

  // serialized protobuf segment or trt engine depending on static_engine_ flag.
  string serialized_segment_;

  // Name of the function for TF native execution of the segment.
  string funcdef_name_;

  // GraphDef representation of the segment.
  GraphDef segment_graph_;

  // Engine Precision mode.
  TrtPrecisionMode precision_mode_;

  // Whether engine is constructed during the conversion or needs to be
  // constructed from protobuf segment.
  bool static_engine_;

  // Whether to calibrate INT8 engine.
  bool calibration_mode_;

  // Batches of the cached engines
  std::vector<int> cached_engine_batches_;

  // Maximum number of cached engines
  int max_cached_engines_;

  int64 workspace_size_;
  mutex engine_mutex_;
  FunctionLibraryRuntime::Handle native_func_;

  // The finalized calibrator for inference.
  std::unique_ptr<TRTInt8Calibrator> calibrator_;

  // If true, create calibration graph for INT8 mode. Otherwise, we are using
  // user-provided quantization ranges.
  bool use_calibration_;
};

#define TYPECASE(dt, X, Y)                                                \
  case dt: {                                                              \
    return (void*)X->flat<tensorflow::EnumToDataType<dt>::Type>().data(); \
  }

void* GetTensorAddress(const Tensor* tensor_ptr) {
  auto tensor_type = tensor_ptr->dtype();
  switch (tensor_type) {
    TYPECASE(tensorflow::DT_FLOAT, tensor_ptr, dest_ptr);
    TYPECASE(tensorflow::DT_HALF, tensor_ptr, dest_ptr);
    TYPECASE(tensorflow::DT_INT8, tensor_ptr, dest_ptr);
    default: {
      LOG(ERROR) << "Unsupported Data type "
                 << tensorflow::DataTypeString(tensor_type);
      return nullptr;
    }
  }
}

tensorflow::Status TRTEngineOp::ConstructFunctionHandle(OpKernelContext* ctx) {
  VLOG(1) << "Constructing function handle";
  auto lib = ctx->function_library();
  if (lib == nullptr) {
    return tensorflow::errors::Internal("Context function library is null");
  }
  auto fdef = lib->GetFunctionLibraryDefinition()->Find(funcdef_name_);
  if (fdef == nullptr) {
    return tensorflow::errors::Internal("Native FunctionDef ", funcdef_name_,
                                        " can't be found in function library");
  }
  tensorflow::FunctionLibraryRuntime::InstantiateOptions inst_ops;
  inst_ops.overlay_lib = nullptr;
  inst_ops.state_handle = "";
  inst_ops.target = ctx->device()->name();
  native_func_ = 0;
  auto status = lib->Instantiate(funcdef_name_, AttrSlice(&fdef->attr()),
                                 inst_ops, &native_func_);
  if (!status.ok()) {
    LOG(ERROR) << " Instantiating native function " << funcdef_name_
               << " failed!";
  }
  return status;
}

TRTEngineOp::TRTEngineOp(OpKernelConstruction* context)
    : AsyncOpKernel(context) {
  // read serialized_engine
  OP_REQUIRES_OK(context,
                 context->GetAttr("serialized_segment", &serialized_segment_));
  OP_REQUIRES_OK(context,
                 context->GetAttr("workspace_size_bytes", &workspace_size_));
  OP_REQUIRES_OK(context, context->GetAttr("static_engine", &static_engine_));
  if (!static_engine_) {
    if (!segment_graph_.ParseFromString(serialized_segment_)) {
      LOG(ERROR) << "Parsing segment graph failed!";
      context->SetStatus(tensorflow::errors::InvalidArgument(
          "Failed to parse segment graphdef!"));
      return;
    }
    serialized_segment_.resize(0);
  }
  VLOG(1) << "Constructing " << name();
  string precision_string;
  OP_REQUIRES_OK(context,
                 context->GetAttr("precision_mode", &precision_string));
  string calibration_data;
  OP_REQUIRES_OK(context,
                 context->GetAttr("calibration_data", &calibration_data));
  OP_REQUIRES_OK(context,
                 context->GetAttr("segment_funcdef_name", &funcdef_name_));
  OP_REQUIRES_OK(context,
                 TrtPrecisionModeFromName(precision_string, &precision_mode_));
  OP_REQUIRES_OK(context,
                 context->GetAttr("use_calibration", &use_calibration_));
  calibration_mode_ =
      (use_calibration_ && precision_mode_ == TrtPrecisionMode::INT8 &&
       calibration_data.size() == 0);
  if (calibration_data.size()) {
    calibrator_.reset(new TRTInt8Calibrator(calibration_data));
    calibration_data.resize(0);
  }
  native_func_ = tensorflow::kInvalidHandle;
  OP_REQUIRES_OK(context, context->GetAttr("max_cached_engines_count",
                                           &max_cached_engines_));
  OP_REQUIRES_OK(context, context->GetAttr("cached_engine_batches",
                                           &cached_engine_batches_));
  std::sort(cached_engine_batches_.begin(), cached_engine_batches_.end());
  if (VLOG_IS_ON(1)) {
    string s("Engine Batches= ");
    for (auto i : cached_engine_batches_) {
      StrAppend(&s, i, " ");
    }
    VLOG(1) << s;
  }
}

void TRTEngineOp::ExecuteNativeSegment(OpKernelContext* ctx,
                                       AsyncHelper* helper) {
  std::vector<Tensor> inputs;
  std::vector<Tensor>* outputs = new std::vector<Tensor>();
  if (native_func_ == tensorflow::kInvalidHandle) {
    auto status = ConstructFunctionHandle(ctx);
    if (!status.ok()) {
      LOG(ERROR) << "Couldn't construct function handle " << funcdef_name_;
      ctx->SetStatus(status);
      return;
    }
  }
  auto lib = ctx->function_library();
  tensorflow::FunctionLibraryRuntime::Options opts;
  opts.step_id = ctx->step_id();
  opts.rendezvous = ctx->rendezvous();
  opts.cancellation_manager = ctx->cancellation_manager();
  opts.runner = ctx->runner();
  for (int i = 0; i < ctx->num_inputs(); i++) {
    inputs.push_back(ctx->input(i));
  }
  helper->Ref();  // Increment count for calculating native graph
  VLOG(1) << "Executing native segment: " << name();
  lib->Run(opts, native_func_, inputs, outputs,
           [this, ctx, outputs, helper](const tensorflow::Status& s) {
             tensorflow::core::ScopedUnref sc(helper);
             if (!s.ok()) {
               LOG(ERROR) << "Failed to execute native segment " << this->name()
                          << ": " << s;
               ctx->SetStatus(s);
               return;
             }
             VLOG(1) << "Native Segment completed";
             for (size_t t = 0; t < outputs->size(); ++t) {
               ctx->set_output(t, outputs->at(t));
             }
             test::AddTestValue(StrCat(this->name(), ":ExecuteNativeSegment"),
                                "done");
             delete outputs;
           });
}

void TRTEngineOp::ExecuteCalibration(OpKernelContext* ctx,
                                     AsyncHelper* helper) {
  VLOG(1) << "Executing TRT calibration: " << name();
  helper->Ref();
  tensorflow::core::ScopedUnref sc(helper);
  auto res_mgr = ctx->resource_manager();
  TRTCalibrationResource* calib_res = nullptr;
  OP_REQUIRES_OK(
      ctx,
      res_mgr->LookupOrCreate(
          "TF_TRT_Calibration", name(),
          reinterpret_cast<SerializableResourceBase**>(&calib_res),
          {[ctx, this](SerializableResourceBase** cr) -> tensorflow::Status {
            return this->AllocateCalibrationResources(ctx, cr);
          }}));
  tensorflow::core::ScopedUnref calib_sc(calib_res);
  // TODO(aaroey): here we also add the resource to the ResourceMgr singleton.
  // This is needed before we migrate all uses of calib_graph_to_infer_graph()
  // to the new calibration workflow. After that we'll remove this block.
  {
    auto deprecated_rm =
        TRTResourceManager::instance()->getManager("TRTCalibration");
    TRTCalibrationResource* copied_resource = nullptr;
    // Check whether the resource exists, and create it if not.
    if (deprecated_rm->Lookup(funcdef_name_, "Calibrator", &copied_resource)
            .ok()) {
      // Do nothing if the resource exists.
      copied_resource->Unref();
    } else {
      copied_resource = calib_res;
      // Increase the refcount by 1 then transfer the ownership of that refcount
      // to the ResourceMgr singleton.
      copied_resource->Ref();
      OP_REQUIRES_OK(ctx, deprecated_rm->Create(funcdef_name_, "Calibrator",
                                                copied_resource));
    }
  }
  int num_inputs = ctx->num_inputs();
  // Pass input data to calibrator
  std::unordered_map<string, void*> input_data;
  for (int i = 0; i < num_inputs; i++) {
    const Tensor& t = ctx->input(i);
    void* data_address = GetTensorAddress(&t);
    if (data_address == nullptr) {
      ctx->SetStatus(tensorflow::errors::InvalidArgument(
          "Unsupported data type encountered in input ", i));
      return;
    }
    // Check the allocated buffer is sufficient for input
    const auto device_tensor =
        calib_res->device_tensors_.at(i).AccessTensor(ctx);
    CHECK_EQ(t.TotalBytes(), device_tensor->TotalBytes());
    input_data.emplace(StrCat(kInputPHName, i), data_address);
  }
  VLOG(2) << "Filled map for sending";
  // copied from cuda_kernel_helper since it seems only valid in *.cu.cc files
  const cudaStream_t* stream = CHECK_NOTNULL(
      reinterpret_cast<const cudaStream_t*>(ctx->op_device_context()
                                                ->stream()
                                                ->implementation()
                                                ->GpuStreamMemberHack()));
  calib_res->calibrator_->setBatch(input_data, *stream);
  test::AddTestValue(StrCat(name(), ":ExecuteCalibration"), "done");
  VLOG(2) << "Passed calibration data";
  ExecuteNativeSegment(ctx, helper);
}

bool TRTEngineOp::GetCompatibleCachedEngine(
    const std::vector<TensorShape>& actual_input_shapes,
    std::vector<TensorShape>* engine_input_shapes) {
  const int batch_size = actual_input_shapes[0].dim_size(0);
  int smallest_batch_size = -1;
  // Output shape will always be the same as the input but we will overwrite the
  // batch size.
  *engine_input_shapes = actual_input_shapes;
  for (const int cached_batch_size : cached_engine_batches_) {
    // Check if compatible: batch <= cached batch.
    //
    // TODO(laigd): here it only compare the first dim a.k.a the batch size,
    // we'll need to to support non-batch dimensions as well. This will be done
    // as part of the offline conversion implementation.
    if (batch_size <= cached_batch_size) {
      // First case: first compatible engine found
      // Second case: smaller batch size engine found
      if ((smallest_batch_size == -1) ||
          (cached_batch_size < smallest_batch_size)) {
        smallest_batch_size = cached_batch_size;
        // Overwrite batch size for output
        for (int i = 0; i < engine_input_shapes->size(); i++) {
          (*engine_input_shapes)[i].set_dim(0, smallest_batch_size);
        }
      }
    }
  }
  return (smallest_batch_size != -1);
}

void TRTEngineOp::ComputeAsync(OpKernelContext* ctx,
                               AsyncOpKernel::DoneCallback done) {
  auto helper = new AsyncHelper(done);
  tensorflow::core::ScopedUnref sc(helper);
  if (calibration_mode_) {
    ExecuteCalibration(ctx, helper);
    return;
  }
  // Get shapes of inputs to engine.
  std::vector<tensorflow::TensorShape> input_shapes;
  for (int i = 0; i < ctx->num_inputs(); ++i) {
    input_shapes.emplace_back(ctx->input(i).shape());
  }
  EngineContext* engine_context = GetEngine(input_shapes, ctx);
  if (!engine_context->cuda_engine) {
    LOG(WARNING) << "Engine retrieval for input shapes: "
                 << TensorShapeUtils::ShapeListString(input_shapes)
                 << " failed. Running native segment for " << name();
    ExecuteNativeSegment(ctx, helper);
    return;
  }
  const bool retry = ExecuteTrtEngine(ctx, engine_context);
  if (retry) {
    LOG(WARNING) << "Failed to execute engine, "
                 << "retrying with native segment for " << name();
    ExecuteNativeSegment(ctx, helper);
    return;
  }
}

bool TRTEngineOp::ExecuteTrtEngine(OpKernelContext* ctx,
                                   EngineContext* engine_context) {
  VLOG(1) << "Executing TRT engine: " << name();
  auto& cuda_engine = engine_context->cuda_engine;
  const bool kRetry = true;
  // All inputs must have the same batch size, so just get it from the first
  // input.
  const int num_batch = ctx->input(0).shape().dim_size(0);
  const int num_binding = ctx->num_inputs() + ctx->num_outputs();
  std::vector<void*> buffers(num_binding);
  for (int i = 0; i < ctx->num_inputs(); i++) {
    const string input_name = StrCat(kInputPHName, i);
    const int binding_index = cuda_engine->getBindingIndex(input_name.c_str());
    if (binding_index == -1) {
      LOG(ERROR) << "Input node not found, at " << input_name;
      return kRetry;
    }

    const Tensor& input_tensor = ctx->input(i);
    const TensorShape& input_shape = input_tensor.shape();
    if (num_batch != input_shape.dim_size(0)) {
      LOG(ERROR) << "Input data has inconsistent batch size: " << num_batch
                 << " vs " << input_shape.dim_size(0);
      return kRetry;
    }
    auto dtype = cuda_engine->getBindingDataType(binding_index);
    switch (dtype) {
      case nvinfer1::DataType::kFLOAT:
        buffers[binding_index] = (void*)(input_tensor.flat<float>().data());
        break;
      case nvinfer1::DataType::kHALF:
        LOG(ERROR) << "FP16 inputs are not supported yet!";
        return kRetry;
      case nvinfer1::DataType::kINT8:
        LOG(ERROR) << "INT8 inputs are not supported yet!";
        return kRetry;
      case nvinfer1::DataType::kINT32:
        buffers[binding_index] = (void*)(input_tensor.flat<int32>().data());
        break;
      default:
        LOG(ERROR) << "Unknown TRT data type: " << int(dtype);
        return kRetry;
    }
  }

  for (int i = 0; i < ctx->num_outputs(); i++) {
    // Create an output tensor
    const string output_name = StrCat(kOutputPHName, i);
    const int binding_index = cuda_engine->getBindingIndex(output_name.c_str());
    Tensor* output_tensor = nullptr;

    TensorShape output_shape;
    if (binding_index != -1) {
      auto dims = cuda_engine->getBindingDimensions(binding_index);
      std::vector<int> trt_shape(dims.nbDims + 1);
      trt_shape[0] = num_batch;
      for (int j = 0; j < dims.nbDims; j++) trt_shape[j + 1] = dims.d[j];
      auto status = TensorShapeUtils::MakeShape(
          trt_shape.data(), trt_shape.size(), &output_shape);
      if (!status.ok()) {
        LOG(ERROR) << "Failed to get output shape: " << status;
        return kRetry;
      }
    } else {
      LOG(ERROR) << "Output node not found, at " << output_name;
      return kRetry;
    }
    auto status = ctx->allocate_output(i, output_shape, &output_tensor);
    if (!status.ok()) {
      LOG(ERROR) << "Allocating output failed with " << status;
      ctx->SetStatus(status);
      // Do not retry since we cannot allocate the same output twice.
      // TODO(aaroey): ideally we should retry, fix this.
      return !kRetry;
    }
    auto dtype = cuda_engine->getBindingDataType(binding_index);
    switch (dtype) {
      case nvinfer1::DataType::kFLOAT:
        buffers[binding_index] =
            reinterpret_cast<void*>(output_tensor->flat<float>().data());
        break;
      case nvinfer1::DataType::kHALF:
        LOG(WARNING) << "half size is not supported yet!";
        return kRetry;
      case nvinfer1::DataType::kINT8:
        LOG(WARNING) << "int8 is not supported yet!";
        return kRetry;
      case nvinfer1::DataType::kINT32:
        buffers[binding_index] =
            reinterpret_cast<void*>(output_tensor->flat<int32>().data());
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

  // nvinfer1::IExecutionContext::enqueue is not thread safe and we need a mutex
  // for it.
  tensorflow::mutex_lock lock(engine_context->mu);
  // TODO(jie): trt enqueue does not return error
  auto ret = engine_context->execution_context->enqueue(num_batch, &buffers[0],
                                                        *stream, nullptr);
  if (!ret) {
    LOG(WARNING) << "Failed to enqueue batch for TRT engine: " << name();
    return kRetry;
  }
  test::AddTestValue(StrCat(name(), ":ExecuteTrtEngine"), "done");
  // Synchronization will be done by TF.
  return !kRetry;
}

EngineContext* TRTEngineOp::GetEngine(
    const std::vector<TensorShape>& input_shapes, OpKernelContext* ctx) {
  static EngineContext empty_context;
  tensorflow::mutex_lock lock(engine_mutex_);
  // TODO(tmorris): using first input to get batch size - is this reliable?
  const int batch_size = input_shapes[0].dim_size(0);

  // Get engine cache
  TRTEngineCacheResource* cache_res = nullptr;
  auto status = ctx->resource_manager()->LookupOrCreate(
      "TRTEngineCache", funcdef_name_, &cache_res,
      {[this, ctx](TRTEngineCacheResource** cr) -> tensorflow::Status {
        *cr = new TRTEngineCacheResource(ctx, this->max_cached_engines_);
        return Status::OK();
      }});
  if (!status.ok()) {
    ctx->SetStatus(status);
    return &empty_context;
  }
  tensorflow::core::ScopedUnref sc(cache_res);
  auto& cache = cache_res->cache_;
  auto allocator = cache_res->allocator_.get();
  if (allocator == nullptr) {
    return &empty_context;
  }

  // Handle the static engine case. For static engines, the cache will have a
  // single element containing the only engine.
  if (static_engine_) {
    if (cache.size()) {
      // Batch size of engine must be >= the input batch size
      // TODO(tmorris): use match compatible function?
      if (cache.begin()->first[0].dim_size(0) >= batch_size) {
        return cache.begin()->second.get();
      }
      return &empty_context;
    }

    TrtUniquePtrType<IRuntime> infer(nvinfer1::createInferRuntime(logger));
    infer->setGpuAllocator(allocator);
    TrtUniquePtrType<nvinfer1::ICudaEngine> static_engine(
        infer->deserializeCudaEngine(serialized_segment_.c_str(),
                                     serialized_segment_.size(),
                                     PluginFactoryTensorRT::GetInstance()));
    auto raw_static_engine = static_engine.get();
    const auto max_batch_size = raw_static_engine->getMaxBatchSize();
    // Static engine will have max_batch_size for batch size so that all inputs
    // will map to this single engine.
    std::vector<TensorShape> engine_input_shapes(input_shapes);
    for (int i = 0; i < engine_input_shapes.size(); i++) {
      // TODO(tmorris): will all inputs have batch size as first dimension??
      engine_input_shapes[i].set_dim(0, max_batch_size);
    }
    // TODO(laigd): here we assume engine_input_shapes matches the actual input
    // shapes of the engine, we should verify that.
    cache.emplace(engine_input_shapes,
                  absl::make_unique<EngineContext>(
                      std::move(static_engine),
                      TrtUniquePtrType<nvinfer1::IExecutionContext>(
                          raw_static_engine->createExecutionContext())));
    // Runtime is safe to delete after engine creation
    serialized_segment_.clear();
    if (max_batch_size < batch_size) {
      return &empty_context;
    }
    return cache.at(engine_input_shapes).get();
  }  // static_engine_

  // Handle the dynamic engine case.
  // See if there is a compatible engine cached. The batch size should be <= the
  // cached batch size.
  std::vector<tensorflow::TensorShape> engine_input_shapes;
  const bool matched_successfully =
      GetCompatibleCachedEngine(input_shapes, &engine_input_shapes);
  // If matched, use that engine. Otherwise, we will look in cache for that
  // exact shape and possibly create a new engine if it is not in cache.
  if (!matched_successfully) {
    engine_input_shapes = input_shapes;
    if (!cached_engine_batches_.empty()) {
      // If user has explicitly defined cached_engine_batches, we should
      // warn them that their input was non-compatible (batch size too high)
      LOG(WARNING) << "No compatible cached engine was found for batch size: "
                   << batch_size << ". A new engine will be created.";
      cached_engine_batches_.push_back(batch_size);
    }
  }

  if (!cache.count(engine_input_shapes)) {
    TrtUniquePtrType<nvinfer1::ICudaEngine> engine;
    bool convert_successfully = false;
    LOG(INFO) << "Building a new TensorRT engine for " << name()
              << " input shapes: "
              << TensorShapeUtils::ShapeListString(engine_input_shapes);
    // Convert to partial shapes
    std::vector<PartialTensorShape> partial_shapes;
    for (int i = 0; i < engine_input_shapes.size(); i++) {
      partial_shapes.emplace_back(engine_input_shapes[i]);
    }
    // Up to this point, calibrator_ can never be empty, since otherwise it
    // means calibration_mode_ is true and this path won't get executed.
    auto status = convert::ConvertGraphDefToEngine(
        segment_graph_, precision_mode_, batch_size, workspace_size_,
        partial_shapes, &logger, allocator, calibrator_.get(), &engine,
        use_calibration_, &convert_successfully);
    if (!status.ok()) {
      if (convert_successfully) {
        // This means it fail to build the engine even when the network is built
        // successfully, probably due to internal issues. In this case we don't
        // retry in the future.
        cache.emplace(engine_input_shapes, absl::make_unique<EngineContext>());
      }
      LOG(WARNING) << "Engine creation for batch size " << batch_size
                   << " failed " << status;
      return &empty_context;
    }
    VLOG(1) << "Conversion is done";
    TrtUniquePtrType<nvinfer1::IExecutionContext> exec_context(
        engine->createExecutionContext());
    cache.emplace(engine_input_shapes,
                  absl::make_unique<EngineContext>(std::move(engine),
                                                   std::move(exec_context)));
  }
  return cache.at(engine_input_shapes).get();
}

tensorflow::Status TRTEngineOp::AllocateCalibrationResources(
    OpKernelContext* ctx, SerializableResourceBase** cr) {
  auto cres = new TRTCalibrationResource();
  *cr = cres;
  // Get the allocator.
  auto alloc = ctx->device()->GetAllocator(tensorflow::AllocatorAttributes());
  if (!alloc) {
    LOG(WARNING) << "Can't get device allocator will not be able to "
                    "allocate memory from TensorFlow memory pool";
    cres->allocator_.reset(new TRTCudaAllocator);
  } else {
    cres->allocator_.reset(new TRTDeviceAllocator(alloc));
  }
  // Get the input shapes.
  const int batch_size = ctx->input(0).dim_size(0);
  const int num_inputs = ctx->num_inputs();
  std::vector<tensorflow::PartialTensorShape> shapes;
  cres->device_tensors_.resize(num_inputs);
  VLOG(1) << " Constructing calibrator";
  for (int i = 0; i < num_inputs; i++) {
    // allocate workspace on device for inputs
    const tensorflow::Tensor& t = ctx->input(i);
    shapes.emplace_back(t.shape());
    Tensor* device_tensor;
    TF_RETURN_IF_ERROR(ctx->allocate_persistent(
        t.dtype(), t.shape(), &cres->device_tensors_.at(i), &device_tensor));
    CHECK_EQ(t.TotalBytes(), device_tensor->TotalBytes());
    void* device_address = GetTensorAddress(device_tensor);
    if (device_address == nullptr) {
      return tensorflow::errors::InvalidArgument(
          "Unsupported data type encountered in input ", i);
    }
    cres->device_buffers_.emplace(
        StrCat(kInputPHName, i),
        std::pair<void*, size_t>(device_address, device_tensor->TotalBytes()));
  }
  cres->calibrator_.reset(
      new TRTInt8Calibrator(cres->device_buffers_, batch_size, name()));
  const string label(name());
  auto segment_graph = &segment_graph_;
  const int platform_gpu_id =
      ctx->device()->tensorflow_gpu_device_info()->gpu_id;
  if (platform_gpu_id < 0) {
    LOG(ERROR) << "Can't get gpu_device_info from context->device()";
    return tensorflow::errors::InvalidArgument(
        "Context->device doesn't contain device info!");
  }
  const int64 workspace_size_bytes = workspace_size_;
  cres->thr_.reset(new std::thread([cres, label, segment_graph, shapes,
                                    platform_gpu_id, workspace_size_bytes]() {
    LOG(INFO) << "Starting calibration thread on device " << platform_gpu_id
              << ", Calibration Resource @ " << cres;
    auto err = cudaSetDevice(platform_gpu_id);
    if (err != cudaSuccess) {
      // TODO(aaroey): should return error here.
      LOG(ERROR) << "Couldn't set cuda device to " << platform_gpu_id
                 << " in calibration thread";
    }
    // ConvertGraphDefToEngine() will try to build the engine. This thread
    // will loop inside buildCudaEngine() consuming the calibration data
    // that is set by the TF op, and drive the builder until calibrator returns
    // false. Engine is discarded after calibration table is generated
    //
    // TODO(aaroey): maybe setting the max batch size using the python
    // calibration wrapper class.
    auto s = convert::ConvertGraphDefToEngine(
        *segment_graph, TrtPrecisionMode::INT8,
        cres->calibrator_->getBatchSize(), workspace_size_bytes, shapes,
        &cres->logger_, cres->allocator_.get(), cres->calibrator_.get(),
        &cres->engine_,
        /*use_calibration=*/true,
        /*convert_successfully=*/nullptr);
    if (!s.ok()) {
      LOG(ERROR) << "Calibration failed: " << s;
      cres->calibrator_->setDone();  // Ignore further pushes
    }
    VLOG(1) << "Calibration loop terminated " << label;
  }));
  VLOG(1) << "initialized calibrator resource";
  return tensorflow::Status::OK();
}

REGISTER_KERNEL_BUILDER(Name("TRTEngineOp").Device(DEVICE_GPU), TRTEngineOp);

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
