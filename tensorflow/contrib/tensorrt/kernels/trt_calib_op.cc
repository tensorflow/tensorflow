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

#include "tensorflow/contrib/tensorrt/kernels/trt_calib_op.h"
#include "tensorflow/contrib/tensorrt/convert/convert_nodes.h"
#include "tensorflow/contrib/tensorrt/resources/trt_int8_calibrator.h"
#include "tensorflow/contrib/tensorrt/resources/trt_resource_manager.h"
#include "tensorflow/contrib/tensorrt/resources/trt_resources.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/stream_executor.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "cuda/include/cuda_runtime_api.h"
#include "tensorrt/include/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

using tensorflow::strings::StrAppend;
using tensorflow::strings::StrCat;
// Helpers from function_test.cc

Status GetOpSig(const string& op, const OpDef** sig) {
  return OpRegistry::Global()->LookUpOpDef(op, sig);
}

// tensorflow::AttrSlice AttrSliceHelper(
//     const std::vector<
//         std::pair<string, tensorflow::FunctionDefHelper::AttrValueWrapper>>&
//         attrs) {
//   tensorflow::AttrValueMap map_;
//   for (const auto& aval : attrs) {
//     map_.insert({aval.first, aval.second.proto});
//   }
//   return tensorflow::AttrSlice(&map_);
// }

TRTCalibOp::TRTCalibOp(OpKernelConstruction* context) : AsyncOpKernel(context) {
  string serialized_segment;
  OP_REQUIRES_OK(context,
                 context->GetAttr("serialized_segment", &serialized_segment));
  if (!segment_graph_.ParseFromString(serialized_segment)) {
    LOG(ERROR) << "Parsing segment graph failed!";
    context->SetStatus(tensorflow::errors::InvalidArgument(
        "Failed to parse segment graphdef!"));
    return;
  }
  serialized_segment.resize(0);
  OP_REQUIRES_OK(context, context->GetAttr("workspace_size_bytes", &workspace_size_));
  OP_REQUIRES_OK(context, context->GetAttr("segment_funcdef_name", &resource_name_));
  auto lib = context->function_library();
  OP_REQUIRES(context, lib != nullptr,
              tensorflow::errors::Internal("Context function library is null"));
  auto fdef = lib->GetFunctionLibraryDefinition()->Find(resource_name_);
  OP_REQUIRES(context, fdef != nullptr,
              tensorflow::errors::Internal(
                  StrCat("Native FunctionDef ", resource_name_,
                         " can't be found in function library")));
  tensorflow::FunctionLibraryRuntime::InstantiateOptions inst_ops;
  inst_ops.overlay_lib = nullptr;
  inst_ops.state_handle = "";
  inst_ops.target = context->device()->name();
  native_func_ = 0;
  OP_REQUIRES_OK(context,
                 lib->Instantiate(resource_name_, AttrSlice(&fdef->attr()),
                                  inst_ops, &native_func_));
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
      LOG(FATAL) << "Unsupported Data type "
                 << tensorflow::DataTypeString(tensor_type);
      return nullptr;
    }
  }
}
tensorflow::Status TRTCalibOp::AllocateCalibrationResources(
    tensorflow::OpKernelContext* ctx,
    tensorflow::tensorrt::TRTCalibrationResource** cr) {
  auto cres = new TRTCalibrationResource();
  *cr = cres;
  cres->logger_ = new tensorflow::tensorrt::Logger();
  cres->builder_ = nvinfer1::createInferBuilder(*(cres->logger_));
#if NV_TENSORRT_MAJOR > 3
  auto dev = ctx->device();
  auto dev_allocator = dev->GetAllocator(tensorflow::AllocatorAttributes());
  if (!dev_allocator) {
    LOG(WARNING) << "Can't get device allocator will not be able to "
                    "allocate memory from TensorFlow memory pool";
    cres->allocator_ =
        std::make_shared<tensorflow::tensorrt::TRTCudaAllocator>();
  } else {
    cres->allocator_ =
        std::make_shared<tensorflow::tensorrt::TRTDeviceAllocator>(
            dev_allocator);
  }
  cres->builder_->setGpuAllocator(cres->allocator_.get());
#endif
  int batch_size = ctx->input(0).dim_size(0);
  cres->builder_->setMaxBatchSize(batch_size);
  cres->builder_->setInt8Mode(true);
  cres->builder_->setMaxWorkspaceSize(workspace_size_);
  cres->engine_ = nullptr;
  std::vector<tensorflow::PartialTensorShape> shapes;
  int num_inputs = ctx->num_inputs();
  // first run instantiate calibrator
  dev_tensors_.resize(num_inputs);
  VLOG(1) << " Constructing calibrator";
  for (int i = 0; i < num_inputs; i++) {
    // allocate workspace on device for inputs
    const tensorflow::Tensor& t = ctx->input(i);
    shapes.emplace_back(t.shape());
    TF_RETURN_IF_ERROR(ctx->allocate_persistent(t.dtype(), t.shape(),
                                                &dev_tensors_.at(i), nullptr));
    const auto device_tensor = dev_tensors_.at(i).AccessTensor(ctx);
    CHECK_EQ(t.TotalBytes(), device_tensor->TotalBytes());
    void* device_address = GetTensorAddress(device_tensor);
    device_buffers_.emplace(
        StrCat("InputPH_", i),
        std::pair<void*, size_t>(device_address, device_tensor->TotalBytes()));
  }
  cres->calibrator_ =
      new TRTInt8Calibrator(device_buffers_, batch_size, name());
  cres->builder_->setInt8Calibrator(cres->calibrator_);
  string label(name());
  auto segment_graph = &segment_graph_;
  cres->thr_ = new std::thread([cres, label, segment_graph, shapes]() {
    VLOG(1) << "Starting calibration thread, Calibration Resource @ " << cres;
    auto s = tensorflow::tensorrt::convert::ConvertSubgraphToEngine(
        *segment_graph, cres->builder_, shapes, &cres->engine_,
        tensorflow::tensorrt::convert::INT8MODE);  // will loop until we
                                                   // terminate calibration
    if (!s.ok()) {
      LOG(ERROR) << "Calibration thread failed with " << s;
    }
    VLOG(1) << "Calibration loop terminated " << label;
  });
  VLOG(1) << "initialized calibrator resource";
  return tensorflow::Status::OK();
}

// Helper Class for ComputeAsync()

class AsyncHelper : public tensorflow::core::RefCounted {
 public:
  AsyncHelper(tensorflow::AsyncOpKernel::DoneCallback done){ done_ = done; }
  ~AsyncHelper() override { done_(); }

 private:
  tensorflow::AsyncOpKernel::DoneCallback done_;
};

void TRTCalibOp::ComputeAsync(tensorflow::OpKernelContext* ctx,
                              tensorflow::AsyncOpKernel::DoneCallback done) {
  // TODO(aaroey): make sure ctx->resource_mgr() is used in future PR.
  auto res_mgr = ctx->resource_manager();
  tensorflow::tensorrt::TRTCalibrationResource* calib_res = nullptr;
  std::function<tensorflow::Status(
      tensorflow::tensorrt::TRTCalibrationResource**)>
      f = [ctx, this](tensorflow::tensorrt::TRTCalibrationResource** cr)
      -> tensorflow::Status {
    return this->AllocateCalibrationResources(ctx, cr);
  };
  auto status = res_mgr->LookupOrCreate(
      name(), "Calibrator", &calib_res,
      {[ctx, this](tensorflow::tensorrt::TRTCalibrationResource** cr)
           -> tensorflow::Status {
        return this->AllocateCalibrationResources(ctx, cr);
      }});

  std::vector<Tensor> inputs;
  std::vector<Tensor>* outputs = new std::vector<Tensor>();
  auto lib = ctx->function_library();
  tensorflow::FunctionLibraryRuntime::Options opts;
  opts.step_id = ctx->step_id();
  opts.rendezvous = ctx->rendezvous();
  opts.cancellation_manager = ctx->cancellation_manager();
  opts.runner = ctx->runner();
  for (int i = 0; i < ctx->num_inputs(); i++) {
    inputs.push_back(ctx->input(i));
  }
  auto ah = new AsyncHelper(done);
  tensorflow::core::ScopedUnref SC(ah);
  ah->Ref();  // Increment count for calculating native graph
  lib->Run(opts, native_func_, inputs, outputs,
           [ctx, outputs, ah](const tensorflow::Status& s) {
             if (!s.ok()) {
               ctx->SetStatus(s);
               ah->Unref();
               return;
             }
             for (size_t t = 0; t < outputs->size(); ++t) {
               ctx->set_output(t, outputs->at(t));
             }
             delete outputs;
             ah->Unref();
           });
  if (!status.ok()) {
    ctx->SetStatus(status);
    return;
  }
  int num_inputs = ctx->num_inputs();
  // Pass input data to calibrator
  std::unordered_map<string, void*> input_data;
  for (int i = 0; i < num_inputs; i++) {
    const Tensor& t = ctx->input(i);
    void* data_address = GetTensorAddress(&t);
    const auto device_tensor = dev_tensors_.at(i).AccessTensor(ctx);
    CHECK_EQ(t.TotalBytes(),
             device_tensor->TotalBytes());  // use the tensor so FW keeps it
    input_data.emplace(StrCat("InputPH_",i), data_address);
  }
  VLOG(2) << "Filled map for sending";
  // copied from cuda_kernel_helper since it seems only valid in *.cu.cc files
  const cudaStream_t* stream = CHECK_NOTNULL(
      reinterpret_cast<const cudaStream_t*>(ctx->op_device_context()
                                                ->stream()
                                                ->implementation()
                                                ->CudaStreamMemberHack()));
  ah->Ref();  // Increment count for calculating calibration data
  calib_res->calibrator_->setBatch(input_data, *stream, ah);
  VLOG(2) << "Passed calibration data";
};

#undef TYPECASE

REGISTER_KERNEL_BUILDER(Name("TRTCalibOp").Device(DEVICE_GPU), TRTCalibOp);

}  // namespace tensorrt
}  // namespace tensorflow
#endif
#endif
