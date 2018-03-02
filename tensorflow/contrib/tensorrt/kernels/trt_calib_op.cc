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
#include "tensorflow/contrib/tensorrt/resources/trt_int8_calibrator.h"
#include "tensorflow/contrib/tensorrt/resources/trt_resource_manager.h"
#include "tensorflow/contrib/tensorrt/resources/trt_resources.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/stream_executor.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "cuda_runtime_api.h"
#include "tensorrt/include/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

TRTCalibOp::TRTCalibOp(OpKernelConstruction* context) : OpKernel(context) {
  OP_REQUIRES_OK(context, context->GetAttr("segment_nodes", &segment_nodes_));
  OP_REQUIRES_OK(context, context->GetAttr("input_names", &input_names_));
  OP_REQUIRES_OK(context, context->GetAttr("resource_name", &resource_name_));
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

void TRTCalibOp::Compute(tensorflow::OpKernelContext* ctx) {
  // TODO(aaroey): make sure ctx->resource_mgr() is used in future PR.
  auto trt_rm = tensorflow::tensorrt::TRTResourceManager::instance();
  auto res_mgr = trt_rm->getManager("TRTCalibOps");
  tensorflow::tensorrt::TRTCalibrationResource* calib_res = nullptr;
  auto status = res_mgr->Lookup(resource_name_, resource_name_, &calib_res);

  if (!status.ok()) {
    ctx->SetStatus(status);
    return;
  }
  int num_inputs = ctx->num_inputs();
  // first run instantiate calibrator
  if (calib_res->calibrator_ == nullptr) {
    dev_tensors_.resize(num_inputs);
    int batch_size = ctx->input(0).dim_size(0);
    VLOG(1) << " Constructing calibrator";
    for (int i = 0; i < num_inputs; i++) {
      // allocate workspace on device for inputs
      const tensorflow::Tensor& t = ctx->input(i);
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_persistent(t.dtype(), t.shape(),
                                              &dev_tensors_.at(i), nullptr));
      const auto device_tensor = dev_tensors_.at(i).AccessTensor(ctx);
      CHECK_EQ(t.TotalBytes(), device_tensor->TotalBytes());
      void* device_address = GetTensorAddress(device_tensor);
      device_buffers_.emplace(input_names_.at(i),
                              std::pair<void*, size_t>(
                                  device_address, device_tensor->TotalBytes()));
    }

    calib_res->calibrator_ =
        new TRTInt8Calibrator(device_buffers_, batch_size, resource_name_);
    string label(resource_name_);
    calib_res->thr_ = new std::thread([calib_res, label]() {
      VLOG(1) << "Starting calibration thread, Calibration Resource @ "
              << calib_res;
      calib_res->builder_->setInt8Calibrator(calib_res->calibrator_);
      calib_res->builder_->setInt8Mode(true);
      calib_res->engine_ = calib_res->builder_->buildCudaEngine(
          *calib_res->network_);  // will loop until we terminate calibrator
      VLOG(1) << "Calibration loop terminated " << label;
    });
    VLOG(1) << "initialized calibrator resource";
  }  //  calibrator initialized

  // Pass input data to calibrator
  std::unordered_map<string, void*> input_data;
  for (int i = 0; i < num_inputs; i++) {
    const Tensor& t = ctx->input(i);
    void* data_address = GetTensorAddress(&t);
    const auto device_tensor = dev_tensors_.at(i).AccessTensor(ctx);
    CHECK_EQ(t.TotalBytes(),
             device_tensor->TotalBytes());  // use the tensor so FW keeps it
    input_data.emplace(input_names_.at(i), data_address);
    ctx->set_output(i, t);
  }
  VLOG(2) << "Filled map for sending";
  // copied from cuda_kernel_helper since it seems only valid in *.cu.cc files
  const cudaStream_t* stream = CHECK_NOTNULL(
      reinterpret_cast<const cudaStream_t*>(ctx->op_device_context()
                                                ->stream()
                                                ->implementation()
                                                ->CudaStreamMemberHack()));
  calib_res->calibrator_->setBatch(input_data,*stream);
  VLOG(2) << "Passed calibration data";
  // TODO(aaroey): make sure we wait for the completion of calibration on the
  // last batch in future PR.
};

#undef TYPECASE

REGISTER_KERNEL_BUILDER(Name("TRTCalibOp").Device(DEVICE_GPU), TRTCalibOp);

}  // namespace tensorrt
}  // namespace tensorflow
#endif
#endif
