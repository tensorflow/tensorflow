/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/cl/cl_kernel.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_program.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

Status GetKernelMaxWorkGroupSize(cl_kernel kernel, cl_device_id device_id,
                                 int* result) {
  size_t max_work_group_size;
  cl_int error_code =
      clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE,
                               sizeof(size_t), &max_work_group_size, nullptr);
  if (error_code != CL_SUCCESS) {
    return UnknownError(
        absl::StrCat("Failed to get info CL_KERNEL_WORK_GROUP_SIZE ",
                     CLErrorCodeToString(error_code)));
  }
  *result = static_cast<int>(max_work_group_size);
  return OkStatus();
}

Status GetKernelPrivateMemorySize(cl_kernel kernel, cl_device_id device_id,
                                  int* result) {
  cl_ulong private_mem_size;
  cl_int error_code =
      clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_PRIVATE_MEM_SIZE,
                               sizeof(cl_ulong), &private_mem_size, nullptr);
  if (error_code != CL_SUCCESS) {
    return UnknownError(
        absl::StrCat("Failed to get info CL_KERNEL_PRIVATE_MEM_SIZE ",
                     CLErrorCodeToString(error_code)));
  }
  *result = static_cast<int>(private_mem_size);
  return OkStatus();
}

}  // namespace

CLKernel::CLKernel(CLKernel&& kernel)
    : private_memory_size_(kernel.private_memory_size_),
      max_work_group_size_(kernel.max_work_group_size_),
      binding_counter_(kernel.binding_counter_),
      function_name_(std::move(kernel.function_name_)),
      program_(kernel.program_),
      kernel_(kernel.kernel_) {
  kernel.kernel_ = nullptr;
}

CLKernel& CLKernel::operator=(CLKernel&& kernel) {
  if (this != &kernel) {
    Release();
    std::swap(private_memory_size_, kernel.private_memory_size_);
    std::swap(max_work_group_size_, kernel.max_work_group_size_);
    std::swap(binding_counter_, kernel.binding_counter_);
    function_name_ = std::move(kernel.function_name_);
    std::swap(program_, kernel.program_);
    std::swap(kernel_, kernel.kernel_);
  }
  return *this;
}

CLKernel::~CLKernel() { Release(); }

Status CLKernel::ReInit() const {
  clReleaseKernel(kernel_);
  cl_kernel* kern_ptr = const_cast<cl_kernel*>(&kernel_);
  int error_code;
  *kern_ptr = clCreateKernel(program_, function_name_.c_str(), &error_code);
  if (!kernel_ || error_code != CL_SUCCESS) {
    *kern_ptr = nullptr;
    return UnknownError(absl::StrCat("Failed to create ", function_name_,
                                     CLErrorCodeToString(error_code)));
  }
  return OkStatus();
}

void CLKernel::Release() {
  if (kernel_) {
    clReleaseKernel(kernel_);
    clReleaseProgram(program_);
    kernel_ = nullptr;
  }
}

Status CLKernel::CreateFromProgram(const CLProgram& program,
                                   const std::string& function_name) {
  int error_code;
  function_name_ = function_name;
  kernel_ =
      clCreateKernel(program.program(), function_name.c_str(), &error_code);
  if (!kernel_ || error_code != CL_SUCCESS) {
    kernel_ = nullptr;
    return UnknownError(absl::StrCat("Failed to create ", function_name,
                                     CLErrorCodeToString(error_code)));
  }

  program_ = program.program();
  clRetainProgram(program_);

  RETURN_IF_ERROR(GetKernelPrivateMemorySize(kernel_, program.GetDeviceId(),
                                             &private_memory_size_));
  RETURN_IF_ERROR(GetKernelMaxWorkGroupSize(kernel_, program.GetDeviceId(),
                                            &max_work_group_size_));
  return OkStatus();
}

Status CLKernel::SetMemory(int index, cl_mem memory) {
  return SetBytes(index, &memory, sizeof(cl_mem));
}

Status CLKernel::SetMemoryAuto(cl_mem memory) {
  return SetBytesAuto(&memory, sizeof(cl_mem));
}

Status CLKernel::SetBytes(int index, const void* ptr, int length) const {
  const int error_code = clSetKernelArg(kernel_, index, length, ptr);
  if (error_code != CL_SUCCESS) {
    return UnknownError(absl::StrCat("Failed to set kernel arguments - ",
                                     CLErrorCodeToString(error_code)));
  }
  return OkStatus();
}

Status CLKernel::SetBytesAuto(const void* ptr, int length) {
  const int error_code = clSetKernelArg(kernel_, binding_counter_, length, ptr);
  if (error_code != CL_SUCCESS) {
    return UnknownError(absl::StrCat("Failed to set kernel arguments - ",
                                     CLErrorCodeToString(error_code),
                                     "(at index - ", binding_counter_, ")"));
  }
  binding_counter_++;
  return OkStatus();
}

template <>
Status CLKernel::SetBytes<FLT>(int index, const FLT& value) const {
  return SetBytes(index, value.GetData(), value.GetSize());
}

template <>
Status CLKernel::SetBytes<FLT2>(int index, const FLT2& value) const {
  return SetBytes(index, value.GetData(), value.GetSize());
}

template <>
Status CLKernel::SetBytes<FLT4>(int index, const FLT4& value) const {
  return SetBytes(index, value.GetData(), value.GetSize());
}

template <>
Status CLKernel::SetBytesAuto<FLT>(const FLT& value) {
  return SetBytesAuto(value.GetData(), value.GetSize());
}

template <>
Status CLKernel::SetBytesAuto<FLT2>(const FLT2& value) {
  return SetBytesAuto(value.GetData(), value.GetSize());
}

template <>
Status CLKernel::SetBytesAuto<FLT4>(const FLT4& value) {
  return SetBytesAuto(value.GetData(), value.GetSize());
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
