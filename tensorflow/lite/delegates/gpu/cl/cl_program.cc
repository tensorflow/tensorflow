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

#include "tensorflow/lite/delegates/gpu/cl/cl_program.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetProgramBuildInfo(cl_program program, cl_device_id id,
                                cl_program_build_info info) {
  size_t size;
  cl_int error_code =
      clGetProgramBuildInfo(program, id, info, 0, nullptr, &size);
  if (error_code != CL_SUCCESS) {
    return absl::StrCat("Failed to GetProgramBuildInfo - ",
                        CLErrorCodeToString(error_code));
  }

  std::string result(size - 1, 0);
  error_code =
      clGetProgramBuildInfo(program, id, info, size, &result[0], nullptr);
  if (error_code != CL_SUCCESS) {
    return absl::StrCat("Failed to GetProgramBuildInfo - ",
                        CLErrorCodeToString(error_code));
  }
  return result;
}

absl::Status GetBinarySize(cl_program program, size_t* binary_size) {
  cl_int error_code = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                                       sizeof(size_t), binary_size, nullptr);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to get program binary size - ",
                     CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

absl::Status BuildProgram(cl_program program, const CLDevice& device,
                          const std::string& compiler_options) {
  const int error_code = clBuildProgram(
      program, 0, nullptr, compiler_options.c_str(), nullptr, nullptr);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(absl::StrCat(
        "Failed to build program executable - ",
        CLErrorCodeToString(error_code),
        GetProgramBuildInfo(program, device.id(), CL_PROGRAM_BUILD_LOG)));
  }

  return absl::OkStatus();
}

std::string CompilerOptionToString(const CLDevice& device,
                                   CompilerOptions option) {
  switch (option) {
    case CompilerOptions::ADRENO_FULL_SIMD_LINE:
      if (device.GetInfo().adreno_info.gpu_version < 500) {
        return "-qcom-accelerate-16-bit";
      } else {
        return "-qcom-accelerate-16-bit=true";
      }
    case CompilerOptions::ADRENO_MORE_WAVES:
      if (device.GetInfo().adreno_info.gpu_version >= 500) {
        return "-qcom-accelerate-16-bit=false";
      } else {
        return "";
      }
    case CompilerOptions::POWERVR_FP16:
      return "-cl-fast-relaxed-math";
    case CompilerOptions::CL_OPT_DISABLE:
      return "-cl-opt-disable";
  }
}

}  // namespace

std::string CompilerOptionsToString(
    const CLDevice& device,
    const std::vector<CompilerOptions>& compiler_options) {
  std::string result;
  for (auto option : compiler_options) {
    absl::StrAppend(&result, CompilerOptionToString(device, option), " ");
  }
  return result;
}

CLProgram::CLProgram(cl_program program, cl_device_id device_id)
    : program_(program), device_id_(device_id) {}

CLProgram::CLProgram(CLProgram&& program)
    : program_(program.program_), device_id_(program.device_id_) {
  program.program_ = nullptr;
}

CLProgram& CLProgram::operator=(CLProgram&& program) {
  if (this != &program) {
    Release();
    std::swap(program_, program.program_);
    std::swap(device_id_, program.device_id_);
  }
  return *this;
}

CLProgram::~CLProgram() { Release(); }

void CLProgram::Release() {
  if (program_) {
    clReleaseProgram(program_);
    program_ = nullptr;
  }
}

absl::Status CLProgram::GetBinary(std::vector<uint8_t>* result) const {
  size_t binary_size;
  RETURN_IF_ERROR(GetBinarySize(program_, &binary_size));
  result->resize(result->size() + binary_size);
  uint8_t* binary_ptr = result->data() + result->size() - binary_size;
  cl_int error_code = clGetProgramInfo(program_, CL_PROGRAM_BINARIES,
                                       binary_size, &binary_ptr, nullptr);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(absl::StrCat("Failed to get program binary - ",
                                           CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

absl::Status CreateCLProgram(const std::string& code,
                             const std::string& compiler_options,
                             const CLContext& context, const CLDevice& device,
                             CLProgram* result) {
  int error_code;
  const char* source = code.c_str();

  cl_program program = clCreateProgramWithSource(context.context(), 1, &source,
                                                 nullptr, &error_code);
  if (!program || error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to create compute program - ",
                     CLErrorCodeToString(error_code)));
  }

  *result = CLProgram(program, device.id());
  RETURN_IF_ERROR(BuildProgram(program, device, compiler_options));
  return absl::OkStatus();
}

absl::Status CreateCLProgramFromBinary(const CLContext& context,
                                       const CLDevice& device,
                                       absl::Span<const uint8_t> binary,
                                       CLProgram* result) {
  cl_int binary_status;
  cl_int error_code;
  cl_device_id devices_list[] = {device.id()};
  size_t binary_size = binary.size();
  const uint8_t* binary_pointer = binary.data();
  cl_program program = clCreateProgramWithBinary(
      context.context(), 1, devices_list, &binary_size, &binary_pointer,
      &binary_status, &error_code);
  if (binary_status != CL_SUCCESS) {
    return absl::UnknownError(absl::StrCat(
        "Something wrong with binary after clCreateProgramWithBinary - ",
        binary_status));
  }
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(absl::StrCat("Failed to create program - ",
                                           CLErrorCodeToString(error_code)));
  }
  *result = CLProgram(program, device.id());
  return BuildProgram(program, device, "");
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
