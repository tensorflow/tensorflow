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

#include "tensorflow/lite/delegates/gpu/cl/util.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

std::string CLErrorCodeToString(cl_int error_code) {
  switch (error_code) {
    case CL_SUCCESS:
      return "Success";
    case CL_DEVICE_NOT_FOUND:
      return "Device not found";
    case CL_DEVICE_NOT_AVAILABLE:
      return "Device not available";
    case CL_COMPILER_NOT_AVAILABLE:
      return "Compiler not available";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "Memory object allocation failure";
    case CL_OUT_OF_RESOURCES:
      return "Out of resources";
    case CL_OUT_OF_HOST_MEMORY:
      return "Out of host memory";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return "Profiling information not available";
    case CL_MEM_COPY_OVERLAP:
      return "Memory copy overlap";
    case CL_IMAGE_FORMAT_MISMATCH:
      return "Image format mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return "Image format not supported";
    case CL_BUILD_PROGRAM_FAILURE:
      return "Build program failure";
    case CL_MAP_FAILURE:
      return "Mapping failure";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
      return "Misaligned sub-buffer offset";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
      return "Execution status error for events in wait list";
    case CL_COMPILE_PROGRAM_FAILURE:
      return "Compile program failure";
    case CL_LINKER_NOT_AVAILABLE:
      return "Linker not available";
    case CL_LINK_PROGRAM_FAILURE:
      return "Link program failure";
    case CL_DEVICE_PARTITION_FAILED:
      return "Device partition failed";
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
      return "Kernel argument information not available";

    case CL_INVALID_VALUE:
      return "Invalid value";
    case CL_INVALID_DEVICE_TYPE:
      return "Invalid device type";
    case CL_INVALID_PLATFORM:
      return "Invalid platform";
    case CL_INVALID_DEVICE:
      return "Invalid device";
    case CL_INVALID_CONTEXT:
      return "Invalid context";
    case CL_INVALID_QUEUE_PROPERTIES:
      return "Invalid queue properties";
    case CL_INVALID_COMMAND_QUEUE:
      return "Invalid command queue";
    case CL_INVALID_HOST_PTR:
      return "Invalid host pointer";
    case CL_INVALID_MEM_OBJECT:
      return "Invalid memory object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return "Invalid image format descriptor";
    case CL_INVALID_IMAGE_SIZE:
      return "Invalid image size";
    case CL_INVALID_SAMPLER:
      return "Invalid sampler";
    case CL_INVALID_BINARY:
      return "Invalid binary";
    case CL_INVALID_BUILD_OPTIONS:
      return "Invalid build options";
    case CL_INVALID_PROGRAM:
      return "Invalid program";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return "Invalid program executable";
    case CL_INVALID_KERNEL_NAME:
      return "Invalid kernel name";
    case CL_INVALID_KERNEL_DEFINITION:
      return "Invalid kernel definition";
    case CL_INVALID_KERNEL:
      return "Invalid kernel";
    case CL_INVALID_ARG_INDEX:
      return "Invalid argument index";
    case CL_INVALID_ARG_VALUE:
      return "Invalid argument value";
    case CL_INVALID_ARG_SIZE:
      return "Invalid argument size";
    case CL_INVALID_KERNEL_ARGS:
      return "Invalid kernel arguments";
    case CL_INVALID_WORK_DIMENSION:
      return "Invalid work dimension";
    case CL_INVALID_WORK_GROUP_SIZE:
      return "Invalid work group size";
    case CL_INVALID_WORK_ITEM_SIZE:
      return "Invalid work item size";
    case CL_INVALID_GLOBAL_OFFSET:
      return "Invalid global offset";
    case CL_INVALID_EVENT_WAIT_LIST:
      return "Invalid event wait list";
    case CL_INVALID_EVENT:
      return "Invalid event";
    case CL_INVALID_OPERATION:
      return "Invalid operation";
    case CL_INVALID_GL_OBJECT:
      return "Invalid GL object";
    case CL_INVALID_BUFFER_SIZE:
      return "Invalid buffer size";
    case CL_INVALID_MIP_LEVEL:
      return "Invalid mip-level";
    case CL_INVALID_GLOBAL_WORK_SIZE:
      return "Invalid global work size";
    case CL_INVALID_PROPERTY:
      return "Invalid property";
    case CL_INVALID_IMAGE_DESCRIPTOR:
      return "Invalid image descriptor";
    case CL_INVALID_COMPILER_OPTIONS:
      return "Invalid compiler options";
    case CL_INVALID_LINKER_OPTIONS:
      return "Invalid linker options";
    case CL_INVALID_DEVICE_PARTITION_COUNT:
      return "Invalid device partition count";
    case CL_INVALID_PIPE_SIZE:
      return "Invalid pipe size";
    case CL_INVALID_DEVICE_QUEUE:
      return "Invalid device queue";
    case CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR:
      return "Invalid GL sharegroup reference KHR";
    case CL_INVALID_COMMAND_BUFFER_KHR:
      return "Invalid command buffer KHR";
    case CL_INVALID_SYNC_POINT_WAIT_LIST_KHR:
      return "Invalid sync point wait list KHR";
    case CL_INCOMPATIBLE_COMMAND_QUEUE_KHR:
      return "Incompatible command queue KHR";

    default:
      return absl::StrCat("Unknown OpenCL error code - ", error_code);
  }
}

int ChannelTypeToSizeInBytes(cl_channel_type type) {
  switch (type) {
    case CL_FLOAT:
      return 4;
    case CL_HALF_FLOAT:
      return 2;
    default:
      return 0;
  }
}

bool OpenCLSupported() { return LoadOpenCL().ok(); }

absl::optional<cl_platform_id> FindPlatformByVendor(
    std::vector<cl_platform_id> const& platform_ids,
    std::string const& vendor_id) {
  auto tolower = [](std::string const& original) {
    std::string str_copy = original;
    std::transform(str_copy.begin(), str_copy.end(), str_copy.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return str_copy;
  };

  auto get_platform_info = [](std::vector<cl_platform_id> const& platform_ids,
                              cl_platform_info info) {
    std::vector<std::string> prop(platform_ids.size());
    std::transform(
        platform_ids.begin(), platform_ids.end(), prop.begin(),
        [info](cl_platform_id id) { return GetPlatformInfo(id, info); });
    return prop;
  };

  auto find_platform_by_vendor = [&](std::vector<std::string> const& candidates)
      -> absl::optional<cl_platform_id> {
    auto it = std::find_if(
        candidates.begin(), candidates.end(),
        [&](std::string const& candidate) {
          if (tolower(candidate).find(vendor_id) == std::string::npos)
            return false;
          return true;
        });

    if (it == candidates.end()) return {};
    return platform_ids[std::distance(candidates.begin(), it)];
  };

  auto platform_id_maybe = find_platform_by_vendor(
      get_platform_info(platform_ids, CL_PLATFORM_VENDOR));
  if (platform_id_maybe.has_value()) return platform_id_maybe;

  return find_platform_by_vendor(
      get_platform_info(platform_ids, CL_PLATFORM_NAME));
}

std::string GetPlatformInfo(cl_platform_id id, cl_platform_info info) {
  size_t size;
  cl_int error = clGetPlatformInfo(id, info, 0, nullptr, &size);
  if (error != CL_SUCCESS) {
    return "";
  }

  std::string result(size - 1, 0);
  error = clGetPlatformInfo(id, info, size, &result[0], nullptr);
  if (error != CL_SUCCESS) {
    return "";
  }
  return result;
}

absl::StatusOr<std::vector<cl_platform_id>> GetOpenCLPlatforms() {
  cl_uint num_platforms;
  cl_int status = clGetPlatformIDs(0, nullptr, &num_platforms);
  if (status != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrFormat("clGetPlatformIDs returned %d", status));
  }
  if (num_platforms == 0) {
    return absl::UnknownError("No supported OpenCL platform.");
  }
  std::vector<cl_platform_id> platforms(num_platforms);
  status = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
  if (status != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrFormat("clGetPlatformIDs returned %d", status));
  }

  return std::move(platforms);
}

absl::StatusOr<std::vector<cl_device_id>> GetOpenCLDevicesForPlatform(
    cl_platform_id platform_id) {
  cl_uint num_devices;
  cl_int status =
      clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
  if (status != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrFormat("clGetDeviceIDs returned %d", status));
  }
  if (num_devices == 0) {
    return absl::UnknownError("No GPU on current platform.");
  }

  std::vector<cl_device_id> devices(num_devices);
  status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, num_devices,
                          devices.data(), nullptr);
  if (status != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrFormat("clGetDeviceIDs returned %d", status));
  }

  return std::move(devices);
}

absl::Status CreateCLBuffer(cl_context context, int size_in_bytes,
                            bool read_only, void* data, cl_mem* result) {
  cl_mem_flags flags = read_only ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE;
  if (data) {
    flags |= CL_MEM_COPY_HOST_PTR;
  }
  cl_int error_code;
  *result = clCreateBuffer(context, flags, size_in_bytes, data, &error_code);
  if (!*result) {
    return absl::UnknownError(
        absl::StrCat("Failed to allocate device memory (clCreateBuffer): ",
                     CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

absl::Status CreateCLSubBuffer(cl_context context, cl_mem parent,
                               size_t origin_in_bytes, size_t size_in_bytes,
                               bool read_only, cl_mem* result) {
  cl_mem_flags flags = read_only ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE;

  cl_buffer_region region{};
  region.origin = origin_in_bytes;
  region.size = size_in_bytes;

  cl_int error_code;
  if (!clCreateSubBuffer) {
    return absl::InternalError("clCreateSubBuffer is not supported.");
  }
  *result = clCreateSubBuffer(parent, flags, CL_BUFFER_CREATE_TYPE_REGION,
                              &region, &error_code);

  if (!*result) {
    return absl::UnknownError(
        absl::StrCat("Failed to allocate device memory (clCreateSubBuffer): ",
                     CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

absl::Status CreateRGBAImage2D(cl_context context, int width, int height,
                               cl_channel_type channel_type, void* data,
                               cl_mem* result) {
  cl_image_desc desc;
  desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  desc.image_width = width;
  desc.image_height = height;
  desc.image_depth = 0;
  desc.image_row_pitch = 0;
  desc.image_slice_pitch = 0;
  desc.num_mip_levels = 0;
  desc.num_samples = 0;
  desc.buffer = nullptr;

  cl_image_format format;
  format.image_channel_order = CL_RGBA;
  format.image_channel_data_type = channel_type;

  cl_mem_flags flags = CL_MEM_READ_WRITE;
  if (data) {
    flags |= CL_MEM_COPY_HOST_PTR;
  }

  cl_int error_code;
  *result =
      CreateImage2DLegacy(context, flags, &format, &desc, data, &error_code);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to create 2D texture (clCreateImage): ",
                     CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
