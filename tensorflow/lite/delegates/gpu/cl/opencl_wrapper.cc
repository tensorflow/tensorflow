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

#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"

#include <dlfcn.h>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

#define LoadFunction(function)                                                 \
  if (is_pixel) {                                                              \
    function = reinterpret_cast<PFN_##function>(loadOpenCLPointer(#function)); \
  } else {                                                                     \
    function = reinterpret_cast<PFN_##function>(dlsym(libopencl, #function));  \
  }

Status LoadOpenCL() {
  void* libopencl = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
  if (libopencl) {
    LoadOpenCLFunctions(libopencl, false);
    return OkStatus();
  } else {
    // Pixel phone?
    libopencl = dlopen("libOpenCL-pixel.so", RTLD_NOW | RTLD_LOCAL);
    if (libopencl) {
      typedef void (*enableOpenCL_t)();
      enableOpenCL_t enableOpenCL =
          reinterpret_cast<enableOpenCL_t>(dlsym(libopencl, "enableOpenCL"));
      enableOpenCL();
      LoadOpenCLFunctions(libopencl, true);
      return OkStatus();
    } else {
      return UnknownError(
          absl::StrCat("OpenCL library not loaded - ", dlerror()));
    }
  }
}

void LoadOpenCLFunctions(void* libopencl, bool is_pixel) {
  typedef void* (*loadOpenCLPointer_t)(const char* name);
  loadOpenCLPointer_t loadOpenCLPointer;
  if (is_pixel) {
    loadOpenCLPointer = reinterpret_cast<loadOpenCLPointer_t>(
        dlsym(libopencl, "loadOpenCLPointer"));
  }

  LoadFunction(clGetPlatformIDs);
  LoadFunction(clGetPlatformInfo);
  LoadFunction(clGetDeviceIDs);
  LoadFunction(clGetDeviceInfo);
  LoadFunction(clCreateSubDevices);
  LoadFunction(clRetainDevice);
  LoadFunction(clReleaseDevice);
  LoadFunction(clCreateContext);
  LoadFunction(clCreateContextFromType);
  LoadFunction(clRetainContext);
  LoadFunction(clReleaseContext);
  LoadFunction(clGetContextInfo);
  LoadFunction(clCreateCommandQueueWithProperties);
  LoadFunction(clRetainCommandQueue);
  LoadFunction(clReleaseCommandQueue);
  LoadFunction(clGetCommandQueueInfo);
  LoadFunction(clCreateBuffer);
  LoadFunction(clCreateSubBuffer);
  LoadFunction(clCreateImage);
  LoadFunction(clCreatePipe);
  LoadFunction(clRetainMemObject);
  LoadFunction(clReleaseMemObject);
  LoadFunction(clGetSupportedImageFormats);
  LoadFunction(clGetMemObjectInfo);
  LoadFunction(clGetImageInfo);
  LoadFunction(clGetPipeInfo);
  LoadFunction(clSetMemObjectDestructorCallback);
  LoadFunction(clSVMAlloc);
  LoadFunction(clSVMFree);
  LoadFunction(clCreateSamplerWithProperties);
  LoadFunction(clRetainSampler);
  LoadFunction(clReleaseSampler);
  LoadFunction(clGetSamplerInfo);
  LoadFunction(clCreateProgramWithSource);
  LoadFunction(clCreateProgramWithBinary);
  LoadFunction(clCreateProgramWithBuiltInKernels);
  LoadFunction(clRetainProgram);
  LoadFunction(clReleaseProgram);
  LoadFunction(clBuildProgram);
  LoadFunction(clCompileProgram);
  LoadFunction(clLinkProgram);
  LoadFunction(clUnloadPlatformCompiler);
  LoadFunction(clGetProgramInfo);
  LoadFunction(clGetProgramBuildInfo);
  LoadFunction(clCreateKernel);
  LoadFunction(clCreateKernelsInProgram);
  LoadFunction(clRetainKernel);
  LoadFunction(clReleaseKernel);
  LoadFunction(clSetKernelArg);
  LoadFunction(clSetKernelArgSVMPointer);
  LoadFunction(clSetKernelExecInfo);
  LoadFunction(clGetKernelInfo);
  LoadFunction(clGetKernelArgInfo);
  LoadFunction(clGetKernelWorkGroupInfo);
  LoadFunction(clWaitForEvents);
  LoadFunction(clGetEventInfo);
  LoadFunction(clCreateUserEvent);
  LoadFunction(clRetainEvent);
  LoadFunction(clReleaseEvent);
  LoadFunction(clSetUserEventStatus);
  LoadFunction(clSetEventCallback);
  LoadFunction(clGetEventProfilingInfo);
  LoadFunction(clFlush);
  LoadFunction(clFinish);
  LoadFunction(clEnqueueReadBuffer);
  LoadFunction(clEnqueueReadBufferRect);
  LoadFunction(clEnqueueWriteBuffer);
  LoadFunction(clEnqueueWriteBufferRect);
  LoadFunction(clEnqueueFillBuffer);
  LoadFunction(clEnqueueCopyBuffer);
  LoadFunction(clEnqueueCopyBufferRect);
  LoadFunction(clEnqueueReadImage);
  LoadFunction(clEnqueueWriteImage);
  LoadFunction(clEnqueueFillImage);
  LoadFunction(clEnqueueCopyImage);
  LoadFunction(clEnqueueCopyImageToBuffer);
  LoadFunction(clEnqueueCopyBufferToImage);
  LoadFunction(clEnqueueMapBuffer);
  LoadFunction(clEnqueueMapImage);
  LoadFunction(clEnqueueUnmapMemObject);
  LoadFunction(clEnqueueMigrateMemObjects);
  LoadFunction(clEnqueueNDRangeKernel);
  LoadFunction(clEnqueueNativeKernel);
  LoadFunction(clEnqueueMarkerWithWaitList);
  LoadFunction(clEnqueueBarrierWithWaitList);
  LoadFunction(clEnqueueSVMFree);
  LoadFunction(clEnqueueSVMMemcpy);
  LoadFunction(clEnqueueSVMMemFill);
  LoadFunction(clEnqueueSVMMap);
  LoadFunction(clEnqueueSVMUnmap);
  LoadFunction(clGetExtensionFunctionAddressForPlatform);
  LoadFunction(clCreateImage2D);
  LoadFunction(clCreateImage3D);
  LoadFunction(clEnqueueMarker);
  LoadFunction(clEnqueueWaitForEvents);
  LoadFunction(clEnqueueBarrier);
  LoadFunction(clUnloadCompiler);
  LoadFunction(clGetExtensionFunctionAddress);
  LoadFunction(clCreateCommandQueue);
  LoadFunction(clCreateSampler);
  LoadFunction(clEnqueueTask);

  // OpenGL sharing
  LoadFunction(clCreateFromGLBuffer);
  LoadFunction(clCreateFromGLTexture);
  LoadFunction(clEnqueueAcquireGLObjects);
  LoadFunction(clEnqueueReleaseGLObjects);

  // cl_khr_egl_event extension
  LoadFunction(clCreateEventFromEGLSyncKHR);
}

// No OpenCL support, do not set function addresses
PFN_clGetPlatformIDs clGetPlatformIDs;
PFN_clGetPlatformInfo clGetPlatformInfo;
PFN_clGetDeviceIDs clGetDeviceIDs;
PFN_clGetDeviceInfo clGetDeviceInfo;
PFN_clCreateSubDevices clCreateSubDevices;
PFN_clRetainDevice clRetainDevice;
PFN_clReleaseDevice clReleaseDevice;
PFN_clCreateContext clCreateContext;
PFN_clCreateContextFromType clCreateContextFromType;
PFN_clRetainContext clRetainContext;
PFN_clReleaseContext clReleaseContext;
PFN_clGetContextInfo clGetContextInfo;
PFN_clCreateCommandQueueWithProperties clCreateCommandQueueWithProperties;
PFN_clRetainCommandQueue clRetainCommandQueue;
PFN_clReleaseCommandQueue clReleaseCommandQueue;
PFN_clGetCommandQueueInfo clGetCommandQueueInfo;
PFN_clCreateBuffer clCreateBuffer;
PFN_clCreateSubBuffer clCreateSubBuffer;
PFN_clCreateImage clCreateImage;
PFN_clCreatePipe clCreatePipe;
PFN_clRetainMemObject clRetainMemObject;
PFN_clReleaseMemObject clReleaseMemObject;
PFN_clGetSupportedImageFormats clGetSupportedImageFormats;
PFN_clGetMemObjectInfo clGetMemObjectInfo;
PFN_clGetImageInfo clGetImageInfo;
PFN_clGetPipeInfo clGetPipeInfo;
PFN_clSetMemObjectDestructorCallback clSetMemObjectDestructorCallback;
PFN_clSVMAlloc clSVMAlloc;
PFN_clSVMFree clSVMFree;
PFN_clCreateSamplerWithProperties clCreateSamplerWithProperties;
PFN_clRetainSampler clRetainSampler;
PFN_clReleaseSampler clReleaseSampler;
PFN_clGetSamplerInfo clGetSamplerInfo;
PFN_clCreateProgramWithSource clCreateProgramWithSource;
PFN_clCreateProgramWithBinary clCreateProgramWithBinary;
PFN_clCreateProgramWithBuiltInKernels clCreateProgramWithBuiltInKernels;
PFN_clRetainProgram clRetainProgram;
PFN_clReleaseProgram clReleaseProgram;
PFN_clBuildProgram clBuildProgram;
PFN_clCompileProgram clCompileProgram;
PFN_clLinkProgram clLinkProgram;
PFN_clUnloadPlatformCompiler clUnloadPlatformCompiler;
PFN_clGetProgramInfo clGetProgramInfo;
PFN_clGetProgramBuildInfo clGetProgramBuildInfo;
PFN_clCreateKernel clCreateKernel;
PFN_clCreateKernelsInProgram clCreateKernelsInProgram;
PFN_clRetainKernel clRetainKernel;
PFN_clReleaseKernel clReleaseKernel;
PFN_clSetKernelArg clSetKernelArg;
PFN_clSetKernelArgSVMPointer clSetKernelArgSVMPointer;
PFN_clSetKernelExecInfo clSetKernelExecInfo;
PFN_clGetKernelInfo clGetKernelInfo;
PFN_clGetKernelArgInfo clGetKernelArgInfo;
PFN_clGetKernelWorkGroupInfo clGetKernelWorkGroupInfo;
PFN_clWaitForEvents clWaitForEvents;
PFN_clGetEventInfo clGetEventInfo;
PFN_clCreateUserEvent clCreateUserEvent;
PFN_clRetainEvent clRetainEvent;
PFN_clReleaseEvent clReleaseEvent;
PFN_clSetUserEventStatus clSetUserEventStatus;
PFN_clSetEventCallback clSetEventCallback;
PFN_clGetEventProfilingInfo clGetEventProfilingInfo;
PFN_clFlush clFlush;
PFN_clFinish clFinish;
PFN_clEnqueueReadBuffer clEnqueueReadBuffer;
PFN_clEnqueueReadBufferRect clEnqueueReadBufferRect;
PFN_clEnqueueWriteBuffer clEnqueueWriteBuffer;
PFN_clEnqueueWriteBufferRect clEnqueueWriteBufferRect;
PFN_clEnqueueFillBuffer clEnqueueFillBuffer;
PFN_clEnqueueCopyBuffer clEnqueueCopyBuffer;
PFN_clEnqueueCopyBufferRect clEnqueueCopyBufferRect;
PFN_clEnqueueReadImage clEnqueueReadImage;
PFN_clEnqueueWriteImage clEnqueueWriteImage;
PFN_clEnqueueFillImage clEnqueueFillImage;
PFN_clEnqueueCopyImage clEnqueueCopyImage;
PFN_clEnqueueCopyImageToBuffer clEnqueueCopyImageToBuffer;
PFN_clEnqueueCopyBufferToImage clEnqueueCopyBufferToImage;
PFN_clEnqueueMapBuffer clEnqueueMapBuffer;
PFN_clEnqueueMapImage clEnqueueMapImage;
PFN_clEnqueueUnmapMemObject clEnqueueUnmapMemObject;
PFN_clEnqueueMigrateMemObjects clEnqueueMigrateMemObjects;
PFN_clEnqueueNDRangeKernel clEnqueueNDRangeKernel;
PFN_clEnqueueNativeKernel clEnqueueNativeKernel;
PFN_clEnqueueMarkerWithWaitList clEnqueueMarkerWithWaitList;
PFN_clEnqueueBarrierWithWaitList clEnqueueBarrierWithWaitList;
PFN_clEnqueueSVMFree clEnqueueSVMFree;
PFN_clEnqueueSVMMemcpy clEnqueueSVMMemcpy;
PFN_clEnqueueSVMMemFill clEnqueueSVMMemFill;
PFN_clEnqueueSVMMap clEnqueueSVMMap;
PFN_clEnqueueSVMUnmap clEnqueueSVMUnmap;
PFN_clGetExtensionFunctionAddressForPlatform
    clGetExtensionFunctionAddressForPlatform;
PFN_clCreateImage2D clCreateImage2D;
PFN_clCreateImage3D clCreateImage3D;
PFN_clEnqueueMarker clEnqueueMarker;
PFN_clEnqueueWaitForEvents clEnqueueWaitForEvents;
PFN_clEnqueueBarrier clEnqueueBarrier;
PFN_clUnloadCompiler clUnloadCompiler;
PFN_clGetExtensionFunctionAddress clGetExtensionFunctionAddress;
PFN_clCreateCommandQueue clCreateCommandQueue;
PFN_clCreateSampler clCreateSampler;
PFN_clEnqueueTask clEnqueueTask;

PFN_clCreateFromGLBuffer clCreateFromGLBuffer;
PFN_clCreateFromGLTexture clCreateFromGLTexture;
PFN_clEnqueueAcquireGLObjects clEnqueueAcquireGLObjects;
PFN_clEnqueueReleaseGLObjects clEnqueueReleaseGLObjects;

PFN_clCreateEventFromEGLSyncKHR clCreateEventFromEGLSyncKHR;

cl_mem CreateImage2DLegacy(cl_context context, cl_mem_flags flags,
                           const cl_image_format* image_format,
                           const cl_image_desc* image_desc, void* host_ptr,
                           cl_int* errcode_ret) {
  if (clCreateImage) {  // clCreateImage available since OpenCL 1.2
    return clCreateImage(context, flags, image_format, image_desc, host_ptr,
                         errcode_ret);
  } else {
    return clCreateImage2D(context, flags, image_format,
                           image_desc->image_width, image_desc->image_height,
                           image_desc->image_row_pitch, host_ptr, errcode_ret);
  }
}
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
