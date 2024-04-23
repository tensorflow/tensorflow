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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_OPENCL_WRAPPER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_OPENCL_WRAPPER_H_

#include <CL/cl.h>
#include <CL/cl_egl.h>
#include <CL/cl_ext.h>
#include <CL/cl_gl.h>
#include <CL/cl_platform.h>
#include "tensorflow/lite/delegates/gpu/cl/default/qcom_wrapper.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

absl::Status LoadOpenCL();
void LoadOpenCLFunctionExtensions(cl_platform_id platform_id);

typedef cl_int(CL_API_CALL *PFN_clGetPlatformIDs)(
    cl_uint /* num_entries */, cl_platform_id * /* platforms */,
    cl_uint * /* num_platforms */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clGetPlatformInfo)(
    cl_platform_id /* platform */, cl_platform_info /* param_name */,
    size_t /* param_value_size */, void * /* param_value */,
    size_t * /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clGetDeviceIDs)(
    cl_platform_id /* platform */, cl_device_type /* device_type */,
    cl_uint /* num_entries */, cl_device_id * /* devices */,
    cl_uint * /* num_devices */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clGetDeviceInfo)(
    cl_device_id /* device */, cl_device_info /* param_name */,
    size_t /* param_value_size */, void * /* param_value */,
    size_t * /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clCreateSubDevices)(
    cl_device_id /* in_device */,
    const cl_device_partition_property * /* properties */,
    cl_uint /* num_devices */, cl_device_id * /* out_devices */,
    cl_uint * /* num_devices_ret */) CL_API_SUFFIX__VERSION_1_2;
typedef cl_int(CL_API_CALL *PFN_clRetainDevice)(cl_device_id /* device */)
    CL_API_SUFFIX__VERSION_1_2;
typedef cl_int(CL_API_CALL *PFN_clReleaseDevice)(cl_device_id /* device */)
    CL_API_SUFFIX__VERSION_1_2;
typedef cl_context(CL_API_CALL *PFN_clCreateContext)(
    const cl_context_properties * /* properties */, cl_uint /* num_devices */,
    const cl_device_id * /* devices */,
    void(CL_CALLBACK * /* pfn_notify */)(const char *, const void *, size_t,
                                         void *),
    void * /* user_data */,
    cl_int * /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_context(CL_API_CALL *PFN_clCreateContextFromType)(
    const cl_context_properties * /* properties */,
    cl_device_type /* device_type */,
    void(CL_CALLBACK * /* pfn_notify*/)(const char *, const void *, size_t,
                                        void *),
    void * /* user_data */,
    cl_int * /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clRetainContext)(cl_context /* context */)
    CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clReleaseContext)(cl_context /* context */)
    CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clGetContextInfo)(
    cl_context /* context */, cl_context_info /* param_name */,
    size_t /* param_value_size */, void * /* param_value */,
    size_t * /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_command_queue(CL_API_CALL *PFN_clCreateCommandQueueWithProperties)(
    cl_context /* context */, cl_device_id /* device */,
    const cl_queue_properties * /* properties */,
    cl_int * /* errcode_ret */) CL_API_SUFFIX__VERSION_2_0;
typedef cl_int(CL_API_CALL *PFN_clRetainCommandQueue)(
    cl_command_queue /* command_queue */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clReleaseCommandQueue)(
    cl_command_queue /* command_queue */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clGetCommandQueueInfo)(
    cl_command_queue /* command_queue */,
    cl_command_queue_info /* param_name */, size_t /* param_value_size */,
    void * /* param_value */,
    size_t * /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_mem(CL_API_CALL *PFN_clCreateBuffer)(
    cl_context /* context */, cl_mem_flags /* flags */, size_t /* size */,
    void * /* host_ptr */,
    cl_int * /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_mem(CL_API_CALL *PFN_clCreateSubBuffer)(
    cl_mem /* buffer */, cl_mem_flags /* flags */,
    cl_buffer_create_type /* buffer_create_type */,
    const void * /* buffer_create_info */,
    cl_int * /* errcode_ret */) CL_API_SUFFIX__VERSION_1_1;
typedef cl_mem(CL_API_CALL *PFN_clCreateImage)(
    cl_context /* context */, cl_mem_flags /* flags */,
    const cl_image_format * /* image_format */,
    const cl_image_desc * /* image_desc */, void * /* host_ptr */,
    cl_int * /* errcode_ret */) CL_API_SUFFIX__VERSION_1_2;
typedef cl_mem(CL_API_CALL *PFN_clCreatePipe)(
    cl_context /* context */, cl_mem_flags /* flags */,
    cl_uint /* pipe_packet_size */, cl_uint /* pipe_max_packets */,
    const cl_pipe_properties * /* properties */,
    cl_int * /* errcode_ret */) CL_API_SUFFIX__VERSION_2_0;
typedef cl_int(CL_API_CALL *PFN_clRetainMemObject)(cl_mem /* memobj */)
    CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clReleaseMemObject)(cl_mem /* memobj */)
    CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clGetSupportedImageFormats)(
    cl_context /* context */, cl_mem_flags /* flags */,
    cl_mem_object_type /* image_type */, cl_uint /* num_entries */,
    cl_image_format * /* image_formats */,
    cl_uint * /* num_image_formats */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clGetMemObjectInfo)(
    cl_mem /* memobj */, cl_mem_info /* param_name */,
    size_t /* param_value_size */, void * /* param_value */,
    size_t * /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clGetImageInfo)(
    cl_mem /* image */, cl_image_info /* param_name */,
    size_t /* param_value_size */, void * /* param_value */,
    size_t * /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clGetPipeInfo)(
    cl_mem /* pipe */, cl_pipe_info /* param_name */,
    size_t /* param_value_size */, void * /* param_value */,
    size_t * /* param_value_size_ret */) CL_API_SUFFIX__VERSION_2_0;
typedef cl_int(CL_API_CALL *PFN_clSetMemObjectDestructorCallback)(
    cl_mem /* memobj */,
    void(CL_CALLBACK * /*pfn_notify*/)(cl_mem /* memobj */,
                                       void * /*user_data*/),
    void * /*user_data */) CL_API_SUFFIX__VERSION_1_1;
typedef void *(CL_API_CALL *PFN_clSVMAlloc)(
    cl_context /* context */, cl_svm_mem_flags /* flags */, size_t /* size */,
    cl_uint /* alignment */)CL_API_SUFFIX__VERSION_2_0;
typedef void(CL_API_CALL *PFN_clSVMFree)(cl_context /* context */,
                                         void * /* svm_pointer */)
    CL_API_SUFFIX__VERSION_2_0;
typedef cl_sampler(CL_API_CALL *PFN_clCreateSamplerWithProperties)(
    cl_context /* context */,
    const cl_sampler_properties * /* normalized_coords */,
    cl_int * /* errcode_ret */) CL_API_SUFFIX__VERSION_2_0;
typedef cl_int(CL_API_CALL *PFN_clRetainSampler)(cl_sampler /* sampler */)
    CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clReleaseSampler)(cl_sampler /* sampler */)
    CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clGetSamplerInfo)(
    cl_sampler /* sampler */, cl_sampler_info /* param_name */,
    size_t /* param_value_size */, void * /* param_value */,
    size_t * /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_program(CL_API_CALL *PFN_clCreateProgramWithSource)(
    cl_context /* context */, cl_uint /* count */, const char ** /* strings */,
    const size_t * /* lengths */,
    cl_int * /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_program(CL_API_CALL *PFN_clCreateProgramWithBinary)(
    cl_context /* context */, cl_uint /* num_devices */,
    const cl_device_id * /* device_list */, const size_t * /* lengths */,
    const unsigned char ** /* binaries */, cl_int * /* binary_status */,
    cl_int * /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_program(CL_API_CALL *PFN_clCreateProgramWithBuiltInKernels)(
    cl_context /* context */, cl_uint /* num_devices */,
    const cl_device_id * /* device_list */, const char * /* kernel_names */,
    cl_int * /* errcode_ret */) CL_API_SUFFIX__VERSION_1_2;
typedef cl_int(CL_API_CALL *PFN_clRetainProgram)(cl_program /* program */)
    CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clReleaseProgram)(cl_program /* program */)
    CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clBuildProgram)(
    cl_program /* program */, cl_uint /* num_devices */,
    const cl_device_id * /* device_list */, const char * /* options */,
    void(CL_CALLBACK * /* pfn_notify */)(cl_program /* program */,
                                         void * /* user_data */),
    void * /* user_data */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clCompileProgram)(
    cl_program /* program */, cl_uint /* num_devices */,
    const cl_device_id * /* device_list */, const char * /* options */,
    cl_uint /* num_input_headers */, const cl_program * /* input_headers */,
    const char ** /* header_include_names */,
    void(CL_CALLBACK * /* pfn_notify */)(cl_program /* program */,
                                         void * /* user_data */),
    void * /* user_data */) CL_API_SUFFIX__VERSION_1_2;
typedef cl_program(CL_API_CALL *PFN_clLinkProgram)(
    cl_context /* context */, cl_uint /* num_devices */,
    const cl_device_id * /* device_list */, const char * /* options */,
    cl_uint /* num_input_programs */, const cl_program * /* input_programs */,
    void(CL_CALLBACK * /* pfn_notify */)(cl_program /* program */,
                                         void * /* user_data */),
    void * /* user_data */,
    cl_int * /* errcode_ret */) CL_API_SUFFIX__VERSION_1_2;
typedef cl_int(CL_API_CALL *PFN_clUnloadPlatformCompiler)(
    cl_platform_id /* platform */) CL_API_SUFFIX__VERSION_1_2;
typedef cl_int(CL_API_CALL *PFN_clGetProgramInfo)(
    cl_program /* program */, cl_program_info /* param_name */,
    size_t /* param_value_size */, void * /* param_value */,
    size_t * /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clGetProgramBuildInfo)(
    cl_program /* program */, cl_device_id /* device */,
    cl_program_build_info /* param_name */, size_t /* param_value_size */,
    void * /* param_value */,
    size_t * /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_kernel(CL_API_CALL *PFN_clCreateKernel)(
    cl_program /* program */, const char * /* kernel_name */,
    cl_int * /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clCreateKernelsInProgram)(
    cl_program /* program */, cl_uint /* num_kernels */,
    cl_kernel * /* kernels */,
    cl_uint * /* num_kernels_ret */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clRetainKernel)(cl_kernel /* kernel */)
    CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clReleaseKernel)(cl_kernel /* kernel */)
    CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clSetKernelArg)(
    cl_kernel /* kernel */, cl_uint /* arg_index */, size_t /* arg_size */,
    const void * /* arg_value */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clSetKernelArgSVMPointer)(
    cl_kernel /* kernel */, cl_uint /* arg_index */,
    const void * /* arg_value */) CL_API_SUFFIX__VERSION_2_0;
typedef cl_int(CL_API_CALL *PFN_clSetKernelExecInfo)(
    cl_kernel /* kernel */, cl_kernel_exec_info /* param_name */,
    size_t /* param_value_size */,
    const void * /* param_value */) CL_API_SUFFIX__VERSION_2_0;
typedef cl_int(CL_API_CALL *PFN_clGetKernelInfo)(
    cl_kernel /* kernel */, cl_kernel_info /* param_name */,
    size_t /* param_value_size */, void * /* param_value */,
    size_t * /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clGetKernelArgInfo)(
    cl_kernel /* kernel */, cl_uint /* arg_indx */,
    cl_kernel_arg_info /* param_name */, size_t /* param_value_size */,
    void * /* param_value */,
    size_t * /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_2;
typedef cl_int(CL_API_CALL *PFN_clGetKernelWorkGroupInfo)(
    cl_kernel /* kernel */, cl_device_id /* device */,
    cl_kernel_work_group_info /* param_name */, size_t /* param_value_size */,
    void * /* param_value */,
    size_t * /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clWaitForEvents)(
    cl_uint /* num_events */,
    const cl_event * /* event_list */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clGetEventInfo)(
    cl_event /* event */, cl_event_info /* param_name */,
    size_t /* param_value_size */, void * /* param_value */,
    size_t * /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_event(CL_API_CALL *PFN_clCreateUserEvent)(cl_context /* context */,
                                                     cl_int * /* errcode_ret */)
    CL_API_SUFFIX__VERSION_1_1;
typedef cl_int(CL_API_CALL *PFN_clRetainEvent)(cl_event /* event */)
    CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clReleaseEvent)(cl_event /* event */)
    CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clSetUserEventStatus)(
    cl_event /* event */,
    cl_int /* execution_status */) CL_API_SUFFIX__VERSION_1_1;
typedef cl_int(CL_API_CALL *PFN_clSetEventCallback)(
    cl_event /* event */, cl_int /* command_exec_callback_type */,
    void(CL_CALLBACK * /* pfn_notify */)(cl_event, cl_int, void *),
    void * /* user_data */) CL_API_SUFFIX__VERSION_1_1;
typedef cl_int(CL_API_CALL *PFN_clGetEventProfilingInfo)(
    cl_event /* event */, cl_profiling_info /* param_name */,
    size_t /* param_value_size */, void * /* param_value */,
    size_t * /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clFlush)(cl_command_queue /* command_queue */)
    CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clFinish)(cl_command_queue /* command_queue */)
    CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clEnqueueReadBuffer)(
    cl_command_queue /* command_queue */, cl_mem /* buffer */,
    cl_bool /* blocking_read */, size_t /* offset */, size_t /* size */,
    void * /* ptr */, cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */,
    cl_event * /* event */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clEnqueueReadBufferRect)(
    cl_command_queue /* command_queue */, cl_mem /* buffer */,
    cl_bool /* blocking_read */, const size_t * /* buffer_offset */,
    const size_t * /* host_offset */, const size_t * /* region */,
    size_t /* buffer_row_pitch */, size_t /* buffer_slice_pitch */,
    size_t /* host_row_pitch */, size_t /* host_slice_pitch */,
    void * /* ptr */, cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */,
    cl_event * /* event */) CL_API_SUFFIX__VERSION_1_1;
typedef cl_int(CL_API_CALL *PFN_clEnqueueWriteBuffer)(
    cl_command_queue /* command_queue */, cl_mem /* buffer */,
    cl_bool /* blocking_write */, size_t /* offset */, size_t /* size */,
    const void * /* ptr */, cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */,
    cl_event * /* event */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clEnqueueWriteBufferRect)(
    cl_command_queue /* command_queue */, cl_mem /* buffer */,
    cl_bool /* blocking_write */, const size_t * /* buffer_offset */,
    const size_t * /* host_offset */, const size_t * /* region */,
    size_t /* buffer_row_pitch */, size_t /* buffer_slice_pitch */,
    size_t /* host_row_pitch */, size_t /* host_slice_pitch */,
    const void * /* ptr */, cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */,
    cl_event * /* event */) CL_API_SUFFIX__VERSION_1_1;
typedef cl_int(CL_API_CALL *PFN_clEnqueueFillBuffer)(
    cl_command_queue /* command_queue */, cl_mem /* buffer */,
    const void * /* pattern */, size_t /* pattern_size */, size_t /* offset */,
    size_t /* size */, cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */,
    cl_event * /* event */) CL_API_SUFFIX__VERSION_1_2;
typedef cl_int(CL_API_CALL *PFN_clEnqueueCopyBuffer)(
    cl_command_queue /* command_queue */, cl_mem /* src_buffer */,
    cl_mem /* dst_buffer */, size_t /* src_offset */, size_t /* dst_offset */,
    size_t /* size */, cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */,
    cl_event * /* event */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clEnqueueCopyBufferRect)(
    cl_command_queue /* command_queue */, cl_mem /* src_buffer */,
    cl_mem /* dst_buffer */, const size_t * /* src_origin */,
    const size_t * /* dst_origin */, const size_t * /* region */,
    size_t /* src_row_pitch */, size_t /* src_slice_pitch */,
    size_t /* dst_row_pitch */, size_t /* dst_slice_pitch */,
    cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */,
    cl_event * /* event */) CL_API_SUFFIX__VERSION_1_1;
typedef cl_int(CL_API_CALL *PFN_clEnqueueReadImage)(
    cl_command_queue /* command_queue */, cl_mem /* image */,
    cl_bool /* blocking_read */, const size_t * /* origin[3] */,
    const size_t * /* region[3] */, size_t /* row_pitch */,
    size_t /* slice_pitch */, void * /* ptr */,
    cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */,
    cl_event * /* event */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clEnqueueWriteImage)(
    cl_command_queue /* command_queue */, cl_mem /* image */,
    cl_bool /* blocking_write */, const size_t * /* origin[3] */,
    const size_t * /* region[3] */, size_t /* input_row_pitch */,
    size_t /* input_slice_pitch */, const void * /* ptr */,
    cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */,
    cl_event * /* event */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clEnqueueFillImage)(
    cl_command_queue /* command_queue */, cl_mem /* image */,
    const void * /* fill_color */, const size_t * /* origin[3] */,
    const size_t * /* region[3] */, cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */,
    cl_event * /* event */) CL_API_SUFFIX__VERSION_1_2;
typedef cl_int(CL_API_CALL *PFN_clEnqueueCopyImage)(
    cl_command_queue /* command_queue */, cl_mem /* src_image */,
    cl_mem /* dst_image */, const size_t * /* src_origin[3] */,
    const size_t * /* dst_origin[3] */, const size_t * /* region[3] */,
    cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */,
    cl_event * /* event */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clEnqueueCopyImageToBuffer)(
    cl_command_queue /* command_queue */, cl_mem /* src_image */,
    cl_mem /* dst_buffer */, const size_t * /* src_origin[3] */,
    const size_t * /* region[3] */, size_t /* dst_offset */,
    cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */,
    cl_event * /* event */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clEnqueueCopyBufferToImage)(
    cl_command_queue /* command_queue */, cl_mem /* src_buffer */,
    cl_mem /* dst_image */, size_t /* src_offset */,
    const size_t * /* dst_origin[3] */, const size_t * /* region[3] */,
    cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */,
    cl_event * /* event */) CL_API_SUFFIX__VERSION_1_0;
typedef void *(CL_API_CALL *PFN_clEnqueueMapBuffer)(
    cl_command_queue /* command_queue */, cl_mem /* buffer */,
    cl_bool /* blocking_map */, cl_map_flags /* map_flags */,
    size_t /* offset */, size_t /* size */,
    cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */, cl_event * /* event */,
    cl_int * /* errcode_ret */)CL_API_SUFFIX__VERSION_1_0;
typedef void *(CL_API_CALL *PFN_clEnqueueMapImage)(
    cl_command_queue /* command_queue */, cl_mem /* image */,
    cl_bool /* blocking_map */, cl_map_flags /* map_flags */,
    const size_t * /* origin[3] */, const size_t * /* region[3] */,
    size_t * /* image_row_pitch */, size_t * /* image_slice_pitch */,
    cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */, cl_event * /* event */,
    cl_int * /* errcode_ret */)CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clEnqueueUnmapMemObject)(
    cl_command_queue /* command_queue */, cl_mem /* memobj */,
    void * /* mapped_ptr */, cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */,
    cl_event * /* event */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clEnqueueMigrateMemObjects)(
    cl_command_queue /* command_queue */, cl_uint /* num_mem_objects */,
    const cl_mem * /* mem_objects */, cl_mem_migration_flags /* flags */,
    cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */,
    cl_event * /* event */) CL_API_SUFFIX__VERSION_1_2;
typedef cl_int(CL_API_CALL *PFN_clEnqueueNDRangeKernel)(
    cl_command_queue /* command_queue */, cl_kernel /* kernel */,
    cl_uint /* work_dim */, const size_t * /* global_work_offset */,
    const size_t * /* global_work_size */, const size_t * /* local_work_size */,
    cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */,
    cl_event * /* event */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clEnqueueNativeKernel)(
    cl_command_queue /* command_queue */,
    void(CL_CALLBACK * /*user_func*/)(void *), void * /* args */,
    size_t /* cb_args */, cl_uint /* num_mem_objects */,
    const cl_mem * /* mem_list */, const void ** /* args_mem_loc */,
    cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */,
    cl_event * /* event */) CL_API_SUFFIX__VERSION_1_0;
typedef cl_int(CL_API_CALL *PFN_clEnqueueMarkerWithWaitList)(
    cl_command_queue /* command_queue */, cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */,
    cl_event * /* event */) CL_API_SUFFIX__VERSION_1_2;
typedef cl_int(CL_API_CALL *PFN_clEnqueueBarrierWithWaitList)(
    cl_command_queue /* command_queue */, cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */,
    cl_event * /* event */) CL_API_SUFFIX__VERSION_1_2;
typedef cl_int(CL_API_CALL *PFN_clEnqueueSVMFree)(
    cl_command_queue /* command_queue */, cl_uint /* num_svm_pointers */,
    void *[] /* svm_pointers[] */,
    void(CL_CALLBACK * /*pfn_free_func*/)(cl_command_queue /* queue */,
                                          cl_uint /* num_svm_pointers */,
                                          void *[] /* svm_pointers[] */,
                                          void * /* user_data */),
    void * /* user_data */, cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */,
    cl_event * /* event */) CL_API_SUFFIX__VERSION_2_0;
typedef cl_int(CL_API_CALL *PFN_clEnqueueSVMMemcpy)(
    cl_command_queue /* command_queue */, cl_bool /* blocking_copy */,
    void * /* dst_ptr */, const void * /* src_ptr */, size_t /* size */,
    cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */,
    cl_event * /* event */) CL_API_SUFFIX__VERSION_2_0;
typedef cl_int(CL_API_CALL *PFN_clEnqueueSVMMemFill)(
    cl_command_queue /* command_queue */, void * /* svm_ptr */,
    const void * /* pattern */, size_t /* pattern_size */, size_t /* size */,
    cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */,
    cl_event * /* event */) CL_API_SUFFIX__VERSION_2_0;
typedef cl_int(CL_API_CALL *PFN_clEnqueueSVMMap)(
    cl_command_queue /* command_queue */, cl_bool /* blocking_map */,
    cl_map_flags /* flags */, void * /* svm_ptr */, size_t /* size */,
    cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */,
    cl_event * /* event */) CL_API_SUFFIX__VERSION_2_0;
typedef cl_int(CL_API_CALL *PFN_clEnqueueSVMUnmap)(
    cl_command_queue /* command_queue */, void * /* svm_ptr */,
    cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */,
    cl_event * /* event */) CL_API_SUFFIX__VERSION_2_0;
typedef void *(CL_API_CALL *PFN_clGetExtensionFunctionAddressForPlatform)(
    cl_platform_id /* platform */,
    const char * /* func_name */)CL_API_SUFFIX__VERSION_1_2;
typedef cl_mem(CL_API_CALL *PFN_clCreateImage2D)(
    cl_context /* context */, cl_mem_flags /* flags */,
    const cl_image_format * /* image_format */, size_t /* image_width */,
    size_t /* image_height */, size_t /* image_row_pitch */,
    void * /* host_ptr */, cl_int * /* errcode_ret */);
typedef cl_mem(CL_API_CALL *PFN_clCreateImage3D)(
    cl_context /* context */, cl_mem_flags /* flags */,
    const cl_image_format * /* image_format */, size_t /* image_width */,
    size_t /* image_height */, size_t /* image_depth */,
    size_t /* image_row_pitch */, size_t /* image_slice_pitch */,
    void * /* host_ptr */, cl_int * /* errcode_ret */);
typedef cl_int(CL_API_CALL *PFN_clEnqueueMarker)(
    cl_command_queue /* command_queue */, cl_event * /* event */);
typedef cl_int(CL_API_CALL *PFN_clEnqueueWaitForEvents)(
    cl_command_queue /* command_queue */, cl_uint /* num_events */,
    const cl_event * /* event_list */);
typedef cl_int(CL_API_CALL *PFN_clEnqueueBarrier)(
    cl_command_queue /* command_queue */);
typedef cl_int(CL_API_CALL *PFN_clUnloadCompiler)();
typedef void *(CL_API_CALL *PFN_clGetExtensionFunctionAddress)(
    const char * /* func_name */);
typedef cl_command_queue(CL_API_CALL *PFN_clCreateCommandQueue)(
    cl_context /* context */, cl_device_id /* device */,
    cl_command_queue_properties /* properties */, cl_int * /* errcode_ret */);
typedef cl_sampler(CL_API_CALL *PFN_clCreateSampler)(
    cl_context /* context */, cl_bool /* normalized_coords */,
    cl_addressing_mode /* addressing_mode */, cl_filter_mode /* filter_mode */,
    cl_int * /* errcode_ret */);
typedef cl_int(CL_API_CALL *PFN_clEnqueueTask)(
    cl_command_queue /* command_queue */, cl_kernel /* kernel */,
    cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */, cl_event * /* event */);

// OpenGL sharing
typedef cl_mem(CL_API_CALL *PFN_clCreateFromGLBuffer)(cl_context, cl_mem_flags,
                                                      cl_GLuint, int *);
typedef cl_mem(CL_API_CALL *PFN_clCreateFromGLTexture)(
    cl_context /* context */, cl_mem_flags /* flags */, cl_GLenum /* target */,
    cl_GLint /* miplevel */, cl_GLuint /* texture */,
    cl_int * /* errcode_ret */) CL_API_SUFFIX__VERSION_1_2;
typedef cl_int(CL_API_CALL *PFN_clEnqueueAcquireGLObjects)(
    cl_command_queue /* command_queue */, cl_uint /* num_objects */,
    const cl_mem * /* mem_objects */, cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */, cl_event * /* event */);
typedef cl_int(CL_API_CALL *PFN_clEnqueueReleaseGLObjects)(
    cl_command_queue /* command_queue */, cl_uint /* num_objects */,
    const cl_mem * /* mem_objects */, cl_uint /* num_events_in_wait_list */,
    const cl_event * /* event_wait_list */,
    cl_event * /* event */) CL_API_SUFFIX__VERSION_1_0;

// cl_khr_egl_event extension

// CLeglDisplayKHR is an opaque handle to an EGLDisplay
typedef void *CLeglDisplayKHR;

// CLeglSyncKHR is an opaque handle to an EGLSync object
typedef void *CLeglSyncKHR;

typedef cl_event(CL_API_CALL *PFN_clCreateEventFromEGLSyncKHR)(
    cl_context /* context */, CLeglSyncKHR /* sync */,
    CLeglDisplayKHR /* display */, cl_int * /* errcode_ret */);

// EGL sharing
typedef cl_mem(CL_API_CALL *PFN_clCreateFromEGLImageKHR)(
    cl_context /*context*/, CLeglDisplayKHR /*display*/,
    CLeglImageKHR /*image*/, cl_mem_flags /*flags*/,
    const cl_egl_image_properties_khr * /*properties*/,
    cl_int * /*errcode_ret*/);
typedef cl_int(CL_API_CALL *PFN_clEnqueueAcquireEGLObjectsKHR)(
    cl_command_queue /*command_queue*/, cl_uint /*num_objects*/,
    const cl_mem * /*mem_objects*/, cl_uint /*num_events_in_wait_list*/,
    const cl_event * /*event_wait_list*/, cl_event * /*event*/);
typedef cl_int(CL_API_CALL *PFN_clEnqueueReleaseEGLObjectsKHR)(
    cl_command_queue /*command_queue*/, cl_uint /*num_objects*/,
    const cl_mem * /*mem_objects*/, cl_uint /*num_events_in_wait_list*/,
    const cl_event * /*event_wait_list*/, cl_event * /*event*/);

// cl_khr_command_buffer
typedef cl_command_buffer_khr(CL_API_CALL *PFN_clCreateCommandBufferKHR)(
    cl_uint /*num_queues*/, const cl_command_queue * /*queues*/,
    const cl_command_buffer_properties_khr * /*properties*/,
    cl_int * /*errcode_ret*/);

typedef cl_int(CL_API_CALL *PFN_clRetainCommandBufferKHR)(
    cl_command_buffer_khr /*command_buffer*/);

typedef cl_int(CL_API_CALL *PFN_clReleaseCommandBufferKHR)(
    cl_command_buffer_khr /*command_buffer*/);

typedef cl_int(CL_API_CALL *PFN_clFinalizeCommandBufferKHR)(
    cl_command_buffer_khr /*command_buffer*/);

typedef cl_int(CL_API_CALL *PFN_clEnqueueCommandBufferKHR)(
    cl_uint /*num_queues*/, cl_command_queue * /*queues*/,
    cl_command_buffer_khr /*command_buffer*/,
    cl_uint /*num_events_in_wait_list*/, const cl_event * /*event_wait_list*/,
    cl_event * /*event*/);

typedef cl_int(CL_API_CALL *PFN_clCommandNDRangeKernelKHR)(
    cl_command_buffer_khr /*command_buffer*/,
    cl_command_queue /*command_queue*/,
    const cl_ndrange_kernel_command_properties_khr * /*properties*/,
    cl_kernel /*kernel*/, cl_uint /*work_dim*/,
    const size_t * /*global_work_offset*/, const size_t * /*global_work_size*/,
    const size_t * /*local_work_size*/,
    cl_uint /*num_sync_points_in_wait_list*/,
    const cl_sync_point_khr * /*sync_point_wait_list*/,
    cl_sync_point_khr * /*sync_point*/,
    cl_mutable_command_khr * /*mutable_handle*/);

typedef cl_int(CL_API_CALL *PFN_clGetCommandBufferInfoKHR)(
    cl_command_buffer_khr /*command_buffer*/,
    cl_command_buffer_info_khr /*param_name*/, size_t /*param_value_size*/,
    void * /*param_value*/, size_t * /*param_value_size_ret*/);

extern PFN_clGetPlatformIDs clGetPlatformIDs;
extern PFN_clGetPlatformInfo clGetPlatformInfo;
extern PFN_clGetDeviceIDs clGetDeviceIDs;
extern PFN_clGetDeviceInfo clGetDeviceInfo;
extern PFN_clCreateSubDevices clCreateSubDevices;
extern PFN_clRetainDevice clRetainDevice;
extern PFN_clReleaseDevice clReleaseDevice;
extern PFN_clCreateContext clCreateContext;
extern PFN_clCreateContextFromType clCreateContextFromType;
extern PFN_clRetainContext clRetainContext;
extern PFN_clReleaseContext clReleaseContext;
extern PFN_clGetContextInfo clGetContextInfo;
extern PFN_clCreateCommandQueueWithProperties
    clCreateCommandQueueWithProperties;
extern PFN_clRetainCommandQueue clRetainCommandQueue;
extern PFN_clReleaseCommandQueue clReleaseCommandQueue;
extern PFN_clGetCommandQueueInfo clGetCommandQueueInfo;
extern PFN_clCreateBuffer clCreateBuffer;
extern PFN_clCreateSubBuffer clCreateSubBuffer;
extern PFN_clCreateImage clCreateImage;
extern PFN_clCreatePipe clCreatePipe;
extern PFN_clRetainMemObject clRetainMemObject;
extern PFN_clReleaseMemObject clReleaseMemObject;
extern PFN_clGetSupportedImageFormats clGetSupportedImageFormats;
extern PFN_clGetMemObjectInfo clGetMemObjectInfo;
extern PFN_clGetImageInfo clGetImageInfo;
extern PFN_clGetPipeInfo clGetPipeInfo;
extern PFN_clSetMemObjectDestructorCallback clSetMemObjectDestructorCallback;
extern PFN_clSVMAlloc clSVMAlloc;
extern PFN_clSVMFree clSVMFree;
extern PFN_clCreateSamplerWithProperties clCreateSamplerWithProperties;
extern PFN_clRetainSampler clRetainSampler;
extern PFN_clReleaseSampler clReleaseSampler;
extern PFN_clGetSamplerInfo clGetSamplerInfo;
extern PFN_clCreateProgramWithSource clCreateProgramWithSource;
extern PFN_clCreateProgramWithBinary clCreateProgramWithBinary;
extern PFN_clCreateProgramWithBuiltInKernels clCreateProgramWithBuiltInKernels;
extern PFN_clRetainProgram clRetainProgram;
extern PFN_clReleaseProgram clReleaseProgram;
extern PFN_clBuildProgram clBuildProgram;
extern PFN_clCompileProgram clCompileProgram;
extern PFN_clLinkProgram clLinkProgram;
extern PFN_clUnloadPlatformCompiler clUnloadPlatformCompiler;
extern PFN_clGetProgramInfo clGetProgramInfo;
extern PFN_clGetProgramBuildInfo clGetProgramBuildInfo;
extern PFN_clCreateKernel clCreateKernel;
extern PFN_clCreateKernelsInProgram clCreateKernelsInProgram;
extern PFN_clRetainKernel clRetainKernel;
extern PFN_clReleaseKernel clReleaseKernel;
extern PFN_clSetKernelArg clSetKernelArg;
extern PFN_clSetKernelArgSVMPointer clSetKernelArgSVMPointer;
extern PFN_clSetKernelExecInfo clSetKernelExecInfo;
extern PFN_clGetKernelInfo clGetKernelInfo;
extern PFN_clGetKernelArgInfo clGetKernelArgInfo;
extern PFN_clGetKernelWorkGroupInfo clGetKernelWorkGroupInfo;
extern PFN_clWaitForEvents clWaitForEvents;
extern PFN_clGetEventInfo clGetEventInfo;
extern PFN_clCreateUserEvent clCreateUserEvent;
extern PFN_clRetainEvent clRetainEvent;
extern PFN_clReleaseEvent clReleaseEvent;
extern PFN_clSetUserEventStatus clSetUserEventStatus;
extern PFN_clSetEventCallback clSetEventCallback;
extern PFN_clGetEventProfilingInfo clGetEventProfilingInfo;
extern PFN_clFlush clFlush;
extern PFN_clFinish clFinish;
extern PFN_clEnqueueReadBuffer clEnqueueReadBuffer;
extern PFN_clEnqueueReadBufferRect clEnqueueReadBufferRect;
extern PFN_clEnqueueWriteBuffer clEnqueueWriteBuffer;
extern PFN_clEnqueueWriteBufferRect clEnqueueWriteBufferRect;
extern PFN_clEnqueueFillBuffer clEnqueueFillBuffer;
extern PFN_clEnqueueCopyBuffer clEnqueueCopyBuffer;
extern PFN_clEnqueueCopyBufferRect clEnqueueCopyBufferRect;
extern PFN_clEnqueueReadImage clEnqueueReadImage;
extern PFN_clEnqueueWriteImage clEnqueueWriteImage;
extern PFN_clEnqueueFillImage clEnqueueFillImage;
extern PFN_clEnqueueCopyImage clEnqueueCopyImage;
extern PFN_clEnqueueCopyImageToBuffer clEnqueueCopyImageToBuffer;
extern PFN_clEnqueueCopyBufferToImage clEnqueueCopyBufferToImage;
extern PFN_clEnqueueMapBuffer clEnqueueMapBuffer;
extern PFN_clEnqueueMapImage clEnqueueMapImage;
extern PFN_clEnqueueUnmapMemObject clEnqueueUnmapMemObject;
extern PFN_clEnqueueMigrateMemObjects clEnqueueMigrateMemObjects;
extern PFN_clEnqueueNDRangeKernel clEnqueueNDRangeKernel;
extern PFN_clEnqueueNativeKernel clEnqueueNativeKernel;
extern PFN_clEnqueueMarkerWithWaitList clEnqueueMarkerWithWaitList;
extern PFN_clEnqueueBarrierWithWaitList clEnqueueBarrierWithWaitList;
extern PFN_clEnqueueSVMFree clEnqueueSVMFree;
extern PFN_clEnqueueSVMMemcpy clEnqueueSVMMemcpy;
extern PFN_clEnqueueSVMMemFill clEnqueueSVMMemFill;
extern PFN_clEnqueueSVMMap clEnqueueSVMMap;
extern PFN_clEnqueueSVMUnmap clEnqueueSVMUnmap;
extern PFN_clGetExtensionFunctionAddressForPlatform
    clGetExtensionFunctionAddressForPlatform;
extern PFN_clCreateImage2D clCreateImage2D;
extern PFN_clCreateImage3D clCreateImage3D;
extern PFN_clEnqueueMarker clEnqueueMarker;
extern PFN_clEnqueueWaitForEvents clEnqueueWaitForEvents;
extern PFN_clEnqueueBarrier clEnqueueBarrier;
extern PFN_clUnloadCompiler clUnloadCompiler;
extern PFN_clGetExtensionFunctionAddress clGetExtensionFunctionAddress;
extern PFN_clCreateCommandQueue clCreateCommandQueue;
extern PFN_clCreateSampler clCreateSampler;
extern PFN_clEnqueueTask clEnqueueTask;

// OpenGL sharing
extern PFN_clCreateFromGLBuffer clCreateFromGLBuffer;
extern PFN_clCreateFromGLTexture clCreateFromGLTexture;
extern PFN_clEnqueueAcquireGLObjects clEnqueueAcquireGLObjects;
extern PFN_clEnqueueReleaseGLObjects clEnqueueReleaseGLObjects;

// cl_khr_egl_event extension
extern PFN_clCreateEventFromEGLSyncKHR clCreateEventFromEGLSyncKHR;

// EGL sharing
extern PFN_clCreateFromEGLImageKHR clCreateFromEGLImageKHR;
extern PFN_clEnqueueAcquireEGLObjectsKHR clEnqueueAcquireEGLObjectsKHR;
extern PFN_clEnqueueReleaseEGLObjectsKHR clEnqueueReleaseEGLObjectsKHR;

// cl_khr_command_buffer extension
extern PFN_clCreateCommandBufferKHR clCreateCommandBufferKHR;
extern PFN_clRetainCommandBufferKHR clRetainCommandBufferKHR;
extern PFN_clReleaseCommandBufferKHR clReleaseCommandBufferKHR;
extern PFN_clFinalizeCommandBufferKHR clFinalizeCommandBufferKHR;
extern PFN_clEnqueueCommandBufferKHR clEnqueueCommandBufferKHR;
extern PFN_clCommandNDRangeKernelKHR clCommandNDRangeKernelKHR;
extern PFN_clGetCommandBufferInfoKHR clGetCommandBufferInfoKHR;

// For convenient image creation
// It uses clCreateImage if it available (clCreateImage available since cl 1.2)
// otherwise it will use legacy clCreateImage2D
cl_mem CreateImage2DLegacy(cl_context context, cl_mem_flags flags,
                           const cl_image_format *image_format,
                           const cl_image_desc *image_desc, void *host_ptr,
                           cl_int *errcode_ret);

// It uses clCreateImage if it available (clCreateImage available since cl 1.2)
// otherwise it will use legacy clCreateImage3D
cl_mem CreateImage3DLegacy(cl_context context, cl_mem_flags flags,
                           const cl_image_format *image_format,
                           const cl_image_desc *image_desc, void *host_ptr,
                           cl_int *errcode_ret);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_OPENCL_WRAPPER_H_
