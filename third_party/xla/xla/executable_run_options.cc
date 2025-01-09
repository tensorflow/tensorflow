/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/executable_run_options.h"

#include <atomic>
#include <cstdint>
#include <string>

namespace xla {

RunId::RunId() {
  static std::atomic<int64_t> counter{0};
  data_ = counter.fetch_add(1);
}

bool operator==(const RunId& a, const RunId& b) { return a.data_ == b.data_; }

std::string RunId::ToString() const {
  return "RunId: " + std::to_string(data_);
}

int64_t RunId::ToInt() const { return data_; }

ExecutableRunOptions& ExecutableRunOptions::set_device_ordinal(
    int device_ordinal) {
  device_ordinal_ = device_ordinal;
  return *this;
}

int ExecutableRunOptions::device_ordinal() const { return device_ordinal_; }

ExecutableRunOptions& ExecutableRunOptions::set_physical_device_ordinal(
    int physical_device_ordinal) {
  physical_device_ordinal_ = physical_device_ordinal;
  return *this;
}

int ExecutableRunOptions::physical_device_ordinal() const {
  return physical_device_ordinal_;
}

ExecutableRunOptions& ExecutableRunOptions::set_allocator(
    stream_executor::DeviceMemoryAllocator* allocator) {
  allocator_ = allocator;
  return *this;
}

stream_executor::DeviceMemoryAllocator* ExecutableRunOptions::allocator()
    const {
  return allocator_;
}

ExecutableRunOptions& ExecutableRunOptions::set_stream(
    stream_executor::Stream* stream) {
  stream_ = stream;
  return *this;
}

stream_executor::Stream* ExecutableRunOptions::stream() const {
  return stream_;
}

ExecutableRunOptions& ExecutableRunOptions::set_host_to_device_stream(
    stream_executor::Stream* stream) {
  host_to_device_stream_ = stream;
  return *this;
}

stream_executor::Stream* ExecutableRunOptions::host_to_device_stream() const {
  return host_to_device_stream_;
}

ExecutableRunOptions& ExecutableRunOptions::set_device_to_host_stream(
    stream_executor::Stream* stream) {
  device_to_host_stream_ = stream;
  return *this;
}

stream_executor::Stream* ExecutableRunOptions::device_to_host_stream() const {
  return device_to_host_stream_;
}

ExecutableRunOptions& ExecutableRunOptions::set_intra_op_thread_pool(
    const Eigen::ThreadPoolDevice* intra_op_thread_pool) {
  intra_op_thread_pool_ = intra_op_thread_pool;
  return *this;
}

const Eigen::ThreadPoolDevice* ExecutableRunOptions::intra_op_thread_pool()
    const {
  return intra_op_thread_pool_;
}

ExecutableRunOptions& ExecutableRunOptions::set_execution_profile(
    ExecutionProfile* profile) {
  execution_profile_ = profile;
  return *this;
}

ExecutionProfile* ExecutableRunOptions::execution_profile() const {
  return execution_profile_;
}

ExecutableRunOptions& ExecutableRunOptions::set_device_assignment(
    const DeviceAssignment* device_assignment) {
  device_assignment_ = device_assignment;
  return *this;
}

const DeviceAssignment* ExecutableRunOptions::device_assignment() const {
  return device_assignment_;
}

ExecutableRunOptions& ExecutableRunOptions::set_gpu_executable_run_options(
    const gpu::GpuExecutableRunOptions* gpu_executable_run_options) {
  gpu_executable_run_options_ = gpu_executable_run_options;
  return *this;
}

const gpu::GpuExecutableRunOptions*
ExecutableRunOptions::gpu_executable_run_options() const {
  return gpu_executable_run_options_;
}

ExecutableRunOptions& ExecutableRunOptions::set_cpu_executable_run_options(
    const cpu::CpuExecutableRunOptions* cpu_executable_run_options) {
  cpu_executable_run_options_ = cpu_executable_run_options;
  return *this;
}

const cpu::CpuExecutableRunOptions*
ExecutableRunOptions::cpu_executable_run_options() const {
  return cpu_executable_run_options_;
}

ExecutableRunOptions& ExecutableRunOptions::set_ffi_execution_context(
    const ffi::ExecutionContext* ffi_execution_context) {
  ffi_execution_context_ = ffi_execution_context;
  return *this;
}

const ffi::ExecutionContext* ExecutableRunOptions::ffi_execution_context()
    const {
  return ffi_execution_context_;
}

ExecutableRunOptions& ExecutableRunOptions::set_rng_seed(int rng_seed) {
  rng_seed_ = rng_seed;
  return *this;
}

int ExecutableRunOptions::rng_seed() const { return rng_seed_; }

ExecutableRunOptions& ExecutableRunOptions::set_run_id(RunId id) {
  run_id_ = id;
  return *this;
}

RunId ExecutableRunOptions::run_id() const { return run_id_; }

ExecutableRunOptions& ExecutableRunOptions::set_local_device_count(
    int local_device_count) {
  local_device_count_ = local_device_count;
  return *this;
}
int ExecutableRunOptions::local_device_count() const {
  return local_device_count_;
}

}  // namespace xla
