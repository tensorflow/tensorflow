/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/cudnn_scratch_allocator.h"

namespace tensorflow {

CudnnAllocatorInTemp::~CudnnAllocatorInTemp() {}

CudnnAllocatorInTemp::CudnnAllocatorInTemp(OpKernelContext* context)
    : context_(context) {}

int64 CudnnAllocatorInTemp::GetMemoryLimitInBytes() {
  return std::numeric_limits<int64>::max();
}

StatusOr<DeviceMemory<uint8>> CudnnAllocatorInTemp::AllocateBytes(
    int64 byte_size) {
  Tensor temporary_memory;
  const DataType tf_data_type = DataTypeToEnum<uint8>::v();
  int64 allocate_count =
      Eigen::divup(byte_size, static_cast<int64>(sizeof(uint8)));
  Status allocation_status(context_->allocate_temp(
      tf_data_type, TensorShape({allocate_count}), &temporary_memory));
  if (!allocation_status.ok()) {
    return allocation_status;
  }
  // Hold the reference of the allocated tensors until the end of the
  // allocator.
  allocated_tensors_.push_back(temporary_memory);
  total_byte_size_ += byte_size;
  return DeviceMemory<uint8>::MakeFromByteSize(
      temporary_memory.template flat<uint8>().data(),
      temporary_memory.template flat<uint8>().size() * sizeof(uint8));
}

int64 CudnnAllocatorInTemp::TotalByteSize() const {
  return total_byte_size_;
}

Tensor CudnnAllocatorInTemp::get_allocated_tensor(int index) const {
  return allocated_tensors_[index];
}

} // namespace tensorflow
