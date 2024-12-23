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
#include "tensorflow/core/util/memmapped_file_system_writer.h"

#include <algorithm>

namespace tensorflow {

absl::Status MemmappedFileSystemWriter::InitializeToFile(
    Env* env, const string& filename) {
  auto status = env->NewWritableFile(filename, &output_file_);
  if (status.ok()) {
    output_file_offset_ = 0;
  }
  return status;
}

absl::Status MemmappedFileSystemWriter::SaveTensor(const Tensor& tensor,
                                                   const string& element_name) {
  if (!output_file_) {
    return errors::FailedPrecondition(
        "MemmappedEnvWritter: saving tensor into not opened file");
  }
  if (!MemmappedFileSystem::IsWellFormedMemmappedPackageFilename(
          element_name)) {
    return errors::InvalidArgument(
        "MemmappedEnvWritter: element_name is invalid: must have memmapped ",
        "package prefix ", MemmappedFileSystem::kMemmappedPackagePrefix,
        " and include [A-Za-z0-9_.]");
  }
  const auto tensor_data = tensor.tensor_data();
  if (tensor_data.empty()) {
    return errors::InvalidArgument(
        "MemmappedEnvWritter: saving tensor with 0 size");
  }
  // Adds pad for correct alignment after memmapping.
  TF_RETURN_IF_ERROR(AdjustAlignment(Allocator::kAllocatorAlignment));
  AddToDirectoryElement(element_name, tensor_data.size());
  const auto result = output_file_->Append(tensor_data);
  if (result.ok()) {
    output_file_offset_ += tensor_data.size();
  }
  return result;
}

absl::Status MemmappedFileSystemWriter::SaveProtobuf(
    const protobuf::MessageLite& message, const string& element_name) {
  if (!output_file_) {
    return errors::FailedPrecondition(
        "MemmappedEnvWritter: saving protobuf into not opened file");
  }
  if (!MemmappedFileSystem::IsWellFormedMemmappedPackageFilename(
          element_name)) {
    return errors::InvalidArgument(
        "MemmappedEnvWritter: element_name is invalid: must have memmapped "
        "package prefix ",
        MemmappedFileSystem::kMemmappedPackagePrefix,
        " and include [A-Za-z0-9_.]");
  }
  const string encoded = message.SerializeAsString();
  AddToDirectoryElement(element_name, encoded.size());
  const auto res = output_file_->Append(encoded);
  if (res.ok()) {
    output_file_offset_ += encoded.size();
  }
  return res;
}

namespace {

absl::string_view EncodeUint64LittleEndian(uint64 val, char* output_buffer) {
  for (unsigned int i = 0; i < sizeof(uint64); ++i) {
    output_buffer[i] = (val >> i * 8);
  }
  return {output_buffer, sizeof(uint64)};
}

}  // namespace

absl::Status MemmappedFileSystemWriter::FlushAndClose() {
  if (!output_file_) {
    return errors::FailedPrecondition(
        "MemmappedEnvWritter: flushing into not opened file");
  }
  const string dir = directory_.SerializeAsString();
  TF_RETURN_IF_ERROR(output_file_->Append(dir));

  // Write the directory offset.
  char buffer[sizeof(uint64)];
  TF_RETURN_IF_ERROR(output_file_->Append(
      EncodeUint64LittleEndian(output_file_offset_, buffer)));

  // Flush and close the file.
  TF_RETURN_IF_ERROR(output_file_->Flush());
  TF_RETURN_IF_ERROR(output_file_->Close());
  output_file_.reset();
  return absl::OkStatus();
}

absl::Status MemmappedFileSystemWriter::AdjustAlignment(uint64 alignment) {
  const uint64 alignment_rest = output_file_offset_ % alignment;
  const uint64 to_write_for_alignment =
      (alignment_rest == 0) ? 0 : alignment - (output_file_offset_ % alignment);
  static constexpr uint64 kFillerBufferSize = 16;
  const char kFillerBuffer[kFillerBufferSize] = {};
  for (uint64 rest = to_write_for_alignment; rest > 0;) {
    absl::string_view sp(kFillerBuffer, std::min(rest, kFillerBufferSize));
    TF_RETURN_IF_ERROR(output_file_->Append(sp));
    rest -= sp.size();
    output_file_offset_ += sp.size();
  }
  return absl::OkStatus();
}

void MemmappedFileSystemWriter::AddToDirectoryElement(const string& name,
                                                      uint64 length) {
  MemmappedFileSystemDirectoryElement* new_directory_element =
      directory_.add_element();
  new_directory_element->set_offset(output_file_offset_);
  new_directory_element->set_name(name);
  new_directory_element->set_length(length);
}

}  // namespace tensorflow
