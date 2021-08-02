/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/util/autotune_maps/conv_parameters.h"

#include <vector>

#include "absl/strings/str_format.h"
#include "tensorflow/core/platform/hash.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/autotune_maps/conv_parameters.pb.h"
#include "tensorflow/stream_executor/gpu/gpu_driver.h"

namespace tensorflow {

namespace {
using ::stream_executor::gpu::GpuDeviceHandle;
using ::stream_executor::gpu::GpuDriver;
using ::tensorflow::protobuf::util::MessageDifferencer;
// Given a device_id, this function computes an identifier string that
// represents the corresponding GPU device type. Currently the identifier is
// computed as
// "<device_name> <compute_compatibility> <GPU_memory> <multiprocessor_count>".
// We cannot simply use <device_name> output by GetDeviceName here because for
// some GPUs the it will output uninformative names like "Graphics Device",
// which cannot identify device types of GPUs.
// TODO(ruochengw): Replace the identifier with something that uniquely
// determines a GPU device type, e.g. PCI device ID.
StatusOr<string> DeviceIdToIdentifierHelper(int device_id) {
  GpuDeviceHandle device;
  TF_RETURN_IF_ERROR(GpuDriver::GetDevice(device_id, &device));
  std::string device_name;
  TF_RETURN_IF_ERROR(GpuDriver::GetDeviceName(device, &device_name));
  int cc_major;
  int cc_minor;
  TF_RETURN_IF_ERROR(
      GpuDriver::GetComputeCapability(&cc_major, &cc_minor, device));

  uint64 device_memory_size = -1;
  if (!GpuDriver::GetDeviceTotalMemory(device, &device_memory_size)) {
    return errors::Internal("Failed to get device's total memory");
  }

  int core_count;
  TF_ASSIGN_OR_RETURN(core_count, GpuDriver::GetMultiprocessorCount(device));
  return absl::StrFormat("%s sm_%d.%d with %dB RAM and %d cores", device_name,
                         cc_major, cc_minor, device_memory_size, core_count);
}

// Precomputes a map storing the results of DeviceIdToIdentifierHelper for all
// device_ids available and outputs "Unknown Graphics Device" when
// DeviceIdToIdentifierHelper returns an error.
std::vector<string> GetDeviceIdToIdentifierMap() {
  int device_count = GpuDriver::GetDeviceCount();
  std::vector<string> map(device_count);
  for (int device_id = 0; device_id < device_count; device_id++) {
    StatusOr<string> device_identifier_or_status =
        DeviceIdToIdentifierHelper(device_id);
    if (device_identifier_or_status.ok()) {
      map[device_id] = device_identifier_or_status.ValueOrDie();
    } else {
      map[device_id] = "Unknown Graphics Device";
    }
  }
  return map;
}

string DeviceIdToIdentifier(int device_id) {
  // Ensure the static variable is trivially destructible and thus safe to be
  // destruct in multi-thread setting.
  static const auto& map =
      *new std::vector<string>(GetDeviceIdToIdentifierMap());
  if (device_id >= map.size()) {
    return "Unknown Graphics Device";
  } else {
    return map[device_id];
  }
}

uint64 ComputeHash(const ConvParametersProto& proto) {
  string proto_serialized_string;
  // Use scope to make sure StringOutputStream is destroyed so that contents
  // have been fully written to proto_serialized_string.
  {
    protobuf::io::StringOutputStream string_stream(&proto_serialized_string);
    protobuf::io::CodedOutputStream stream(&string_stream);
    // Ensure the serialization is deterministic so that equal ConvParameters
    // have equal serialized strings and therefore equal hash codes.
    stream.SetSerializationDeterministic(true);
    proto.SerializeToCodedStream(&stream);
  }
  return Hash64(proto_serialized_string);
}
}  // namespace

ConvParameters::ConvParameters(
    int64_t batch, int64_t in_depths, const absl::Span<const int64_t> in,
    int data_format, int64_t out_depths, const absl::Span<const int64_t> filter,
    const absl::Span<const int64_t> dilation,
    const absl::Span<const int64_t> stride,
    const absl::Span<const int64_t> padding, DataType dtype, int device_id,
    int group_count, bool has_side_input,
    stream_executor::dnn::ActivationMode activation_mode) {
  proto_.set_batch(batch);
  proto_.set_in_depths(in_depths);
  *proto_.mutable_in() = {in.begin(), in.end()};
  proto_.set_data_format(static_cast<int>(data_format));
  proto_.set_out_depths(out_depths);
  *proto_.mutable_filter() = {filter.begin(), filter.end()};
  *proto_.mutable_dilation() = {dilation.begin(), dilation.end()};
  *proto_.mutable_stride() = {stride.begin(), stride.end()};
  *proto_.mutable_padding() = {padding.begin(), padding.end()};
  proto_.set_dtype(dtype);
  proto_.set_group_count(group_count);
  proto_.mutable_fusion()->set_has_side_input(has_side_input);
  proto_.mutable_fusion()->set_activation_mode(activation_mode);
  proto_.set_device_identifier(DeviceIdToIdentifier(device_id));
  hash_code_ = ComputeHash(proto_);
}

ConvParameters::ConvParameters(const ConvParametersProto& proto)
    : proto_(proto) {
  hash_code_ = ComputeHash(proto_);
}

bool ConvParameters::operator==(const ConvParameters& other) const {
  return MessageDifferencer::Equals(this->proto_, other.proto_);
}

string ConvParameters::ToString() const { return proto_.DebugString(); }

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
