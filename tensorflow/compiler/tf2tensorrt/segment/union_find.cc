/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2tensorrt/segment/union_find.h"

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {
namespace segment {

namespace {
template <typename T>
inline bool CheckIfCompatible(const std::optional<T>& a,
                              const std::optional<T>& b) {
  if (a.has_value() && b.has_value()) {
    return *a == *b;
  }
  return true;
}

template <typename T>
inline bool UnifyValues(std::optional<T>& a, std::optional<T>& b) {
  if (a.has_value()) {
    b = a;
  } else {
    a = b;
  }
  return true;
}

template <typename T>
inline std::optional<T> MergeCompatible(const std::optional<T>& a,
                                        const std::optional<T>& b) {
  DCHECK(CheckIfCompatible(a, b));
  return a.has_value() ? a : b;
}

}  // namespace

ClusterBatchSize::ClusterBatchSize()
    : batch_size_(std::nullopt), max_batch_size_(std::nullopt) {}

bool ClusterBatchSize::operator==(const ClusterBatchSize& other) {
  return batch_size_ == other.batch_size_ &&
         max_batch_size_ == other.max_batch_size_;
}

ClusterBatchSize& ClusterBatchSize::SetBatchSize(int batch_size) {
  SetBatchSize(static_cast<std::optional<int>>(batch_size));
  return *this;
}

ClusterBatchSize& ClusterBatchSize::SetBatchSize(
    const std::optional<int>& batch_size) {
  batch_size_ = MergeCompatible<int>(batch_size_, batch_size);
  if (batch_size_.has_value() && batch_size_.value() >= 0) {
    SetMaxBatchSize(batch_size_);
  }
  return *this;
}

bool ClusterBatchSize::HasBatchSize() const { return batch_size_.has_value(); }

int ClusterBatchSize::GetBatchSize() const {
  DCHECK(HasBatchSize());
  return batch_size_.value();
}

ClusterBatchSize& ClusterBatchSize::SetMaxBatchSize(int max_batch_size) {
  SetBatchSize(static_cast<std::optional<int>>(max_batch_size));
  return *this;
}

ClusterBatchSize& ClusterBatchSize::SetMaxBatchSize(
    const std::optional<int>& max_batch_size) {
  max_batch_size_ = MergeCompatible<int>(max_batch_size_, max_batch_size);
  return *this;
}

std::optional<int> ClusterBatchSize::GetOptionalMaxBatchSize() const {
  return max_batch_size_;
}

bool ClusterBatchSize::MergeIfCompatible(const ClusterBatchSize& other) {
  if (!CheckIfCompatible(batch_size_, other.batch_size_) ||
      !CheckIfCompatible(max_batch_size_, other.max_batch_size_)) {
    return false;
  }

  SetBatchSize(other.batch_size_);
  SetMaxBatchSize(other.max_batch_size_);
  return true;
}

string ClusterBatchSize::ToString() const {
  string s;
  const auto append_optional_num = [&](const std::optional<int>& num) {
    if (num.has_value()) {
      absl::StrAppendFormat(&s, "%d", num.value());
    } else {
      absl::StrAppendFormat(&s, "?");
    }
  };
  absl::StrAppendFormat(&s, "batch_size=");
  append_optional_num(batch_size_);
  absl::StrAppendFormat(&s, ", max_batch_size=");
  append_optional_num(max_batch_size_);
  return s;
}

ClusterProperty::ClusterProperty(const ClusterBatchSize& batch_size,
                                 const DeviceNameUtils::ParsedName& device_name)
    : batch_size_(batch_size), device_name_(device_name) {}

Status ClusterProperty::Merge(const ClusterProperty& other) {
  ClusterBatchSize merged_batch_size(batch_size_);
  if (!merged_batch_size.MergeIfCompatible(other.batch_size_)) {
    return errors::Internal(
        "trying to merge clusters with incompatible batch sizes.");
  }

  std::optional<DeviceNameUtils::ParsedName> merged_device_name =
      MergeIfCompatible(device_name_, other.device_name_);
  if (!merged_device_name.has_value()) {
    return errors::Internal(
        "trying to merge clusters with incompatible device assignment.");
  }

  batch_size_ = std::move(merged_batch_size);
  device_name_ = std::move(merged_device_name.value());
  return OkStatus();
}

}  // namespace segment
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
