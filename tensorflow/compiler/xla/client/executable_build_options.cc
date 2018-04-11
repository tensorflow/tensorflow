/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/executable_build_options.h"

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace xla {

ExecutableBuildOptions& ExecutableBuildOptions::set_device_allocator(
    DeviceMemoryAllocator* allocator) {
  device_allocator_ = allocator;
  return *this;
}

DeviceMemoryAllocator* ExecutableBuildOptions::device_allocator() const {
  return device_allocator_;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_device_ordinal(
    int device_ordinal) {
  CHECK_GE(device_ordinal, 0);
  device_ordinal_ = device_ordinal;
  return *this;
}

int ExecutableBuildOptions::device_ordinal() const { return device_ordinal_; }

ExecutableBuildOptions& ExecutableBuildOptions::set_result_layout(
    const Shape& shape_with_layout) {
  result_layout_set_ = true;
  result_layout_ = shape_with_layout;
  return *this;
}

const Shape* ExecutableBuildOptions::result_layout() const {
  return result_layout_set_ ? &result_layout_ : nullptr;
}

string ExecutableBuildOptions::ToString() const {
  string result_layout = "nullopt";
  if (result_layout_set_) {
    result_layout = ShapeUtil::HumanStringWithLayout(result_layout_);
  }
  string generate_hlo_graph = "nullopt";
  if (generate_hlo_graph_.has_value()) {
    generate_hlo_graph = generate_hlo_graph_.value();
  }
  return tensorflow::strings::Printf(
      "ExecutableBuildOptions{device_ordinal=%d, result_layout=%s, "
      "generate_hlo_graph=%s}",
      device_ordinal_, result_layout.c_str(), generate_hlo_graph.c_str());
}

ExecutableBuildOptions& ExecutableBuildOptions::set_generate_hlo_graph(
    string regex) {
  generate_hlo_graph_ = std::move(regex);
  return *this;
}

const tensorflow::gtl::optional<string>&
ExecutableBuildOptions::generate_hlo_graph() const {
  return generate_hlo_graph_;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_dump_optimized_hlo_proto_to(
    tensorflow::StringPiece dirpath) {
  dump_optimized_hlo_proto_to_ = dirpath.ToString();
  return *this;
}

const tensorflow::gtl::optional<string>&
ExecutableBuildOptions::dump_optimized_hlo_proto_to() const {
  return dump_optimized_hlo_proto_to_;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_dump_per_pass_hlo_proto_to(
    tensorflow::StringPiece dirpath) {
  dump_per_pass_hlo_proto_to_ = dirpath.ToString();
  return *this;
}

const tensorflow::gtl::optional<string>&
ExecutableBuildOptions::dump_per_pass_hlo_proto_to() const {
  return dump_per_pass_hlo_proto_to_;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_hlo_profile(bool enabled) {
  hlo_profile_ = enabled;
  return *this;
}

tensorflow::gtl::optional<bool> ExecutableBuildOptions::hlo_profile() const {
  return hlo_profile_;
}

}  // namespace xla
