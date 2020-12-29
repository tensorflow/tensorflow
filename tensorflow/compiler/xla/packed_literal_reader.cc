/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/packed_literal_reader.h"

#include <limits>
#include <string>
#include <utility>

#include "absl/base/casts.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

PackedLiteralReader::PackedLiteralReader(tensorflow::RandomAccessFile* file)
    : file_(file), offset_(0) {}

PackedLiteralReader::~PackedLiteralReader() { delete file_; }

StatusOr<Literal> PackedLiteralReader::Read(const Shape& shape,
                                            const Layout* layout) {
  VLOG(3) << "reading shape from file: " << ShapeUtil::HumanString(shape)
          << " layout: " << (layout == nullptr ? "<none>" : layout->ToString());
  Shape literal_shape = shape;
  if (layout != nullptr) {
    TF_RETURN_IF_ERROR(
        LayoutUtil::ValidateLayoutForShape(*layout, literal_shape));
    *literal_shape.mutable_layout() = *layout;
  }

  if (shape.element_type() != F32) {
    return Unimplemented(
        "not yet implemented element type for packed literal reading: %s",
        PrimitiveType_Name(shape.element_type()));
  }

  Literal result(literal_shape);
  result.PopulateWithValue(std::numeric_limits<float>::quiet_NaN());

  int64 elements = ShapeUtil::ElementsIn(shape);
  absl::Span<const float> field = result.data<float>();
  char* data = absl::bit_cast<char*>(field.data());
  uint64 bytes = elements * sizeof(float);
  absl::string_view sp;
  auto s = file_->Read(offset_, bytes, &sp, data);
  offset_ += sp.size();
  if (!s.ok()) {
    return s;
  } else {
    // Success: make sure we move the data into the right place if the Read
    // call decided to return data in somewhere other than "data".
    CHECK_EQ(sp.size(), bytes);
    if (sp.data() != data) {
      memcpy(data, sp.data(), sp.size());
    }
  }
  VLOG(3) << "read shape from file: " << ShapeUtil::HumanString(shape);
  return std::move(result);
}

bool PackedLiteralReader::IsExhausted() const {
  // Try to read a single byte from offset_.  If we can't, we've
  // exhausted the data.
  char single_byte[1];
  absl::string_view sp;
  auto s = file_->Read(offset_, sizeof(single_byte), &sp, single_byte);
  return !s.ok();
}

}  // namespace xla
