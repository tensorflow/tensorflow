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

#include "tensorflow/compiler/xla/layout.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/layout_util.h"

namespace xla {

TileProto Tile::ToProto() const {
  TileProto tile_proto;
  for (int64 i : dimensions()) {
    tile_proto.add_dimensions(i);
  }
  return tile_proto;
}

string Tile::ToString() const {
  return absl::StrCat("(", absl::StrJoin(dimensions(), ","), ")");
}

/* static */ Layout Layout::CreateFromProto(const LayoutProto& proto) {
  Layout layout;
  layout.set_format(proto.format());
  layout.minor_to_major_.reserve(proto.minor_to_major_size());
  for (const int64 dimension : proto.minor_to_major()) {
    layout.add_minor_to_major(dimension);
  }
  layout.set_max_sparse_elements(proto.max_sparse_elements());
  for (const TileProto& tile_proto : proto.tiles()) {
    *layout.add_tiles() = Tile::CreateFromProto(tile_proto);
  }
  layout.set_element_size_in_bits(proto.element_size_in_bits());
  return layout;
}

LayoutProto Layout::ToProto() const {
  LayoutProto proto;
  proto.set_format(format_);
  proto.mutable_minor_to_major()->Reserve(minor_to_major_size());
  for (const int64 dimension : minor_to_major()) {
    proto.add_minor_to_major(dimension);
  }
  proto.set_max_sparse_elements(max_sparse_elements_);
  for (const Tile& tile : tiles()) {
    *proto.add_tiles() = tile.ToProto();
  }
  proto.set_element_size_in_bits(element_size_in_bits());
  return proto;
}

string Layout::ToString() const {
  // TODO(b/119839262): Emit tiles in string.
  if (format() == SPARSE) {
    return absl::StrCat("sparse{", max_sparse_elements(), "}");
  } else if (format() == DENSE) {
    return absl::StrCat("{", absl::StrJoin(minor_to_major(), ","), "}");
  } else {
    CHECK_EQ(format(), INVALID_FORMAT);
    return "invalid{}";
  }
}

bool Layout::operator==(const Layout& other) const {
  return (other.format() == format() &&
          other.minor_to_major() == minor_to_major() &&
          other.element_size_in_bits() == element_size_in_bits() &&
          other.max_sparse_elements() == max_sparse_elements() &&
          other.tiles() == tiles());
}

std::ostream& operator<<(std::ostream& out, const Tile& tile) {
  out << tile.ToString();
  return out;
}

std::ostream& operator<<(std::ostream& out, const Layout& layout) {
  out << layout.ToString();
  return out;
}

}  // namespace xla
