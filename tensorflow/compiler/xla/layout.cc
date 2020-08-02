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
  std::vector<string> elements;
  for (auto dim : dimensions()) {
    if (dim >= 0) {
      elements.push_back(std::to_string(dim));
    } else {
      if (dim == kCombineDimension) {
        elements.push_back("*");
      } else {
        elements.push_back(absl::StrCat("Invalid value ", dim));
      }
    }
  }
  return absl::StrCat("(", absl::StrJoin(elements, ","), ")");
}

/* static */ Layout Layout::CreateFromProto(const LayoutProto& proto) {
  Layout layout;
  layout.set_format(proto.format());
  layout.minor_to_major_.reserve(proto.minor_to_major_size());
  for (const int64 dimension : proto.minor_to_major()) {
    layout.add_minor_to_major(dimension);
  }
  for (const TileProto& tile_proto : proto.tiles()) {
    *layout.add_tiles() = Tile::CreateFromProto(tile_proto);
  }
  layout.set_element_size_in_bits(proto.element_size_in_bits());
  layout.set_memory_space(proto.memory_space());
  return layout;
}

LayoutProto Layout::ToProto() const {
  LayoutProto proto;
  proto.set_format(format_);
  proto.mutable_minor_to_major()->Reserve(minor_to_major_size());
  for (const int64 dimension : minor_to_major()) {
    proto.add_minor_to_major(dimension);
  }
  for (const Tile& tile : tiles()) {
    *proto.add_tiles() = tile.ToProto();
  }
  proto.set_element_size_in_bits(element_size_in_bits());
  proto.set_memory_space(memory_space_);
  return proto;
}

string Layout::ToString() const {
  if (format() == DENSE) {
    string colon_string = tiles().empty() ? "" : "T";
    for (const Tile& tile : tiles()) {
      absl::StrAppend(&colon_string, tile.ToString());
    }
    if (element_size_in_bits() != 0) {
      absl::StrAppend(&colon_string, "E(", element_size_in_bits(), ")");
    }
    if (memory_space() != 0) {
      absl::StrAppend(&colon_string, "S(", memory_space(), ")");
    }
    return absl::StrCat("{", absl::StrJoin(minor_to_major(), ","),
                        colon_string.empty() ? "" : ":", colon_string, "}");
  } else {
    CHECK_EQ(format(), INVALID_FORMAT);
    return "invalid{}";
  }
}

bool Layout::Equal::operator()(const Layout& lhs, const Layout& rhs) {
  if (lhs.format() != rhs.format()) {
    return false;
  }
  if (lhs.format() == DENSE && lhs.minor_to_major() != rhs.minor_to_major()) {
    return false;
  }
  if (!ignore_tiles_ && lhs.tiles() != rhs.tiles()) {
    return false;
  }
  if (!ignore_element_size_ &&
      lhs.element_size_in_bits() != rhs.element_size_in_bits()) {
    return false;
  }
  if (!ignore_memory_space_ && lhs.memory_space() != rhs.memory_space()) {
    return false;
  }
  return true;
}

bool Layout::operator==(const Layout& other) const {
  return Equal()(*this, other);
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
