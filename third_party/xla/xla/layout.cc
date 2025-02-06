/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/layout.h"

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/printer.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep

namespace xla {

TileProto Tile::ToProto() const {
  TileProto tile_proto;
  SetProto(tile_proto);
  return tile_proto;
}

void Tile::SetProto(TileProto& tile_proto) const {
  tile_proto.Clear();
  for (int64_t i : dimensions()) {
    tile_proto.add_dimensions(i);
  }
}

void Tile::Print(Printer* printer) const {
  printer->Append("(");
  AppendJoin(printer, dimensions(), ",", [&](Printer* printer, int64_t dim) {
    if (dim >= 0) {
      printer->Append(dim);
    } else {
      if (dim == kCombineDimension) {
        printer->Append("*");
      } else {
        printer->Append("Invalid value ");
        printer->Append(dim);
      }
    }
  });
  printer->Append(")");
}

std::string Tile::ToString() const {
  StringPrinter printer;
  Print(&printer);
  return std::move(printer).ToString();
}

Layout::Layout()
    : index_primitive_type_(PRIMITIVE_TYPE_INVALID),
      pointer_primitive_type_(PRIMITIVE_TYPE_INVALID) {}

SplitConfigProto SplitConfig::ToProto() const {
  SplitConfigProto split_config_proto;
  split_config_proto.set_dimension(dimension_);
  for (int64_t i : split_indices_) {
    split_config_proto.add_split_indices(i);
  }
  return split_config_proto;
}

void SplitConfig::SetProto(SplitConfigProto& split_config_proto) const {
  split_config_proto.Clear();
  split_config_proto.set_dimension(dimension_);
  for (int64_t i : split_indices_) {
    split_config_proto.add_split_indices(i);
  }
}

std::string SplitConfig::ToString() const {
  return absl::StrCat("(", dimension_, ":", absl::StrJoin(split_indices_, ","),
                      ")");
}

Layout::Layout(absl::Span<const int64_t> minor_to_major)
    : index_primitive_type_(PRIMITIVE_TYPE_INVALID),
      pointer_primitive_type_(PRIMITIVE_TYPE_INVALID),
      minor_to_major_(minor_to_major.begin(), minor_to_major.end()) {}

Layout::Layout(absl::Span<const int64_t> minor_to_major,
               absl::Span<const Tile> tiles, int64_t element_size_in_bits)
    : index_primitive_type_(PRIMITIVE_TYPE_INVALID),
      pointer_primitive_type_(PRIMITIVE_TYPE_INVALID),
      element_size_in_bits_(element_size_in_bits),
      minor_to_major_(minor_to_major.begin(), minor_to_major.end()),
      tiles_(tiles.begin(), tiles.end()) {}

Layout::Layout(absl::Span<const int64_t> minor_to_major,
               absl::Span<const DimLevelType> dim_level_types,
               absl::Span<const bool> dim_unique,
               absl::Span<const bool> dim_ordered, absl::Span<const Tile> tiles,
               int64_t tail_padding_alignment_in_elements,
               PrimitiveType index_primitive_type,
               PrimitiveType element_primitive_type,
               int64_t element_size_in_bits, int64_t memory_space,
               absl::Span<const SplitConfig> split_configs,
               std::unique_ptr<Shape> physical_shape,
               int64_t dynamic_shape_metadata_prefix_bytes)
    : index_primitive_type_(index_primitive_type),
      pointer_primitive_type_(element_primitive_type),
      memory_space_(memory_space),
      element_size_in_bits_(element_size_in_bits),
      minor_to_major_(minor_to_major.begin(), minor_to_major.end()),
      tiles_(tiles.begin(), tiles.end()),
      split_configs_(split_configs.begin(), split_configs.end()),
      tail_padding_alignment_in_elements_(tail_padding_alignment_in_elements),
      physical_shape_(std::move(physical_shape)),
      dynamic_shape_metadata_prefix_bytes_(
          dynamic_shape_metadata_prefix_bytes) {
  // Grow dim_attributes_ to the maximum length of "dim_level_types",
  // "dim_unique", and "dim_ordered", and then initialize the attributes that
  // should exist.
  n_dim_level_types_ = dim_level_types.size();
  n_dim_unique_ = dim_unique.size();
  n_dim_ordered_ = dim_ordered.size();
  const int n_attributes = std::max<int>(
      n_dim_level_types_, std::max<int>(n_dim_unique_, n_dim_ordered_));
  dim_attributes_.resize(n_attributes);
  for (int i = 0; i < n_attributes; i++) {
    if (i < n_dim_level_types_)
      dim_attributes_[i].dim_level_type = dim_level_types[i];
    if (i < n_dim_unique_) dim_attributes_[i].dim_unique = dim_unique[i];
    if (i < n_dim_ordered_) dim_attributes_[i].dim_ordered = dim_ordered[i];
  }
}

Layout::Layout(const Layout& other)
    : dim_attributes_(other.dim_attributes_),
      n_dim_level_types_(other.n_dim_level_types_),
      n_dim_unique_(other.n_dim_unique_),
      n_dim_ordered_(other.n_dim_ordered_),
      index_primitive_type_(other.index_primitive_type_),
      pointer_primitive_type_(other.pointer_primitive_type_),
      memory_space_(other.memory_space_),
      element_size_in_bits_(other.element_size_in_bits_),
      minor_to_major_(other.minor_to_major_),
      tiles_(other.tiles_),
      split_configs_(other.split_configs_),
      tail_padding_alignment_in_elements_(
          other.tail_padding_alignment_in_elements_),
      physical_shape_(other.physical_shape_ != nullptr
                          ? std::make_unique<Shape>(*other.physical_shape_)
                          : nullptr),
      dynamic_shape_metadata_prefix_bytes_(
          other.dynamic_shape_metadata_prefix_bytes_) {}

Layout::Layout(Layout&& other) = default;

Layout::~Layout() = default;

Layout& Layout::operator=(const Layout& other) {
  if (this != &other) {
    dim_attributes_ = other.dim_attributes_;
    n_dim_level_types_ = other.n_dim_level_types_;
    n_dim_unique_ = other.n_dim_unique_;
    n_dim_ordered_ = other.n_dim_ordered_;
    minor_to_major_ = other.minor_to_major_;
    tiles_ = other.tiles_;
    tail_padding_alignment_in_elements_ =
        other.tail_padding_alignment_in_elements_;
    index_primitive_type_ = other.index_primitive_type_;
    pointer_primitive_type_ = other.pointer_primitive_type_;
    element_size_in_bits_ = other.element_size_in_bits_;
    memory_space_ = other.memory_space_;
    split_configs_ = other.split_configs_;
    if (other.physical_shape_ != nullptr) {
      physical_shape_ = std::make_unique<Shape>(*other.physical_shape_);
    } else {
      physical_shape_ = nullptr;
    }
    dynamic_shape_metadata_prefix_bytes_ =
        other.dynamic_shape_metadata_prefix_bytes_;
  }
  return *this;
}

Layout& Layout::operator=(Layout&& other) = default;

/* static */ Layout Layout::CreateFromProto(const LayoutProto& proto) {
  Layout layout;
  for (int dim_level_type : proto.dim_level_types()) {
    layout.add_dim_level_type(static_cast<DimLevelType>(dim_level_type));
  }
  for (bool dim_unique : proto.dim_unique()) {
    layout.add_dim_unique(dim_unique);
  }
  for (bool dim_ordered : proto.dim_ordered()) {
    layout.add_dim_ordered(dim_ordered);
  }
  layout.minor_to_major_.reserve(proto.minor_to_major_size());
  for (const int64_t dimension : proto.minor_to_major()) {
    layout.add_minor_to_major(dimension);
  }
  for (const TileProto& tile_proto : proto.tiles()) {
    *layout.add_tiles() = Tile::CreateFromProto(tile_proto);
  }
  if (proto.tail_padding_alignment_in_elements() != 0) {
    layout.set_tail_padding_alignment_in_elements(
        proto.tail_padding_alignment_in_elements());
  } else {
    layout.set_tail_padding_alignment_in_elements(1);
  }
  layout.set_index_primitive_type(proto.index_primitive_type());
  layout.set_pointer_primitive_type(proto.pointer_primitive_type());
  layout.set_element_size_in_bits(proto.element_size_in_bits());
  layout.set_memory_space(proto.memory_space());
  for (const SplitConfigProto& split_config_proto : proto.split_configs()) {
    layout.add_split_configs(SplitConfig::CreateFromProto(split_config_proto));
  }
  if (proto.has_physical_shape()) {
    *layout.mutable_physical_shape() = Shape(proto.physical_shape());
  }
  layout.set_dynamic_shape_metadata_prefix_bytes(
      proto.dynamic_shape_metadata_prefix_bytes());
  return layout;
}

LayoutProto Layout::ToProto() const {
  LayoutProto proto;
  SetProto(proto);
  return proto;
}

void Layout::SetProto(LayoutProto& proto) const {
  proto.Clear();
  for (int i = 0; i < n_dim_level_types_; i++) {
    proto.add_dim_level_types(dim_level_type(i));
  }
  for (int i = 0; i < n_dim_unique_; i++) {
    proto.add_dim_unique(dim_unique(i));
  }
  for (int i = 0; i < n_dim_ordered_; i++) {
    proto.add_dim_ordered(dim_ordered(i));
  }
  proto.mutable_minor_to_major()->Reserve(minor_to_major_size());
  for (const int64_t dimension : minor_to_major()) {
    proto.add_minor_to_major(dimension);
  }
  for (const Tile& tile : tiles()) {
    tile.SetProto(*proto.add_tiles());
  }
  proto.set_tail_padding_alignment_in_elements(
      tail_padding_alignment_in_elements());
  proto.set_index_primitive_type(index_primitive_type());
  proto.set_pointer_primitive_type(pointer_primitive_type());
  proto.set_element_size_in_bits(element_size_in_bits_);
  proto.set_memory_space(memory_space_);
  for (const SplitConfig& split_config : split_configs()) {
    split_config.SetProto(*proto.add_split_configs());
  }
  if (has_physical_shape()) {
    *proto.mutable_physical_shape() = physical_shape_->ToProto();
  }
  proto.set_dynamic_shape_metadata_prefix_bytes(
      dynamic_shape_metadata_prefix_bytes_);
}

namespace {
absl::string_view DimLevelTypeAbbrev(DimLevelType dim_level_type) {
  switch (dim_level_type) {
    case DIM_DENSE:
      return "D";
    case DIM_COMPRESSED:
      return "C";
    case DIM_SINGLETON:
      return "S";
    case xla::DIM_LOOSE_COMPRESSED:
      return "H";
    default:
      LOG(FATAL) << "Invalid DimLevelType value: " << dim_level_type;
  }
}
}  // namespace

void Layout::Print(Printer* printer) const {
  printer->Append("{");
  AppendJoin(printer, minor_to_major(), ",");

  bool colon_printed = false;
  auto print_colon = [&]() {
    if (colon_printed) return;
    printer->Append(":");
    colon_printed = true;
  };

  if (n_dim_level_types_ > 0) {
    auto print_one = [&](int i) {
      printer->Append(DimLevelTypeAbbrev(dim_level_type(i)));
      if (n_dim_unique_ > 0 && !dim_unique(i)) {
        printer->Append("+");
      }
      if (n_dim_ordered_ > 0 && !dim_ordered(i)) {
        printer->Append("~");
      }
    };
    print_colon();
    printer->Append("D(");
    print_one(0);
    for (int i = 1; i < n_dim_level_types_; ++i) {
      printer->Append(",");
      print_one(i);
    }
    printer->Append(")");
  }

  if (!tiles().empty()) {
    print_colon();
    printer->Append("T");
    for (const Tile& tile : tiles()) {
      tile.Print(printer);
    }
  }

  if (tail_padding_alignment_in_elements() != 1) {
    print_colon();
    printer->Append("L(");
    printer->Append(tail_padding_alignment_in_elements());
    printer->Append(")");
  }

  if (index_primitive_type() != PRIMITIVE_TYPE_INVALID) {
    print_colon();
    if (primitive_util::IsIntegralType(index_primitive_type())) {
      printer->Append("#(");
      printer->Append(
          primitive_util::LowercasePrimitiveTypeName(index_primitive_type()));
      printer->Append(")");
    } else {
      printer->Append("#(invalid)");
    }
  }

  if (pointer_primitive_type() != PRIMITIVE_TYPE_INVALID) {
    print_colon();
    if (primitive_util::IsIntegralType(pointer_primitive_type())) {
      printer->Append("*(");
      printer->Append(
          primitive_util::LowercasePrimitiveTypeName(pointer_primitive_type()));
      printer->Append(")");
    } else {
      printer->Append("*(invalid)");
    }
  }

  if (element_size_in_bits() != 0) {
    print_colon();
    printer->Append("E(");
    printer->Append(element_size_in_bits());
    printer->Append(")");
  }

  if (memory_space() != 0) {
    print_colon();
    printer->Append("S(");
    printer->Append(memory_space());
    printer->Append(")");
  }
  if (!split_configs().empty()) {
    print_colon();
    printer->Append("SC");
    for (const auto& split_config : split_configs()) {
      printer->Append(split_config.ToString());
    }
  }

  if (has_physical_shape()) {
    print_colon();
    printer->Append("P(");
    physical_shape_->Print(printer, /*print_layout=*/true);
    printer->Append(")");
  }

  if (dynamic_shape_metadata_prefix_bytes_ > 0) {
    print_colon();
    printer->Append("M(");
    printer->Append(dynamic_shape_metadata_prefix_bytes());
    printer->Append(")");
  }

  printer->Append("}");
}

std::string Layout::ToString() const {
  StringPrinter printer;
  Print(&printer);
  return std::move(printer).ToString();
}

bool Layout::Equal::operator()(const Layout& lhs, const Layout& rhs) {
  if (!LayoutUtil::IsDense(lhs) || !LayoutUtil::IsDense(rhs)) {
    // dim_level_types
    if (lhs.dim_level_types_size() != rhs.dim_level_types_size()) {
      return false;
    }
    for (int i = 0; i < lhs.dim_level_types_size(); i++) {
      if (lhs.dim_level_type(i) != rhs.dim_level_type(i)) {
        return false;
      }
    }
    // dim_unique
    if (lhs.dim_unique_size() != rhs.dim_unique_size()) {
      return false;
    }
    for (int i = 0; i < lhs.dim_unique_size(); i++) {
      if (lhs.dim_unique(i) != rhs.dim_unique(i)) {
        return false;
      }
    }
    // dim_ordered
    if (lhs.dim_ordered_size() != rhs.dim_ordered_size()) {
      return false;
    }
    for (int i = 0; i < lhs.dim_ordered_size(); i++) {
      if (lhs.dim_ordered(i) != rhs.dim_ordered(i)) {
        return false;
      }
    }
  }
  if (lhs.minor_to_major() != rhs.minor_to_major()) {
    return false;
  }
  if (!ignore_tiles_ && lhs.tiles() != rhs.tiles()) {
    return false;
  }
  if (!ignore_tail_padding_alignment_in_elements_ &&
      lhs.tail_padding_alignment_in_elements() !=
          rhs.tail_padding_alignment_in_elements()) {
    return false;
  }
  if (!ignore_index_primitive_type_ &&
      lhs.index_primitive_type() != rhs.index_primitive_type()) {
    return false;
  }
  if (!ignore_pointer_primitive_type_ &&
      lhs.pointer_primitive_type() != rhs.pointer_primitive_type()) {
    return false;
  }
  if (!ignore_element_size_ &&
      lhs.element_size_in_bits() != rhs.element_size_in_bits()) {
    return false;
  }
  if (!ignore_memory_space_ && lhs.memory_space() != rhs.memory_space()) {
    return false;
  }
  if (!ignore_split_configs_ && lhs.split_configs() != rhs.split_configs()) {
    return false;
  }
  if (!ignore_physical_shape_) {
    if (lhs.has_physical_shape() || rhs.has_physical_shape()) {
      if (!lhs.has_physical_shape() || !rhs.has_physical_shape()) {
        return false;
      }
      if (lhs.physical_shape() != rhs.physical_shape()) {
        return false;
      }
    }
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

Shape* Layout::mutable_physical_shape() {
  if (physical_shape_ == nullptr) {
    physical_shape_ = std::make_unique<Shape>();
  }
  return physical_shape_.get();
}

void Layout::clear_physical_shape() { physical_shape_ = nullptr; }

Layout& Layout::DeleteDimension(int64_t dim_to_delete) {
  for (int64_t i = 0; i < minor_to_major_.size();) {
    if (minor_to_major_[i] == dim_to_delete) {
      minor_to_major_.erase(minor_to_major_.begin() + i);
      continue;
    }
    if (minor_to_major_[i] > dim_to_delete) {
      minor_to_major_[i] -= 1;
    }
    ++i;
  }
  // Delete the corresponding dim level types.
  if (LayoutUtil::IsSparse(*this)) {
    if (dim_to_delete < n_dim_level_types_) n_dim_level_types_--;
    if (dim_to_delete < n_dim_unique_) n_dim_unique_--;
    if (dim_to_delete < n_dim_ordered_) n_dim_ordered_--;
    dim_attributes_.erase(dim_attributes_.begin() + dim_to_delete);
  }
  return *this;
}

}  // namespace xla
