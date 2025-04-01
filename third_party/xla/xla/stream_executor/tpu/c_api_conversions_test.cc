/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/stream_executor/tpu/c_api_conversions.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/layout.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

namespace ApiConverter {

namespace {

constexpr absl::string_view kHloString =
    R"(
HloModule TupleCreate_module:
ENTRY %TupleCreate.v4 (v1: f32[], v2: f32[3], v3: f32[2,3]) -> (f32[], f32[3], f32[2,3]) {
  %v1 = f32[] parameter(0)
  %v2 = f32[3]{0} parameter(1)
  %v3 = f32[2,3]{1,0} parameter(2)
  ROOT %tuple = (f32[], f32[3]{0}, f32[2,3]{1,0}) tuple(f32[] %v1, f32[3]{0} %v2, f32[2,3]{1,0} %v3)
}
)";

TEST(XlaTile, ToCInlined) {
  std::vector<int64_t> tile_dimensions{2, 3, 4, 5};
  xla::Tile cpp_tile(tile_dimensions);
  XLA_Tile c_tile;
  ToC(cpp_tile, &c_tile);

  absl::Span<const int64_t> cpp_tile_dimensions = cpp_tile.dimensions();
  ASSERT_EQ(cpp_tile_dimensions, tile_dimensions);
  absl::Span<const int64_t> c_tile_dimensions = MakeSpan(c_tile.dimensions);
  EXPECT_EQ(cpp_tile_dimensions, c_tile_dimensions);

  Destroy(&c_tile);
}

TEST(XlaTile, ToCDynamic) {
  std::vector<int64_t> tile_dimensions{2, 3, 4, 5, 6, 7, 8, 9};
  xla::Tile cpp_tile(tile_dimensions);
  XLA_Tile c_tile;
  ToC(cpp_tile, &c_tile);

  absl::Span<const int64_t> cpp_tile_dimensions = cpp_tile.dimensions();
  ASSERT_EQ(cpp_tile_dimensions, tile_dimensions);
  absl::Span<const int64_t> c_tile_dimensions = MakeSpan(c_tile.dimensions);
  EXPECT_EQ(cpp_tile_dimensions, c_tile_dimensions);

  Destroy(&c_tile);
}

TEST(XlaTile, FromCInlined) {
  constexpr size_t kInlinedSize = 4;
  Int64List tile_dimensions;
  tile_dimensions.size = kInlinedSize;
  for (int i = 0; i < kInlinedSize; ++i) {
    tile_dimensions.inlined[i] = i + 2;
  }
  XLA_Tile c_tile{tile_dimensions};
  xla::Tile cpp_tile = FromC(&c_tile);
  auto cpp_dimensions = cpp_tile.dimensions();
  EXPECT_EQ(cpp_dimensions.size(), kInlinedSize);
  for (int i = 0; i < kInlinedSize; ++i) {
    EXPECT_EQ(cpp_dimensions[i], i + 2);
  }
  Destroy(&c_tile);
}

TEST(XlaTile, FromCDynamic) {
  constexpr size_t kDynamicSize = 8;
  int64_t* dynamic = new int64_t[kDynamicSize];
  for (int i = 0; i < kDynamicSize; ++i) {
    dynamic[i] = i + 2;
  }
  Int64List tile_dimensions;
  tile_dimensions.size = kDynamicSize;
  tile_dimensions.heap = dynamic;
  XLA_Tile c_tile{tile_dimensions};
  xla::Tile cpp_tile = FromC(&c_tile);
  auto cpp_dimensions = cpp_tile.dimensions();
  EXPECT_EQ(cpp_dimensions.size(), kDynamicSize);
  for (int i = 0; i < kDynamicSize; ++i) {
    EXPECT_EQ(cpp_dimensions[i], i + 2);
  }
  Destroy(&c_tile);
}

namespace TestImpl {

void XlaLayout_ToC(const xla::Layout& cpp_layout) {
  XLA_Layout c_layout;
  ToC(cpp_layout, &c_layout);

  absl::Span<const int64_t> cpp_minor_to_major = cpp_layout.minor_to_major();
  absl::Span<const int64_t> c_minor_to_major =
      MakeSpan(c_layout.minor_to_major);
  EXPECT_EQ(cpp_minor_to_major, c_minor_to_major);

  absl::Span<const int> c_dim_level_types = MakeSpan(c_layout.dim_level_types);
  EXPECT_EQ(cpp_layout.dim_level_types_size(), c_dim_level_types.size());
  for (int i = 0; i < c_dim_level_types.size(); ++i) {
    EXPECT_EQ(static_cast<int>(cpp_layout.dim_level_type(i)),
              c_dim_level_types[i]);
  }

  absl::Span<const int> c_dim_unique = MakeSpan(c_layout.dim_unique);
  EXPECT_EQ(cpp_layout.dim_unique_size(), c_dim_unique.size());
  for (int i = 0; i < c_dim_unique.size(); ++i) {
    EXPECT_EQ(cpp_layout.dim_unique(i), static_cast<bool>(c_dim_unique[i]));
  }

  absl::Span<const int> c_dim_ordered = MakeSpan(c_layout.dim_ordered);
  EXPECT_EQ(cpp_layout.dim_ordered_size(), c_dim_ordered.size());
  for (int i = 0; i < c_dim_ordered.size(); ++i) {
    EXPECT_EQ(cpp_layout.dim_ordered(i), static_cast<bool>(c_dim_ordered[i]));
  }

  absl::Span<const xla::Tile> cpp_tiles = cpp_layout.tiles();
  TileList c_tiles = c_layout.tiles;
  EXPECT_EQ(cpp_tiles.size(), c_tiles.size);
  XLA_Tile* tile_base =
      (c_tiles.size > TPU_C_API_MAX_INLINED) ? c_tiles.heap : c_tiles.inlined;
  for (int i = 0; i < c_tiles.size; ++i) {
    xla::Tile converted_c_tile = FromC(&tile_base[i]);
    EXPECT_EQ(cpp_tiles[i], converted_c_tile);
  }

  EXPECT_EQ(cpp_layout.index_primitive_type(), c_layout.index_primitive_type);
  EXPECT_EQ(cpp_layout.pointer_primitive_type(),
            c_layout.pointer_primitive_type);
  EXPECT_EQ(cpp_layout.element_size_in_bits(), c_layout.element_size_in_bits);
  EXPECT_EQ(cpp_layout.memory_space(), c_layout.memory_space);
  EXPECT_EQ(cpp_layout.dynamic_shape_metadata_prefix_bytes(),
            c_layout.dynamic_shape_metadata_prefix_bytes);

  Destroy(&c_layout);
}

}  // namespace TestImpl

TEST(XlaLayout, ToCScalar) {
  xla::Shape cpp_shape = xla::ShapeUtil::MakeShapeWithType<float>({4});
  xla::Layout cpp_layout = cpp_shape.layout();
  TestImpl::XlaLayout_ToC(cpp_layout);
}

TEST(XlaLayout, ToCNested) {
  xla::Shape cpp_shape = xla::ShapeUtil::MakeShapeWithType<float>({4, 3, 2});
  xla::Layout cpp_layout = cpp_shape.layout();
  TestImpl::XlaLayout_ToC(cpp_layout);
}

TEST(XlaLayout, FromCScalar) {
  xla::Shape cpp_shape = xla::ShapeUtil::MakeShapeWithType<float>({4});
  xla::Layout in_layout = cpp_shape.layout();
  XLA_Layout c_layout;
  ToC(in_layout, &c_layout);
  xla::Layout out_layout = FromC(&c_layout);
  EXPECT_EQ(in_layout, out_layout);
  Destroy(&c_layout);
}

TEST(XlaLayout, FromCNested) {
  xla::Shape cpp_shape = xla::ShapeUtil::MakeShapeWithType<float>({4, 3, 2});
  xla::Layout in_layout = cpp_shape.layout();
  XLA_Layout c_layout;
  ToC(in_layout, &c_layout);
  xla::Layout out_layout = FromC(&c_layout);
  EXPECT_EQ(in_layout, out_layout);
  Destroy(&c_layout);
}

TEST(XlaShape, ToCScalar) {
  xla::Shape cpp_shape = xla::ShapeUtil::MakeShapeWithType<float>({4});
  XLA_Shape c_shape;
  ToC(cpp_shape, &c_shape);

  EXPECT_EQ(cpp_shape.element_type(), c_shape.element_type);

  absl::Span<const int64_t> cpp_dimensions = cpp_shape.dimensions();
  absl::Span<const int64_t> c_dimensions = MakeSpan(c_shape.dimensions);
  EXPECT_EQ(cpp_dimensions, c_dimensions);

  absl::Span<const bool> cpp_dynamic_dimensions =
      cpp_shape.dynamic_dimensions();
  absl::Span<const bool> c_dynamic_dimensions =
      MakeSpan(c_shape.dynamic_dimensions);
  EXPECT_EQ(cpp_dynamic_dimensions, c_dynamic_dimensions);

  EXPECT_FALSE(cpp_shape.IsTuple());
  EXPECT_EQ(c_shape.ntuple_shapes, 0);

  bool cpp_has_layout = cpp_shape.has_layout();
  bool c_has_layout = c_shape.has_layout;
  EXPECT_EQ(cpp_has_layout, c_has_layout);

  Destroy(&c_shape);
}

TEST(XlaShape, ToCNested) {
  const xla::Shape cpp_shape =
      xla::ShapeUtil::MakeShapeWithType<float>({4, 3, 2});
  XLA_Shape c_shape;
  ToC(cpp_shape, &c_shape);

  EXPECT_EQ(cpp_shape.element_type(), c_shape.element_type);

  absl::Span<const int64_t> cpp_dimensions = cpp_shape.dimensions();
  absl::Span<const int64_t> c_dimensions = MakeSpan(c_shape.dimensions);
  EXPECT_EQ(cpp_dimensions, c_dimensions);

  absl::Span<const bool> cpp_dynamic_dimensions =
      cpp_shape.dynamic_dimensions();
  absl::Span<const bool> c_dynamic_dimensions =
      MakeSpan(c_shape.dynamic_dimensions);
  EXPECT_EQ(cpp_dynamic_dimensions, c_dynamic_dimensions);

  EXPECT_FALSE(cpp_shape.IsTuple());
  EXPECT_EQ(c_shape.ntuple_shapes, 0);

  const int c_ntuple_shapes = c_shape.ntuple_shapes;
  const std::vector<xla::Shape>& cpp_tuple_shapes = cpp_shape.tuple_shapes();
  absl::Span<const XLA_Shape> c_tuple_shapes(c_shape.tuple_shapes,
                                             c_ntuple_shapes);
  for (int i = 0; i < c_ntuple_shapes; ++i) {
    xla::Shape converted_c_shape = FromC(&c_tuple_shapes[i]);
    EXPECT_EQ(cpp_tuple_shapes[i], converted_c_shape);
  }

  bool cpp_has_layout = cpp_shape.has_layout();
  bool c_has_layout = c_shape.has_layout;
  EXPECT_EQ(cpp_has_layout, c_has_layout);

  if (c_has_layout) {
    xla::Layout converted_c_layout = FromC(&c_shape.layout);
    EXPECT_EQ(cpp_shape.layout(), converted_c_layout);
  }

  Destroy(&c_shape);
}

TEST(XlaShape, FromCScalar) {
  xla::Shape in_shape = xla::ShapeUtil::MakeShapeWithType<float>({4});
  XLA_Shape c_shape;
  ToC(in_shape, &c_shape);
  xla::Shape out_shape = FromC(&c_shape);
  EXPECT_EQ(in_shape, out_shape);
  Destroy(&c_shape);
}

TEST(XlaShape, FromCNested) {
  xla::Shape in_shape = xla::ShapeUtil::MakeShapeWithType<float>({4, 3, 2});
  XLA_Shape c_shape;
  ToC(in_shape, &c_shape);
  xla::Shape out_shape = FromC(&c_shape);
  EXPECT_EQ(in_shape, out_shape);
  Destroy(&c_shape);
}

// TODO(b/290654348): xla::ShapeIndex, xla::Literal, xla::ShapedBuffer

TEST(XlaHloModuleConfig, ToAndFromC) {
  absl::StatusOr<std::unique_ptr<xla::HloModule>> hlo_module =
      xla::ParseAndReturnUnverifiedModule(kHloString);
  ASSERT_TRUE(hlo_module.ok());
  xla::HloModule& cpp_module = *hlo_module.value();
  xla::HloModuleConfig in_config = cpp_module.config();

  XLA_HloModuleConfig c_config = ToC(in_config);
  xla::HloModuleConfig out_config = FromC(c_config);

  xla::HloModuleConfigProto in_config_proto = in_config.ToProto();
  xla::HloModuleConfigProto out_config_proto = out_config.ToProto();

  tsl::protobuf::util::MessageDifferencer diff;
  diff.set_message_field_comparison(
      tsl::protobuf::util::MessageDifferencer::EQUIVALENT);
  EXPECT_TRUE(diff.Equals(in_config_proto, out_config_proto));

  Destroy(&c_config);
}

TEST(XlaHloModule, ToAndFromC) {
  absl::StatusOr<std::unique_ptr<xla::HloModule>> hlo_module =
      xla::ParseAndReturnUnverifiedModule(kHloString);
  ASSERT_TRUE(hlo_module.ok());
  xla::HloModule& in_module = *hlo_module.value();

  XLA_HloModule c_module = ToC(in_module);
  absl::StatusOr<std::unique_ptr<xla::HloModule>> out_module_ptr =
      FromC(c_module);
  ASSERT_TRUE(out_module_ptr.ok());
  xla::HloModule& out_module = *out_module_ptr.value();

  xla::HloModuleProtoWithConfig in_module_proto = in_module.ToProtoWithConfig();
  xla::HloModuleProtoWithConfig out_module_proto =
      out_module.ToProtoWithConfig();

  tsl::protobuf::util::MessageDifferencer diff;
  diff.set_message_field_comparison(
      tsl::protobuf::util::MessageDifferencer::EQUIVALENT);
  const auto* ignore_unique_id =
      xla::HloModuleProto::GetDescriptor()->FindFieldByName("id");
  diff.IgnoreField(ignore_unique_id);
  EXPECT_TRUE(diff.Compare(in_module_proto, out_module_proto));

  Destroy(&c_module);
}

// TODO(b/290654348): SE_DeviceMemoryBase, SE_DeviceMemoryAllocator,
// SE_MaybeOwningDeviceMemory

}  // namespace

}  // namespace ApiConverter
