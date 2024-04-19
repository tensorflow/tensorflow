/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/tensor_slice.h"

#include <limits>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace {

// Basic tests
TEST(TensorSliceTest, Basic) {
  {
    // Repeatedly setting FullSlice should work.
    TensorSlice s(3);
    EXPECT_EQ("-:-:-", s.DebugString());
    EXPECT_TRUE(s.IsFull());

    s.SetFullSlice(4);
    EXPECT_EQ("-:-:-:-", s.DebugString());
    EXPECT_TRUE(s.IsFull());
  }
}

// Testing for serialization and parsing for the string format of slices.
TEST(TensorSliceTest, Serialization) {
  // Serialization
  {
    TensorSlice s({{0, -1}, {0, 10}, {14, 1}, {0, -1}});
    EXPECT_EQ("-:0,10:14,1:-", s.DebugString());
    EXPECT_TRUE(!s.IsFull());
  }

  {
    TensorSliceProto proto;
    // Define ptxt outside ASSERT_TRUE call to work around bug in some
    // versions of gcc that breaks when you use raw string literals
    // inside macro expansions.
    //   See: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=55971
    const char* ptxt = R"pb(
      extent {}
      extent { start: 0 length: 10 }
      extent { start: 14 length: 1 }
      extent {}
    )pb";
    ASSERT_TRUE(protobuf::TextFormat::ParseFromString(ptxt, &proto));
    TensorSlice s(proto);
    EXPECT_EQ("-:0,10:14,1:-", s.DebugString());
    EXPECT_TRUE(!s.IsFull());
  }

  // Parsing
  {
    TensorSlice s = TensorSlice::ParseOrDie("-:-:1,3:4,5");
    TensorSliceProto proto;
    s.AsProto(&proto);
    TensorSliceProto expected_slice_proto;
    protobuf::TextFormat::ParseFromString(
        "extent { } "
        "extent { } "
        "extent { start: 1 length: 3 } "
        "extent { start: 4 length: 5 }",
        &expected_slice_proto);
    EXPECT_EQ(proto.ShortDebugString(),
              expected_slice_proto.ShortDebugString());
    EXPECT_TRUE(!s.IsFull());
  }

  // Failed parsing
  {
    TensorSlice slice;
    Status s = TensorSlice::Parse("-:-:1,3:4:5", &slice);
    EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
    EXPECT_TRUE(
        absl::StrContains(s.message(),
                          "Expected a pair of numbers or '-' but got '4': "
                          "string = -:-:1,3:4:5"));
  }
  {
    TensorSlice slice;
    Status s = TensorSlice::Parse("-:-1,3", &slice);
    EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
    EXPECT_TRUE(absl::StrContains(
        s.message(),
        "Expected non-negative start and positive length but got "
        "start = -1, length = 3: string = -:-1,3"));
  }

  // int64 parsing
  {
    TensorSlice s =
        TensorSlice::ParseOrDie("9223372036854775807,9223372036854775807");
    TensorSliceProto proto;
    s.AsProto(&proto);
    TensorSliceProto expected_slice_proto;
    protobuf::TextFormat::ParseFromString(
        "extent { start: 9223372036854775807 length: 9223372036854775807 }",
        &expected_slice_proto);
    EXPECT_EQ(proto.ShortDebugString(),
              expected_slice_proto.ShortDebugString());
    EXPECT_TRUE(!s.IsFull());
  }

  // int64 parsing failure
  {
    TensorSlice slice;
    Status s =
        TensorSlice::Parse("19223372036854775808,19223372036854775808", &slice);
    EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
    EXPECT_TRUE(absl::StrContains(
        s.message(),
        "Expected a pair of numbers or '-' but got "
        "'19223372036854775808,19223372036854775808': string = "
        "19223372036854775808,19223372036854775808"));
  }
}

// Testing `BuildTensorSlice` with valid and invalid input protos.
TEST(TensorSliceTest, BuildTensorSlice) {
  TensorSliceProto proto;
  TensorSlice({{0, -1}, {0, 10}, {14, 1}}).AsProto(&proto);
  TensorSlice s;

  // Successful building.
  {
    TF_ASSERT_OK(TensorSlice::BuildTensorSlice(proto, &s));
    EXPECT_EQ("-:0,10:14,1", s.DebugString());
  }

  // Failed building due to negative extent start.
  {
    TensorSliceProto invalid_proto = proto;
    invalid_proto.mutable_extent(0)->set_start(-1);
    EXPECT_FALSE(TensorSlice::BuildTensorSlice(invalid_proto, &s).ok());
  }

  // Failed building due to negative extent length.
  {
    TensorSliceProto invalid_proto = proto;
    invalid_proto.mutable_extent(2)->set_length(-1);
    EXPECT_FALSE(TensorSlice::BuildTensorSlice(invalid_proto, &s).ok());
  }

  // Failed building due to missing extent length.
  {
    TensorSliceProto invalid_proto = proto;
    invalid_proto.mutable_extent(2)->clear_length();
    EXPECT_FALSE(TensorSlice::BuildTensorSlice(invalid_proto, &s).ok());
  }

  // Failed building due to extent end overflowing.
  {
    TensorSliceProto invalid_proto = proto;
    invalid_proto.mutable_extent(2)->set_length(
        std::numeric_limits<int64_t>::max());
    EXPECT_FALSE(TensorSlice::BuildTensorSlice(invalid_proto, &s).ok());
  }
}

// Testing the slice intersection
TEST(TensorSliceTest, Intersection) {
  // "EVERYTHING" intersects with everything
  {
    TensorSlice a = TensorSlice::ParseOrDie("-:-");
    TensorSlice b = TensorSlice::ParseOrDie("1,2:3,4");
    TensorSlice c;
    EXPECT_TRUE(a.Intersect(b, &c));
    EXPECT_EQ("1,2:3,4", c.DebugString());
  }

  {
    TensorSlice a = TensorSlice::ParseOrDie("-:-");
    TensorSlice b = TensorSlice::ParseOrDie("1,2:3,4");
    TensorSlice c;
    EXPECT_TRUE(b.Intersect(a, &c));
    EXPECT_EQ("1,2:3,4", c.DebugString());
  }

  // Overlap at all dimensions
  {
    TensorSlice a = TensorSlice::ParseOrDie("1,5:2,6:3,7:5,10");
    TensorSlice b = TensorSlice::ParseOrDie("1,2:3,4:9,10:12,1");
    TensorSlice c;
    EXPECT_TRUE(a.Intersect(b, &c));
    EXPECT_EQ("1,2:3,4:9,1:12,1", c.DebugString());
  }

  // A mixture of everything and non-trivial slices
  {
    TensorSlice a = TensorSlice::ParseOrDie("-:1,1");
    TensorSlice b = TensorSlice::ParseOrDie("-:0,2");
    TensorSlice c;
    EXPECT_TRUE(a.Intersect(b, &c));
    EXPECT_EQ("-:1,1", c.DebugString());
  }

  // No overlap on dimension 3: "3,1" and "4,5" don't intersect
  {
    TensorSlice a = TensorSlice::ParseOrDie("1,2:3,1:5,6");
    TensorSlice b = TensorSlice::ParseOrDie("1,3:4,5:1,6");
    TensorSlice c;
    EXPECT_FALSE(a.Intersect(b, &c));
    EXPECT_EQ("", c.DebugString());
  }
  // No intersection when there are different dimensions
  {
    TensorSlice a = TensorSlice::ParseOrDie("1,2:3,1:-");
    TensorSlice b = TensorSlice::ParseOrDie("-:-");
    TensorSlice c;
    EXPECT_FALSE(a.Intersect(b, &c));
    EXPECT_EQ("", c.DebugString());
  }
}

// Testing applying a slice to a tensor shape
TEST(TensorSliceTest, SliceTensorShape) {
  // A proper application
  {
    TensorSlice a = TensorSlice::ParseOrDie("1,1:-:4,1:2,6");
    TensorShape x({2, 4, 5, 8});
    TensorShape y;
    TF_EXPECT_OK(a.SliceTensorShape(x, &y));
    EXPECT_EQ("[1,4,1,6]", y.DebugString());
  }

  // An invalid application -- dimension 2 is out of bounds
  {
    TensorSlice a = TensorSlice::ParseOrDie("1,1:1,4:-:-");
    TensorShape x({2, 4, 5, 8});
    TensorShape y;
    Status s = a.SliceTensorShape(x, &y);
    EXPECT_EQ(s.code(), error::INTERNAL);
    EXPECT_TRUE(absl::StrContains(s.message(),
                                  "Extent in dimension 1 out of bounds: "
                                  "shape = [2,4,5,8], slice = 1,1:1,4:-:-"));
    EXPECT_EQ("[]", y.DebugString());
  }
}

// Testing the computation of relative slices.
TEST(TensorSliceTest, ComputeRelative) {
  // Easy case: base is "everything"
  {
    TensorSlice base = TensorSlice::ParseOrDie("-:-:-:-");
    TensorSlice sub = TensorSlice::ParseOrDie("-:1,2:-:3,4");
    TensorSlice relative;
    base.ComputeRelative(sub, &relative);
    EXPECT_EQ("-:1,2:-:3,4", relative.DebugString());
  }

  // A slightly more complicated case
  {
    TensorSlice base = TensorSlice::ParseOrDie("1,2:3,4:-:5,1");
    TensorSlice sub = TensorSlice::ParseOrDie("1,1:4,2:3,3:5,1");
    TensorSlice relative;
    base.ComputeRelative(sub, &relative);
    EXPECT_EQ("0,1:1,2:3,3:0,1", relative.DebugString());
  }
}

TEST(TensorSliceTest, ExtentLength) {
  TensorSliceProto proto;
  // Define ptxt outside ASSERT_TRUE call to work around bug in some
  // versions of gcc that breaks when you use raw string literals
  // inside macro expansions.
  //   See: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=55971
  const char* ptxt = R"pb(
    extent {}
    extent { start: 0 length: 10 }
    extent { start: 14 length: 1 }
    extent {}
  )pb";
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(ptxt, &proto));
  EXPECT_FALSE(TensorSlice::HasExtentLength(proto.extent(0)));
  EXPECT_TRUE(TensorSlice::HasExtentLength(proto.extent(1)));
  EXPECT_TRUE(TensorSlice::HasExtentLength(proto.extent(2)));
  EXPECT_FALSE(TensorSlice::HasExtentLength(proto.extent(3)));
  EXPECT_EQ(-1, TensorSlice::GetExtentLength(proto.extent(0)));
  EXPECT_EQ(10, TensorSlice::GetExtentLength(proto.extent(1)));
  EXPECT_EQ(1, TensorSlice::GetExtentLength(proto.extent(2)));
  EXPECT_EQ(-1, TensorSlice::GetExtentLength(proto.extent(3)));
}

TEST(TensorSliceTest, Deserialization) {
  // Serialization of
  //     extent { length: 5 }
  //     extent { start: 0 length: 10 }
  //     extent { start: 14 length: 1 }
  //     extent { start: 1 }
  //     extent { }
  // in proto2 and proto3:
  const char pb2[] =
      "\x0A\x02\x10\x05\x0A\x04\x08\x00"
      "\x10\x0A\x0A\x04\x08\x0E\x10\x01\x0A\x02\x08\x01\x0A\x00";
  const char pb3[] =
      "\x0A\x02\x10\x05\x0A\x02"
      "\x10\x0A\x0A\x04\x08\x0E\x10\x01\x0A\x02\x08\x01\x0A\x00";
  // (The difference is that in the proto3 version, "start: 0" isn't included
  // since 0 is start's default value.)

  TensorSliceProto proto2;
  ASSERT_TRUE(proto2.ParseFromArray(pb2, sizeof(pb2) - 1));
  TensorSlice ts2(proto2);

  TensorSliceProto proto3;
  ASSERT_TRUE(proto3.ParseFromArray(pb3, sizeof(pb3) - 1));
  TensorSlice ts3(proto3);

  // Both serializations should be interpreted the same.
  EXPECT_EQ("0,5:0,10:14,1:1,-1:-", ts2.DebugString());
  EXPECT_EQ("0,5:0,10:14,1:1,-1:-", ts3.DebugString());
}

TEST(TensorSliceTest, UpdateToCover) {
  // [2:4, :, 3:]
  TensorSlice s({{2, 2}, {0, -1}, {3, 7}});
  // [:, 1:4, 2:4]
  TensorSlice other({{0, -1}, {1, 3}, {2, 2}});

  s.UpdateToCover(other);
  // [:, :, 2:]
  EXPECT_EQ("-:-:2,8", s.DebugString());
}

TEST(TensorSliceTest, IsFull) {
  TensorSlice slice(3);
  EXPECT_TRUE(slice.IsFull());

  TensorSlice slice2({{0, -1}});
  EXPECT_TRUE(slice2.IsFull());

  TensorSlice slice3({{0, -1}, {0, -1}, {14, 1}});
  EXPECT_TRUE(!slice3.IsFull());
}

TEST(TensorSliceTest, Equality) {
  {  // Dims are different.
    TensorSlice slice1(3);
    TensorSlice slice2(2);
    EXPECT_TRUE(slice1 != slice2);
    EXPECT_TRUE(slice2 != slice1);
  }
  {  // Both are 3-dim full slices.
    TensorSlice slice1(3);
    TensorSlice slice2({{0, -1}, {0, -1}, {0, -1}});
    EXPECT_TRUE(slice1 == slice2);
    EXPECT_TRUE(slice2 == slice1);
  }
  {  // Differs in one dimension.
    TensorSlice slice1(3);
    TensorSlice slice2({{0, -1}, {0, 1}, {0, -1}});
    EXPECT_TRUE(slice1 != slice2);
    EXPECT_TRUE(slice2 != slice1);
  }
}

}  // namespace
}  // namespace tensorflow
