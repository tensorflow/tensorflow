#include "tensorflow/core/framework/tensor_slice.h"

#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/logging.h"
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace {

// Basic tests
TEST(TensorSliceTest, Basic) {
  {
    // Repeatedly setting FullSlice should work.
    TensorSlice s(3);
    EXPECT_EQ("-:-:-", s.DebugString());

    s.SetFullSlice(4);
    EXPECT_EQ("-:-:-:-", s.DebugString());
  }
}

// Testing for serialization and parsing for the string format of slices.
TEST(TensorSliceTest, Serialization) {
  // Serialization
  {
    TensorSlice s({{0, -1}, {0, 10}, {14, 1}, {0, -1}});
    EXPECT_EQ("-:0,10:14,1:-", s.DebugString());
  }

  {
    TensorSliceProto proto;
    // Define ptxt outside ASSERT_TRUE call to work around bug in some
    // versions of gcc that breaks when you use raw string literals
    // inside macro expansions.
    //   See: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=55971
    const char* ptxt = R"PROTO(
      extent { }
      extent { start: 0 length: 10 }
      extent { start: 14 length: 1 }
      extent { }
    )PROTO";
    ASSERT_TRUE(protobuf::TextFormat::ParseFromString(ptxt, &proto));
    TensorSlice s(proto);
    EXPECT_EQ("-:0,10:14,1:-", s.DebugString());
  }

  // Parsing
  {
    TensorSlice s = TensorSlice::ParseOrDie("-:-:1,3:4,5");
    TensorSliceProto proto;
    s.AsProto(&proto);
    EXPECT_EQ(
        "extent { } "
        "extent { } "
        "extent { start: 1 length: 3 } "
        "extent { start: 4 length: 5 }",
        proto.ShortDebugString());
  }

  // Failed parsing
  {
    TensorSlice slice;
    Status s = TensorSlice::Parse("-:-:1,3:4:5", &slice);
    EXPECT_EQ(
        "Invalid argument: "
        "Expected a pair of numbers or '-' but got '4': "
        "string = -:-:1,3:4:5",
        s.ToString());
  }
  {
    TensorSlice slice;
    Status s = TensorSlice::Parse("-:-1,3", &slice);
    EXPECT_EQ(
        "Invalid argument: "
        "Expected non-negative start and positive length but got "
        "start = -1, length = 3: string = -:-1,3",
        s.ToString());
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
    EXPECT_OK(a.SliceTensorShape(x, &y));
    EXPECT_EQ(
        "dim { size: 1 } "
        "dim { size: 4 } "
        "dim { size: 1 } "
        "dim { size: 6 }",
        y.DebugString());
  }

  // An invalid application -- dimension 2 is out of bound
  {
    TensorSlice a = TensorSlice::ParseOrDie("1,1:1,4:-:-");
    TensorShape x({2, 4, 5, 8});
    TensorShape y;
    EXPECT_EQ(
        "Internal: "
        "Extent in dimension 1 out of bounds: "
        "shape = dim { size: 2 } "
        "dim { size: 4 } "
        "dim { size: 5 } "
        "dim { size: 8 }, "
        "slice = 1,1:1,4:-:-",
        a.SliceTensorShape(x, &y).ToString());
    EXPECT_EQ("", y.DebugString());
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
  const char* ptxt = R"PROTO(
    extent { }
    extent { start: 0 length: 10 }
    extent { start: 14 length: 1 }
    extent { }
  )PROTO";
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
  EXPECT_EQ("0,5:0,10:14,1:-:-", ts2.DebugString());
  EXPECT_EQ("0,5:0,10:14,1:-:-", ts3.DebugString());
}

}  // namespace
}  // namespace tensorflow
