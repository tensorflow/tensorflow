#include "tensorflow/core/framework/kernel_def_builder.h"

#include <gtest/gtest.h>
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace {

TEST(KernelDefBuilderTest, Basic) {
  const KernelDef* def = KernelDefBuilder("A").Device(DEVICE_CPU).Build();
  KernelDef expected;
  protobuf::TextFormat::ParseFromString("op: 'A' device_type: 'CPU'",
                                        &expected);
  EXPECT_EQ(def->DebugString(), expected.DebugString());
  delete def;
}

TEST(KernelDefBuilderTest, TypeConstraint) {
  const KernelDef* def = KernelDefBuilder("B")
                             .Device(DEVICE_GPU)
                             .TypeConstraint<float>("T")
                             .Build();
  KernelDef expected;
  protobuf::TextFormat::ParseFromString(R"proto(
    op: 'B' device_type: 'GPU'
    constraint { name: 'T' allowed_values { list { type: DT_FLOAT } } } )proto",
                                        &expected);

  EXPECT_EQ(def->DebugString(), expected.DebugString());
  delete def;

  def = KernelDefBuilder("C")
            .Device(DEVICE_GPU)
            .TypeConstraint<int32>("U")
            .TypeConstraint<bool>("V")
            .Build();

  protobuf::TextFormat::ParseFromString(R"proto(
    op: 'C' device_type: 'GPU'
    constraint { name: 'U' allowed_values { list { type: DT_INT32 } } }
    constraint { name: 'V' allowed_values { list { type: DT_BOOL } } } )proto",
                                        &expected);
  EXPECT_EQ(def->DebugString(), expected.DebugString());
  delete def;

  def = KernelDefBuilder("D")
            .Device(DEVICE_CPU)
            .TypeConstraint("W", {DT_DOUBLE, DT_STRING})
            .Build();
  protobuf::TextFormat::ParseFromString(R"proto(
    op: 'D' device_type: 'CPU'
    constraint { name: 'W'
        allowed_values { list { type: [DT_DOUBLE, DT_STRING] } } } )proto",
                                        &expected);
  EXPECT_EQ(def->DebugString(), expected.DebugString());
  delete def;
}

TEST(KernelDefBuilderTest, HostMemory) {
  const KernelDef* def = KernelDefBuilder("E")
                             .Device(DEVICE_GPU)
                             .HostMemory("in")
                             .HostMemory("out")
                             .Build();
  KernelDef expected;
  protobuf::TextFormat::ParseFromString(
      "op: 'E' device_type: 'GPU' "
      "host_memory_arg: ['in', 'out']",
      &expected);
  EXPECT_EQ(def->DebugString(), expected.DebugString());
  delete def;
}

}  // namespace
}  // namespace tensorflow
