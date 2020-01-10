#include "tensorflow/core/platform/port.h"

#include <gtest/gtest.h>

namespace tensorflow {
namespace {

TEST(IntegralTypes, Basic) {
  EXPECT_EQ(1, sizeof(int8));
  EXPECT_EQ(2, sizeof(int16));
  EXPECT_EQ(4, sizeof(int32));
  EXPECT_EQ(8, sizeof(int64));

  EXPECT_EQ(1, sizeof(uint8));
  EXPECT_EQ(2, sizeof(uint16));
  EXPECT_EQ(4, sizeof(uint32));
  EXPECT_EQ(8, sizeof(uint64));
}

TEST(IntegralTypes, MinAndMaxConstants) {
  EXPECT_EQ(static_cast<uint8>(kint8min), static_cast<uint8>(kint8max) + 1);
  EXPECT_EQ(static_cast<uint16>(kint16min), static_cast<uint16>(kint16max) + 1);
  EXPECT_EQ(static_cast<uint32>(kint32min), static_cast<uint32>(kint32max) + 1);
  EXPECT_EQ(static_cast<uint64>(kint64min), static_cast<uint64>(kint64max) + 1);

  EXPECT_EQ(0, static_cast<uint8>(kuint8max + 1));
  EXPECT_EQ(0, static_cast<uint16>(kuint16max + 1));
  EXPECT_EQ(0, static_cast<uint32>(kuint32max + 1));
  EXPECT_EQ(0, static_cast<uint64>(kuint64max + 1));
}

}  // namespace
}  // namespace tensorflow
