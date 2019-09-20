#ifndef TEST_BUILDER_H
#define TEST_BUILDER_H

#include <set>
#include <type_traits>
#include "monster_test_generated.h"
#include "flatbuffers/flatbuffers.h"
#include "test_assert.h"

using MyGame::Example::Color;
using MyGame::Example::Monster;

namespace flatbuffers {
namespace grpc {
class MessageBuilder;
}
}

template <class T, class U>
struct is_same {
  static const bool value = false;
};

template <class T>
struct is_same<T, T> {
  static const bool value = true;
};

extern const std::string m1_name;
extern const Color m1_color;
extern const std::string m2_name;
extern const Color m2_color;

flatbuffers::Offset<Monster> populate1(flatbuffers::FlatBufferBuilder &builder);
flatbuffers::Offset<Monster> populate2(flatbuffers::FlatBufferBuilder &builder);

uint8_t *release_raw_base(flatbuffers::FlatBufferBuilder &fbb, size_t &size, size_t &offset);

void free_raw(flatbuffers::grpc::MessageBuilder &mbb, uint8_t *buf);
void free_raw(flatbuffers::FlatBufferBuilder &fbb, uint8_t *buf);

bool verify(const flatbuffers::DetachedBuffer &buf, const std::string &expected_name, Color color);
bool verify(const uint8_t *buf, size_t offset, const std::string &expected_name, Color color);

bool release_n_verify(flatbuffers::FlatBufferBuilder &fbb, const std::string &expected_name, Color color);
bool release_n_verify(flatbuffers::grpc::MessageBuilder &mbb, const std::string &expected_name, Color color);

// clang-format off
#if !defined(FLATBUFFERS_CPP98_STL)
// clang-format on
// Invokes this function when testing the following Builder types
// FlatBufferBuilder, TestHeapBuilder, and GrpcLikeMessageBuilder
template <class Builder>
void builder_move_assign_after_releaseraw_test(Builder b1) {
  auto root_offset1 = populate1(b1);
  b1.Finish(root_offset1);
  size_t size, offset;
  std::shared_ptr<uint8_t> raw(b1.ReleaseRaw(size, offset), [size](uint8_t *ptr) {
    flatbuffers::DefaultAllocator::dealloc(ptr, size);
  });
  Builder src;
  auto root_offset2 = populate2(src);
  src.Finish(root_offset2);
  auto src_size = src.GetSize();
  // Move into a released builder.
  b1 = std::move(src);
  TEST_EQ_FUNC(b1.GetSize(), src_size);
  TEST_ASSERT_FUNC(release_n_verify(b1, m2_name, m2_color));
  TEST_EQ_FUNC(src.GetSize(), 0);
}
// clang-format off
#endif  // !defined(FLATBUFFERS_CPP98_STL)
// clang-format on

void builder_move_assign_after_releaseraw_test(flatbuffers::grpc::MessageBuilder b1);

template <class DestBuilder, class SrcBuilder = DestBuilder>
struct BuilderTests {
  // clang-format off
  #if !defined(FLATBUFFERS_CPP98_STL)
  // clang-format on
  static void empty_builder_movector_test() {
    SrcBuilder src;
    size_t src_size = src.GetSize();
    DestBuilder dst(std::move(src));
    size_t dst_size = dst.GetSize();
    TEST_EQ_FUNC(src_size, 0);
    TEST_EQ_FUNC(src_size, dst_size);
  }

  static void nonempty_builder_movector_test() {
    SrcBuilder src;
    populate1(src);
    size_t src_size = src.GetSize();
    DestBuilder dst(std::move(src));
    TEST_EQ_FUNC(src_size, dst.GetSize());
    TEST_EQ_FUNC(src.GetSize(), 0);
  }

  static void builder_movector_before_finish_test() {
    SrcBuilder src;
    auto root_offset1 = populate1(src);
    DestBuilder dst(std::move(src));
    dst.Finish(root_offset1);
    TEST_ASSERT_FUNC(release_n_verify(dst, m1_name, m1_color));
    TEST_EQ_FUNC(src.GetSize(), 0);
  }

  static void builder_movector_after_finish_test() {
    SrcBuilder src;
    auto root_offset1 = populate1(src);
    src.Finish(root_offset1);
    auto src_size = src.GetSize();
    DestBuilder dst(std::move(src));
    TEST_EQ_FUNC(dst.GetSize(), src_size);
    TEST_ASSERT_FUNC(release_n_verify(dst, m1_name, m1_color));
    TEST_EQ_FUNC(src.GetSize(), 0);
  }

  static void builder_move_assign_before_finish_test() {
    SrcBuilder src;
    auto root_offset1 = populate1(src);
    DestBuilder dst;
    populate2(dst);
    dst = std::move(src);
    dst.Finish(root_offset1);
    TEST_ASSERT_FUNC(release_n_verify(dst, m1_name, m1_color));
    TEST_EQ_FUNC(src.GetSize(), 0);
  }

  static void builder_move_assign_after_finish_test() {
    SrcBuilder src;
    auto root_offset1 = populate1(src);
    src.Finish(root_offset1);
    auto src_size = src.GetSize();
    DestBuilder dst;
    auto root_offset2 = populate2(dst);
    dst.Finish(root_offset2);
    dst = std::move(src);
    TEST_EQ_FUNC(dst.GetSize(), src_size);
    TEST_ASSERT_FUNC(release_n_verify(dst, m1_name, m1_color));
    TEST_EQ_FUNC(src.GetSize(), 0);
  }

  static void builder_move_assign_after_release_test() {
    DestBuilder dst;
    auto root_offset1 = populate1(dst);
    dst.Finish(root_offset1);
    {
      flatbuffers::DetachedBuffer dst_detached = dst.Release();
      // detached buffer is deleted
    }
    SrcBuilder src;
    auto root_offset2 = populate2(src);
    src.Finish(root_offset2);
    auto src_size = src.GetSize();
    // Move into a released builder.
    dst = std::move(src);
    TEST_EQ_FUNC(dst.GetSize(), src_size);
    TEST_ASSERT_FUNC(release_n_verify(dst, m2_name, m2_color));
    TEST_EQ_FUNC(src.GetSize(), 0);
  }
  // clang-format off
  #endif  // !defined(FLATBUFFERS_CPP98_STL)
  // clang-format on

  static void builder_swap_before_finish_test(bool run = is_same<DestBuilder, SrcBuilder>::value) {
    /// Swap is allowed only when lhs and rhs are the same concrete type.
    if(run) {
      SrcBuilder src;
      auto root_offset1 = populate1(src);
      auto size1 = src.GetSize();
      DestBuilder dst;
      auto root_offset2 = populate2(dst);
      auto size2 = dst.GetSize();
      src.Swap(dst);
      src.Finish(root_offset2);
      dst.Finish(root_offset1);
      TEST_EQ_FUNC(src.GetSize() > size2, true);
      TEST_EQ_FUNC(dst.GetSize() > size1, true);
      TEST_ASSERT_FUNC(release_n_verify(src, m2_name, m2_color));
      TEST_ASSERT_FUNC(release_n_verify(dst, m1_name, m1_color));
    }
  }

  static void builder_swap_after_finish_test(bool run = is_same<DestBuilder, SrcBuilder>::value) {
    /// Swap is allowed only when lhs and rhs are the same concrete type.
    if(run) {
      SrcBuilder src;
      auto root_offset1 = populate1(src);
      src.Finish(root_offset1);
      auto size1 = src.GetSize();
      DestBuilder dst;
      auto root_offset2 = populate2(dst);
      dst.Finish(root_offset2);
      auto size2 = dst.GetSize();
      src.Swap(dst);
      TEST_EQ_FUNC(src.GetSize(), size2);
      TEST_EQ_FUNC(dst.GetSize(), size1);
      TEST_ASSERT_FUNC(release_n_verify(src, m2_name, m2_color));
      TEST_ASSERT_FUNC(release_n_verify(dst, m1_name, m1_color));
    }
  }

  static void all_tests() {
    // clang-format off
    #if !defined(FLATBUFFERS_CPP98_STL)
    // clang-format on
    empty_builder_movector_test();
    nonempty_builder_movector_test();
    builder_movector_before_finish_test();
    builder_movector_after_finish_test();
    builder_move_assign_before_finish_test();
    builder_move_assign_after_finish_test();
    builder_move_assign_after_release_test();
    builder_move_assign_after_releaseraw_test(DestBuilder());
    // clang-format off
    #endif   // !defined(FLATBUFFERS_CPP98_STL)
    // clang-format on
    builder_swap_before_finish_test();
    builder_swap_after_finish_test();
  }
};

enum BuilderReuseTestSelector {
  REUSABLE_AFTER_RELEASE = 1,
  REUSABLE_AFTER_RELEASE_RAW = 2,
  REUSABLE_AFTER_RELEASE_MESSAGE = 3,
  REUSABLE_AFTER_RELEASE_AND_MOVE_ASSIGN = 4,
  REUSABLE_AFTER_RELEASE_RAW_AND_MOVE_ASSIGN = 5,
  REUSABLE_AFTER_RELEASE_MESSAGE_AND_MOVE_ASSIGN = 6
};

typedef std::set<BuilderReuseTestSelector> TestSelector;

template <class DestBuilder, class SrcBuilder>
struct BuilderReuseTests {
  static void builder_reusable_after_release_test(TestSelector selector) {
    if (!selector.count(REUSABLE_AFTER_RELEASE)) {
      return;
    }

    DestBuilder fbb;
    std::vector<flatbuffers::DetachedBuffer> buffers;
    for (int i = 0; i < 5; ++i) {
      auto root_offset1 = populate1(fbb);
      fbb.Finish(root_offset1);
      buffers.push_back(fbb.Release());
      TEST_ASSERT_FUNC(verify(buffers[i], m1_name, m1_color));
    }
  }

  static void builder_reusable_after_releaseraw_test(TestSelector selector) {
    if (!selector.count(REUSABLE_AFTER_RELEASE_RAW)) {
      return;
    }

    DestBuilder fbb;
    for (int i = 0; i < 5; ++i) {
      auto root_offset1 = populate1(fbb);
      fbb.Finish(root_offset1);
      size_t size, offset;
      uint8_t *buf = release_raw_base(fbb, size, offset);
      TEST_ASSERT_FUNC(verify(buf, offset, m1_name, m1_color));
      free_raw(fbb, buf);
    }
  }

  // clang-format off
  #if !defined(FLATBUFFERS_CPP98_STL)
  // clang-format on
  static void builder_reusable_after_release_and_move_assign_test(TestSelector selector) {
    if (!selector.count(REUSABLE_AFTER_RELEASE_AND_MOVE_ASSIGN)) {
      return;
    }

    DestBuilder dst;
    std::vector<flatbuffers::DetachedBuffer> buffers;
    for (int i = 0; i < 5; ++i) {
      auto root_offset1 = populate1(dst);
      dst.Finish(root_offset1);
      buffers.push_back(dst.Release());
      TEST_ASSERT_FUNC(verify(buffers[i], m1_name, m1_color));
      SrcBuilder src;
      dst = std::move(src);
      TEST_EQ_FUNC(src.GetSize(), 0);
    }
  }

  static void builder_reusable_after_releaseraw_and_move_assign_test(TestSelector selector) {
    if (!selector.count(REUSABLE_AFTER_RELEASE_RAW_AND_MOVE_ASSIGN)) {
      return;
    }

    DestBuilder dst;
    for (int i = 0; i < 5; ++i) {
      auto root_offset1 = populate1(dst);
      dst.Finish(root_offset1);
      size_t size, offset;
      uint8_t *buf = release_raw_base(dst, size, offset);
      TEST_ASSERT_FUNC(verify(buf, offset, m1_name, m1_color));
      free_raw(dst, buf);
      SrcBuilder src;
      dst = std::move(src);
      TEST_EQ_FUNC(src.GetSize(), 0);
    }
  }
  // clang-format off
  #endif  // !defined(FLATBUFFERS_CPP98_STL)
  // clang-format on

  static void run_tests(TestSelector selector) {
    builder_reusable_after_release_test(selector);
    builder_reusable_after_releaseraw_test(selector);
    // clang-format off
    #if !defined(FLATBUFFERS_CPP98_STL)
    // clang-format on
    builder_reusable_after_release_and_move_assign_test(selector);
    builder_reusable_after_releaseraw_and_move_assign_test(selector);
    // clang-format off
    #endif  // !defined(FLATBUFFERS_CPP98_STL)
    // clang-format on
  }
};

#endif // TEST_BUILDER_H
