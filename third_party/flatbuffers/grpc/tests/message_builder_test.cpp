#include "flatbuffers/grpc.h"
#include "monster_test_generated.h"
#include "test_assert.h"
#include "test_builder.h"

using MyGame::Example::Vec3;
using MyGame::Example::CreateStat;
using MyGame::Example::Any_NONE;

bool verify(flatbuffers::grpc::Message<Monster> &msg, const std::string &expected_name, Color color) {
  const Monster *monster = msg.GetRoot();
  return (monster->name()->str() == expected_name) && (monster->color() == color);
}

bool release_n_verify(flatbuffers::grpc::MessageBuilder &mbb, const std::string &expected_name, Color color) {
  flatbuffers::grpc::Message<Monster> msg = mbb.ReleaseMessage<Monster>();
  const Monster *monster = msg.GetRoot();
  return (monster->name()->str() == expected_name) && (monster->color() == color);
}

void builder_move_assign_after_releaseraw_test(flatbuffers::grpc::MessageBuilder dst) {
  auto root_offset1 = populate1(dst);
  dst.Finish(root_offset1);
  size_t size, offset;
  grpc_slice slice;
  dst.ReleaseRaw(size, offset, slice);
  flatbuffers::FlatBufferBuilder src;
  auto root_offset2 = populate2(src);
  src.Finish(root_offset2);
  auto src_size = src.GetSize();
  // Move into a released builder.
  dst = std::move(src);
  TEST_EQ(dst.GetSize(), src_size);
  TEST_ASSERT(release_n_verify(dst, m2_name, m2_color));
  TEST_EQ(src.GetSize(), 0);
  grpc_slice_unref(slice);
}

template <class SrcBuilder>
struct BuilderReuseTests<flatbuffers::grpc::MessageBuilder, SrcBuilder> {
  static void builder_reusable_after_release_message_test(TestSelector selector) {
    if (!selector.count(REUSABLE_AFTER_RELEASE_MESSAGE)) {
      return;
    }

    flatbuffers::grpc::MessageBuilder mb;
    std::vector<flatbuffers::grpc::Message<Monster>> buffers;
    for (int i = 0; i < 5; ++i) {
      auto root_offset1 = populate1(mb);
      mb.Finish(root_offset1);
      buffers.push_back(mb.ReleaseMessage<Monster>());
      TEST_ASSERT_FUNC(verify(buffers[i], m1_name, m1_color));
    }
  }

  static void builder_reusable_after_release_test(TestSelector selector) {
    if (!selector.count(REUSABLE_AFTER_RELEASE)) {
      return;
    }

    // FIXME: Populate-Release loop fails assert(GRPC_SLICE_IS_EMPTY(slice_)) in SliceAllocator::allocate
    // in the second iteration.

    flatbuffers::grpc::MessageBuilder mb;
    std::vector<flatbuffers::DetachedBuffer> buffers;
    for (int i = 0; i < 2; ++i) {
      auto root_offset1 = populate1(mb);
      mb.Finish(root_offset1);
      buffers.push_back(mb.Release());
      TEST_ASSERT_FUNC(verify(buffers[i], m1_name, m1_color));
    }
  }

  static void builder_reusable_after_releaseraw_test(TestSelector selector) {
    if (!selector.count(REUSABLE_AFTER_RELEASE_RAW)) {
      return;
    }

    flatbuffers::grpc::MessageBuilder mb;
    for (int i = 0; i < 5; ++i) {
      auto root_offset1 = populate1(mb);
      mb.Finish(root_offset1);
      size_t size, offset;
      grpc_slice slice;
      const uint8_t *buf = mb.ReleaseRaw(size, offset, slice);
      TEST_ASSERT_FUNC(verify(buf, offset, m1_name, m1_color));
      grpc_slice_unref(slice);
    }
  }

  static void builder_reusable_after_release_and_move_assign_test(TestSelector selector) {
    if (!selector.count(REUSABLE_AFTER_RELEASE_AND_MOVE_ASSIGN)) {
      return;
    }

    // FIXME: Release-move_assign loop fails assert(p == GRPC_SLICE_START_PTR(slice_))
    // in DetachedBuffer destructor after all the iterations

    flatbuffers::grpc::MessageBuilder dst;
    std::vector<flatbuffers::DetachedBuffer> buffers;

    for (int i = 0; i < 2; ++i) {
      auto root_offset1 = populate1(dst);
      dst.Finish(root_offset1);
      buffers.push_back(dst.Release());
      TEST_ASSERT_FUNC(verify(buffers[i], m1_name, m1_color));

      // bring dst back to life.
      SrcBuilder src;
      dst = std::move(src);
      TEST_EQ_FUNC(dst.GetSize(), 0);
      TEST_EQ_FUNC(src.GetSize(), 0);
    }
  }

  static void builder_reusable_after_release_message_and_move_assign_test(TestSelector selector) {
    if (!selector.count(REUSABLE_AFTER_RELEASE_MESSAGE_AND_MOVE_ASSIGN)) {
      return;
    }

    flatbuffers::grpc::MessageBuilder dst;
    std::vector<flatbuffers::grpc::Message<Monster>> buffers;

    for (int i = 0; i < 5; ++i) {
      auto root_offset1 = populate1(dst);
      dst.Finish(root_offset1);
      buffers.push_back(dst.ReleaseMessage<Monster>());
      TEST_ASSERT_FUNC(verify(buffers[i], m1_name, m1_color));

      // bring dst back to life.
      SrcBuilder src;
      dst = std::move(src);
      TEST_EQ_FUNC(dst.GetSize(), 0);
      TEST_EQ_FUNC(src.GetSize(), 0);
    }
  }

  static void builder_reusable_after_releaseraw_and_move_assign_test(TestSelector selector) {
    if (!selector.count(REUSABLE_AFTER_RELEASE_RAW_AND_MOVE_ASSIGN)) {
      return;
    }

    flatbuffers::grpc::MessageBuilder dst;
    for (int i = 0; i < 5; ++i) {
      auto root_offset1 = populate1(dst);
      dst.Finish(root_offset1);
      size_t size, offset;
      grpc_slice slice = grpc_empty_slice();
      const uint8_t *buf = dst.ReleaseRaw(size, offset, slice);
      TEST_ASSERT_FUNC(verify(buf, offset, m1_name, m1_color));
      grpc_slice_unref(slice);

      SrcBuilder src;
      dst = std::move(src);
      TEST_EQ_FUNC(dst.GetSize(), 0);
      TEST_EQ_FUNC(src.GetSize(), 0);
    }
  }

  static void run_tests(TestSelector selector) {
    builder_reusable_after_release_test(selector);
    builder_reusable_after_release_message_test(selector);
    builder_reusable_after_releaseraw_test(selector);
    builder_reusable_after_release_and_move_assign_test(selector);
    builder_reusable_after_releaseraw_and_move_assign_test(selector);
    builder_reusable_after_release_message_and_move_assign_test(selector);
  }
};

void slice_allocator_tests() {
  // move-construct no-delete test
  {
    size_t size = 2048;
    flatbuffers::grpc::SliceAllocator sa1;
    uint8_t *buf = sa1.allocate(size);
    TEST_ASSERT_FUNC(buf != 0);
    buf[0] = 100;
    buf[size-1] = 200;
    flatbuffers::grpc::SliceAllocator sa2(std::move(sa1));
    // buf should not be deleted after move-construct
    TEST_EQ_FUNC(buf[0], 100);
    TEST_EQ_FUNC(buf[size-1], 200);
    // buf is freed here
  }

  // move-assign test
  {
    flatbuffers::grpc::SliceAllocator sa1, sa2;
    uint8_t *buf = sa1.allocate(2048);
    sa1 = std::move(sa2);
    // sa1 deletes previously allocated memory in move-assign.
    // So buf is no longer usable here.
    TEST_ASSERT_FUNC(buf != 0);
  }
}

/// This function does not populate exactly the first half of the table. But it could.
void populate_first_half(MyGame::Example::MonsterBuilder &wrapper, flatbuffers::Offset<flatbuffers::String> name_offset) {
  wrapper.add_name(name_offset);
  wrapper.add_color(m1_color);
}

/// This function does not populate exactly the second half of the table. But it could.
void populate_second_half(MyGame::Example::MonsterBuilder &wrapper) {
  wrapper.add_hp(77);
  wrapper.add_mana(88);
  Vec3 vec3;
  wrapper.add_pos(&vec3);
}

/// This function is a hack to update the FlatBufferBuilder reference (fbb_) in the MonsterBuilder object.
/// This function will break if fbb_ is not the first member in MonsterBuilder. In that case, some offset must be added.
/// This function is used exclusively for testing correctness of move operations between FlatBufferBuilders.
/// If MonsterBuilder had a fbb_ pointer, this hack would be unnecessary. That involves a code-generator change though.
void test_only_hack_update_fbb_reference(MyGame::Example::MonsterBuilder &monsterBuilder,
                                         flatbuffers::grpc::MessageBuilder &mb) {
  *reinterpret_cast<flatbuffers::FlatBufferBuilder **>(&monsterBuilder) = &mb;
}

/// This test validates correctness of move conversion of FlatBufferBuilder to a MessageBuilder DURING
/// a table construction. Half of the table is constructed using FlatBufferBuilder and the other half
/// of the table is constructed using a MessageBuilder.
void builder_move_ctor_conversion_before_finish_half_n_half_table_test() {
  for (size_t initial_size = 4 ; initial_size <= 2048; initial_size *= 2) {
    flatbuffers::FlatBufferBuilder fbb(initial_size);
    auto name_offset = fbb.CreateString(m1_name);
    MyGame::Example::MonsterBuilder monsterBuilder(fbb);     // starts a table in FlatBufferBuilder
    populate_first_half(monsterBuilder, name_offset);
    flatbuffers::grpc::MessageBuilder mb(std::move(fbb));
    test_only_hack_update_fbb_reference(monsterBuilder, mb); // hack
    populate_second_half(monsterBuilder);
    mb.Finish(monsterBuilder.Finish());                      // ends the table in MessageBuilder
    TEST_ASSERT_FUNC(release_n_verify(mb, m1_name, m1_color));
    TEST_EQ_FUNC(fbb.GetSize(), 0);
  }
}

/// This test populates a COMPLETE inner table before move conversion and later populates more members in the outer table.
void builder_move_ctor_conversion_before_finish_test() {
  for (size_t initial_size = 4 ; initial_size <= 2048; initial_size *= 2) {
    flatbuffers::FlatBufferBuilder fbb(initial_size);
    auto stat_offset = CreateStat(fbb, fbb.CreateString("SomeId"), 0, 0);
    flatbuffers::grpc::MessageBuilder mb(std::move(fbb));
    auto monster_offset = CreateMonster(mb, 0, 150, 100, mb.CreateString(m1_name), 0, m1_color, Any_NONE, 0, 0, 0, 0, 0, 0, stat_offset);
    mb.Finish(monster_offset);
    TEST_ASSERT_FUNC(release_n_verify(mb, m1_name, m1_color));
    TEST_EQ_FUNC(fbb.GetSize(), 0);
  }
}

/// This test validates correctness of move conversion of FlatBufferBuilder to a MessageBuilder DURING
/// a table construction. Half of the table is constructed using FlatBufferBuilder and the other half
/// of the table is constructed using a MessageBuilder.
void builder_move_assign_conversion_before_finish_half_n_half_table_test() {
  flatbuffers::FlatBufferBuilder fbb;
  flatbuffers::grpc::MessageBuilder mb;

  for (int i = 0;i < 5; ++i) {
    flatbuffers::FlatBufferBuilder fbb;
    auto name_offset = fbb.CreateString(m1_name);
    MyGame::Example::MonsterBuilder monsterBuilder(fbb);     // starts a table in FlatBufferBuilder
    populate_first_half(monsterBuilder, name_offset);
    mb = std::move(fbb);
    test_only_hack_update_fbb_reference(monsterBuilder, mb); // hack
    populate_second_half(monsterBuilder);
    mb.Finish(monsterBuilder.Finish());                      // ends the table in MessageBuilder
    TEST_ASSERT_FUNC(release_n_verify(mb, m1_name, m1_color));
    TEST_EQ_FUNC(fbb.GetSize(), 0);
  }
}

/// This test populates a COMPLETE inner table before move conversion and later populates more members in the outer table.
void builder_move_assign_conversion_before_finish_test() {
  flatbuffers::FlatBufferBuilder fbb;
  flatbuffers::grpc::MessageBuilder mb;

  for (int i = 0;i < 5; ++i) {
    auto stat_offset = CreateStat(fbb, fbb.CreateString("SomeId"), 0, 0);
    mb = std::move(fbb);
    auto monster_offset = CreateMonster(mb, 0, 150, 100, mb.CreateString(m1_name), 0, m1_color, Any_NONE, 0, 0, 0, 0, 0, 0, stat_offset);
    mb.Finish(monster_offset);
    TEST_ASSERT_FUNC(release_n_verify(mb, m1_name, m1_color));
    TEST_EQ_FUNC(fbb.GetSize(), 0);
  }
}

/// This test populates data, finishes the buffer, and does move conversion after.
void builder_move_ctor_conversion_after_finish_test() {
  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(populate1(fbb));
  flatbuffers::grpc::MessageBuilder mb(std::move(fbb));
  TEST_ASSERT_FUNC(release_n_verify(mb, m1_name, m1_color));
  TEST_EQ_FUNC(fbb.GetSize(), 0);
}

/// This test populates data, finishes the buffer, and does move conversion after.
void builder_move_assign_conversion_after_finish_test() {
  flatbuffers::FlatBufferBuilder fbb;
  flatbuffers::grpc::MessageBuilder mb;

  for (int i = 0;i < 5; ++i) {
    fbb.Finish(populate1(fbb));
    mb = std::move(fbb);
    TEST_ASSERT_FUNC(release_n_verify(mb, m1_name, m1_color));
    TEST_EQ_FUNC(fbb.GetSize(), 0);
  }
}

void message_builder_tests() {
  using flatbuffers::grpc::MessageBuilder;
  using flatbuffers::FlatBufferBuilder;

  slice_allocator_tests();

#ifndef __APPLE__
  builder_move_ctor_conversion_before_finish_half_n_half_table_test();
  builder_move_assign_conversion_before_finish_half_n_half_table_test();
#endif // __APPLE__
  builder_move_ctor_conversion_before_finish_test();
  builder_move_assign_conversion_before_finish_test();

  builder_move_ctor_conversion_after_finish_test();
  builder_move_assign_conversion_after_finish_test();

  BuilderTests<MessageBuilder, MessageBuilder>::all_tests();
  BuilderTests<MessageBuilder, FlatBufferBuilder>::all_tests();

  BuilderReuseTestSelector tests[6] = {
    //REUSABLE_AFTER_RELEASE,                 // Assertion failed: (GRPC_SLICE_IS_EMPTY(slice_))
    //REUSABLE_AFTER_RELEASE_AND_MOVE_ASSIGN, // Assertion failed: (p == GRPC_SLICE_START_PTR(slice_)

    REUSABLE_AFTER_RELEASE_RAW,
    REUSABLE_AFTER_RELEASE_MESSAGE,
    REUSABLE_AFTER_RELEASE_MESSAGE_AND_MOVE_ASSIGN,
    REUSABLE_AFTER_RELEASE_RAW_AND_MOVE_ASSIGN
  };

  BuilderReuseTests<MessageBuilder, MessageBuilder>::run_tests(TestSelector(tests, tests+6));
  BuilderReuseTests<MessageBuilder, FlatBufferBuilder>::run_tests(TestSelector(tests, tests+6));
}
