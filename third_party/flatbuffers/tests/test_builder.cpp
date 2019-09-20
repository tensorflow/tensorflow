#include "flatbuffers/stl_emulation.h"

#include "monster_test_generated.h"
#include "test_builder.h"

using namespace MyGame::Example;

const std::string m1_name = "Cyberdemon";
const Color m1_color = Color_Red;
const std::string m2_name = "Imp";
const Color m2_color = Color_Green;

struct OwnedAllocator : public flatbuffers::DefaultAllocator {};

class TestHeapBuilder : public flatbuffers::FlatBufferBuilder {
private:
  // clang-format off
  #if !defined(FLATBUFFERS_CPP98_STL)
  TestHeapBuilder(const TestHeapBuilder &);
  TestHeapBuilder &operator=(const TestHeapBuilder &);
  #endif  // !defined(FLATBUFFERS_CPP98_STL)
  // clang-format on

public:
  TestHeapBuilder()
    : flatbuffers::FlatBufferBuilder(2048, new OwnedAllocator(), true) {}

  // clang-format off
  #if !defined(FLATBUFFERS_CPP98_STL)
  // clang-format on
  TestHeapBuilder(TestHeapBuilder &&other)
    : FlatBufferBuilder(std::move(other)) { }

  TestHeapBuilder &operator=(TestHeapBuilder &&other) {
    FlatBufferBuilder::operator=(std::move(other));
    return *this;
  }
  // clang-format off
  #endif  // !defined(FLATBUFFERS_CPP98_STL)
  // clang-format on
};

// This class simulates flatbuffers::grpc::detail::SliceAllocatorMember
struct AllocatorMember {
  flatbuffers::DefaultAllocator member_allocator_;
};

struct GrpcLikeMessageBuilder : private AllocatorMember,
                                public flatbuffers::FlatBufferBuilder {
private:
  GrpcLikeMessageBuilder(const GrpcLikeMessageBuilder &);
  GrpcLikeMessageBuilder &operator=(const GrpcLikeMessageBuilder &);

public:
  GrpcLikeMessageBuilder()
    : flatbuffers::FlatBufferBuilder(1024, &member_allocator_, false) {}

  GrpcLikeMessageBuilder(GrpcLikeMessageBuilder &&other)
    : FlatBufferBuilder(1024, &member_allocator_, false) {
    // Default construct and swap idiom.
    Swap(other);
  }

  // clang-format off
  #if !defined(FLATBUFFERS_CPP98_STL)
  // clang-format on
  GrpcLikeMessageBuilder &operator=(GrpcLikeMessageBuilder &&other) {
    // Construct temporary and swap idiom
    GrpcLikeMessageBuilder temp(std::move(other));
    Swap(temp);
    return *this;
  }
  // clang-format off
  #endif  // !defined(FLATBUFFERS_CPP98_STL)
  // clang-format on

  void Swap(GrpcLikeMessageBuilder &other) {
    // No need to swap member_allocator_ because it's stateless.
    FlatBufferBuilder::Swap(other);
    // After swapping the FlatBufferBuilder, we swap back the allocator, which restores
    // the original allocator back in place. This is necessary because MessageBuilder's
    // allocator is its own member (SliceAllocatorMember). The allocator passed to
    // FlatBufferBuilder::vector_downward must point to this member.
    buf_.swap_allocator(other.buf_);
  }
};

flatbuffers::Offset<Monster> populate1(flatbuffers::FlatBufferBuilder &builder) {
  auto name_offset = builder.CreateString(m1_name);
  return CreateMonster(builder, nullptr, 0, 0, name_offset, 0, m1_color);
}

flatbuffers::Offset<Monster> populate2(flatbuffers::FlatBufferBuilder &builder) {
  auto name_offset = builder.CreateString(m2_name);
  return CreateMonster(builder, nullptr, 0, 0, name_offset, 0, m2_color);
}

uint8_t *release_raw_base(flatbuffers::FlatBufferBuilder &fbb, size_t &size, size_t &offset) {
  return fbb.ReleaseRaw(size, offset);
}

void free_raw(flatbuffers::grpc::MessageBuilder &, uint8_t *) {
  // release_raw_base calls FlatBufferBuilder::ReleaseRaw on the argument MessageBuilder.
  // It's semantically wrong as MessageBuilder has its own ReleaseRaw member function that
  // takes three arguments. In such cases though, ~MessageBuilder() invokes
  // ~SliceAllocator() that takes care of deleting memory as it calls grpc_slice_unref.
  // Obviously, this behavior is very surprising as the pointer returned by
  // FlatBufferBuilder::ReleaseRaw is not valid as soon as MessageBuilder goes out of scope.
  // This problem does not occur with FlatBufferBuilder.
}

void free_raw(flatbuffers::FlatBufferBuilder &, uint8_t *buf) {
  flatbuffers::DefaultAllocator().deallocate(buf, 0);
}

bool verify(const flatbuffers::DetachedBuffer &buf, const std::string &expected_name, Color color) {
  const Monster *monster = flatbuffers::GetRoot<Monster>(buf.data());
  return (monster->name()->str() == expected_name) && (monster->color() == color);
}

bool verify(const uint8_t *buf, size_t offset, const std::string &expected_name, Color color) {
  const Monster *monster = flatbuffers::GetRoot<Monster>(buf+offset);
  return (monster->name()->str() == expected_name) && (monster->color() == color);
}

bool release_n_verify(flatbuffers::FlatBufferBuilder &fbb, const std::string &expected_name, Color color) {
  flatbuffers::DetachedBuffer buf = fbb.Release();
  return verify(buf, expected_name, color);
}

void FlatBufferBuilderTest() {
  using flatbuffers::FlatBufferBuilder;

  BuilderTests<FlatBufferBuilder>::all_tests();
  BuilderTests<TestHeapBuilder>::all_tests();
  BuilderTests<GrpcLikeMessageBuilder>::all_tests();

  BuilderReuseTestSelector tests[4] = {
    REUSABLE_AFTER_RELEASE,
    REUSABLE_AFTER_RELEASE_RAW,
    REUSABLE_AFTER_RELEASE_AND_MOVE_ASSIGN,
    REUSABLE_AFTER_RELEASE_RAW_AND_MOVE_ASSIGN
  };

  BuilderReuseTests<FlatBufferBuilder, FlatBufferBuilder>::run_tests(TestSelector(tests, tests+4));
  BuilderReuseTests<TestHeapBuilder, TestHeapBuilder>::run_tests(TestSelector(tests, tests+4));
  BuilderReuseTests<GrpcLikeMessageBuilder, GrpcLikeMessageBuilder>::run_tests(TestSelector(tests, tests+4));
}
