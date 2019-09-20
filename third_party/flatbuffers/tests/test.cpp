/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cmath>
#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/minireflect.h"
#include "flatbuffers/registry.h"
#include "flatbuffers/util.h"

// clang-format off
#ifdef FLATBUFFERS_CPP98_STL
  #include "flatbuffers/stl_emulation.h"
  namespace std {
    using flatbuffers::unique_ptr;
  }
#endif
// clang-format on

#include "monster_test_generated.h"
#include "namespace_test/namespace_test1_generated.h"
#include "namespace_test/namespace_test2_generated.h"
#include "union_vector/union_vector_generated.h"
#include "monster_extra_generated.h"
#if !defined(_MSC_VER) || _MSC_VER >= 1700
#include "arrays_test_generated.h"
#endif

#include "native_type_test_generated.h"
#include "test_assert.h"

#include "flatbuffers/flexbuffers.h"


// clang-format off
// Check that char* and uint8_t* are interoperable types.
// The reinterpret_cast<> between the pointers are used to simplify data loading.
static_assert(flatbuffers::is_same<uint8_t, char>::value ||
              flatbuffers::is_same<uint8_t, unsigned char>::value,
              "unexpected uint8_t type");

#if defined(FLATBUFFERS_HAS_NEW_STRTOD) && (FLATBUFFERS_HAS_NEW_STRTOD > 0)
  // Ensure IEEE-754 support if tests of floats with NaN/Inf will run.
  static_assert(std::numeric_limits<float>::is_iec559 &&
                std::numeric_limits<double>::is_iec559,
                "IEC-559 (IEEE-754) standard required");
#endif
// clang-format on

// Shortcuts for the infinity.
static const auto infinityf = std::numeric_limits<float>::infinity();
static const auto infinityd = std::numeric_limits<double>::infinity();

using namespace MyGame::Example;

void FlatBufferBuilderTest();

// Include simple random number generator to ensure results will be the
// same cross platform.
// http://en.wikipedia.org/wiki/Park%E2%80%93Miller_random_number_generator
uint32_t lcg_seed = 48271;
uint32_t lcg_rand() {
  return lcg_seed = (static_cast<uint64_t>(lcg_seed) * 279470273UL) % 4294967291UL;
}
void lcg_reset() { lcg_seed = 48271; }

std::string test_data_path =
#ifdef BAZEL_TEST_DATA_PATH
    "../com_github_google_flatbuffers/tests/";
#else
    "tests/";
#endif

// example of how to build up a serialized buffer algorithmically:
flatbuffers::DetachedBuffer CreateFlatBufferTest(std::string &buffer) {
  flatbuffers::FlatBufferBuilder builder;

  auto vec = Vec3(1, 2, 3, 0, Color_Red, Test(10, 20));

  auto name = builder.CreateString("MyMonster");

  unsigned char inv_data[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
  auto inventory = builder.CreateVector(inv_data, 10);

  // Alternatively, create the vector first, and fill in data later:
  // unsigned char *inv_buf = nullptr;
  // auto inventory = builder.CreateUninitializedVector<unsigned char>(
  //                                                              10, &inv_buf);
  // memcpy(inv_buf, inv_data, 10);

  Test tests[] = { Test(10, 20), Test(30, 40) };
  auto testv = builder.CreateVectorOfStructs(tests, 2);

  // clang-format off
  #ifndef FLATBUFFERS_CPP98_STL
    // Create a vector of structures from a lambda.
    auto testv2 = builder.CreateVectorOfStructs<Test>(
          2, [&](size_t i, Test* s) -> void {
            *s = tests[i];
          });
  #else
    // Create a vector of structures using a plain old C++ function.
    auto testv2 = builder.CreateVectorOfStructs<Test>(
          2, [](size_t i, Test* s, void *state) -> void {
            *s = (reinterpret_cast<Test*>(state))[i];
          }, tests);
  #endif  // FLATBUFFERS_CPP98_STL
  // clang-format on

  // create monster with very few fields set:
  // (same functionality as CreateMonster below, but sets fields manually)
  flatbuffers::Offset<Monster> mlocs[3];
  auto fred = builder.CreateString("Fred");
  auto barney = builder.CreateString("Barney");
  auto wilma = builder.CreateString("Wilma");
  MonsterBuilder mb1(builder);
  mb1.add_name(fred);
  mlocs[0] = mb1.Finish();
  MonsterBuilder mb2(builder);
  mb2.add_name(barney);
  mb2.add_hp(1000);
  mlocs[1] = mb2.Finish();
  MonsterBuilder mb3(builder);
  mb3.add_name(wilma);
  mlocs[2] = mb3.Finish();

  // Create an array of strings. Also test string pooling, and lambdas.
  auto vecofstrings =
      builder.CreateVector<flatbuffers::Offset<flatbuffers::String>>(
          4,
          [](size_t i, flatbuffers::FlatBufferBuilder *b)
              -> flatbuffers::Offset<flatbuffers::String> {
            static const char *names[] = { "bob", "fred", "bob", "fred" };
            return b->CreateSharedString(names[i]);
          },
          &builder);

  // Creating vectors of strings in one convenient call.
  std::vector<std::string> names2;
  names2.push_back("jane");
  names2.push_back("mary");
  auto vecofstrings2 = builder.CreateVectorOfStrings(names2);

  // Create an array of sorted tables, can be used with binary search when read:
  auto vecoftables = builder.CreateVectorOfSortedTables(mlocs, 3);

  // Create an array of sorted structs,
  // can be used with binary search when read:
  std::vector<Ability> abilities;
  abilities.push_back(Ability(4, 40));
  abilities.push_back(Ability(3, 30));
  abilities.push_back(Ability(2, 20));
  abilities.push_back(Ability(1, 10));
  auto vecofstructs = builder.CreateVectorOfSortedStructs(&abilities);

  // Create a nested FlatBuffer.
  // Nested FlatBuffers are stored in a ubyte vector, which can be convenient
  // since they can be memcpy'd around much easier than other FlatBuffer
  // values. They have little overhead compared to storing the table directly.
  // As a test, create a mostly empty Monster buffer:
  flatbuffers::FlatBufferBuilder nested_builder;
  auto nmloc = CreateMonster(nested_builder, nullptr, 0, 0,
                             nested_builder.CreateString("NestedMonster"));
  FinishMonsterBuffer(nested_builder, nmloc);
  // Now we can store the buffer in the parent. Note that by default, vectors
  // are only aligned to their elements or size field, so in this case if the
  // buffer contains 64-bit elements, they may not be correctly aligned. We fix
  // that with:
  builder.ForceVectorAlignment(nested_builder.GetSize(), sizeof(uint8_t),
                               nested_builder.GetBufferMinAlignment());
  // If for whatever reason you don't have the nested_builder available, you
  // can substitute flatbuffers::largest_scalar_t (64-bit) for the alignment, or
  // the largest force_align value in your schema if you're using it.
  auto nested_flatbuffer_vector = builder.CreateVector(
      nested_builder.GetBufferPointer(), nested_builder.GetSize());

  // Test a nested FlexBuffer:
  flexbuffers::Builder flexbuild;
  flexbuild.Int(1234);
  flexbuild.Finish();
  auto flex = builder.CreateVector(flexbuild.GetBuffer());

  // Test vector of enums.
  Color colors[] = { Color_Blue, Color_Green };
  // We use this special creation function because we have an array of
  // pre-C++11 (enum class) enums whose size likely is int, yet its declared
  // type in the schema is byte.
  auto vecofcolors = builder.CreateVectorScalarCast<uint8_t, Color>(colors, 2);

  // shortcut for creating monster with all fields set:
  auto mloc = CreateMonster(builder, &vec, 150, 80, name, inventory, Color_Blue,
                            Any_Monster, mlocs[1].Union(),  // Store a union.
                            testv, vecofstrings, vecoftables, 0,
                            nested_flatbuffer_vector, 0, false, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 3.14159f, 3.0f, 0.0f, vecofstrings2,
                            vecofstructs, flex, testv2, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, AnyUniqueAliases_NONE, 0,
                            AnyAmbiguousAliases_NONE, 0, vecofcolors);

  FinishMonsterBuffer(builder, mloc);

  // clang-format off
  #ifdef FLATBUFFERS_TEST_VERBOSE
  // print byte data for debugging:
  auto p = builder.GetBufferPointer();
  for (flatbuffers::uoffset_t i = 0; i < builder.GetSize(); i++)
    printf("%d ", p[i]);
  #endif
  // clang-format on

  // return the buffer for the caller to use.
  auto bufferpointer =
      reinterpret_cast<const char *>(builder.GetBufferPointer());
  buffer.assign(bufferpointer, bufferpointer + builder.GetSize());

  return builder.Release();
}

//  example of accessing a buffer loaded in memory:
void AccessFlatBufferTest(const uint8_t *flatbuf, size_t length,
                          bool pooled = true) {
  // First, verify the buffers integrity (optional)
  flatbuffers::Verifier verifier(flatbuf, length);
  TEST_EQ(VerifyMonsterBuffer(verifier), true);

  // clang-format off
  #ifdef FLATBUFFERS_TRACK_VERIFIER_BUFFER_SIZE
    std::vector<uint8_t> test_buff;
    test_buff.resize(length * 2);
    std::memcpy(&test_buff[0], flatbuf, length);
    std::memcpy(&test_buff[length], flatbuf, length);

    flatbuffers::Verifier verifier1(&test_buff[0], length);
    TEST_EQ(VerifyMonsterBuffer(verifier1), true);
    TEST_EQ(verifier1.GetComputedSize(), length);

    flatbuffers::Verifier verifier2(&test_buff[length], length);
    TEST_EQ(VerifyMonsterBuffer(verifier2), true);
    TEST_EQ(verifier2.GetComputedSize(), length);
  #endif
  // clang-format on

  TEST_EQ(strcmp(MonsterIdentifier(), "MONS"), 0);
  TEST_EQ(MonsterBufferHasIdentifier(flatbuf), true);
  TEST_EQ(strcmp(MonsterExtension(), "mon"), 0);

  // Access the buffer from the root.
  auto monster = GetMonster(flatbuf);

  TEST_EQ(monster->hp(), 80);
  TEST_EQ(monster->mana(), 150);  // default
  TEST_EQ_STR(monster->name()->c_str(), "MyMonster");
  // Can't access the following field, it is deprecated in the schema,
  // which means accessors are not generated:
  // monster.friendly()

  auto pos = monster->pos();
  TEST_NOTNULL(pos);
  TEST_EQ(pos->z(), 3);
  TEST_EQ(pos->test3().a(), 10);
  TEST_EQ(pos->test3().b(), 20);

  auto inventory = monster->inventory();
  TEST_EQ(VectorLength(inventory), 10UL);  // Works even if inventory is null.
  TEST_NOTNULL(inventory);
  unsigned char inv_data[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
  // Check compatibilty of iterators with STL.
  std::vector<unsigned char> inv_vec(inventory->begin(), inventory->end());
  int n = 0;
  for (auto it = inventory->begin(); it != inventory->end(); ++it, ++n) {
    auto indx = it - inventory->begin();
    TEST_EQ(*it, inv_vec.at(indx));  // Use bounds-check.
    TEST_EQ(*it, inv_data[indx]);
  }
  TEST_EQ(n, inv_vec.size());

  n = 0;
  for (auto it = inventory->cbegin(); it != inventory->cend(); ++it, ++n) {
    auto indx = it - inventory->cbegin();
    TEST_EQ(*it, inv_vec.at(indx));  // Use bounds-check.
    TEST_EQ(*it, inv_data[indx]);
  }
  TEST_EQ(n, inv_vec.size());

  n = 0;
  for (auto it = inventory->rbegin(); it != inventory->rend(); ++it, ++n) {
    auto indx = inventory->rend() - it - 1;
    TEST_EQ(*it, inv_vec.at(indx));  // Use bounds-check.
    TEST_EQ(*it, inv_data[indx]);
  }
  TEST_EQ(n, inv_vec.size());

  n = 0;
  for (auto it = inventory->crbegin(); it != inventory->crend(); ++it, ++n) {
    auto indx = inventory->crend() - it - 1;
    TEST_EQ(*it, inv_vec.at(indx));  // Use bounds-check.
    TEST_EQ(*it, inv_data[indx]);
  }
  TEST_EQ(n, inv_vec.size());

  TEST_EQ(monster->color(), Color_Blue);

  // Example of accessing a union:
  TEST_EQ(monster->test_type(), Any_Monster);  // First make sure which it is.
  auto monster2 = reinterpret_cast<const Monster *>(monster->test());
  TEST_NOTNULL(monster2);
  TEST_EQ_STR(monster2->name()->c_str(), "Fred");

  // Example of accessing a vector of strings:
  auto vecofstrings = monster->testarrayofstring();
  TEST_EQ(vecofstrings->size(), 4U);
  TEST_EQ_STR(vecofstrings->Get(0)->c_str(), "bob");
  TEST_EQ_STR(vecofstrings->Get(1)->c_str(), "fred");
  if (pooled) {
    // These should have pointer equality because of string pooling.
    TEST_EQ(vecofstrings->Get(0)->c_str(), vecofstrings->Get(2)->c_str());
    TEST_EQ(vecofstrings->Get(1)->c_str(), vecofstrings->Get(3)->c_str());
  }

  auto vecofstrings2 = monster->testarrayofstring2();
  if (vecofstrings2) {
    TEST_EQ(vecofstrings2->size(), 2U);
    TEST_EQ_STR(vecofstrings2->Get(0)->c_str(), "jane");
    TEST_EQ_STR(vecofstrings2->Get(1)->c_str(), "mary");
  }

  // Example of accessing a vector of tables:
  auto vecoftables = monster->testarrayoftables();
  TEST_EQ(vecoftables->size(), 3U);
  for (auto it = vecoftables->begin(); it != vecoftables->end(); ++it)
    TEST_EQ(strlen(it->name()->c_str()) >= 4, true);
  TEST_EQ_STR(vecoftables->Get(0)->name()->c_str(), "Barney");
  TEST_EQ(vecoftables->Get(0)->hp(), 1000);
  TEST_EQ_STR(vecoftables->Get(1)->name()->c_str(), "Fred");
  TEST_EQ_STR(vecoftables->Get(2)->name()->c_str(), "Wilma");
  TEST_NOTNULL(vecoftables->LookupByKey("Barney"));
  TEST_NOTNULL(vecoftables->LookupByKey("Fred"));
  TEST_NOTNULL(vecoftables->LookupByKey("Wilma"));

  // Test accessing a vector of sorted structs
  auto vecofstructs = monster->testarrayofsortedstruct();
  if (vecofstructs) {  // not filled in monster_test.bfbs
    for (flatbuffers::uoffset_t i = 0; i < vecofstructs->size() - 1; i++) {
      auto left = vecofstructs->Get(i);
      auto right = vecofstructs->Get(i + 1);
      TEST_EQ(true, (left->KeyCompareLessThan(right)));
    }
    TEST_NOTNULL(vecofstructs->LookupByKey(3));
    TEST_EQ(static_cast<const Ability *>(nullptr),
            vecofstructs->LookupByKey(5));
  }

  // Test nested FlatBuffers if available:
  auto nested_buffer = monster->testnestedflatbuffer();
  if (nested_buffer) {
    // nested_buffer is a vector of bytes you can memcpy. However, if you
    // actually want to access the nested data, this is a convenient
    // accessor that directly gives you the root table:
    auto nested_monster = monster->testnestedflatbuffer_nested_root();
    TEST_EQ_STR(nested_monster->name()->c_str(), "NestedMonster");
  }

  // Test flexbuffer if available:
  auto flex = monster->flex();
  // flex is a vector of bytes you can memcpy etc.
  TEST_EQ(flex->size(), 4);  // Encoded FlexBuffer bytes.
  // However, if you actually want to access the nested data, this is a
  // convenient accessor that directly gives you the root value:
  TEST_EQ(monster->flex_flexbuffer_root().AsInt16(), 1234);

  // Test vector of enums:
  auto colors = monster->vector_of_enums();
  if (colors) {
    TEST_EQ(colors->size(), 2);
    TEST_EQ(colors->Get(0), Color_Blue);
    TEST_EQ(colors->Get(1), Color_Green);
  }

  // Since Flatbuffers uses explicit mechanisms to override the default
  // compiler alignment, double check that the compiler indeed obeys them:
  // (Test consists of a short and byte):
  TEST_EQ(flatbuffers::AlignOf<Test>(), 2UL);
  TEST_EQ(sizeof(Test), 4UL);

  const flatbuffers::Vector<const Test *> *tests_array[] = {
    monster->test4(),
    monster->test5(),
  };
  for (size_t i = 0; i < sizeof(tests_array) / sizeof(tests_array[0]); ++i) {
    auto tests = tests_array[i];
    TEST_NOTNULL(tests);
    auto test_0 = tests->Get(0);
    auto test_1 = tests->Get(1);
    TEST_EQ(test_0->a(), 10);
    TEST_EQ(test_0->b(), 20);
    TEST_EQ(test_1->a(), 30);
    TEST_EQ(test_1->b(), 40);
    for (auto it = tests->begin(); it != tests->end(); ++it) {
      TEST_EQ(it->a() == 10 || it->a() == 30, true);  // Just testing iterators.
    }
  }

  // Checking for presence of fields:
  TEST_EQ(flatbuffers::IsFieldPresent(monster, Monster::VT_HP), true);
  TEST_EQ(flatbuffers::IsFieldPresent(monster, Monster::VT_MANA), false);

  // Obtaining a buffer from a root:
  TEST_EQ(GetBufferStartFromRootPointer(monster), flatbuf);
}

// Change a FlatBuffer in-place, after it has been constructed.
void MutateFlatBuffersTest(uint8_t *flatbuf, std::size_t length) {
  // Get non-const pointer to root.
  auto monster = GetMutableMonster(flatbuf);

  // Each of these tests mutates, then tests, then set back to the original,
  // so we can test that the buffer in the end still passes our original test.
  auto hp_ok = monster->mutate_hp(10);
  TEST_EQ(hp_ok, true);  // Field was present.
  TEST_EQ(monster->hp(), 10);
  // Mutate to default value
  auto hp_ok_default = monster->mutate_hp(100);
  TEST_EQ(hp_ok_default, true);  // Field was present.
  TEST_EQ(monster->hp(), 100);
  // Test that mutate to default above keeps field valid for further mutations
  auto hp_ok_2 = monster->mutate_hp(20);
  TEST_EQ(hp_ok_2, true);
  TEST_EQ(monster->hp(), 20);
  monster->mutate_hp(80);

  // Monster originally at 150 mana (default value)
  auto mana_default_ok = monster->mutate_mana(150);  // Mutate to default value.
  TEST_EQ(mana_default_ok,
          true);  // Mutation should succeed, because default value.
  TEST_EQ(monster->mana(), 150);
  auto mana_ok = monster->mutate_mana(10);
  TEST_EQ(mana_ok, false);  // Field was NOT present, because default value.
  TEST_EQ(monster->mana(), 150);

  // Mutate structs.
  auto pos = monster->mutable_pos();
  auto test3 = pos->mutable_test3();  // Struct inside a struct.
  test3.mutate_a(50);                 // Struct fields never fail.
  TEST_EQ(test3.a(), 50);
  test3.mutate_a(10);

  // Mutate vectors.
  auto inventory = monster->mutable_inventory();
  inventory->Mutate(9, 100);
  TEST_EQ(inventory->Get(9), 100);
  inventory->Mutate(9, 9);

  auto tables = monster->mutable_testarrayoftables();
  auto first = tables->GetMutableObject(0);
  TEST_EQ(first->hp(), 1000);
  first->mutate_hp(0);
  TEST_EQ(first->hp(), 0);
  first->mutate_hp(1000);

  // Run the verifier and the regular test to make sure we didn't trample on
  // anything.
  AccessFlatBufferTest(flatbuf, length);
}

// Unpack a FlatBuffer into objects.
void ObjectFlatBuffersTest(uint8_t *flatbuf) {
  // Optional: we can specify resolver and rehasher functions to turn hashed
  // strings into object pointers and back, to implement remote references
  // and such.
  auto resolver = flatbuffers::resolver_function_t(
      [](void **pointer_adr, flatbuffers::hash_value_t hash) {
        (void)pointer_adr;
        (void)hash;
        // Don't actually do anything, leave variable null.
      });
  auto rehasher = flatbuffers::rehasher_function_t(
      [](void *pointer) -> flatbuffers::hash_value_t {
        (void)pointer;
        return 0;
      });

  // Turn a buffer into C++ objects.
  auto monster1 = UnPackMonster(flatbuf, &resolver);

  // Re-serialize the data.
  flatbuffers::FlatBufferBuilder fbb1;
  fbb1.Finish(CreateMonster(fbb1, monster1.get(), &rehasher),
              MonsterIdentifier());

  // Unpack again, and re-serialize again.
  auto monster2 = UnPackMonster(fbb1.GetBufferPointer(), &resolver);
  flatbuffers::FlatBufferBuilder fbb2;
  fbb2.Finish(CreateMonster(fbb2, monster2.get(), &rehasher),
              MonsterIdentifier());

  // Now we've gone full round-trip, the two buffers should match.
  auto len1 = fbb1.GetSize();
  auto len2 = fbb2.GetSize();
  TEST_EQ(len1, len2);
  TEST_EQ(memcmp(fbb1.GetBufferPointer(), fbb2.GetBufferPointer(), len1), 0);

  // Test it with the original buffer test to make sure all data survived.
  AccessFlatBufferTest(fbb2.GetBufferPointer(), len2, false);

  // Test accessing fields, similar to AccessFlatBufferTest above.
  TEST_EQ(monster2->hp, 80);
  TEST_EQ(monster2->mana, 150);  // default
  TEST_EQ_STR(monster2->name.c_str(), "MyMonster");

  auto &pos = monster2->pos;
  TEST_NOTNULL(pos);
  TEST_EQ(pos->z(), 3);
  TEST_EQ(pos->test3().a(), 10);
  TEST_EQ(pos->test3().b(), 20);

  auto &inventory = monster2->inventory;
  TEST_EQ(inventory.size(), 10UL);
  unsigned char inv_data[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
  for (auto it = inventory.begin(); it != inventory.end(); ++it)
    TEST_EQ(*it, inv_data[it - inventory.begin()]);

  TEST_EQ(monster2->color, Color_Blue);

  auto monster3 = monster2->test.AsMonster();
  TEST_NOTNULL(monster3);
  TEST_EQ_STR(monster3->name.c_str(), "Fred");

  auto &vecofstrings = monster2->testarrayofstring;
  TEST_EQ(vecofstrings.size(), 4U);
  TEST_EQ_STR(vecofstrings[0].c_str(), "bob");
  TEST_EQ_STR(vecofstrings[1].c_str(), "fred");

  auto &vecofstrings2 = monster2->testarrayofstring2;
  TEST_EQ(vecofstrings2.size(), 2U);
  TEST_EQ_STR(vecofstrings2[0].c_str(), "jane");
  TEST_EQ_STR(vecofstrings2[1].c_str(), "mary");

  auto &vecoftables = monster2->testarrayoftables;
  TEST_EQ(vecoftables.size(), 3U);
  TEST_EQ_STR(vecoftables[0]->name.c_str(), "Barney");
  TEST_EQ(vecoftables[0]->hp, 1000);
  TEST_EQ_STR(vecoftables[1]->name.c_str(), "Fred");
  TEST_EQ_STR(vecoftables[2]->name.c_str(), "Wilma");

  auto &tests = monster2->test4;
  TEST_EQ(tests[0].a(), 10);
  TEST_EQ(tests[0].b(), 20);
  TEST_EQ(tests[1].a(), 30);
  TEST_EQ(tests[1].b(), 40);
}

// Prefix a FlatBuffer with a size field.
void SizePrefixedTest() {
  // Create size prefixed buffer.
  flatbuffers::FlatBufferBuilder fbb;
  FinishSizePrefixedMonsterBuffer(
      fbb,
      CreateMonster(fbb, 0, 200, 300, fbb.CreateString("bob")));

  // Verify it.
  flatbuffers::Verifier verifier(fbb.GetBufferPointer(), fbb.GetSize());
  TEST_EQ(VerifySizePrefixedMonsterBuffer(verifier), true);

  // Access it.
  auto m = GetSizePrefixedMonster(fbb.GetBufferPointer());
  TEST_EQ(m->mana(), 200);
  TEST_EQ(m->hp(), 300);
  TEST_EQ_STR(m->name()->c_str(), "bob");
}

void TriviallyCopyableTest() {
  // clang-format off
  #if __GNUG__ && __GNUC__ < 5
    TEST_EQ(__has_trivial_copy(Vec3), true);
  #else
    #if __cplusplus >= 201103L
      TEST_EQ(std::is_trivially_copyable<Vec3>::value, true);
    #endif
  #endif
  // clang-format on
}

// Check stringify of an default enum value to json
void JsonDefaultTest() {
  // load FlatBuffer schema (.fbs) from disk
  std::string schemafile;
  TEST_EQ(flatbuffers::LoadFile((test_data_path + "monster_test.fbs").c_str(),
                                false, &schemafile), true);
  // parse schema first, so we can use it to parse the data after
  flatbuffers::Parser parser;
  auto include_test_path =
      flatbuffers::ConCatPathFileName(test_data_path, "include_test");
  const char *include_directories[] = { test_data_path.c_str(),
                                        include_test_path.c_str(), nullptr };

  TEST_EQ(parser.Parse(schemafile.c_str(), include_directories), true);
  // create incomplete monster and store to json
  parser.opts.output_default_scalars_in_json = true;
  parser.opts.output_enum_identifiers = true;
  flatbuffers::FlatBufferBuilder builder;
  auto name = builder.CreateString("default_enum");
  MonsterBuilder color_monster(builder);
  color_monster.add_name(name);
  FinishMonsterBuffer(builder, color_monster.Finish());
  std::string jsongen;
  auto result = GenerateText(parser, builder.GetBufferPointer(), &jsongen);
  TEST_EQ(result, true);
  // default value of the "color" field is Blue
  TEST_EQ(std::string::npos != jsongen.find("color: \"Blue\""), true);
  // default value of the "testf" field is 3.14159
  TEST_EQ(std::string::npos != jsongen.find("testf: 3.14159"), true);
}

void JsonEnumsTest() {
  // load FlatBuffer schema (.fbs) from disk
  std::string schemafile;
  TEST_EQ(flatbuffers::LoadFile((test_data_path + "monster_test.fbs").c_str(),
                                false, &schemafile),
          true);
  // parse schema first, so we can use it to parse the data after
  flatbuffers::Parser parser;
  auto include_test_path =
      flatbuffers::ConCatPathFileName(test_data_path, "include_test");
  const char *include_directories[] = { test_data_path.c_str(),
                                        include_test_path.c_str(), nullptr };
  parser.opts.output_enum_identifiers = true;
  TEST_EQ(parser.Parse(schemafile.c_str(), include_directories), true);
  flatbuffers::FlatBufferBuilder builder;
  auto name = builder.CreateString("bitflag_enum");
  MonsterBuilder color_monster(builder);
  color_monster.add_name(name);
  color_monster.add_color(Color(Color_Blue | Color_Red));
  FinishMonsterBuffer(builder, color_monster.Finish());
  std::string jsongen;
  auto result = GenerateText(parser, builder.GetBufferPointer(), &jsongen);
  TEST_EQ(result, true);
  TEST_EQ(std::string::npos != jsongen.find("color: \"Red Blue\""), true);
}

#if defined(FLATBUFFERS_HAS_NEW_STRTOD) && (FLATBUFFERS_HAS_NEW_STRTOD > 0)
// The IEEE-754 quiet_NaN is not simple binary constant.
// All binary NaN bit strings have all the bits of the biased exponent field E
// set to 1. A quiet NaN bit string should be encoded with the first bit d[1]
// of the trailing significand field T being 1 (d[0] is implicit bit).
// It is assumed that endianness of floating-point is same as integer.
template<typename T, typename U, U qnan_base> bool is_quiet_nan_impl(T v) {
  static_assert(sizeof(T) == sizeof(U), "unexpected");
  U b = 0;
  std::memcpy(&b, &v, sizeof(T));
  return ((b & qnan_base) == qnan_base);
}
static bool is_quiet_nan(float v) {
  return is_quiet_nan_impl<float, uint32_t, 0x7FC00000u>(v);
}
static bool is_quiet_nan(double v) {
  return is_quiet_nan_impl<double, uint64_t, 0x7FF8000000000000ul>(v);
}

void TestMonsterExtraFloats() {
  TEST_EQ(is_quiet_nan(1.0), false);
  TEST_EQ(is_quiet_nan(infinityd), false);
  TEST_EQ(is_quiet_nan(-infinityf), false);
  TEST_EQ(is_quiet_nan(std::numeric_limits<float>::quiet_NaN()), true);
  TEST_EQ(is_quiet_nan(std::numeric_limits<double>::quiet_NaN()), true);

  using namespace flatbuffers;
  using namespace MyGame;
  // Load FlatBuffer schema (.fbs) from disk.
  std::string schemafile;
  TEST_EQ(LoadFile((test_data_path + "monster_extra.fbs").c_str(), false,
                   &schemafile),
          true);
  // Parse schema first, so we can use it to parse the data after.
  Parser parser;
  auto include_test_path = ConCatPathFileName(test_data_path, "include_test");
  const char *include_directories[] = { test_data_path.c_str(),
                                        include_test_path.c_str(), nullptr };
  TEST_EQ(parser.Parse(schemafile.c_str(), include_directories), true);
  // Create empty extra and store to json.
  parser.opts.output_default_scalars_in_json = true;
  parser.opts.output_enum_identifiers = true;
  FlatBufferBuilder builder;
  const auto def_root = MonsterExtraBuilder(builder).Finish();
  FinishMonsterExtraBuffer(builder, def_root);
  const auto def_obj = builder.GetBufferPointer();
  const auto def_extra = GetMonsterExtra(def_obj);
  TEST_NOTNULL(def_extra);
  TEST_EQ(is_quiet_nan(def_extra->f0()), true);
  TEST_EQ(is_quiet_nan(def_extra->f1()), true);
  TEST_EQ(def_extra->f2(), +infinityf);
  TEST_EQ(def_extra->f3(), -infinityf);
  TEST_EQ(is_quiet_nan(def_extra->d0()), true);
  TEST_EQ(is_quiet_nan(def_extra->d1()), true);
  TEST_EQ(def_extra->d2(), +infinityd);
  TEST_EQ(def_extra->d3(), -infinityd);
  std::string jsongen;
  auto result = GenerateText(parser, def_obj, &jsongen);
  TEST_EQ(result, true);
  // Check expected default values.
  TEST_EQ(std::string::npos != jsongen.find("f0: nan"), true);
  TEST_EQ(std::string::npos != jsongen.find("f1: nan"), true);
  TEST_EQ(std::string::npos != jsongen.find("f2: inf"), true);
  TEST_EQ(std::string::npos != jsongen.find("f3: -inf"), true);
  TEST_EQ(std::string::npos != jsongen.find("d0: nan"), true);
  TEST_EQ(std::string::npos != jsongen.find("d1: nan"), true);
  TEST_EQ(std::string::npos != jsongen.find("d2: inf"), true);
  TEST_EQ(std::string::npos != jsongen.find("d3: -inf"), true);
  // Parse 'mosterdata_extra.json'.
  const auto extra_base = test_data_path + "monsterdata_extra";
  jsongen = "";
  TEST_EQ(LoadFile((extra_base + ".json").c_str(), false, &jsongen), true);
  TEST_EQ(parser.Parse(jsongen.c_str()), true);
  const auto test_file = parser.builder_.GetBufferPointer();
  const auto test_size = parser.builder_.GetSize();
  Verifier verifier(test_file, test_size);
  TEST_ASSERT(VerifyMonsterExtraBuffer(verifier));
  const auto extra = GetMonsterExtra(test_file);
  TEST_NOTNULL(extra);
  TEST_EQ(is_quiet_nan(extra->f0()), true);
  TEST_EQ(is_quiet_nan(extra->f1()), true);
  TEST_EQ(extra->f2(), +infinityf);
  TEST_EQ(extra->f3(), -infinityf);
  TEST_EQ(is_quiet_nan(extra->d0()), true);
  TEST_EQ(extra->d1(), +infinityd);
  TEST_EQ(extra->d2(), -infinityd);
  TEST_EQ(is_quiet_nan(extra->d3()), true);
  TEST_NOTNULL(extra->fvec());
  TEST_EQ(extra->fvec()->size(), 4);
  TEST_EQ(extra->fvec()->Get(0), 1.0f);
  TEST_EQ(extra->fvec()->Get(1), -infinityf);
  TEST_EQ(extra->fvec()->Get(2), +infinityf);
  TEST_EQ(is_quiet_nan(extra->fvec()->Get(3)), true);
  TEST_NOTNULL(extra->dvec());
  TEST_EQ(extra->dvec()->size(), 4);
  TEST_EQ(extra->dvec()->Get(0), 2.0);
  TEST_EQ(extra->dvec()->Get(1), +infinityd);
  TEST_EQ(extra->dvec()->Get(2), -infinityd);
  TEST_EQ(is_quiet_nan(extra->dvec()->Get(3)), true);
}
#else
void TestMonsterExtraFloats() {}
#endif

// example of parsing text straight into a buffer, and generating
// text back from it:
void ParseAndGenerateTextTest(bool binary) {
  // load FlatBuffer schema (.fbs) and JSON from disk
  std::string schemafile;
  std::string jsonfile;
  TEST_EQ(flatbuffers::LoadFile(
              (test_data_path + "monster_test." + (binary ? "bfbs" : "fbs"))
                  .c_str(),
              binary, &schemafile),
          true);
  TEST_EQ(flatbuffers::LoadFile(
              (test_data_path + "monsterdata_test.golden").c_str(), false,
              &jsonfile),
          true);

  auto include_test_path =
    flatbuffers::ConCatPathFileName(test_data_path, "include_test");
  const char *include_directories[] = { test_data_path.c_str(),
                                        include_test_path.c_str(), nullptr };

  // parse schema first, so we can use it to parse the data after
  flatbuffers::Parser parser;
  if (binary) {
    flatbuffers::Verifier verifier(
        reinterpret_cast<const uint8_t *>(schemafile.c_str()),
        schemafile.size());
    TEST_EQ(reflection::VerifySchemaBuffer(verifier), true);
    //auto schema = reflection::GetSchema(schemafile.c_str());
    TEST_EQ(parser.Deserialize((const uint8_t *)schemafile.c_str(), schemafile.size()), true);
  } else {
    TEST_EQ(parser.Parse(schemafile.c_str(), include_directories), true);
  }
  TEST_EQ(parser.Parse(jsonfile.c_str(), include_directories), true);

  // here, parser.builder_ contains a binary buffer that is the parsed data.

  // First, verify it, just in case:
  flatbuffers::Verifier verifier(parser.builder_.GetBufferPointer(),
                                 parser.builder_.GetSize());
  TEST_EQ(VerifyMonsterBuffer(verifier), true);

  AccessFlatBufferTest(parser.builder_.GetBufferPointer(),
                       parser.builder_.GetSize(), false);

  // to ensure it is correct, we now generate text back from the binary,
  // and compare the two:
  std::string jsongen;
  auto result =
      GenerateText(parser, parser.builder_.GetBufferPointer(), &jsongen);
  TEST_EQ(result, true);
  TEST_EQ_STR(jsongen.c_str(), jsonfile.c_str());

  // We can also do the above using the convenient Registry that knows about
  // a set of file_identifiers mapped to schemas.
  flatbuffers::Registry registry;
  // Make sure schemas can find their includes.
  registry.AddIncludeDirectory(test_data_path.c_str());
  registry.AddIncludeDirectory(include_test_path.c_str());
  // Call this with many schemas if possible.
  registry.Register(MonsterIdentifier(),
                    (test_data_path + "monster_test.fbs").c_str());
  // Now we got this set up, we can parse by just specifying the identifier,
  // the correct schema will be loaded on the fly:
  auto buf = registry.TextToFlatBuffer(jsonfile.c_str(), MonsterIdentifier());
  // If this fails, check registry.lasterror_.
  TEST_NOTNULL(buf.data());
  // Test the buffer, to be sure:
  AccessFlatBufferTest(buf.data(), buf.size(), false);
  // We can use the registry to turn this back into text, in this case it
  // will get the file_identifier from the binary:
  std::string text;
  auto ok = registry.FlatBufferToText(buf.data(), buf.size(), &text);
  // If this fails, check registry.lasterror_.
  TEST_EQ(ok, true);
  TEST_EQ_STR(text.c_str(), jsonfile.c_str());

  // Generate text for UTF-8 strings without escapes.
  std::string jsonfile_utf8;
  TEST_EQ(flatbuffers::LoadFile((test_data_path + "unicode_test.json").c_str(),
                                false, &jsonfile_utf8),
          true);
  TEST_EQ(parser.Parse(jsonfile_utf8.c_str(), include_directories), true);
  // To ensure it is correct, generate utf-8 text back from the binary.
  std::string jsongen_utf8;
  // request natural printing for utf-8 strings
  parser.opts.natural_utf8 = true;
  parser.opts.strict_json = true;
  TEST_EQ(
      GenerateText(parser, parser.builder_.GetBufferPointer(), &jsongen_utf8),
      true);
  TEST_EQ_STR(jsongen_utf8.c_str(), jsonfile_utf8.c_str());
}

void ReflectionTest(uint8_t *flatbuf, size_t length) {
  // Load a binary schema.
  std::string bfbsfile;
  TEST_EQ(flatbuffers::LoadFile((test_data_path + "monster_test.bfbs").c_str(),
                                true, &bfbsfile),
          true);

  // Verify it, just in case:
  flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t *>(bfbsfile.c_str()), bfbsfile.length());
  TEST_EQ(reflection::VerifySchemaBuffer(verifier), true);

  // Make sure the schema is what we expect it to be.
  auto &schema = *reflection::GetSchema(bfbsfile.c_str());
  auto root_table = schema.root_table();
  TEST_EQ_STR(root_table->name()->c_str(), "MyGame.Example.Monster");
  auto fields = root_table->fields();
  auto hp_field_ptr = fields->LookupByKey("hp");
  TEST_NOTNULL(hp_field_ptr);
  auto &hp_field = *hp_field_ptr;
  TEST_EQ_STR(hp_field.name()->c_str(), "hp");
  TEST_EQ(hp_field.id(), 2);
  TEST_EQ(hp_field.type()->base_type(), reflection::Short);
  auto friendly_field_ptr = fields->LookupByKey("friendly");
  TEST_NOTNULL(friendly_field_ptr);
  TEST_NOTNULL(friendly_field_ptr->attributes());
  TEST_NOTNULL(friendly_field_ptr->attributes()->LookupByKey("priority"));

  // Make sure the table index is what we expect it to be.
  auto pos_field_ptr = fields->LookupByKey("pos");
  TEST_NOTNULL(pos_field_ptr);
  TEST_EQ(pos_field_ptr->type()->base_type(), reflection::Obj);
  auto pos_table_ptr = schema.objects()->Get(pos_field_ptr->type()->index());
  TEST_NOTNULL(pos_table_ptr);
  TEST_EQ_STR(pos_table_ptr->name()->c_str(), "MyGame.Example.Vec3");

  // Now use it to dynamically access a buffer.
  auto &root = *flatbuffers::GetAnyRoot(flatbuf);

  // Verify the buffer first using reflection based verification
  TEST_EQ(flatbuffers::Verify(schema, *schema.root_table(), flatbuf, length),
          true);

  auto hp = flatbuffers::GetFieldI<uint16_t>(root, hp_field);
  TEST_EQ(hp, 80);

  // Rather than needing to know the type, we can also get the value of
  // any field as an int64_t/double/string, regardless of what it actually is.
  auto hp_int64 = flatbuffers::GetAnyFieldI(root, hp_field);
  TEST_EQ(hp_int64, 80);
  auto hp_double = flatbuffers::GetAnyFieldF(root, hp_field);
  TEST_EQ(hp_double, 80.0);
  auto hp_string = flatbuffers::GetAnyFieldS(root, hp_field, &schema);
  TEST_EQ_STR(hp_string.c_str(), "80");

  // Get struct field through reflection
  auto pos_struct = flatbuffers::GetFieldStruct(root, *pos_field_ptr);
  TEST_NOTNULL(pos_struct);
  TEST_EQ(flatbuffers::GetAnyFieldF(*pos_struct,
                                    *pos_table_ptr->fields()->LookupByKey("z")),
          3.0f);

  auto test3_field = pos_table_ptr->fields()->LookupByKey("test3");
  auto test3_struct = flatbuffers::GetFieldStruct(*pos_struct, *test3_field);
  TEST_NOTNULL(test3_struct);
  auto test3_object = schema.objects()->Get(test3_field->type()->index());

  TEST_EQ(flatbuffers::GetAnyFieldF(*test3_struct,
                                    *test3_object->fields()->LookupByKey("a")),
          10);

  // We can also modify it.
  flatbuffers::SetField<uint16_t>(&root, hp_field, 200);
  hp = flatbuffers::GetFieldI<uint16_t>(root, hp_field);
  TEST_EQ(hp, 200);

  // We can also set fields generically:
  flatbuffers::SetAnyFieldI(&root, hp_field, 300);
  hp_int64 = flatbuffers::GetAnyFieldI(root, hp_field);
  TEST_EQ(hp_int64, 300);
  flatbuffers::SetAnyFieldF(&root, hp_field, 300.5);
  hp_int64 = flatbuffers::GetAnyFieldI(root, hp_field);
  TEST_EQ(hp_int64, 300);
  flatbuffers::SetAnyFieldS(&root, hp_field, "300");
  hp_int64 = flatbuffers::GetAnyFieldI(root, hp_field);
  TEST_EQ(hp_int64, 300);

  // Test buffer is valid after the modifications
  TEST_EQ(flatbuffers::Verify(schema, *schema.root_table(), flatbuf, length),
          true);

  // Reset it, for further tests.
  flatbuffers::SetField<uint16_t>(&root, hp_field, 80);

  // More advanced functionality: changing the size of items in-line!
  // First we put the FlatBuffer inside an std::vector.
  std::vector<uint8_t> resizingbuf(flatbuf, flatbuf + length);
  // Find the field we want to modify.
  auto &name_field = *fields->LookupByKey("name");
  // Get the root.
  // This time we wrap the result from GetAnyRoot in a smartpointer that
  // will keep rroot valid as resizingbuf resizes.
  auto rroot = flatbuffers::piv(
      flatbuffers::GetAnyRoot(flatbuffers::vector_data(resizingbuf)),
      resizingbuf);
  SetString(schema, "totally new string", GetFieldS(**rroot, name_field),
            &resizingbuf);
  // Here resizingbuf has changed, but rroot is still valid.
  TEST_EQ_STR(GetFieldS(**rroot, name_field)->c_str(), "totally new string");
  // Now lets extend a vector by 100 elements (10 -> 110).
  auto &inventory_field = *fields->LookupByKey("inventory");
  auto rinventory = flatbuffers::piv(
      flatbuffers::GetFieldV<uint8_t>(**rroot, inventory_field), resizingbuf);
  flatbuffers::ResizeVector<uint8_t>(schema, 110, 50, *rinventory,
                                     &resizingbuf);
  // rinventory still valid, so lets read from it.
  TEST_EQ(rinventory->Get(10), 50);

  // For reflection uses not covered already, there is a more powerful way:
  // we can simply generate whatever object we want to add/modify in a
  // FlatBuffer of its own, then add that to an existing FlatBuffer:
  // As an example, let's add a string to an array of strings.
  // First, find our field:
  auto &testarrayofstring_field = *fields->LookupByKey("testarrayofstring");
  // Find the vector value:
  auto rtestarrayofstring = flatbuffers::piv(
      flatbuffers::GetFieldV<flatbuffers::Offset<flatbuffers::String>>(
          **rroot, testarrayofstring_field),
      resizingbuf);
  // It's a vector of 2 strings, to which we add one more, initialized to
  // offset 0.
  flatbuffers::ResizeVector<flatbuffers::Offset<flatbuffers::String>>(
      schema, 3, 0, *rtestarrayofstring, &resizingbuf);
  // Here we just create a buffer that contans a single string, but this
  // could also be any complex set of tables and other values.
  flatbuffers::FlatBufferBuilder stringfbb;
  stringfbb.Finish(stringfbb.CreateString("hank"));
  // Add the contents of it to our existing FlatBuffer.
  // We do this last, so the pointer doesn't get invalidated (since it is
  // at the end of the buffer):
  auto string_ptr = flatbuffers::AddFlatBuffer(
      resizingbuf, stringfbb.GetBufferPointer(), stringfbb.GetSize());
  // Finally, set the new value in the vector.
  rtestarrayofstring->MutateOffset(2, string_ptr);
  TEST_EQ_STR(rtestarrayofstring->Get(0)->c_str(), "bob");
  TEST_EQ_STR(rtestarrayofstring->Get(2)->c_str(), "hank");
  // Test integrity of all resize operations above.
  flatbuffers::Verifier resize_verifier(
      reinterpret_cast<const uint8_t *>(flatbuffers::vector_data(resizingbuf)),
      resizingbuf.size());
  TEST_EQ(VerifyMonsterBuffer(resize_verifier), true);

  // Test buffer is valid using reflection as well
  TEST_EQ(flatbuffers::Verify(schema, *schema.root_table(),
                              flatbuffers::vector_data(resizingbuf),
                              resizingbuf.size()),
          true);

  // As an additional test, also set it on the name field.
  // Note: unlike the name change above, this just overwrites the offset,
  // rather than changing the string in-place.
  SetFieldT(*rroot, name_field, string_ptr);
  TEST_EQ_STR(GetFieldS(**rroot, name_field)->c_str(), "hank");

  // Using reflection, rather than mutating binary FlatBuffers, we can also copy
  // tables and other things out of other FlatBuffers into a FlatBufferBuilder,
  // either part or whole.
  flatbuffers::FlatBufferBuilder fbb;
  auto root_offset = flatbuffers::CopyTable(
      fbb, schema, *root_table, *flatbuffers::GetAnyRoot(flatbuf), true);
  fbb.Finish(root_offset, MonsterIdentifier());
  // Test that it was copied correctly:
  AccessFlatBufferTest(fbb.GetBufferPointer(), fbb.GetSize());

  // Test buffer is valid using reflection as well
  TEST_EQ(flatbuffers::Verify(schema, *schema.root_table(),
                              fbb.GetBufferPointer(), fbb.GetSize()),
          true);
}

void MiniReflectFlatBuffersTest(uint8_t *flatbuf) {
  auto s = flatbuffers::FlatBufferToString(flatbuf, Monster::MiniReflectTypeTable());
  TEST_EQ_STR(
      s.c_str(),
      "{ "
      "pos: { x: 1.0, y: 2.0, z: 3.0, test1: 0.0, test2: Red, test3: "
      "{ a: 10, b: 20 } }, "
      "hp: 80, "
      "name: \"MyMonster\", "
      "inventory: [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ], "
      "test_type: Monster, "
      "test: { name: \"Fred\" }, "
      "test4: [ { a: 10, b: 20 }, { a: 30, b: 40 } ], "
      "testarrayofstring: [ \"bob\", \"fred\", \"bob\", \"fred\" ], "
      "testarrayoftables: [ { hp: 1000, name: \"Barney\" }, { name: \"Fred\" "
      "}, "
      "{ name: \"Wilma\" } ], "
      // TODO(wvo): should really print this nested buffer correctly.
      "testnestedflatbuffer: [ 20, 0, 0, 0, 77, 79, 78, 83, 12, 0, 12, 0, 0, "
      "0, "
      "4, 0, 6, 0, 8, 0, 12, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 13, 0, 0, 0, 78, "
      "101, 115, 116, 101, 100, 77, 111, 110, 115, 116, 101, 114, 0, 0, 0 ], "
      "testarrayofstring2: [ \"jane\", \"mary\" ], "
      "testarrayofsortedstruct: [ { id: 1, distance: 10 }, "
      "{ id: 2, distance: 20 }, { id: 3, distance: 30 }, "
      "{ id: 4, distance: 40 } ], "
      "flex: [ 210, 4, 5, 2 ], "
      "test5: [ { a: 10, b: 20 }, { a: 30, b: 40 } ], "
      "vector_of_enums: [ Blue, Green ] "
      "}");

  Test test(16, 32);
  Vec3 vec(1,2,3, 1.5, Color_Red, test);
  flatbuffers::FlatBufferBuilder vec_builder;
  vec_builder.Finish(vec_builder.CreateStruct(vec));
  auto vec_buffer = vec_builder.Release();
  auto vec_str = flatbuffers::FlatBufferToString(vec_buffer.data(),
                                                 Vec3::MiniReflectTypeTable());
  TEST_EQ_STR(
      vec_str.c_str(),
      "{ x: 1.0, y: 2.0, z: 3.0, test1: 1.5, test2: Red, test3: { a: 16, b: 32 } }");
}

// Parse a .proto schema, output as .fbs
void ParseProtoTest() {
  // load the .proto and the golden file from disk
  std::string protofile;
  std::string goldenfile;
  std::string goldenunionfile;
  TEST_EQ(
      flatbuffers::LoadFile((test_data_path + "prototest/test.proto").c_str(),
                            false, &protofile),
      true);
  TEST_EQ(
      flatbuffers::LoadFile((test_data_path + "prototest/test.golden").c_str(),
                            false, &goldenfile),
      true);
  TEST_EQ(
      flatbuffers::LoadFile((test_data_path +
                            "prototest/test_union.golden").c_str(),
                            false, &goldenunionfile),
      true);

  flatbuffers::IDLOptions opts;
  opts.include_dependence_headers = false;
  opts.proto_mode = true;

  // Parse proto.
  flatbuffers::Parser parser(opts);
  auto protopath = test_data_path + "prototest/";
  const char *include_directories[] = { protopath.c_str(), nullptr };
  TEST_EQ(parser.Parse(protofile.c_str(), include_directories), true);

  // Generate fbs.
  auto fbs = flatbuffers::GenerateFBS(parser, "test");

  // Ensure generated file is parsable.
  flatbuffers::Parser parser2;
  TEST_EQ(parser2.Parse(fbs.c_str(), nullptr), true);
  TEST_EQ_STR(fbs.c_str(), goldenfile.c_str());

  // Parse proto with --oneof-union option.
  opts.proto_oneof_union = true;
  flatbuffers::Parser parser3(opts);
  TEST_EQ(parser3.Parse(protofile.c_str(), include_directories), true);

  // Generate fbs.
  auto fbs_union = flatbuffers::GenerateFBS(parser3, "test");

  // Ensure generated file is parsable.
  flatbuffers::Parser parser4;
  TEST_EQ(parser4.Parse(fbs_union.c_str(), nullptr), true);
  TEST_EQ_STR(fbs_union.c_str(), goldenunionfile.c_str());
}

template<typename T>
void CompareTableFieldValue(flatbuffers::Table *table,
                            flatbuffers::voffset_t voffset, T val) {
  T read = table->GetField(voffset, static_cast<T>(0));
  TEST_EQ(read, val);
}

// Low level stress/fuzz test: serialize/deserialize a variety of
// different kinds of data in different combinations
void FuzzTest1() {
  // Values we're testing against: chosen to ensure no bits get chopped
  // off anywhere, and also be different from eachother.
  const uint8_t bool_val = true;
  const int8_t char_val = -127;  // 0x81
  const uint8_t uchar_val = 0xFF;
  const int16_t short_val = -32222;  // 0x8222;
  const uint16_t ushort_val = 0xFEEE;
  const int32_t int_val = 0x83333333;
  const uint32_t uint_val = 0xFDDDDDDD;
  const int64_t long_val = 0x8444444444444444LL;
  const uint64_t ulong_val = 0xFCCCCCCCCCCCCCCCULL;
  const float float_val = 3.14159f;
  const double double_val = 3.14159265359;

  const int test_values_max = 11;
  const flatbuffers::voffset_t fields_per_object = 4;
  const int num_fuzz_objects = 10000;  // The higher, the more thorough :)

  flatbuffers::FlatBufferBuilder builder;

  lcg_reset();  // Keep it deterministic.

  flatbuffers::uoffset_t objects[num_fuzz_objects];

  // Generate num_fuzz_objects random objects each consisting of
  // fields_per_object fields, each of a random type.
  for (int i = 0; i < num_fuzz_objects; i++) {
    auto start = builder.StartTable();
    for (flatbuffers::voffset_t f = 0; f < fields_per_object; f++) {
      int choice = lcg_rand() % test_values_max;
      auto off = flatbuffers::FieldIndexToOffset(f);
      switch (choice) {
        case 0: builder.AddElement<uint8_t>(off, bool_val, 0); break;
        case 1: builder.AddElement<int8_t>(off, char_val, 0); break;
        case 2: builder.AddElement<uint8_t>(off, uchar_val, 0); break;
        case 3: builder.AddElement<int16_t>(off, short_val, 0); break;
        case 4: builder.AddElement<uint16_t>(off, ushort_val, 0); break;
        case 5: builder.AddElement<int32_t>(off, int_val, 0); break;
        case 6: builder.AddElement<uint32_t>(off, uint_val, 0); break;
        case 7: builder.AddElement<int64_t>(off, long_val, 0); break;
        case 8: builder.AddElement<uint64_t>(off, ulong_val, 0); break;
        case 9: builder.AddElement<float>(off, float_val, 0); break;
        case 10: builder.AddElement<double>(off, double_val, 0); break;
      }
    }
    objects[i] = builder.EndTable(start);
  }
  builder.PreAlign<flatbuffers::largest_scalar_t>(0);  // Align whole buffer.

  lcg_reset();  // Reset.

  uint8_t *eob = builder.GetCurrentBufferPointer() + builder.GetSize();

  // Test that all objects we generated are readable and return the
  // expected values. We generate random objects in the same order
  // so this is deterministic.
  for (int i = 0; i < num_fuzz_objects; i++) {
    auto table = reinterpret_cast<flatbuffers::Table *>(eob - objects[i]);
    for (flatbuffers::voffset_t f = 0; f < fields_per_object; f++) {
      int choice = lcg_rand() % test_values_max;
      flatbuffers::voffset_t off = flatbuffers::FieldIndexToOffset(f);
      switch (choice) {
        case 0: CompareTableFieldValue(table, off, bool_val); break;
        case 1: CompareTableFieldValue(table, off, char_val); break;
        case 2: CompareTableFieldValue(table, off, uchar_val); break;
        case 3: CompareTableFieldValue(table, off, short_val); break;
        case 4: CompareTableFieldValue(table, off, ushort_val); break;
        case 5: CompareTableFieldValue(table, off, int_val); break;
        case 6: CompareTableFieldValue(table, off, uint_val); break;
        case 7: CompareTableFieldValue(table, off, long_val); break;
        case 8: CompareTableFieldValue(table, off, ulong_val); break;
        case 9: CompareTableFieldValue(table, off, float_val); break;
        case 10: CompareTableFieldValue(table, off, double_val); break;
      }
    }
  }
}

// High level stress/fuzz test: generate a big schema and
// matching json data in random combinations, then parse both,
// generate json back from the binary, and compare with the original.
void FuzzTest2() {
  lcg_reset();  // Keep it deterministic.

  const int num_definitions = 30;
  const int num_struct_definitions = 5;  // Subset of num_definitions.
  const int fields_per_definition = 15;
  const int instances_per_definition = 5;
  const int deprecation_rate = 10;  // 1 in deprecation_rate fields will
                                    // be deprecated.

  std::string schema = "namespace test;\n\n";

  struct RndDef {
    std::string instances[instances_per_definition];

    // Since we're generating schema and corresponding data in tandem,
    // this convenience function adds strings to both at once.
    static void Add(RndDef (&definitions_l)[num_definitions],
                    std::string &schema_l, const int instances_per_definition_l,
                    const char *schema_add, const char *instance_add,
                    int definition) {
      schema_l += schema_add;
      for (int i = 0; i < instances_per_definition_l; i++)
        definitions_l[definition].instances[i] += instance_add;
    }
  };

  // clang-format off
  #define AddToSchemaAndInstances(schema_add, instance_add) \
    RndDef::Add(definitions, schema, instances_per_definition, \
                schema_add, instance_add, definition)

  #define Dummy() \
    RndDef::Add(definitions, schema, instances_per_definition, \
                "byte", "1", definition)
  // clang-format on

  RndDef definitions[num_definitions];

  // We are going to generate num_definitions, the first
  // num_struct_definitions will be structs, the rest tables. For each
  // generate random fields, some of which may be struct/table types
  // referring to previously generated structs/tables.
  // Simultanenously, we generate instances_per_definition JSON data
  // definitions, which will have identical structure to the schema
  // being generated. We generate multiple instances such that when creating
  // hierarchy, we get some variety by picking one randomly.
  for (int definition = 0; definition < num_definitions; definition++) {
    std::string definition_name = "D" + flatbuffers::NumToString(definition);

    bool is_struct = definition < num_struct_definitions;

    AddToSchemaAndInstances(
        ((is_struct ? "struct " : "table ") + definition_name + " {\n").c_str(),
        "{\n");

    for (int field = 0; field < fields_per_definition; field++) {
      const bool is_last_field = field == fields_per_definition - 1;

      // Deprecate 1 in deprecation_rate fields. Only table fields can be
      // deprecated.
      // Don't deprecate the last field to avoid dangling commas in JSON.
      const bool deprecated =
          !is_struct && !is_last_field && (lcg_rand() % deprecation_rate == 0);

      std::string field_name = "f" + flatbuffers::NumToString(field);
      AddToSchemaAndInstances(("  " + field_name + ":").c_str(),
                              deprecated ? "" : (field_name + ": ").c_str());
      // Pick random type:
      auto base_type = static_cast<flatbuffers::BaseType>(
          lcg_rand() % (flatbuffers::BASE_TYPE_UNION + 1));
      switch (base_type) {
        case flatbuffers::BASE_TYPE_STRING:
          if (is_struct) {
            Dummy();  // No strings in structs.
          } else {
            AddToSchemaAndInstances("string", deprecated ? "" : "\"hi\"");
          }
          break;
        case flatbuffers::BASE_TYPE_VECTOR:
          if (is_struct) {
            Dummy();  // No vectors in structs.
          } else {
            AddToSchemaAndInstances("[ubyte]",
                                    deprecated ? "" : "[\n0,\n1,\n255\n]");
          }
          break;
        case flatbuffers::BASE_TYPE_NONE:
        case flatbuffers::BASE_TYPE_UTYPE:
        case flatbuffers::BASE_TYPE_STRUCT:
        case flatbuffers::BASE_TYPE_UNION:
          if (definition) {
            // Pick a random previous definition and random data instance of
            // that definition.
            int defref = lcg_rand() % definition;
            int instance = lcg_rand() % instances_per_definition;
            AddToSchemaAndInstances(
                ("D" + flatbuffers::NumToString(defref)).c_str(),
                deprecated ? ""
                           : definitions[defref].instances[instance].c_str());
          } else {
            // If this is the first definition, we have no definition we can
            // refer to.
            Dummy();
          }
          break;
        case flatbuffers::BASE_TYPE_BOOL:
          AddToSchemaAndInstances(
              "bool", deprecated ? "" : (lcg_rand() % 2 ? "true" : "false"));
          break;
        case flatbuffers::BASE_TYPE_ARRAY:
          if (!is_struct) {
            AddToSchemaAndInstances(
                "ubyte",
                deprecated ? "" : "255");  // No fixed-length arrays in tables.
          } else {
            AddToSchemaAndInstances("[int:3]", deprecated ? "" : "[\n,\n,\n]");
          }
          break;
        default:
          // All the scalar types.
          schema += flatbuffers::kTypeNames[base_type];

          if (!deprecated) {
            // We want each instance to use its own random value.
            for (int inst = 0; inst < instances_per_definition; inst++)
              definitions[definition].instances[inst] +=
                  flatbuffers::IsFloat(base_type)
                      ? flatbuffers::NumToString<double>(lcg_rand() % 128)
                            .c_str()
                      : flatbuffers::NumToString<int>(lcg_rand() % 128).c_str();
          }
      }
      AddToSchemaAndInstances(deprecated ? "(deprecated);\n" : ";\n",
                              deprecated ? "" : is_last_field ? "\n" : ",\n");
    }
    AddToSchemaAndInstances("}\n\n", "}");
  }

  schema += "root_type D" + flatbuffers::NumToString(num_definitions - 1);
  schema += ";\n";

  flatbuffers::Parser parser;

  // Will not compare against the original if we don't write defaults
  parser.builder_.ForceDefaults(true);

  // Parse the schema, parse the generated data, then generate text back
  // from the binary and compare against the original.
  TEST_EQ(parser.Parse(schema.c_str()), true);

  const std::string &json =
      definitions[num_definitions - 1].instances[0] + "\n";

  TEST_EQ(parser.Parse(json.c_str()), true);

  std::string jsongen;
  parser.opts.indent_step = 0;
  auto result =
      GenerateText(parser, parser.builder_.GetBufferPointer(), &jsongen);
  TEST_EQ(result, true);

  if (jsongen != json) {
    // These strings are larger than a megabyte, so we show the bytes around
    // the first bytes that are different rather than the whole string.
    size_t len = std::min(json.length(), jsongen.length());
    for (size_t i = 0; i < len; i++) {
      if (json[i] != jsongen[i]) {
        i -= std::min(static_cast<size_t>(10), i);  // show some context;
        size_t end = std::min(len, i + 20);
        for (; i < end; i++)
          TEST_OUTPUT_LINE("at %d: found \"%c\", expected \"%c\"\n",
                           static_cast<int>(i), jsongen[i], json[i]);
        break;
      }
    }
    TEST_NOTNULL(NULL);
  }

  // clang-format off
  #ifdef FLATBUFFERS_TEST_VERBOSE
    TEST_OUTPUT_LINE("%dk schema tested with %dk of json\n",
                     static_cast<int>(schema.length() / 1024),
                     static_cast<int>(json.length() / 1024));
  #endif
  // clang-format on
}

// Test that parser errors are actually generated.
void TestError_(const char *src, const char *error_substr, bool strict_json,
                const char *file, int line, const char *func) {
  flatbuffers::IDLOptions opts;
  opts.strict_json = strict_json;
  flatbuffers::Parser parser(opts);
  if (parser.Parse(src)) {
    TestFail("true", "false",
             ("parser.Parse(\"" + std::string(src) + "\")").c_str(), file, line,
             func);
  } else if (!strstr(parser.error_.c_str(), error_substr)) {
    TestFail(parser.error_.c_str(), error_substr,
             ("parser.Parse(\"" + std::string(src) + "\")").c_str(), file, line,
             func);
  }
}

void TestError_(const char *src, const char *error_substr, const char *file,
                int line, const char *func) {
  TestError_(src, error_substr, false, file, line, func);
}

#ifdef _WIN32
#  define TestError(src, ...) \
    TestError_(src, __VA_ARGS__, __FILE__, __LINE__, __FUNCTION__)
#else
#  define TestError(src, ...) \
    TestError_(src, __VA_ARGS__, __FILE__, __LINE__, __PRETTY_FUNCTION__)
#endif

// Test that parsing errors occur as we'd expect.
// Also useful for coverage, making sure these paths are run.
void ErrorTest() {
  // In order they appear in idl_parser.cpp
  TestError("table X { Y:byte; } root_type X; { Y: 999 }", "does not fit");
  TestError("\"\0", "illegal");
  TestError("\"\\q", "escape code");
  TestError("table ///", "documentation");
  TestError("@", "illegal");
  TestError("table 1", "expecting");
  TestError("table X { Y:[[int]]; }", "nested vector");
  TestError("table X { Y:1; }", "illegal type");
  TestError("table X { Y:int; Y:int; }", "field already");
  TestError("table Y {} table X { Y:int; }", "same as table");
  TestError("struct X { Y:string; }", "only scalar");
  TestError("table X { Y:string = \"\"; }", "default values");
  TestError("struct X { a:uint = 42; }", "default values");
  TestError("enum Y:byte { Z = 1 } table X { y:Y; }", "not part of enum");
  TestError("struct X { Y:int (deprecated); }", "deprecate");
  TestError("union Z { X } table X { Y:Z; } root_type X; { Y: {}, A:1 }",
            "missing type field");
  TestError("union Z { X } table X { Y:Z; } root_type X; { Y_type: 99, Y: {",
            "type id");
  TestError("table X { Y:int; } root_type X; { Z:", "unknown field");
  TestError("table X { Y:int; } root_type X; { Y:", "string constant", true);
  TestError("table X { Y:int; } root_type X; { \"Y\":1, }", "string constant",
            true);
  TestError(
      "struct X { Y:int; Z:int; } table W { V:X; } root_type W; "
      "{ V:{ Y:1 } }",
      "wrong number");
  TestError("enum E:byte { A } table X { Y:E; } root_type X; { Y:U }",
            "unknown enum value");
  TestError("table X { Y:byte; } root_type X; { Y:; }", "starting");
  TestError("enum X:byte { Y } enum X {", "enum already");
  TestError("enum X:float {}", "underlying");
  TestError("enum X:byte { Y, Y }", "value already");
  TestError("enum X:byte { Y=2, Z=1 }", "ascending");
  TestError("table X { Y:int; } table X {", "datatype already");
  TestError("struct X (force_align: 7) { Y:int; }", "force_align");
  TestError("struct X {}", "size 0");
  TestError("{}", "no root");
  TestError("table X { Y:byte; } root_type X; { Y:1 } { Y:1 }", "end of file");
  TestError("table X { Y:byte; } root_type X; { Y:1 } table Y{ Z:int }",
            "end of file");
  TestError("root_type X;", "unknown root");
  TestError("struct X { Y:int; } root_type X;", "a table");
  TestError("union X { Y }", "referenced");
  TestError("union Z { X } struct X { Y:int; }", "only tables");
  TestError("table X { Y:[int]; YLength:int; }", "clash");
  TestError("table X { Y:byte; } root_type X; { Y:1, Y:2 }", "more than once");
  // float to integer conversion is forbidden
  TestError("table X { Y:int; } root_type X; { Y:1.0 }", "float");
  TestError("table X { Y:bool; } root_type X; { Y:1.0 }", "float");
  TestError("enum X:bool { Y = true }", "must be integral");
}

template<typename T>
T TestValue(const char *json, const char *type_name,
            const char *decls = nullptr) {
  flatbuffers::Parser parser;
  parser.builder_.ForceDefaults(true);  // return defaults
  auto check_default = json ? false : true;
  if (check_default) { parser.opts.output_default_scalars_in_json = true; }
  // Simple schema.
  std::string schema = std::string(decls ? decls : "") + "\n" +
                       "table X { Y:" + std::string(type_name) +
                       "; } root_type X;";
  auto schema_done = parser.Parse(schema.c_str());
  TEST_EQ_STR(parser.error_.c_str(), "");
  TEST_EQ(schema_done, true);

  auto done = parser.Parse(check_default ? "{}" : json);
  TEST_EQ_STR(parser.error_.c_str(), "");
  TEST_EQ(done, true);

  // Check with print.
  std::string print_back;
  parser.opts.indent_step = -1;
  TEST_EQ(GenerateText(parser, parser.builder_.GetBufferPointer(), &print_back),
          true);
  // restore value from its default
  if (check_default) { TEST_EQ(parser.Parse(print_back.c_str()), true); }

  auto root = flatbuffers::GetRoot<flatbuffers::Table>(
      parser.builder_.GetBufferPointer());
  return root->GetField<T>(flatbuffers::FieldIndexToOffset(0), 0);
}

bool FloatCompare(float a, float b) { return fabs(a - b) < 0.001; }

// Additional parser testing not covered elsewhere.
void ValueTest() {
  // Test scientific notation numbers.
  TEST_EQ(FloatCompare(TestValue<float>("{ Y:0.0314159e+2 }", "float"),
                       3.14159f),
          true);
  // number in string
  TEST_EQ(FloatCompare(TestValue<float>("{ Y:\"0.0314159e+2\" }", "float"),
                       3.14159f),
          true);

  // Test conversion functions.
  TEST_EQ(FloatCompare(TestValue<float>("{ Y:cos(rad(180)) }", "float"), -1),
          true);

  // int embedded to string
  TEST_EQ(TestValue<int>("{ Y:\"-876\" }", "int=-123"), -876);
  TEST_EQ(TestValue<int>("{ Y:\"876\" }", "int=-123"), 876);

  // Test negative hex constant.
  TEST_EQ(TestValue<int>("{ Y:-0x8ea0 }", "int=-0x8ea0"), -36512);
  TEST_EQ(TestValue<int>(nullptr, "int=-0x8ea0"), -36512);

  // positive hex constant
  TEST_EQ(TestValue<int>("{ Y:0x1abcdef }", "int=0x1"), 0x1abcdef);
  // with optional '+' sign
  TEST_EQ(TestValue<int>("{ Y:+0x1abcdef }", "int=+0x1"), 0x1abcdef);
  // hex in string
  TEST_EQ(TestValue<int>("{ Y:\"0x1abcdef\" }", "int=+0x1"), 0x1abcdef);

  // Make sure we do unsigned 64bit correctly.
  TEST_EQ(TestValue<uint64_t>("{ Y:12335089644688340133 }", "ulong"),
          12335089644688340133ULL);

  // bool in string
  TEST_EQ(TestValue<bool>("{ Y:\"false\" }", "bool=true"), false);
  TEST_EQ(TestValue<bool>("{ Y:\"true\" }", "bool=\"true\""), true);
  TEST_EQ(TestValue<bool>("{ Y:'false' }", "bool=true"), false);
  TEST_EQ(TestValue<bool>("{ Y:'true' }", "bool=\"true\""), true);

  // check comments before and after json object
  TEST_EQ(TestValue<int>("/*before*/ { Y:1 } /*after*/", "int"), 1);
  TEST_EQ(TestValue<int>("//before \n { Y:1 } //after", "int"), 1);

}

void NestedListTest() {
  flatbuffers::Parser parser1;
  TEST_EQ(parser1.Parse("struct Test { a:short; b:byte; } table T { F:[Test]; }"
                        "root_type T;"
                        "{ F:[ [10,20], [30,40]] }"),
          true);
}

void EnumStringsTest() {
  flatbuffers::Parser parser1;
  TEST_EQ(parser1.Parse("enum E:byte { A, B, C } table T { F:[E]; }"
                        "root_type T;"
                        "{ F:[ A, B, \"C\", \"A B C\" ] }"),
          true);
  flatbuffers::Parser parser2;
  TEST_EQ(parser2.Parse("enum E:byte { A, B, C } table T { F:[int]; }"
                        "root_type T;"
                        "{ F:[ \"E.C\", \"E.A E.B E.C\" ] }"),
          true);
  // unsigned bit_flags
  flatbuffers::Parser parser3;
  TEST_EQ(
      parser3.Parse("enum E:uint16 (bit_flags) { F0, F07=7, F08, F14=14, F15 }"
                    " table T { F: E = \"F15 F08\"; }"
                    "root_type T;"),
      true);
}

void EnumNamesTest() {
  TEST_EQ_STR("Red", EnumNameColor(Color_Red));
  TEST_EQ_STR("Green", EnumNameColor(Color_Green));
  TEST_EQ_STR("Blue", EnumNameColor(Color_Blue));
  // Check that Color to string don't crash while decode a mixture of Colors.
  // 1) Example::Color enum is enum with unfixed underlying type.
  // 2) Valid enum range: [0; 2^(ceil(log2(Color_ANY))) - 1].
  // Consequence: A value is out of this range will lead to UB (since C++17).
  // For details see C++17 standard or explanation on the SO:
  // stackoverflow.com/questions/18195312/what-happens-if-you-static-cast-invalid-value-to-enum-class
  TEST_EQ_STR("", EnumNameColor(static_cast<Color>(0)));
  TEST_EQ_STR("", EnumNameColor(static_cast<Color>(Color_ANY-1)));
  TEST_EQ_STR("", EnumNameColor(static_cast<Color>(Color_ANY+1)));
}

void EnumOutOfRangeTest() {
  TestError("enum X:byte { Y = 128 }", "enum value does not fit");
  TestError("enum X:byte { Y = -129 }", "enum value does not fit");
  TestError("enum X:byte { Y = 126, Z0, Z1 }", "enum value does not fit");
  TestError("enum X:ubyte { Y = -1 }", "enum value does not fit");
  TestError("enum X:ubyte { Y = 256 }", "enum value does not fit");
  TestError("enum X:ubyte { Y = 255, Z }", "enum value does not fit");
  // Unions begin with an implicit "NONE = 0".
  TestError("table Y{} union X { Y = -1 }",
            "enum values must be specified in ascending order");
  TestError("table Y{} union X { Y = 256 }", "enum value does not fit");
  TestError("table Y{} union X { Y = 255, Z:Y }", "enum value does not fit");
  TestError("enum X:int { Y = -2147483649 }", "enum value does not fit");
  TestError("enum X:int { Y = 2147483648 }", "enum value does not fit");
  TestError("enum X:uint { Y = -1 }", "enum value does not fit");
  TestError("enum X:uint { Y = 4294967297 }", "enum value does not fit");
  TestError("enum X:long { Y = 9223372036854775808 }", "does not fit");
  TestError("enum X:long { Y = 9223372036854775807, Z }", "enum value does not fit");
  TestError("enum X:ulong { Y = -1 }", "does not fit");
  TestError("enum X:ubyte (bit_flags) { Y=8 }", "bit flag out");
  TestError("enum X:byte (bit_flags) { Y=7 }", "must be unsigned"); // -128
  // bit_flgs out of range
  TestError("enum X:ubyte (bit_flags) { Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8 }", "out of range");
}

void EnumValueTest() {
  // json: "{ Y:0 }", schema: table X { Y : "E"}
  // 0 in enum (V=0) E then Y=0 is valid.
  TEST_EQ(TestValue<int>("{ Y:0 }", "E", "enum E:int { V }"), 0);
  TEST_EQ(TestValue<int>("{ Y:V }", "E", "enum E:int { V }"), 0);
  // A default value of Y is 0.
  TEST_EQ(TestValue<int>("{ }", "E", "enum E:int { V }"), 0);
  TEST_EQ(TestValue<int>("{ Y:5 }", "E=V", "enum E:int { V=5 }"), 5);
  // Generate json with defaults and check.
  TEST_EQ(TestValue<int>(nullptr, "E=V", "enum E:int { V=5 }"), 5);
  // 5 in enum
  TEST_EQ(TestValue<int>("{ Y:5 }", "E", "enum E:int { Z, V=5 }"), 5);
  TEST_EQ(TestValue<int>("{ Y:5 }", "E=V", "enum E:int { Z, V=5 }"), 5);
  // Generate json with defaults and check.
  TEST_EQ(TestValue<int>(nullptr, "E", "enum E:int { Z, V=5 }"), 0);
  TEST_EQ(TestValue<int>(nullptr, "E=V", "enum E:int { Z, V=5 }"), 5);
  // u84 test
  TEST_EQ(TestValue<uint64_t>(nullptr, "E=V",
                              "enum E:ulong { V = 13835058055282163712 }"),
          13835058055282163712ULL);
  TEST_EQ(TestValue<uint64_t>(nullptr, "E=V",
                              "enum E:ulong { V = 18446744073709551615 }"),
          18446744073709551615ULL);
  // Assign non-enum value to enum field. Is it right?
  TEST_EQ(TestValue<int>("{ Y:7 }", "E", "enum E:int { V = 0 }"), 7);
}

void IntegerOutOfRangeTest() {
  TestError("table T { F:byte; } root_type T; { F:128 }",
            "constant does not fit");
  TestError("table T { F:byte; } root_type T; { F:-129 }",
            "constant does not fit");
  TestError("table T { F:ubyte; } root_type T; { F:256 }",
            "constant does not fit");
  TestError("table T { F:ubyte; } root_type T; { F:-1 }",
            "constant does not fit");
  TestError("table T { F:short; } root_type T; { F:32768 }",
            "constant does not fit");
  TestError("table T { F:short; } root_type T; { F:-32769 }",
            "constant does not fit");
  TestError("table T { F:ushort; } root_type T; { F:65536 }",
            "constant does not fit");
  TestError("table T { F:ushort; } root_type T; { F:-1 }",
            "constant does not fit");
  TestError("table T { F:int; } root_type T; { F:2147483648 }",
            "constant does not fit");
  TestError("table T { F:int; } root_type T; { F:-2147483649 }",
            "constant does not fit");
  TestError("table T { F:uint; } root_type T; { F:4294967296 }",
            "constant does not fit");
  TestError("table T { F:uint; } root_type T; { F:-1 }",
            "constant does not fit");
  // Check fixed width aliases
  TestError("table X { Y:uint8; } root_type X; { Y: -1 }", "does not fit");
  TestError("table X { Y:uint8; } root_type X; { Y: 256 }", "does not fit");
  TestError("table X { Y:uint16; } root_type X; { Y: -1 }", "does not fit");
  TestError("table X { Y:uint16; } root_type X; { Y: 65536 }", "does not fit");
  TestError("table X { Y:uint32; } root_type X; { Y: -1 }", "");
  TestError("table X { Y:uint32; } root_type X; { Y: 4294967296 }",
            "does not fit");
  TestError("table X { Y:uint64; } root_type X; { Y: -1 }", "");
  TestError("table X { Y:uint64; } root_type X; { Y: -9223372036854775809 }",
            "does not fit");
  TestError("table X { Y:uint64; } root_type X; { Y: 18446744073709551616 }",
            "does not fit");

  TestError("table X { Y:int8; } root_type X; { Y: -129 }", "does not fit");
  TestError("table X { Y:int8; } root_type X; { Y: 128 }", "does not fit");
  TestError("table X { Y:int16; } root_type X; { Y: -32769 }", "does not fit");
  TestError("table X { Y:int16; } root_type X; { Y: 32768 }", "does not fit");
  TestError("table X { Y:int32; } root_type X; { Y: -2147483649 }", "");
  TestError("table X { Y:int32; } root_type X; { Y: 2147483648 }",
            "does not fit");
  TestError("table X { Y:int64; } root_type X; { Y: -9223372036854775809 }",
            "does not fit");
  TestError("table X { Y:int64; } root_type X; { Y: 9223372036854775808 }",
            "does not fit");
  // check out-of-int64 as int8
  TestError("table X { Y:int8; } root_type X; { Y: -9223372036854775809 }",
            "does not fit");
  TestError("table X { Y:int8; } root_type X; { Y: 9223372036854775808 }",
            "does not fit");

  // Check default values
  TestError("table X { Y:int64=-9223372036854775809; } root_type X; {}",
            "does not fit");
  TestError("table X { Y:int64= 9223372036854775808; } root_type X; {}",
            "does not fit");
  TestError("table X { Y:uint64; } root_type X; { Y: -1 }", "");
  TestError("table X { Y:uint64=-9223372036854775809; } root_type X; {}",
            "does not fit");
  TestError("table X { Y:uint64= 18446744073709551616; } root_type X; {}",
            "does not fit");
}

void IntegerBoundaryTest() {
  // Check numerical compatibility with non-C++ languages.
  // By the C++ standard, std::numerical_limits<int64_t>::min() == -9223372036854775807 (-2^63+1) or less*
  // The Flatbuffers grammar and most of the languages (C#, Java, Rust) expect
  // that minimum values are: -128, -32768,.., -9223372036854775808.
  // Since C++20, static_cast<int64>(0x8000000000000000ULL) is well-defined two's complement cast.
  // Therefore -9223372036854775808 should be valid negative value.
  TEST_EQ(flatbuffers::numeric_limits<int8_t>::min(), -128);
  TEST_EQ(flatbuffers::numeric_limits<int8_t>::max(), 127);
  TEST_EQ(flatbuffers::numeric_limits<int16_t>::min(), -32768);
  TEST_EQ(flatbuffers::numeric_limits<int16_t>::max(), 32767);
  TEST_EQ(flatbuffers::numeric_limits<int32_t>::min() + 1, -2147483647);
  TEST_EQ(flatbuffers::numeric_limits<int32_t>::max(), 2147483647ULL);
  TEST_EQ(flatbuffers::numeric_limits<int64_t>::min() + 1LL,
          -9223372036854775807LL);
  TEST_EQ(flatbuffers::numeric_limits<int64_t>::max(), 9223372036854775807ULL);
  TEST_EQ(flatbuffers::numeric_limits<uint8_t>::max(), 255);
  TEST_EQ(flatbuffers::numeric_limits<uint16_t>::max(), 65535);
  TEST_EQ(flatbuffers::numeric_limits<uint32_t>::max(), 4294967295ULL);
  TEST_EQ(flatbuffers::numeric_limits<uint64_t>::max(),
          18446744073709551615ULL);

  TEST_EQ(TestValue<int8_t>("{ Y:127 }", "byte"), 127);
  TEST_EQ(TestValue<int8_t>("{ Y:-128 }", "byte"), -128);
  TEST_EQ(TestValue<uint8_t>("{ Y:255 }", "ubyte"), 255);
  TEST_EQ(TestValue<uint8_t>("{ Y:0 }", "ubyte"), 0);
  TEST_EQ(TestValue<int16_t>("{ Y:32767 }", "short"), 32767);
  TEST_EQ(TestValue<int16_t>("{ Y:-32768 }", "short"), -32768);
  TEST_EQ(TestValue<uint16_t>("{ Y:65535 }", "ushort"), 65535);
  TEST_EQ(TestValue<uint16_t>("{ Y:0 }", "ushort"), 0);
  TEST_EQ(TestValue<int32_t>("{ Y:2147483647 }", "int"), 2147483647);
  TEST_EQ(TestValue<int32_t>("{ Y:-2147483648 }", "int") + 1, -2147483647);
  TEST_EQ(TestValue<uint32_t>("{ Y:4294967295 }", "uint"), 4294967295);
  TEST_EQ(TestValue<uint32_t>("{ Y:0 }", "uint"), 0);
  TEST_EQ(TestValue<int64_t>("{ Y:9223372036854775807 }", "long"),
          9223372036854775807LL);
  TEST_EQ(TestValue<int64_t>("{ Y:-9223372036854775808 }", "long") + 1LL,
          -9223372036854775807LL);
  TEST_EQ(TestValue<uint64_t>("{ Y:18446744073709551615 }", "ulong"),
          18446744073709551615ULL);
  TEST_EQ(TestValue<uint64_t>("{ Y:0 }", "ulong"), 0);
  TEST_EQ(TestValue<uint64_t>("{ Y: 18446744073709551615 }", "uint64"),
          18446744073709551615ULL);
  // check that the default works
  TEST_EQ(TestValue<uint64_t>(nullptr, "uint64 = 18446744073709551615"),
          18446744073709551615ULL);
}

void ValidFloatTest() {
  // check rounding to infinity
  TEST_EQ(TestValue<float>("{ Y:+3.4029e+38 }", "float"), +infinityf);
  TEST_EQ(TestValue<float>("{ Y:-3.4029e+38 }", "float"), -infinityf);
  TEST_EQ(TestValue<double>("{ Y:+1.7977e+308 }", "double"), +infinityd);
  TEST_EQ(TestValue<double>("{ Y:-1.7977e+308 }", "double"), -infinityd);

  TEST_EQ(
      FloatCompare(TestValue<float>("{ Y:0.0314159e+2 }", "float"), 3.14159f),
      true);
  // float in string
  TEST_EQ(FloatCompare(TestValue<float>("{ Y:\" 0.0314159e+2  \" }", "float"),
                       3.14159f),
          true);

  TEST_EQ(TestValue<float>("{ Y:1 }", "float"), 1.0f);
  TEST_EQ(TestValue<float>("{ Y:1.0 }", "float"), 1.0f);
  TEST_EQ(TestValue<float>("{ Y:1. }", "float"), 1.0f);
  TEST_EQ(TestValue<float>("{ Y:+1. }", "float"), 1.0f);
  TEST_EQ(TestValue<float>("{ Y:-1. }", "float"), -1.0f);
  TEST_EQ(TestValue<float>("{ Y:1.e0 }", "float"), 1.0f);
  TEST_EQ(TestValue<float>("{ Y:1.e+0 }", "float"), 1.0f);
  TEST_EQ(TestValue<float>("{ Y:1.e-0 }", "float"), 1.0f);
  TEST_EQ(TestValue<float>("{ Y:0.125 }", "float"), 0.125f);
  TEST_EQ(TestValue<float>("{ Y:.125 }", "float"), 0.125f);
  TEST_EQ(TestValue<float>("{ Y:-.125 }", "float"), -0.125f);
  TEST_EQ(TestValue<float>("{ Y:+.125 }", "float"), +0.125f);
  TEST_EQ(TestValue<float>("{ Y:5 }", "float"), 5.0f);
  TEST_EQ(TestValue<float>("{ Y:\"5\" }", "float"), 5.0f);

  #if defined(FLATBUFFERS_HAS_NEW_STRTOD) && (FLATBUFFERS_HAS_NEW_STRTOD > 0)
  // Old MSVC versions may have problem with this check.
  // https://www.exploringbinary.com/visual-c-plus-plus-strtod-still-broken/
  TEST_EQ(TestValue<double>("{ Y:6.9294956446009195e15 }", "double"),
    6929495644600920.0);
  // check nan's
  TEST_EQ(std::isnan(TestValue<double>("{ Y:nan }", "double")), true);
  TEST_EQ(std::isnan(TestValue<float>("{ Y:nan }", "float")), true);
  TEST_EQ(std::isnan(TestValue<float>("{ Y:\"nan\" }", "float")), true);
  TEST_EQ(std::isnan(TestValue<float>("{ Y:+nan }", "float")), true);
  TEST_EQ(std::isnan(TestValue<float>("{ Y:-nan }", "float")), true);
  TEST_EQ(std::isnan(TestValue<float>(nullptr, "float=nan")), true);
  TEST_EQ(std::isnan(TestValue<float>(nullptr, "float=-nan")), true);
  // check inf
  TEST_EQ(TestValue<float>("{ Y:inf }", "float"), infinityf);
  TEST_EQ(TestValue<float>("{ Y:\"inf\" }", "float"), infinityf);
  TEST_EQ(TestValue<float>("{ Y:+inf }", "float"), infinityf);
  TEST_EQ(TestValue<float>("{ Y:-inf }", "float"), -infinityf);
  TEST_EQ(TestValue<float>(nullptr, "float=inf"), infinityf);
  TEST_EQ(TestValue<float>(nullptr, "float=-inf"), -infinityf);
  TestValue<double>(
      "{ Y : [0.2, .2, 1.0, -1.0, -2., 2., 1e0, -1e0, 1.0e0, -1.0e0, -3.e2, "
      "3.0e2] }",
      "[double]");
  TestValue<float>(
      "{ Y : [0.2, .2, 1.0, -1.0, -2., 2., 1e0, -1e0, 1.0e0, -1.0e0, -3.e2, "
      "3.0e2] }",
      "[float]");

  // Test binary format of float point.
  // https://en.cppreference.com/w/cpp/language/floating_literal
  // 0x11.12p-1 = (1*16^1 + 2*16^0 + 3*16^-1 + 4*16^-2) * 2^-1 =
  TEST_EQ(TestValue<double>("{ Y:0x12.34p-1 }", "double"), 9.1015625);
  // hex fraction 1.2 (decimal 1.125) scaled by 2^3, that is 9.0
  TEST_EQ(TestValue<float>("{ Y:-0x0.2p0 }", "float"), -0.125f);
  TEST_EQ(TestValue<float>("{ Y:-0x.2p1 }", "float"), -0.25f);
  TEST_EQ(TestValue<float>("{ Y:0x1.2p3 }", "float"), 9.0f);
  TEST_EQ(TestValue<float>("{ Y:0x10.1p0 }", "float"), 16.0625f);
  TEST_EQ(TestValue<double>("{ Y:0x1.2p3 }", "double"), 9.0);
  TEST_EQ(TestValue<double>("{ Y:0x10.1p0 }", "double"), 16.0625);
  TEST_EQ(TestValue<double>("{ Y:0xC.68p+2 }", "double"), 49.625);
  TestValue<double>("{ Y : [0x20.4ep1, +0x20.4ep1, -0x20.4ep1] }", "[double]");
  TestValue<float>("{ Y : [0x20.4ep1, +0x20.4ep1, -0x20.4ep1] }", "[float]");

#else   // FLATBUFFERS_HAS_NEW_STRTOD
  TEST_OUTPUT_LINE("FLATBUFFERS_HAS_NEW_STRTOD tests skipped");
#endif  // !FLATBUFFERS_HAS_NEW_STRTOD
}

void InvalidFloatTest() {
  auto invalid_msg = "invalid number";
  auto comma_msg = "expecting: ,";
  TestError("table T { F:float; } root_type T; { F:1,0 }", "");
  TestError("table T { F:float; } root_type T; { F:. }", "");
  TestError("table T { F:float; } root_type T; { F:- }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:+ }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:-. }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:+. }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:.e }", "");
  TestError("table T { F:float; } root_type T; { F:-e }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:+e }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:-.e }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:+.e }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:-e1 }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:+e1 }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:1.0e+ }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:1.0e- }", invalid_msg);
  // exponent pP is mandatory for hex-float
  TestError("table T { F:float; } root_type T; { F:0x0 }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:-0x. }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:0x. }", invalid_msg);
  // eE not exponent in hex-float!
  TestError("table T { F:float; } root_type T; { F:0x0.0e+ }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:0x0.0e- }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:0x0.0p }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:0x0.0p+ }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:0x0.0p- }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:0x0.0pa1 }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:0x0.0e+ }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:0x0.0e- }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:0x0.0e+0 }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:0x0.0e-0 }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:0x0.0ep+ }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:0x0.0ep- }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:1.2.3 }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:1.2.e3 }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:1.2e.3 }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:1.2e0.3 }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:1.2e3. }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:1.2e3.0 }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:+-1.0 }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:1.0e+-1 }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:\"1.0e+-1\" }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:1.e0e }", comma_msg);
  TestError("table T { F:float; } root_type T; { F:0x1.p0e }", comma_msg);
  TestError("table T { F:float; } root_type T; { F:\" 0x10 \" }", invalid_msg);
  // floats in string
  TestError("table T { F:float; } root_type T; { F:\"1,2.\" }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:\"1.2e3.\" }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:\"0x1.p0e\" }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:\"0x1.0\" }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:\" 0x1.0\" }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:\"+ 0\" }", invalid_msg);
  // disable escapes for "number-in-string"
  TestError("table T { F:float; } root_type T; { F:\"\\f1.2e3.\" }", "invalid");
  TestError("table T { F:float; } root_type T; { F:\"\\t1.2e3.\" }", "invalid");
  TestError("table T { F:float; } root_type T; { F:\"\\n1.2e3.\" }", "invalid");
  TestError("table T { F:float; } root_type T; { F:\"\\r1.2e3.\" }", "invalid");
  TestError("table T { F:float; } root_type T; { F:\"4\\x005\" }", "invalid");
  TestError("table T { F:float; } root_type T; { F:\"\'12\'\" }", invalid_msg);
  // null is not a number constant!
  TestError("table T { F:float; } root_type T; { F:\"null\" }", invalid_msg);
  TestError("table T { F:float; } root_type T; { F:null }", invalid_msg);
}

void GenerateTableTextTest() {
  std::string schemafile;
  std::string jsonfile;
  bool ok =
      flatbuffers::LoadFile((test_data_path + "monster_test.fbs").c_str(),
                            false, &schemafile) &&
      flatbuffers::LoadFile((test_data_path + "monsterdata_test.json").c_str(),
                            false, &jsonfile);
  TEST_EQ(ok, true);
  auto include_test_path =
      flatbuffers::ConCatPathFileName(test_data_path, "include_test");
  const char *include_directories[] = {test_data_path.c_str(),
                                       include_test_path.c_str(), nullptr};
  flatbuffers::IDLOptions opt;
  opt.indent_step = -1;
  flatbuffers::Parser parser(opt);
  ok = parser.Parse(schemafile.c_str(), include_directories) &&
       parser.Parse(jsonfile.c_str(), include_directories);
  TEST_EQ(ok, true);
  // Test root table
  const Monster *monster = GetMonster(parser.builder_.GetBufferPointer());
  std::string jsongen;
  auto result = GenerateTextFromTable(parser, monster, "MyGame.Example.Monster",
                                      &jsongen);
  TEST_EQ(result, true);
  // Test sub table
  const Vec3 *pos = monster->pos();
  jsongen.clear();
  result = GenerateTextFromTable(parser, pos, "MyGame.Example.Vec3", &jsongen);
  TEST_EQ(result, true);
  TEST_EQ_STR(
      jsongen.c_str(),
      "{x: 1.0,y: 2.0,z: 3.0,test1: 3.0,test2: \"Green\",test3: {a: 5,b: 6}}");
  const Test &test3 = pos->test3();
  jsongen.clear();
  result =
      GenerateTextFromTable(parser, &test3, "MyGame.Example.Test", &jsongen);
  TEST_EQ(result, true);
  TEST_EQ_STR(jsongen.c_str(), "{a: 5,b: 6}");
  const Test *test4 = monster->test4()->Get(0);
  jsongen.clear();
  result =
      GenerateTextFromTable(parser, test4, "MyGame.Example.Test", &jsongen);
  TEST_EQ(result, true);
  TEST_EQ_STR(jsongen.c_str(), "{a: 10,b: 20}");
}

template<typename T>
void NumericUtilsTestInteger(const char *lower, const char *upper) {
  T x;
  TEST_EQ(flatbuffers::StringToNumber("1q", &x), false);
  TEST_EQ(x, 0);
  TEST_EQ(flatbuffers::StringToNumber(upper, &x), false);
  TEST_EQ(x, flatbuffers::numeric_limits<T>::max());
  TEST_EQ(flatbuffers::StringToNumber(lower, &x), false);
  auto expval = flatbuffers::is_unsigned<T>::value
                    ? flatbuffers::numeric_limits<T>::max()
                    : flatbuffers::numeric_limits<T>::lowest();
  TEST_EQ(x, expval);
}

template<typename T>
void NumericUtilsTestFloat(const char *lower, const char *upper) {
  T f;
  TEST_EQ(flatbuffers::StringToNumber("", &f), false);
  TEST_EQ(flatbuffers::StringToNumber("1q", &f), false);
  TEST_EQ(f, 0);
  TEST_EQ(flatbuffers::StringToNumber(upper, &f), true);
  TEST_EQ(f, +flatbuffers::numeric_limits<T>::infinity());
  TEST_EQ(flatbuffers::StringToNumber(lower, &f), true);
  TEST_EQ(f, -flatbuffers::numeric_limits<T>::infinity());
}

void NumericUtilsTest() {
  NumericUtilsTestInteger<uint64_t>("-1", "18446744073709551616");
  NumericUtilsTestInteger<uint8_t>("-1", "256");
  NumericUtilsTestInteger<int64_t>("-9223372036854775809",
                                   "9223372036854775808");
  NumericUtilsTestInteger<int8_t>("-129", "128");
  NumericUtilsTestFloat<float>("-3.4029e+38", "+3.4029e+38");
  NumericUtilsTestFloat<float>("-1.7977e+308", "+1.7977e+308");
}

void IsAsciiUtilsTest() {
  char c = -128;
  for (int cnt = 0; cnt < 256; cnt++) {
    auto alpha = (('a' <= c) && (c <= 'z')) || (('A' <= c) && (c <= 'Z'));
    auto dec = (('0' <= c) && (c <= '9'));
    auto hex = (('a' <= c) && (c <= 'f')) || (('A' <= c) && (c <= 'F'));
    TEST_EQ(flatbuffers::is_alpha(c), alpha);
    TEST_EQ(flatbuffers::is_alnum(c), alpha || dec);
    TEST_EQ(flatbuffers::is_digit(c), dec);
    TEST_EQ(flatbuffers::is_xdigit(c), dec || hex);
    c += 1;
  }
}

void UnicodeTest() {
  flatbuffers::Parser parser;
  // Without setting allow_non_utf8 = true, we treat \x sequences as byte
  // sequences which are then validated as UTF-8.
  TEST_EQ(parser.Parse("table T { F:string; }"
                       "root_type T;"
                       "{ F:\"\\u20AC\\u00A2\\u30E6\\u30FC\\u30B6\\u30FC"
                       "\\u5225\\u30B5\\u30A4\\u30C8\\xE2\\x82\\xAC\\u0080\\uD8"
                       "3D\\uDE0E\" }"),
          true);
  std::string jsongen;
  parser.opts.indent_step = -1;
  auto result =
      GenerateText(parser, parser.builder_.GetBufferPointer(), &jsongen);
  TEST_EQ(result, true);
  TEST_EQ_STR(jsongen.c_str(),
              "{F: \"\\u20AC\\u00A2\\u30E6\\u30FC\\u30B6\\u30FC"
              "\\u5225\\u30B5\\u30A4\\u30C8\\u20AC\\u0080\\uD83D\\uDE0E\"}");
}

void UnicodeTestAllowNonUTF8() {
  flatbuffers::Parser parser;
  parser.opts.allow_non_utf8 = true;
  TEST_EQ(
      parser.Parse(
          "table T { F:string; }"
          "root_type T;"
          "{ F:\"\\u20AC\\u00A2\\u30E6\\u30FC\\u30B6\\u30FC"
          "\\u5225\\u30B5\\u30A4\\u30C8\\x01\\x80\\u0080\\uD83D\\uDE0E\" }"),
      true);
  std::string jsongen;
  parser.opts.indent_step = -1;
  auto result =
      GenerateText(parser, parser.builder_.GetBufferPointer(), &jsongen);
  TEST_EQ(result, true);
  TEST_EQ_STR(
      jsongen.c_str(),
      "{F: \"\\u20AC\\u00A2\\u30E6\\u30FC\\u30B6\\u30FC"
      "\\u5225\\u30B5\\u30A4\\u30C8\\u0001\\x80\\u0080\\uD83D\\uDE0E\"}");
}

void UnicodeTestGenerateTextFailsOnNonUTF8() {
  flatbuffers::Parser parser;
  // Allow non-UTF-8 initially to model what happens when we load a binary
  // flatbuffer from disk which contains non-UTF-8 strings.
  parser.opts.allow_non_utf8 = true;
  TEST_EQ(
      parser.Parse(
          "table T { F:string; }"
          "root_type T;"
          "{ F:\"\\u20AC\\u00A2\\u30E6\\u30FC\\u30B6\\u30FC"
          "\\u5225\\u30B5\\u30A4\\u30C8\\x01\\x80\\u0080\\uD83D\\uDE0E\" }"),
      true);
  std::string jsongen;
  parser.opts.indent_step = -1;
  // Now, disallow non-UTF-8 (the default behavior) so GenerateText indicates
  // failure.
  parser.opts.allow_non_utf8 = false;
  auto result =
      GenerateText(parser, parser.builder_.GetBufferPointer(), &jsongen);
  TEST_EQ(result, false);
}

void UnicodeSurrogatesTest() {
  flatbuffers::Parser parser;

  TEST_EQ(parser.Parse("table T { F:string (id: 0); }"
                       "root_type T;"
                       "{ F:\"\\uD83D\\uDCA9\"}"),
          true);
  auto root = flatbuffers::GetRoot<flatbuffers::Table>(
      parser.builder_.GetBufferPointer());
  auto string = root->GetPointer<flatbuffers::String *>(
      flatbuffers::FieldIndexToOffset(0));
  TEST_EQ_STR(string->c_str(), "\xF0\x9F\x92\xA9");
}

void UnicodeInvalidSurrogatesTest() {
  TestError(
      "table T { F:string; }"
      "root_type T;"
      "{ F:\"\\uD800\"}",
      "unpaired high surrogate");
  TestError(
      "table T { F:string; }"
      "root_type T;"
      "{ F:\"\\uD800abcd\"}",
      "unpaired high surrogate");
  TestError(
      "table T { F:string; }"
      "root_type T;"
      "{ F:\"\\uD800\\n\"}",
      "unpaired high surrogate");
  TestError(
      "table T { F:string; }"
      "root_type T;"
      "{ F:\"\\uD800\\uD800\"}",
      "multiple high surrogates");
  TestError(
      "table T { F:string; }"
      "root_type T;"
      "{ F:\"\\uDC00\"}",
      "unpaired low surrogate");
}

void InvalidUTF8Test() {
  // "1 byte" pattern, under min length of 2 bytes
  TestError(
      "table T { F:string; }"
      "root_type T;"
      "{ F:\"\x80\"}",
      "illegal UTF-8 sequence");
  // 2 byte pattern, string too short
  TestError(
      "table T { F:string; }"
      "root_type T;"
      "{ F:\"\xDF\"}",
      "illegal UTF-8 sequence");
  // 3 byte pattern, string too short
  TestError(
      "table T { F:string; }"
      "root_type T;"
      "{ F:\"\xEF\xBF\"}",
      "illegal UTF-8 sequence");
  // 4 byte pattern, string too short
  TestError(
      "table T { F:string; }"
      "root_type T;"
      "{ F:\"\xF7\xBF\xBF\"}",
      "illegal UTF-8 sequence");
  // "5 byte" pattern, string too short
  TestError(
      "table T { F:string; }"
      "root_type T;"
      "{ F:\"\xFB\xBF\xBF\xBF\"}",
      "illegal UTF-8 sequence");
  // "6 byte" pattern, string too short
  TestError(
      "table T { F:string; }"
      "root_type T;"
      "{ F:\"\xFD\xBF\xBF\xBF\xBF\"}",
      "illegal UTF-8 sequence");
  // "7 byte" pattern, string too short
  TestError(
      "table T { F:string; }"
      "root_type T;"
      "{ F:\"\xFE\xBF\xBF\xBF\xBF\xBF\"}",
      "illegal UTF-8 sequence");
  // "5 byte" pattern, over max length of 4 bytes
  TestError(
      "table T { F:string; }"
      "root_type T;"
      "{ F:\"\xFB\xBF\xBF\xBF\xBF\"}",
      "illegal UTF-8 sequence");
  // "6 byte" pattern, over max length of 4 bytes
  TestError(
      "table T { F:string; }"
      "root_type T;"
      "{ F:\"\xFD\xBF\xBF\xBF\xBF\xBF\"}",
      "illegal UTF-8 sequence");
  // "7 byte" pattern, over max length of 4 bytes
  TestError(
      "table T { F:string; }"
      "root_type T;"
      "{ F:\"\xFE\xBF\xBF\xBF\xBF\xBF\xBF\"}",
      "illegal UTF-8 sequence");

  // Three invalid encodings for U+000A (\n, aka NEWLINE)
  TestError(
      "table T { F:string; }"
      "root_type T;"
      "{ F:\"\xC0\x8A\"}",
      "illegal UTF-8 sequence");
  TestError(
      "table T { F:string; }"
      "root_type T;"
      "{ F:\"\xE0\x80\x8A\"}",
      "illegal UTF-8 sequence");
  TestError(
      "table T { F:string; }"
      "root_type T;"
      "{ F:\"\xF0\x80\x80\x8A\"}",
      "illegal UTF-8 sequence");

  // Two invalid encodings for U+00A9 (COPYRIGHT SYMBOL)
  TestError(
      "table T { F:string; }"
      "root_type T;"
      "{ F:\"\xE0\x81\xA9\"}",
      "illegal UTF-8 sequence");
  TestError(
      "table T { F:string; }"
      "root_type T;"
      "{ F:\"\xF0\x80\x81\xA9\"}",
      "illegal UTF-8 sequence");

  // Invalid encoding for U+20AC (EURO SYMBOL)
  TestError(
      "table T { F:string; }"
      "root_type T;"
      "{ F:\"\xF0\x82\x82\xAC\"}",
      "illegal UTF-8 sequence");

  // UTF-16 surrogate values between U+D800 and U+DFFF cannot be encoded in
  // UTF-8
  TestError(
      "table T { F:string; }"
      "root_type T;"
      // U+10400 "encoded" as U+D801 U+DC00
      "{ F:\"\xED\xA0\x81\xED\xB0\x80\"}",
      "illegal UTF-8 sequence");

  // Check independence of identifier from locale.
  std::string locale_ident;
  locale_ident += "table T { F";
  locale_ident += static_cast<char>(-32); // unsigned 0xE0
  locale_ident += " :string; }";
  locale_ident += "root_type T;";
  locale_ident += "{}";
  TestError(locale_ident.c_str(), "");
}

void UnknownFieldsTest() {
  flatbuffers::IDLOptions opts;
  opts.skip_unexpected_fields_in_json = true;
  flatbuffers::Parser parser(opts);

  TEST_EQ(parser.Parse("table T { str:string; i:int;}"
                       "root_type T;"
                       "{ str:\"test\","
                       "unknown_string:\"test\","
                       "\"unknown_string\":\"test\","
                       "unknown_int:10,"
                       "unknown_float:1.0,"
                       "unknown_array: [ 1, 2, 3, 4],"
                       "unknown_object: { i: 10 },"
                       "\"unknown_object\": { \"i\": 10 },"
                       "i:10}"),
          true);

  std::string jsongen;
  parser.opts.indent_step = -1;
  auto result =
      GenerateText(parser, parser.builder_.GetBufferPointer(), &jsongen);
  TEST_EQ(result, true);
  TEST_EQ_STR(jsongen.c_str(), "{str: \"test\",i: 10}");
}

void ParseUnionTest() {
  // Unions must be parseable with the type field following the object.
  flatbuffers::Parser parser;
  TEST_EQ(parser.Parse("table T { A:int; }"
                       "union U { T }"
                       "table V { X:U; }"
                       "root_type V;"
                       "{ X:{ A:1 }, X_type: T }"),
          true);
  // Unions must be parsable with prefixed namespace.
  flatbuffers::Parser parser2;
  TEST_EQ(parser2.Parse("namespace N; table A {} namespace; union U { N.A }"
                        "table B { e:U; } root_type B;"
                        "{ e_type: N_A, e: {} }"),
          true);
}

void InvalidNestedFlatbufferTest() {
  // First, load and parse FlatBuffer schema (.fbs)
  std::string schemafile;
  TEST_EQ(flatbuffers::LoadFile((test_data_path + "monster_test.fbs").c_str(),
                                false, &schemafile),
          true);
  auto include_test_path =
      flatbuffers::ConCatPathFileName(test_data_path, "include_test");
  const char *include_directories[] = { test_data_path.c_str(),
                                        include_test_path.c_str(), nullptr };
  flatbuffers::Parser parser1;
  TEST_EQ(parser1.Parse(schemafile.c_str(), include_directories), true);

  // "color" inside nested flatbuffer contains invalid enum value
  TEST_EQ(parser1.Parse("{ name: \"Bender\", testnestedflatbuffer: { name: "
                        "\"Leela\", color: \"nonexistent\"}}"),
          false);
  // Check that Parser is destroyed correctly after parsing invalid json
}

void UnionVectorTest() {
  // load FlatBuffer fbs schema and json.
  std::string schemafile, jsonfile;
  TEST_EQ(flatbuffers::LoadFile(
              (test_data_path + "union_vector/union_vector.fbs").c_str(),
              false, &schemafile),
          true);
  TEST_EQ(flatbuffers::LoadFile(
              (test_data_path + "union_vector/union_vector.json").c_str(),
              false, &jsonfile),
          true);

  // parse schema.
  flatbuffers::IDLOptions idl_opts;
  idl_opts.lang_to_generate |= flatbuffers::IDLOptions::kBinary;
  flatbuffers::Parser parser(idl_opts);
  TEST_EQ(parser.Parse(schemafile.c_str()), true);

  flatbuffers::FlatBufferBuilder fbb;

  // union types.
  std::vector<uint8_t> types;
  types.push_back(static_cast<uint8_t>(Character_Belle));
  types.push_back(static_cast<uint8_t>(Character_MuLan));
  types.push_back(static_cast<uint8_t>(Character_BookFan));
  types.push_back(static_cast<uint8_t>(Character_Other));
  types.push_back(static_cast<uint8_t>(Character_Unused));

  // union values.
  std::vector<flatbuffers::Offset<void>> characters;
  characters.push_back(fbb.CreateStruct(BookReader(/*books_read=*/7)).Union());
  characters.push_back(CreateAttacker(fbb, /*sword_attack_damage=*/5).Union());
  characters.push_back(fbb.CreateStruct(BookReader(/*books_read=*/2)).Union());
  characters.push_back(fbb.CreateString("Other").Union());
  characters.push_back(fbb.CreateString("Unused").Union());

  // create Movie.
  const auto movie_offset =
      CreateMovie(fbb, Character_Rapunzel,
                  fbb.CreateStruct(Rapunzel(/*hair_length=*/6)).Union(),
                  fbb.CreateVector(types), fbb.CreateVector(characters));
  FinishMovieBuffer(fbb, movie_offset);
  auto buf = fbb.GetBufferPointer();

  flatbuffers::Verifier verifier(buf, fbb.GetSize());
  TEST_EQ(VerifyMovieBuffer(verifier), true);

  auto flat_movie = GetMovie(buf);

  auto TestMovie = [](const Movie *movie) {
    TEST_EQ(movie->main_character_type() == Character_Rapunzel, true);

    auto cts = movie->characters_type();
    TEST_EQ(movie->characters_type()->size(), 5);
    TEST_EQ(cts->GetEnum<Character>(0) == Character_Belle, true);
    TEST_EQ(cts->GetEnum<Character>(1) == Character_MuLan, true);
    TEST_EQ(cts->GetEnum<Character>(2) == Character_BookFan, true);
    TEST_EQ(cts->GetEnum<Character>(3) == Character_Other, true);
    TEST_EQ(cts->GetEnum<Character>(4) == Character_Unused, true);

    auto rapunzel = movie->main_character_as_Rapunzel();
    TEST_NOTNULL(rapunzel);
    TEST_EQ(rapunzel->hair_length(), 6);

    auto cs = movie->characters();
    TEST_EQ(cs->size(), 5);
    auto belle = cs->GetAs<BookReader>(0);
    TEST_EQ(belle->books_read(), 7);
    auto mu_lan = cs->GetAs<Attacker>(1);
    TEST_EQ(mu_lan->sword_attack_damage(), 5);
    auto book_fan = cs->GetAs<BookReader>(2);
    TEST_EQ(book_fan->books_read(), 2);
    auto other = cs->GetAsString(3);
    TEST_EQ_STR(other->c_str(), "Other");
    auto unused = cs->GetAsString(4);
    TEST_EQ_STR(unused->c_str(), "Unused");
  };

  TestMovie(flat_movie);

  // Also test the JSON we loaded above.
  TEST_EQ(parser.Parse(jsonfile.c_str()), true);
  auto jbuf = parser.builder_.GetBufferPointer();
  flatbuffers::Verifier jverifier(jbuf, parser.builder_.GetSize());
  TEST_EQ(VerifyMovieBuffer(jverifier), true);
  TestMovie(GetMovie(jbuf));

  auto movie_object = flat_movie->UnPack();
  TEST_EQ(movie_object->main_character.AsRapunzel()->hair_length(), 6);
  TEST_EQ(movie_object->characters[0].AsBelle()->books_read(), 7);
  TEST_EQ(movie_object->characters[1].AsMuLan()->sword_attack_damage, 5);
  TEST_EQ(movie_object->characters[2].AsBookFan()->books_read(), 2);
  TEST_EQ_STR(movie_object->characters[3].AsOther()->c_str(), "Other");
  TEST_EQ_STR(movie_object->characters[4].AsUnused()->c_str(), "Unused");

  fbb.Clear();
  fbb.Finish(Movie::Pack(fbb, movie_object));

  delete movie_object;

  auto repacked_movie = GetMovie(fbb.GetBufferPointer());

  TestMovie(repacked_movie);

  auto s =
      flatbuffers::FlatBufferToString(fbb.GetBufferPointer(), MovieTypeTable());
  TEST_EQ_STR(
      s.c_str(),
      "{ main_character_type: Rapunzel, main_character: { hair_length: 6 }, "
      "characters_type: [ Belle, MuLan, BookFan, Other, Unused ], "
      "characters: [ { books_read: 7 }, { sword_attack_damage: 5 }, "
      "{ books_read: 2 }, \"Other\", \"Unused\" ] }");


  flatbuffers::ToStringVisitor visitor("\n", true, "  ");
  IterateFlatBuffer(fbb.GetBufferPointer(), MovieTypeTable(), &visitor);
  TEST_EQ_STR(
      visitor.s.c_str(),
      "{\n"
      "  \"main_character_type\": \"Rapunzel\",\n"
      "  \"main_character\": {\n"
      "    \"hair_length\": 6\n"
      "  },\n"
      "  \"characters_type\": [\n"
      "    \"Belle\",\n"
      "    \"MuLan\",\n"
      "    \"BookFan\",\n"
      "    \"Other\",\n"
      "    \"Unused\"\n"
      "  ],\n"
      "  \"characters\": [\n"
      "    {\n"
      "      \"books_read\": 7\n"
      "    },\n"
      "    {\n"
      "      \"sword_attack_damage\": 5\n"
      "    },\n"
      "    {\n"
      "      \"books_read\": 2\n"
      "    },\n"
      "    \"Other\",\n"
      "    \"Unused\"\n"
      "  ]\n"
      "}");

  flatbuffers::Parser parser2(idl_opts);
  TEST_EQ(parser2.Parse("struct Bool { b:bool; }"
                        "union Any { Bool }"
                        "table Root { a:Any; }"
                        "root_type Root;"), true);
  TEST_EQ(parser2.Parse("{a_type:Bool,a:{b:true}}"), true);
}

void ConformTest() {
  flatbuffers::Parser parser;
  TEST_EQ(parser.Parse("table T { A:int; } enum E:byte { A }"), true);

  auto test_conform = [](flatbuffers::Parser &parser1, const char *test,
                         const char *expected_err) {
    flatbuffers::Parser parser2;
    TEST_EQ(parser2.Parse(test), true);
    auto err = parser2.ConformTo(parser1);
    TEST_NOTNULL(strstr(err.c_str(), expected_err));
  };

  test_conform(parser, "table T { A:byte; }", "types differ for field");
  test_conform(parser, "table T { B:int; A:int; }", "offsets differ for field");
  test_conform(parser, "table T { A:int = 1; }", "defaults differ for field");
  test_conform(parser, "table T { B:float; }",
               "field renamed to different type");
  test_conform(parser, "enum E:byte { B, A }", "values differ for enum");
}

void ParseProtoBufAsciiTest() {
  // We can put the parser in a mode where it will accept JSON that looks more
  // like Protobuf ASCII, for users that have data in that format.
  // This uses no "" for field names (which we already support by default,
  // omits `,`, `:` before `{` and a couple of other features.
  flatbuffers::Parser parser;
  parser.opts.protobuf_ascii_alike = true;
  TEST_EQ(
      parser.Parse("table S { B:int; } table T { A:[int]; C:S; } root_type T;"),
      true);
  TEST_EQ(parser.Parse("{ A [1 2] C { B:2 }}"), true);
  // Similarly, in text output, it should omit these.
  std::string text;
  auto ok = flatbuffers::GenerateText(
      parser, parser.builder_.GetBufferPointer(), &text);
  TEST_EQ(ok, true);
  TEST_EQ_STR(text.c_str(),
              "{\n  A [\n    1\n    2\n  ]\n  C {\n    B: 2\n  }\n}\n");
}

void FlexBuffersTest() {
  flexbuffers::Builder slb(512,
                           flexbuffers::BUILDER_FLAG_SHARE_KEYS_AND_STRINGS);

  // Write the equivalent of:
  // { vec: [ -100, "Fred", 4.0, false ], bar: [ 1, 2, 3 ], bar3: [ 1, 2, 3 ],
  // foo: 100, bool: true, mymap: { foo: "Fred" } }
  // clang-format off
  #ifndef FLATBUFFERS_CPP98_STL
    // It's possible to do this without std::function support as well.
    slb.Map([&]() {
       slb.Vector("vec", [&]() {
        slb += -100;  // Equivalent to slb.Add(-100) or slb.Int(-100);
        slb += "Fred";
        slb.IndirectFloat(4.0f);
        auto i_f = slb.LastValue();
        uint8_t blob[] = { 77 };
        slb.Blob(blob, 1);
        slb += false;
        slb.ReuseValue(i_f);
      });
      int ints[] = { 1, 2, 3 };
      slb.Vector("bar", ints, 3);
      slb.FixedTypedVector("bar3", ints, 3);
      bool bools[] = {true, false, true, false};
      slb.Vector("bools", bools, 4);
      slb.Bool("bool", true);
      slb.Double("foo", 100);
      slb.Map("mymap", [&]() {
        slb.String("foo", "Fred");  // Testing key and string reuse.
      });
    });
    slb.Finish();
  #else
    // It's possible to do this without std::function support as well.
    slb.Map([](flexbuffers::Builder& slb2) {
       slb2.Vector("vec", [](flexbuffers::Builder& slb3) {
        slb3 += -100;  // Equivalent to slb.Add(-100) or slb.Int(-100);
        slb3 += "Fred";
        slb3.IndirectFloat(4.0f);
        auto i_f = slb3.LastValue();
        uint8_t blob[] = { 77 };
        slb3.Blob(blob, 1);
        slb3 += false;
        slb3.ReuseValue(i_f);
      }, slb2);
      int ints[] = { 1, 2, 3 };
      slb2.Vector("bar", ints, 3);
      slb2.FixedTypedVector("bar3", ints, 3);
      slb2.Bool("bool", true);
      slb2.Double("foo", 100);
      slb2.Map("mymap", [](flexbuffers::Builder& slb3) {
        slb3.String("foo", "Fred");  // Testing key and string reuse.
      }, slb2);
    }, slb);
    slb.Finish();
  #endif  // FLATBUFFERS_CPP98_STL

  #ifdef FLATBUFFERS_TEST_VERBOSE
    for (size_t i = 0; i < slb.GetBuffer().size(); i++)
      printf("%d ", flatbuffers::vector_data(slb.GetBuffer())[i]);
    printf("\n");
  #endif
  // clang-format on

  auto map = flexbuffers::GetRoot(slb.GetBuffer()).AsMap();
  TEST_EQ(map.size(), 7);
  auto vec = map["vec"].AsVector();
  TEST_EQ(vec.size(), 6);
  TEST_EQ(vec[0].AsInt64(), -100);
  TEST_EQ_STR(vec[1].AsString().c_str(), "Fred");
  TEST_EQ(vec[1].AsInt64(), 0);  // Number parsing failed.
  TEST_EQ(vec[2].AsDouble(), 4.0);
  TEST_EQ(vec[2].AsString().IsTheEmptyString(), true);  // Wrong Type.
  TEST_EQ_STR(vec[2].AsString().c_str(), "");     // This still works though.
  TEST_EQ_STR(vec[2].ToString().c_str(), "4.0");  // Or have it converted.
  // Few tests for templated version of As.
  TEST_EQ(vec[0].As<int64_t>(), -100);
  TEST_EQ_STR(vec[1].As<std::string>().c_str(), "Fred");
  TEST_EQ(vec[1].As<int64_t>(), 0);  // Number parsing failed.
  TEST_EQ(vec[2].As<double>(), 4.0);
  // Test that the blob can be accessed.
  TEST_EQ(vec[3].IsBlob(), true);
  auto blob = vec[3].AsBlob();
  TEST_EQ(blob.size(), 1);
  TEST_EQ(blob.data()[0], 77);
  TEST_EQ(vec[4].IsBool(), true);   // Check if type is a bool
  TEST_EQ(vec[4].AsBool(), false);  // Check if value is false
  TEST_EQ(vec[5].AsDouble(), 4.0);  // This is shared with vec[2] !
  auto tvec = map["bar"].AsTypedVector();
  TEST_EQ(tvec.size(), 3);
  TEST_EQ(tvec[2].AsInt8(), 3);
  auto tvec3 = map["bar3"].AsFixedTypedVector();
  TEST_EQ(tvec3.size(), 3);
  TEST_EQ(tvec3[2].AsInt8(), 3);
  TEST_EQ(map["bool"].AsBool(), true);
  auto tvecb = map["bools"].AsTypedVector();
  TEST_EQ(tvecb.ElementType(), flexbuffers::FBT_BOOL);
  TEST_EQ(map["foo"].AsUInt8(), 100);
  TEST_EQ(map["unknown"].IsNull(), true);
  auto mymap = map["mymap"].AsMap();
  // These should be equal by pointer equality, since key and value are shared.
  TEST_EQ(mymap.Keys()[0].AsKey(), map.Keys()[4].AsKey());
  TEST_EQ(mymap.Values()[0].AsString().c_str(), vec[1].AsString().c_str());
  // We can mutate values in the buffer.
  TEST_EQ(vec[0].MutateInt(-99), true);
  TEST_EQ(vec[0].AsInt64(), -99);
  TEST_EQ(vec[1].MutateString("John"), true);  // Size must match.
  TEST_EQ_STR(vec[1].AsString().c_str(), "John");
  TEST_EQ(vec[1].MutateString("Alfred"), false);  // Too long.
  TEST_EQ(vec[2].MutateFloat(2.0f), true);
  TEST_EQ(vec[2].AsFloat(), 2.0f);
  TEST_EQ(vec[2].MutateFloat(3.14159), false);  // Double does not fit in float.
  TEST_EQ(vec[4].AsBool(), false);              // Is false before change
  TEST_EQ(vec[4].MutateBool(true), true);       // Can change a bool
  TEST_EQ(vec[4].AsBool(), true);               // Changed bool is now true

  // Parse from JSON:
  flatbuffers::Parser parser;
  slb.Clear();
  auto jsontest = "{ a: [ 123, 456.0 ], b: \"hello\", c: true, d: false }";
  TEST_EQ(parser.ParseFlexBuffer(jsontest, nullptr, &slb), true);
  auto jroot = flexbuffers::GetRoot(slb.GetBuffer());
  auto jmap = jroot.AsMap();
  auto jvec = jmap["a"].AsVector();
  TEST_EQ(jvec[0].AsInt64(), 123);
  TEST_EQ(jvec[1].AsDouble(), 456.0);
  TEST_EQ_STR(jmap["b"].AsString().c_str(), "hello");
  TEST_EQ(jmap["c"].IsBool(), true);   // Parsed correctly to a bool
  TEST_EQ(jmap["c"].AsBool(), true);   // Parsed correctly to true
  TEST_EQ(jmap["d"].IsBool(), true);   // Parsed correctly to a bool
  TEST_EQ(jmap["d"].AsBool(), false);  // Parsed correctly to false
  // And from FlexBuffer back to JSON:
  auto jsonback = jroot.ToString();
  TEST_EQ_STR(jsontest, jsonback.c_str());
}

void TypeAliasesTest() {
  flatbuffers::FlatBufferBuilder builder;

  builder.Finish(CreateTypeAliases(
      builder, flatbuffers::numeric_limits<int8_t>::min(),
      flatbuffers::numeric_limits<uint8_t>::max(),
      flatbuffers::numeric_limits<int16_t>::min(),
      flatbuffers::numeric_limits<uint16_t>::max(),
      flatbuffers::numeric_limits<int32_t>::min(),
      flatbuffers::numeric_limits<uint32_t>::max(),
      flatbuffers::numeric_limits<int64_t>::min(),
      flatbuffers::numeric_limits<uint64_t>::max(), 2.3f, 2.3));

  auto p = builder.GetBufferPointer();
  auto ta = flatbuffers::GetRoot<TypeAliases>(p);

  TEST_EQ(ta->i8(), flatbuffers::numeric_limits<int8_t>::min());
  TEST_EQ(ta->u8(), flatbuffers::numeric_limits<uint8_t>::max());
  TEST_EQ(ta->i16(), flatbuffers::numeric_limits<int16_t>::min());
  TEST_EQ(ta->u16(), flatbuffers::numeric_limits<uint16_t>::max());
  TEST_EQ(ta->i32(), flatbuffers::numeric_limits<int32_t>::min());
  TEST_EQ(ta->u32(), flatbuffers::numeric_limits<uint32_t>::max());
  TEST_EQ(ta->i64(), flatbuffers::numeric_limits<int64_t>::min());
  TEST_EQ(ta->u64(), flatbuffers::numeric_limits<uint64_t>::max());
  TEST_EQ(ta->f32(), 2.3f);
  TEST_EQ(ta->f64(), 2.3);
  using namespace flatbuffers; // is_same
  static_assert(is_same<decltype(ta->i8()), int8_t>::value, "invalid type");
  static_assert(is_same<decltype(ta->i16()), int16_t>::value, "invalid type");
  static_assert(is_same<decltype(ta->i32()), int32_t>::value, "invalid type");
  static_assert(is_same<decltype(ta->i64()), int64_t>::value, "invalid type");
  static_assert(is_same<decltype(ta->u8()), uint8_t>::value, "invalid type");
  static_assert(is_same<decltype(ta->u16()), uint16_t>::value, "invalid type");
  static_assert(is_same<decltype(ta->u32()), uint32_t>::value, "invalid type");
  static_assert(is_same<decltype(ta->u64()), uint64_t>::value, "invalid type");
  static_assert(is_same<decltype(ta->f32()), float>::value, "invalid type");
  static_assert(is_same<decltype(ta->f64()), double>::value, "invalid type");
}

void EndianSwapTest() {
  TEST_EQ(flatbuffers::EndianSwap(static_cast<int16_t>(0x1234)), 0x3412);
  TEST_EQ(flatbuffers::EndianSwap(static_cast<int32_t>(0x12345678)),
          0x78563412);
  TEST_EQ(flatbuffers::EndianSwap(static_cast<int64_t>(0x1234567890ABCDEF)),
          0xEFCDAB9078563412);
  TEST_EQ(flatbuffers::EndianSwap(flatbuffers::EndianSwap(3.14f)), 3.14f);
}

void UninitializedVectorTest() {
  flatbuffers::FlatBufferBuilder builder;

  Test *buf = nullptr;
  auto vector_offset = builder.CreateUninitializedVectorOfStructs<Test>(2, &buf);
  TEST_NOTNULL(buf);
  buf[0] = Test(10, 20);
  buf[1] = Test(30, 40);

  auto required_name = builder.CreateString("myMonster");
  auto monster_builder = MonsterBuilder(builder);
  monster_builder.add_name(required_name); // required field mandated for monster.
  monster_builder.add_test4(vector_offset);
  builder.Finish(monster_builder.Finish());

  auto p = builder.GetBufferPointer();
  auto uvt = flatbuffers::GetRoot<Monster>(p);
  TEST_NOTNULL(uvt);
  auto vec = uvt->test4();
  TEST_NOTNULL(vec);
  auto test_0 = vec->Get(0);
  auto test_1 = vec->Get(1);
  TEST_EQ(test_0->a(), 10);
  TEST_EQ(test_0->b(), 20);
  TEST_EQ(test_1->a(), 30);
  TEST_EQ(test_1->b(), 40);
}

void EqualOperatorTest() {
  MonsterT a;
  MonsterT b;
  TEST_EQ(b == a, true);
  TEST_EQ(b != a, false);

  b.mana = 33;
  TEST_EQ(b == a, false);
  TEST_EQ(b != a, true);
  b.mana = 150;
  TEST_EQ(b == a, true);
  TEST_EQ(b != a, false);

  b.inventory.push_back(3);
  TEST_EQ(b == a, false);
  TEST_EQ(b != a, true);
  b.inventory.clear();
  TEST_EQ(b == a, true);
  TEST_EQ(b != a, false);

  b.test.type = Any_Monster;
  TEST_EQ(b == a, false);
  TEST_EQ(b != a, true);
}

// For testing any binaries, e.g. from fuzzing.
void LoadVerifyBinaryTest() {
  std::string binary;
  if (flatbuffers::LoadFile((test_data_path +
                             "fuzzer/your-filename-here").c_str(),
                            true, &binary)) {
    flatbuffers::Verifier verifier(
          reinterpret_cast<const uint8_t *>(binary.data()), binary.size());
    TEST_EQ(VerifyMonsterBuffer(verifier), true);
  }
}

void CreateSharedStringTest() {
  flatbuffers::FlatBufferBuilder builder;
  const auto one1 = builder.CreateSharedString("one");
  const auto two = builder.CreateSharedString("two");
  const auto one2 = builder.CreateSharedString("one");
  TEST_EQ(one1.o, one2.o);
  const auto onetwo = builder.CreateSharedString("onetwo");
  TEST_EQ(onetwo.o != one1.o, true);
  TEST_EQ(onetwo.o != two.o, true);

  // Support for embedded nulls
  const char chars_b[] = {'a', '\0', 'b'};
  const char chars_c[] = {'a', '\0', 'c'};
  const auto null_b1 = builder.CreateSharedString(chars_b, sizeof(chars_b));
  const auto null_c = builder.CreateSharedString(chars_c, sizeof(chars_c));
  const auto null_b2 = builder.CreateSharedString(chars_b, sizeof(chars_b));
  TEST_EQ(null_b1.o != null_c.o, true); // Issue#5058 repro
  TEST_EQ(null_b1.o, null_b2.o);

  // Put the strings into an array for round trip verification.
  const flatbuffers::Offset<flatbuffers::String> array[7] = { one1, two, one2, onetwo, null_b1, null_c, null_b2 };
  const auto vector_offset = builder.CreateVector(array, flatbuffers::uoffset_t(7));
  MonsterBuilder monster_builder(builder);
  monster_builder.add_name(two);
  monster_builder.add_testarrayofstring(vector_offset);
  builder.Finish(monster_builder.Finish());

  // Read the Monster back.
  const auto *monster = flatbuffers::GetRoot<Monster>(builder.GetBufferPointer());
  TEST_EQ_STR(monster->name()->c_str(), "two");
  const auto *testarrayofstring = monster->testarrayofstring();
  TEST_EQ(testarrayofstring->size(), flatbuffers::uoffset_t(7));
  const auto &a = *testarrayofstring;
  TEST_EQ_STR(a[0]->c_str(), "one");
  TEST_EQ_STR(a[1]->c_str(), "two");
  TEST_EQ_STR(a[2]->c_str(), "one");
  TEST_EQ_STR(a[3]->c_str(), "onetwo");
  TEST_EQ(a[4]->str(), (std::string(chars_b, sizeof(chars_b))));
  TEST_EQ(a[5]->str(), (std::string(chars_c, sizeof(chars_c))));
  TEST_EQ(a[6]->str(), (std::string(chars_b, sizeof(chars_b))));

  // Make sure String::operator< works, too, since it is related to StringOffsetCompare.
  TEST_EQ((*a[0]) < (*a[1]), true);
  TEST_EQ((*a[1]) < (*a[0]), false);
  TEST_EQ((*a[1]) < (*a[2]), false);
  TEST_EQ((*a[2]) < (*a[1]), true);
  TEST_EQ((*a[4]) < (*a[3]), true);
  TEST_EQ((*a[5]) < (*a[4]), false);
  TEST_EQ((*a[5]) < (*a[4]), false);
  TEST_EQ((*a[6]) < (*a[5]), true);
}

void FixedLengthArrayTest() {
  // VS10 does not support typed enums, exclude from tests
#if !defined(_MSC_VER) || _MSC_VER >= 1700
  // Generate an ArrayTable containing one ArrayStruct.
  flatbuffers::FlatBufferBuilder fbb;
  MyGame::Example::NestedStruct nStruct0(MyGame::Example::TestEnum::B);
  TEST_NOTNULL(nStruct0.mutable_a());
  nStruct0.mutable_a()->Mutate(0, 1);
  nStruct0.mutable_a()->Mutate(1, 2);
  TEST_NOTNULL(nStruct0.mutable_c());
  nStruct0.mutable_c()->Mutate(0, MyGame::Example::TestEnum::C);
  nStruct0.mutable_c()->Mutate(1, MyGame::Example::TestEnum::A);
  TEST_NOTNULL(nStruct0.mutable_d());
  nStruct0.mutable_d()->Mutate(0, flatbuffers::numeric_limits<int64_t>::max());
  nStruct0.mutable_d()->Mutate(1, flatbuffers::numeric_limits<int64_t>::min());
  MyGame::Example::NestedStruct nStruct1(MyGame::Example::TestEnum::C);
  TEST_NOTNULL(nStruct1.mutable_a());
  nStruct1.mutable_a()->Mutate(0, 3);
  nStruct1.mutable_a()->Mutate(1, 4);
  TEST_NOTNULL(nStruct1.mutable_c());
  nStruct1.mutable_c()->Mutate(0, MyGame::Example::TestEnum::C);
  nStruct1.mutable_c()->Mutate(1, MyGame::Example::TestEnum::A);
  TEST_NOTNULL(nStruct1.mutable_d());
  nStruct1.mutable_d()->Mutate(0, flatbuffers::numeric_limits<int64_t>::min());
  nStruct1.mutable_d()->Mutate(1, flatbuffers::numeric_limits<int64_t>::max());
  MyGame::Example::ArrayStruct aStruct(2, 12, 1);
  TEST_NOTNULL(aStruct.b());
  TEST_NOTNULL(aStruct.mutable_b());
  TEST_NOTNULL(aStruct.mutable_d());
  TEST_NOTNULL(aStruct.mutable_f());
  for (int i = 0; i < aStruct.b()->size(); i++)
    aStruct.mutable_b()->Mutate(i, i + 1);
  aStruct.mutable_d()->Mutate(0, nStruct0);
  aStruct.mutable_d()->Mutate(1, nStruct1);
  auto aTable = MyGame::Example::CreateArrayTable(fbb, &aStruct);
  fbb.Finish(aTable);

  // Verify correctness of the ArrayTable.
  flatbuffers::Verifier verifier(fbb.GetBufferPointer(), fbb.GetSize());
  MyGame::Example::VerifyArrayTableBuffer(verifier);
  auto p = MyGame::Example::GetMutableArrayTable(fbb.GetBufferPointer());
  auto mArStruct = p->mutable_a();
  TEST_NOTNULL(mArStruct);
  TEST_NOTNULL(mArStruct->b());
  TEST_NOTNULL(mArStruct->d());
  TEST_NOTNULL(mArStruct->f());
  TEST_NOTNULL(mArStruct->mutable_b());
  TEST_NOTNULL(mArStruct->mutable_d());
  TEST_NOTNULL(mArStruct->mutable_f());
  mArStruct->mutable_b()->Mutate(14, -14);
  TEST_EQ(mArStruct->a(), 2);
  TEST_EQ(mArStruct->b()->size(), 15);
  TEST_EQ(mArStruct->b()->Get(aStruct.b()->size() - 1), -14);
  TEST_EQ(mArStruct->c(), 12);
  TEST_NOTNULL(mArStruct->d()->Get(0).a());
  TEST_EQ(mArStruct->d()->Get(0).a()->Get(0), 1);
  TEST_EQ(mArStruct->d()->Get(0).a()->Get(1), 2);
  TEST_NOTNULL(mArStruct->d()->Get(1).a());
  TEST_EQ(mArStruct->d()->Get(1).a()->Get(0), 3);
  TEST_EQ(mArStruct->d()->Get(1).a()->Get(1), 4);
  TEST_NOTNULL(mArStruct->mutable_d()->GetMutablePointer(1));
  TEST_NOTNULL(mArStruct->mutable_d()->GetMutablePointer(1)->mutable_a());
  mArStruct->mutable_d()->GetMutablePointer(1)->mutable_a()->Mutate(1, 5);
  TEST_EQ(mArStruct->d()->Get(1).a()->Get(1), 5);
  TEST_EQ(mArStruct->d()->Get(0).b() == MyGame::Example::TestEnum::B, true);
  TEST_NOTNULL(mArStruct->d()->Get(0).c());
  TEST_EQ(mArStruct->d()->Get(0).c()->Get(0) == MyGame::Example::TestEnum::C,
          true);
  TEST_EQ(mArStruct->d()->Get(0).c()->Get(1) == MyGame::Example::TestEnum::A,
          true);
  TEST_EQ(mArStruct->d()->Get(0).d()->Get(0),
          flatbuffers::numeric_limits<int64_t>::max());
  TEST_EQ(mArStruct->d()->Get(0).d()->Get(1),
          flatbuffers::numeric_limits<int64_t>::min());
  TEST_EQ(mArStruct->d()->Get(1).b() == MyGame::Example::TestEnum::C, true);
  TEST_NOTNULL(mArStruct->d()->Get(1).c());
  TEST_EQ(mArStruct->d()->Get(1).c()->Get(0) == MyGame::Example::TestEnum::C,
          true);
  TEST_EQ(mArStruct->d()->Get(1).c()->Get(1) == MyGame::Example::TestEnum::A,
          true);
  TEST_EQ(mArStruct->d()->Get(1).d()->Get(0),
          flatbuffers::numeric_limits<int64_t>::min());
  TEST_EQ(mArStruct->d()->Get(1).d()->Get(1),
          flatbuffers::numeric_limits<int64_t>::max());
  for (int i = 0; i < mArStruct->b()->size() - 1; i++)
    TEST_EQ(mArStruct->b()->Get(i), i + 1);
  // Check alignment
  TEST_EQ(reinterpret_cast<uintptr_t>(mArStruct->d()) % 8, 0);
  TEST_EQ(reinterpret_cast<uintptr_t>(mArStruct->f()) % 8, 0);
#endif
}

void NativeTypeTest() {
  const int N = 3;

  Geometry::ApplicationDataT src_data;
  src_data.vectors.reserve(N);

  for (int i = 0; i < N; ++i) {
    src_data.vectors.push_back (Native::Vector3D(10 * i + 0.1f, 10 * i + 0.2f, 10 * i + 0.3f));
  }

  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(Geometry::ApplicationData::Pack(fbb, &src_data));

  auto dstDataT =  Geometry::UnPackApplicationData(fbb.GetBufferPointer());

  for (int i = 0; i < N; ++i) {
    Native::Vector3D& v = dstDataT->vectors[i];
    TEST_EQ(v.x, 10 * i + 0.1f);
    TEST_EQ(v.y, 10 * i + 0.2f);
    TEST_EQ(v.z, 10 * i + 0.3f);
  }
}

void FixedLengthArrayJsonTest(bool binary) {
  // VS10 does not support typed enums, exclude from tests
#if !defined(_MSC_VER) || _MSC_VER >= 1700
  // load FlatBuffer schema (.fbs) and JSON from disk
  std::string schemafile;
  std::string jsonfile;
  TEST_EQ(
      flatbuffers::LoadFile(
          (test_data_path + "arrays_test." + (binary ? "bfbs" : "fbs")).c_str(),
          binary, &schemafile),
      true);
  TEST_EQ(flatbuffers::LoadFile((test_data_path + "arrays_test.golden").c_str(),
                                false, &jsonfile),
          true);

  // parse schema first, so we can use it to parse the data after
  flatbuffers::Parser parserOrg, parserGen;
  if (binary) {
    flatbuffers::Verifier verifier(
        reinterpret_cast<const uint8_t *>(schemafile.c_str()),
        schemafile.size());
    TEST_EQ(reflection::VerifySchemaBuffer(verifier), true);
    TEST_EQ(parserOrg.Deserialize((const uint8_t *)schemafile.c_str(),
                                  schemafile.size()),
            true);
    TEST_EQ(parserGen.Deserialize((const uint8_t *)schemafile.c_str(),
                                  schemafile.size()),
            true);
  } else {
    TEST_EQ(parserOrg.Parse(schemafile.c_str()), true);
    TEST_EQ(parserGen.Parse(schemafile.c_str()), true);
  }
  TEST_EQ(parserOrg.Parse(jsonfile.c_str()), true);

  // First, verify it, just in case:
  flatbuffers::Verifier verifierOrg(parserOrg.builder_.GetBufferPointer(),
                                    parserOrg.builder_.GetSize());
  TEST_EQ(VerifyArrayTableBuffer(verifierOrg), true);

  // Export to JSON
  std::string jsonGen;
  TEST_EQ(
      GenerateText(parserOrg, parserOrg.builder_.GetBufferPointer(), &jsonGen),
      true);

  // Import from JSON
  TEST_EQ(parserGen.Parse(jsonGen.c_str()), true);

  // Verify buffer from generated JSON
  flatbuffers::Verifier verifierGen(parserGen.builder_.GetBufferPointer(),
                                    parserGen.builder_.GetSize());
  TEST_EQ(VerifyArrayTableBuffer(verifierGen), true);

  // Compare generated buffer to original
  TEST_EQ(parserOrg.builder_.GetSize(), parserGen.builder_.GetSize());
  TEST_EQ(std::memcmp(parserOrg.builder_.GetBufferPointer(),
                      parserGen.builder_.GetBufferPointer(),
                      parserOrg.builder_.GetSize()),
          0);
#else
  (void)binary;
#endif
}

int FlatBufferTests() {
  // clang-format off

  // Run our various test suites:

  std::string rawbuf;
  auto flatbuf1 = CreateFlatBufferTest(rawbuf);
  #if !defined(FLATBUFFERS_CPP98_STL)
    auto flatbuf = std::move(flatbuf1);  // Test move assignment.
  #else
    auto &flatbuf = flatbuf1;
  #endif // !defined(FLATBUFFERS_CPP98_STL)

  TriviallyCopyableTest();

  AccessFlatBufferTest(reinterpret_cast<const uint8_t *>(rawbuf.c_str()),
                       rawbuf.length());
  AccessFlatBufferTest(flatbuf.data(), flatbuf.size());

  MutateFlatBuffersTest(flatbuf.data(), flatbuf.size());

  ObjectFlatBuffersTest(flatbuf.data());

  MiniReflectFlatBuffersTest(flatbuf.data());

  SizePrefixedTest();

  #ifndef FLATBUFFERS_NO_FILE_TESTS
    #ifdef FLATBUFFERS_TEST_PATH_PREFIX
      test_data_path = FLATBUFFERS_STRING(FLATBUFFERS_TEST_PATH_PREFIX) +
                       test_data_path;
    #endif
    ParseAndGenerateTextTest(false);
    ParseAndGenerateTextTest(true);
    FixedLengthArrayJsonTest(false);
    FixedLengthArrayJsonTest(true);
    ReflectionTest(flatbuf.data(), flatbuf.size());
    ParseProtoTest();
    UnionVectorTest();
    LoadVerifyBinaryTest();
    GenerateTableTextTest();
  #endif
  // clang-format on

  FuzzTest1();
  FuzzTest2();

  ErrorTest();
  ValueTest();
  EnumValueTest();
  EnumStringsTest();
  EnumNamesTest();
  EnumOutOfRangeTest();
  IntegerOutOfRangeTest();
  IntegerBoundaryTest();
  UnicodeTest();
  UnicodeTestAllowNonUTF8();
  UnicodeTestGenerateTextFailsOnNonUTF8();
  UnicodeSurrogatesTest();
  UnicodeInvalidSurrogatesTest();
  InvalidUTF8Test();
  UnknownFieldsTest();
  ParseUnionTest();
  InvalidNestedFlatbufferTest();
  ConformTest();
  ParseProtoBufAsciiTest();
  TypeAliasesTest();
  EndianSwapTest();
  CreateSharedStringTest();
  JsonDefaultTest();
  JsonEnumsTest();
  FlexBuffersTest();
  UninitializedVectorTest();
  EqualOperatorTest();
  NumericUtilsTest();
  IsAsciiUtilsTest();
  ValidFloatTest();
  InvalidFloatTest();
  TestMonsterExtraFloats();
  FixedLengthArrayTest();
  NativeTypeTest();
  return 0;
}

int main(int /*argc*/, const char * /*argv*/ []) {
  InitTestEngine();

  std::string req_locale;
  if (flatbuffers::ReadEnvironmentVariable("FLATBUFFERS_TEST_LOCALE",
                                           &req_locale)) {
    TEST_OUTPUT_LINE("The environment variable FLATBUFFERS_TEST_LOCALE=%s",
                     req_locale.c_str());
    req_locale = flatbuffers::RemoveStringQuotes(req_locale);
    std::string the_locale;
    TEST_ASSERT_FUNC(
        flatbuffers::SetGlobalTestLocale(req_locale.c_str(), &the_locale));
    TEST_OUTPUT_LINE("The global C-locale changed: %s", the_locale.c_str());
  }

  FlatBufferTests();
  FlatBufferBuilderTest();

  if (!testing_fails) {
    TEST_OUTPUT_LINE("ALL TESTS PASSED");
  } else {
    TEST_OUTPUT_LINE("%d FAILED TESTS", testing_fails);
  }
  return CloseTestEngine();
}
