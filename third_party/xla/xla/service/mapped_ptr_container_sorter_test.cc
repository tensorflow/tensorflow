/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/mapped_ptr_container_sorter.h"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <list>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/bind_front.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/testlib/test.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::Pointee;

std::vector<std::unique_ptr<std::string>> CreateUniquePtrContainer(
    const std::vector<std::string>& values) {
  std::vector<std::unique_ptr<std::string>> container;
  for (auto value : values) {
    container.push_back(std::make_unique<std::string>(value));
  }
  return container;
}

class MappedPtrContainerSorterTest : public ::testing::Test {
 public:
  using Sorter = MappedPtrContainerSorter<std::string>;

  MappedPtrContainerSorterTest()
      : ordered_unique_ptrs_(CreateUniquePtrContainer(
            {"m0", "m1", "m2", "m3", "not_in_unordered"})),
        unordered_unique_ptrs_(
            CreateUniquePtrContainer({"m3", "m1", "m0", "m2"})) {
    for (auto& unique : ordered_unique_ptrs_) {
      ordered_raw_ptrs_.push_back(unique.get());
      ordered_const_raw_ptrs_.push_back(unique.get());
    }
    for (auto& unique : unordered_unique_ptrs_) {
      unordered_raw_ptrs_.push_back(unique.get());
      unordered_const_raw_ptrs_.push_back(unique.get());
    }
  }

 protected:
  const std::string* MapPtr(const std::string* ordered) const {
    for (size_t i = 0; i < unordered_unique_ptrs_.size(); ++i) {
      if (*ordered == *unordered_unique_ptrs_[i]) {
        return unordered_unique_ptrs_[i].get();
      }
    }
    return nullptr;
  }

  auto MapPtrFn() const {
    return absl::bind_front(&MappedPtrContainerSorterTest::MapPtr, this);
  }

  // unordered_unique_ptrs_: u0, m3, u1, u2, m2, m0, m2, u3
  void AddUnmappedElementsToUnorderedUniquePtrs() {
    unordered_unique_ptrs_.insert(unordered_unique_ptrs_.begin(),
                                  std::make_unique<std::string>("u0"));
    unordered_unique_ptrs_.insert(unordered_unique_ptrs_.begin() + 2,
                                  std::make_unique<std::string>("u1"));
    unordered_unique_ptrs_.insert(unordered_unique_ptrs_.begin() + 3,
                                  std::make_unique<std::string>("u2"));
    unordered_unique_ptrs_.insert(unordered_unique_ptrs_.end(),
                                  std::make_unique<std::string>("u3"));
  }

  std::vector<std::unique_ptr<std::string>> ordered_unique_ptrs_;
  std::vector<std::unique_ptr<std::string>> unordered_unique_ptrs_;
  std::vector<std::string*> ordered_raw_ptrs_;
  std::vector<std::string*> unordered_raw_ptrs_;
  std::vector<const std::string*> ordered_const_raw_ptrs_;
  std::vector<const std::string*> unordered_const_raw_ptrs_;
};

TEST_F(MappedPtrContainerSorterTest, SortUniquePtrs) {
  EXPECT_OK(Sorter::Sort(MapPtrFn(), Sorter::InvalidIndexFn(),
                         ordered_unique_ptrs_, unordered_unique_ptrs_));
  EXPECT_THAT(
      unordered_unique_ptrs_,
      ElementsAre(Pointee(std::string("m0")), Pointee(std::string("m1")),
                  Pointee(std::string("m2")), Pointee(std::string("m3"))));
}

TEST_F(MappedPtrContainerSorterTest, RawPtrs) {
  EXPECT_OK(Sorter::Sort(MapPtrFn(), Sorter::InvalidIndexFn(),
                         ordered_raw_ptrs_, unordered_raw_ptrs_));
  EXPECT_THAT(
      unordered_raw_ptrs_,
      ElementsAre(Pointee(std::string("m0")), Pointee(std::string("m1")),
                  Pointee(std::string("m2")), Pointee(std::string("m3"))));
}

TEST_F(MappedPtrContainerSorterTest, ConstRawPtrs) {
  EXPECT_OK(Sorter::Sort(MapPtrFn(), Sorter::InvalidIndexFn(),
                         ordered_const_raw_ptrs_, unordered_const_raw_ptrs_));
  EXPECT_THAT(
      unordered_const_raw_ptrs_,
      ElementsAre(Pointee(std::string("m0")), Pointee(std::string("m1")),
                  Pointee(std::string("m2")), Pointee(std::string("m3"))));
}

TEST_F(MappedPtrContainerSorterTest, DifferentContainerTypes) {
  std::list<std::unique_ptr<std::string>> ordered_ptrs;
  for (auto& ptr : ordered_unique_ptrs_) {
    ordered_ptrs.push_back(std::move(ptr));
  }

  EXPECT_OK(Sorter::Sort(MapPtrFn(), Sorter::InvalidIndexFn(), ordered_ptrs,
                         unordered_unique_ptrs_));
  EXPECT_THAT(
      unordered_unique_ptrs_,
      ElementsAre(Pointee(std::string("m0")), Pointee(std::string("m1")),
                  Pointee(std::string("m2")), Pointee(std::string("m3"))));
}

TEST_F(MappedPtrContainerSorterTest, WithUnmappedPtrsAfterMappedPtrs) {
  AddUnmappedElementsToUnorderedUniquePtrs();

  EXPECT_OK(Sorter::Sort(MapPtrFn(), Sorter::IndexAfterMappedElementsFn(),
                         ordered_unique_ptrs_, unordered_unique_ptrs_));
  EXPECT_THAT(
      unordered_unique_ptrs_,
      ElementsAre(Pointee(std::string("m0")), Pointee(std::string("m1")),
                  Pointee(std::string("m2")), Pointee(std::string("m3")),
                  // Unmapped pointers come after mapped ptrs
                  Pointee(std::string("u0")), Pointee(std::string("u1")),
                  Pointee(std::string("u2")), Pointee(std::string("u3"))));
}

TEST_F(MappedPtrContainerSorterTest, WithUnmappedPtrsBeforeMappedPtrs) {
  AddUnmappedElementsToUnorderedUniquePtrs();

  EXPECT_OK(Sorter::Sort(MapPtrFn(), Sorter::IndexBeforeMappedElementsFn(),
                         ordered_unique_ptrs_, unordered_unique_ptrs_));
  EXPECT_THAT(unordered_unique_ptrs_,
              ElementsAre(
                  // Unmapped pointers come before mapped ptrs
                  Pointee(std::string("u0")), Pointee(std::string("u1")),
                  Pointee(std::string("u2")), Pointee(std::string("u3")),
                  Pointee(std::string("m0")), Pointee(std::string("m1")),
                  Pointee(std::string("m2")), Pointee(std::string("m3"))));
}

TEST_F(MappedPtrContainerSorterTest, WithUnmappedPtrsInCustomLocations) {
  auto unmapped_ptr_index = [](const std::string* s) -> size_t {
    if (*s == "u0") {
      return Sorter::IndexAfterMappedElementsFn()(s);
    }
    if (*s == "u1") {
      return 2;
    }
    if (*s == "u2") {
      return 2;
    }
    if (*s == "u3") {
      return Sorter::IndexBeforeMappedElementsFn()(s);
    }
    LOG(FATAL) << "We should not be getting an unmapped ptr index for " << *s;
  };
  AddUnmappedElementsToUnorderedUniquePtrs();

  EXPECT_OK(Sorter::Sort(MapPtrFn(), unmapped_ptr_index, ordered_unique_ptrs_,
                         unordered_unique_ptrs_));
  EXPECT_THAT(
      unordered_unique_ptrs_,
      ElementsAre(
          Pointee(std::string("u3")),  // unmapped u3 comes before mapped ptrs
          Pointee(std::string("m0")),  // mapped index 0
          Pointee(std::string("m1")),  // mapped index 1
          Pointee(std::string("m2")),  // mapped index 2
          Pointee(std::string("u1")),  // unmapped u1 comes after mapped index 2
          Pointee(std::string("u2")),  // unmapped u2 comes after mapped index 2
          Pointee(std::string("m3")),  // mapped index 3
          Pointee(std::string("u0"))   // unmapped u0 comes after mapped ptrs
          ));
}

TEST_F(MappedPtrContainerSorterTest,
       ManyOrderedElementsMapToFewUnorderedElements) {
  std::string* ordered_m1 = nullptr;
  for (auto ptr : ordered_raw_ptrs_) {
    if (*ptr == "m1") {
      ordered_m1 = ptr;
      break;
    }
  }
  ASSERT_NE(ordered_m1, nullptr);
  std::string* unordered_m1 = nullptr;
  for (auto ptr : unordered_raw_ptrs_) {
    if (*ptr == "m1") {
      unordered_m1 = ptr;
      break;
    }
  }
  ASSERT_NE(unordered_m1, nullptr);

  // Add 2 more instances of m1 to the ordered container and 1 more to the
  // unordered container.
  ordered_raw_ptrs_.insert(ordered_raw_ptrs_.begin(), ordered_m1);
  ordered_raw_ptrs_.push_back(ordered_m1);
  unordered_raw_ptrs_.push_back(unordered_m1);

  EXPECT_OK(Sorter::Sort(MapPtrFn(), Sorter::IndexBeforeMappedElementsFn(),
                         ordered_raw_ptrs_, unordered_raw_ptrs_));
  EXPECT_THAT(
      unordered_raw_ptrs_,
      ElementsAre(
          Pointee(std::string("m1")),  // Corresponds to 1st m1 in ordered
          Pointee(std::string("m0")),
          Pointee(std::string("m1")),  // Corresponds to 2nd m1 in ordered
          Pointee(std::string("m2")), Pointee(std::string("m3"))));
}

TEST_F(MappedPtrContainerSorterTest,
       FewOrderedElementsMapToManyUnorderedElements) {
  std::string* ordered_m1 = nullptr;
  for (auto ptr : ordered_raw_ptrs_) {
    if (*ptr == "m1") {
      ordered_m1 = ptr;
      break;
    }
  }
  ASSERT_NE(ordered_m1, nullptr);
  std::string* unordered_m1 = nullptr;
  for (auto ptr : unordered_raw_ptrs_) {
    if (*ptr == "m1") {
      unordered_m1 = ptr;
      break;
    }
  }
  ASSERT_NE(unordered_m1, nullptr);

  // Add 1 more instances of m1 to the ordered container and 2 more to the
  // unordered container.
  ordered_raw_ptrs_.insert(ordered_raw_ptrs_.begin(), ordered_m1);
  unordered_raw_ptrs_.push_back(unordered_m1);
  unordered_raw_ptrs_.push_back(unordered_m1);

  EXPECT_OK(Sorter::Sort(MapPtrFn(), Sorter::IndexBeforeMappedElementsFn(),
                         ordered_raw_ptrs_, unordered_raw_ptrs_));
  EXPECT_THAT(
      unordered_raw_ptrs_,
      ElementsAre(
          Pointee(std::string("m1")),  // Corresponds to 1st m1 in ordered
          Pointee(std::string("m0")),
          Pointee(std::string("m1")),  // Corresponds to 2nd m1 in ordered
          Pointee(std::string("m1")),  // Reuse position of 2nd m1 in ordered
          Pointee(std::string("m2")), Pointee(std::string("m3"))));
}

// Returns "m0", "m1", ... "m<size-1>".
std::vector<std::string> Names(size_t size) {
  std::vector<std::string> names;
  names.reserve(size);
  for (size_t i = 0; i < size; ++i) {
    names.push_back(absl::StrCat("m", i));
  }
  return names;
}

std::vector<std::string> Values(
    const std::vector<std::unique_ptr<std::string>>& container) {
  std::vector<std::string> values;
  for (const auto& element : container) {
    values.push_back(*element);
  }
  return values;
}

// Maps an ordered element to the unordered element of the same value, "x" to
// the outside pointer, and anything else to null. Counts its calls: the direct
// path maps every ordered element once, and a fallback after a failed attempt
// twice.
class NameMapper {
 public:
  explicit NameMapper(
      const std::vector<std::unique_ptr<std::string>>& unordered,
      const std::string* outside = nullptr)
      : unordered_(unordered), outside_(outside) {}

  const std::string* operator()(const std::string* ordered) {
    ++calls_;
    if (*ordered == "x") {
      return outside_;
    }
    for (const auto& u : unordered_) {
      if (*u == *ordered) {
        return u.get();
      }
    }
    return nullptr;
  }

  size_t calls() const { return calls_; }

 private:
  const std::vector<std::unique_ptr<std::string>>& unordered_;
  const std::string* outside_;
  size_t calls_ = 0;
};

// Mapped containers are sorted without the partial order bookkeeping, and a
// bijection never takes the fallback. This covers both lookups of the direct
// path, ordered elements that map to nothing and the identity permutation.
TEST_F(MappedPtrContainerSorterTest, BijectionsOfEverySmallSize) {
  std::minstd_rand rng(20260923);
  for (size_t size = 1; size <= 2 * Sorter::kLinearSearchLimit + 6; ++size) {
    for (int trial = 0; trial < 3; ++trial) {
      const std::vector<std::string> names = Names(size);
      std::vector<std::unique_ptr<std::string>> unordered =
          CreateUniquePtrContainer(names);
      if (trial > 0) {
        std::shuffle(unordered.begin(), unordered.end(), rng);
      }
      // The ordered container is m0, m1, ..., interleaved with pointers that
      // map to no unordered element.
      std::vector<std::unique_ptr<std::string>> ordered;
      for (size_t i = 0; i < size; ++i) {
        if (i % 3 == 1) {
          ordered.push_back(std::make_unique<std::string>("not_in_unordered"));
        }
        ordered.push_back(std::make_unique<std::string>(names[i]));
      }
      NameMapper mapper(unordered);
      ASSERT_OK(
          Sorter::Sort(mapper, Sorter::InvalidIndexFn(), ordered, unordered));
      EXPECT_EQ(Values(unordered), names)
          << "size " << size << " trial " << trial;
      EXPECT_EQ(mapper.calls(), ordered.size())
          << "size " << size << " trial " << trial;
    }
  }
}

// An empty unordered container needs no mapping at all.
TEST_F(MappedPtrContainerSorterTest, EmptyUnorderedContainer) {
  std::vector<std::unique_ptr<std::string>> unordered;
  NameMapper mapper(unordered);
  EXPECT_OK(Sorter::Sort(mapper, Sorter::InvalidIndexFn(), ordered_unique_ptrs_,
                         unordered));
  EXPECT_TRUE(unordered.empty());
  EXPECT_EQ(mapper.calls(), 0);
}

// An ordered element may map to a live pointer that is not in the unordered
// container; it is skipped like a null mapping, on both lookup paths.
TEST_F(MappedPtrContainerSorterTest, MappingOutsideTheUnorderedContainer) {
  const std::string outside = "outside";
  for (size_t size : {size_t{4}, 2 * Sorter::kLinearSearchLimit}) {
    const std::vector<std::string> names = Names(size);
    std::vector<std::unique_ptr<std::string>> unordered =
        CreateUniquePtrContainer(names);
    std::reverse(unordered.begin(), unordered.end());
    std::vector<std::unique_ptr<std::string>> ordered =
        CreateUniquePtrContainer(names);
    ordered.insert(ordered.begin() + 1, std::make_unique<std::string>("x"));
    NameMapper mapper(unordered, &outside);
    ASSERT_OK(
        Sorter::Sort(mapper, Sorter::InvalidIndexFn(), ordered, unordered));
    EXPECT_EQ(Values(unordered), names) << "size " << size;
    EXPECT_EQ(mapper.calls(), ordered.size()) << "size " << size;
  }
}

// A container above the linear search limit with one unmapped element takes
// the fallback, which places the element by the unmapped index policy.
TEST_F(MappedPtrContainerSorterTest, LargeContainerWithUnmappedElement) {
  const std::vector<std::string> names = Names(2 * Sorter::kLinearSearchLimit);
  std::vector<std::unique_ptr<std::string>> unordered =
      CreateUniquePtrContainer(names);
  std::reverse(unordered.begin(), unordered.end());
  unordered.insert(unordered.begin() + 7, std::make_unique<std::string>("u0"));
  std::vector<std::unique_ptr<std::string>> ordered =
      CreateUniquePtrContainer(names);
  NameMapper mapper(unordered);
  ASSERT_OK(Sorter::Sort(mapper, Sorter::IndexBeforeMappedElementsFn(), ordered,
                         unordered));
  std::vector<std::string> expected = {"u0"};
  expected.insert(expected.end(), names.begin(), names.end());
  EXPECT_EQ(Values(unordered), expected);
  EXPECT_EQ(mapper.calls(), 2 * ordered.size());
}

TEST_F(MappedPtrContainerSorterTest, InvalidUnmappedIndex) {
  unordered_unique_ptrs_.push_back(std::make_unique<std::string>("u0"));
  auto unmapped_index_fn = [](const std::string* unmapped) -> size_t {
    if (*unmapped == "u0") {
      // There are 4 mapped elements, so index 3 is the highest valid index,
      // (excluding special indices)
      return 4;
    }
    return Sorter::IndexBeforeMappedElementsFn()(unmapped);
  };

  EXPECT_FALSE(Sorter::Sort(MapPtrFn(), unmapped_index_fn, ordered_unique_ptrs_,
                            unordered_unique_ptrs_)
                   .ok());
}

}  // namespace
}  // namespace xla
