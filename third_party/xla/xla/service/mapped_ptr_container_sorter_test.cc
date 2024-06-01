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

#include <cstddef>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/bind_front.h"
#include "absl/log/log.h"
#include "xla/test.h"
#include "tsl/lib/core/status_test_util.h"

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
  TF_EXPECT_OK(Sorter::Sort(MapPtrFn(), Sorter::InvalidIndexFn(),
                            ordered_unique_ptrs_, unordered_unique_ptrs_));
  EXPECT_THAT(
      unordered_unique_ptrs_,
      ElementsAre(Pointee(std::string("m0")), Pointee(std::string("m1")),
                  Pointee(std::string("m2")), Pointee(std::string("m3"))));
}

TEST_F(MappedPtrContainerSorterTest, RawPtrs) {
  TF_EXPECT_OK(Sorter::Sort(MapPtrFn(), Sorter::InvalidIndexFn(),
                            ordered_raw_ptrs_, unordered_raw_ptrs_));
  EXPECT_THAT(
      unordered_raw_ptrs_,
      ElementsAre(Pointee(std::string("m0")), Pointee(std::string("m1")),
                  Pointee(std::string("m2")), Pointee(std::string("m3"))));
}

TEST_F(MappedPtrContainerSorterTest, ConstRawPtrs) {
  TF_EXPECT_OK(Sorter::Sort(MapPtrFn(), Sorter::InvalidIndexFn(),
                            ordered_const_raw_ptrs_,
                            unordered_const_raw_ptrs_));
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

  TF_EXPECT_OK(Sorter::Sort(MapPtrFn(), Sorter::InvalidIndexFn(), ordered_ptrs,
                            unordered_unique_ptrs_));
  EXPECT_THAT(
      unordered_unique_ptrs_,
      ElementsAre(Pointee(std::string("m0")), Pointee(std::string("m1")),
                  Pointee(std::string("m2")), Pointee(std::string("m3"))));
}

TEST_F(MappedPtrContainerSorterTest, WithUnmappedPtrsAfterMappedPtrs) {
  AddUnmappedElementsToUnorderedUniquePtrs();

  TF_EXPECT_OK(Sorter::Sort(MapPtrFn(), Sorter::IndexAfterMappedElementsFn(),
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

  TF_EXPECT_OK(Sorter::Sort(MapPtrFn(), Sorter::IndexBeforeMappedElementsFn(),
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

  TF_EXPECT_OK(Sorter::Sort(MapPtrFn(), unmapped_ptr_index,
                            ordered_unique_ptrs_, unordered_unique_ptrs_));
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

  TF_EXPECT_OK(Sorter::Sort(MapPtrFn(), Sorter::IndexBeforeMappedElementsFn(),
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

  TF_EXPECT_OK(Sorter::Sort(MapPtrFn(), Sorter::IndexBeforeMappedElementsFn(),
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
