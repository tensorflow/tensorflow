/* Copyright 2024 The OpenXLA Authors.

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

#include <cstdint>
#include <memory>
#include <utility>
#include <variant>

#include <gtest/gtest.h>
// #include "xla/hlo/ir/hlo_module.h"
// #include "tsl/platform/test.h"

#include "xla/hlo/utils/copy_on_write.h"

namespace xla {
namespace {

using std::make_unique;
using std::shared_ptr;
using std::unique_ptr;

class Config {
 public:
  explicit Config(int8_t value) : copy_num_(0), value_(value) {};
  Config(const Config& other)
      : copy_num_(other.copy_num_ + 1), value_(other.value_) {}
  Config(Config&& other) : copy_num_(other.copy_num_), value_(other.value_) {
    other.value_ = -1;
  }
  ~Config() = default;

  Config& operator=(const Config& other) {
    if (this != &other) {
      copy_num_ = other.copy_num_ + 1;
      value_ = other.value_;
    }
    return *this;
  }

  Config& operator=(Config&& other) {
    if (this != &other) {
      copy_num_ = other.copy_num_;
      other.copy_num_ = -1;
      value_ = other.value_;
    }
    return *this;
  }

  int8_t copy_num_;
  int8_t value_;
};

// Holding class which ultimately uses CopyOnWrite. Modeled after HloModule.
class Holder {
 public:
  explicit Holder(Config config)
      : Holder(make_unique<Config>(std::move(config))) {}

  explicit Holder(
      std::variant<unique_ptr<Config>, shared_ptr<const Config>> config)
      : config_(std::move(config)) {}

  Config& mutable_config() { return config_.get_mutable(); }
  const Config& config() const { return config_.get(); }
  void set_config(Config config) { config_.set(std::move(config)); }

  const std::shared_ptr<const Config>& shared_config() const {
    return config_.FreezeAndShare();
  }

  CopyOnWrite<Config> config_;
};

template <typename T>
void setExp(T&& value, unique_ptr<T>& ptr) {
  *ptr = std::forward<T>(value);
}

TEST(CopyOnWriteTest, Exp) {
  auto pc1 = make_unique<Config>(7);
  Config c2(10);
  setExp(std::move(c2), pc1);
  EXPECT_EQ(pc1->copy_num_, 0);
  EXPECT_EQ(pc1->value_, 10);
}

// assure that proper copy and move constructors behavior.
TEST(CopyOnWriteTest, PreconditionConstructors) {
  Config orig(7);
  Config copy = orig;
  EXPECT_EQ(orig.copy_num_, 0);
  EXPECT_EQ(copy.copy_num_, 1);
  EXPECT_EQ(orig.value_, copy.value_);
  Config dest = std::move(copy);
  EXPECT_EQ(dest.copy_num_, 1);
  EXPECT_EQ(dest.value_, 7);
  EXPECT_EQ(copy.value_, -1);
}

// Intended usage and the most common case.
TEST(CopyOnWriteTest, Basic) {
  Config input(7);
  // move is required to avoid copy.
  CopyOnWrite<Config> orig(make_unique<Config>(std::move(input)));
  EXPECT_EQ(orig.get().copy_num_, 0) << "Original object. No copies";
  EXPECT_EQ(orig.get().value_, 7);

  // Moves the object from unique_ptr to shared_ptr and shares with second
  // container.
  CopyOnWrite<Config> share1(orig.FreezeAndShare());
  EXPECT_EQ(orig.get().copy_num_, 0) << "No copies.";
  EXPECT_TRUE(&orig.get() == &share1.get())
      << "Both containers refer to the same object.";

  orig.get_mutable().value_ = 10;  // Mutate the original.

  EXPECT_TRUE(&orig.get() != &share1.get())
      << "Containers diverge on mutation.";
  EXPECT_EQ(orig.get().value_, 10);
  EXPECT_EQ(share1.get().value_, 7)
      << "Second container is not affected by mutation.";
  EXPECT_EQ(share1.get().copy_num_, 0)
      << "Second container refers to the original object.";
  EXPECT_EQ(orig.get().copy_num_, 1) << "Orig refers to a new copy.";
}

// Common pattern across XLA which creates a copy.
TEST(CopyOnWriteTest, GetModifySet) {
  Config input(7);
  CopyOnWrite<Config> orig(make_unique<Config>(std::move(input)));
  Config temp = orig.get();  // Common across XLA. Creates a copy.
  temp.value_ = 10;
  orig.set(std::move(temp));
  EXPECT_EQ(orig.get().value_, 10);
  EXPECT_EQ(orig.get().copy_num_, 1) << "New copy is created.";
}

// Common pattern of using HloModule across XLA which creates 2 copies.
TEST(CopyOnWriteTest, HolderGetModifySet) {
  Config orig(7);
  Holder holder(orig);
  EXPECT_EQ(holder.config().copy_num_, 1)
      << "Passed by value. First copy created.";
  // holder.config().value_ = 10; // GOOD: not possible due to const.
  Config config = holder.config();
  EXPECT_EQ(config.copy_num_, 2) << "Second copy created.";
  config.value_ = 10;
  // Third copy created when passing by value.
  holder.set_config(config);
  EXPECT_EQ(holder.config().value_, 10);
  EXPECT_EQ(holder.config().copy_num_, 3) << "Third copy created.";
}

}  // namespace
}  // namespace xla
