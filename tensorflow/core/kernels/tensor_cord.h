/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_TENSOR_CORD_H_
#define TENSORFLOW_CORE_KERNELS_TENSOR_CORD_H_

#include <array>
#include <numeric>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/variant_tensor_data.h"

namespace tensorflow {

typedef void (*CordRepReleaser)(void*);

class TensorCord {
  // A TensorCord keeps a view into some data, and a cleanup method to clean up
  // that data when the TensorCord destructor is called.  Copying a TensorCord
  // increments a reference count to the cleanup method, and so the cleanup
  // method is only called when all copies of the original TensorCord are
  // cleared.
  //
  // Example:
  //
  // const string& s = t.scalar<string>()();
  // TensorCord tc(s, &t);
  // ASSERT_EQ(s, tc.view());
  // TensorCord copy(tc);
  // tc = TensorCord();  // cleanup not called; the reference is held by `copy`.
  // copy = TensorCord();  // cleanup happens now, the reference is destroyed.
  //
  // Another example:
  //
  // void TensorProtoDeleter(void* ptr) {
  //   delete static_cast<TensorProto*>(ptr);
  // }
  //
  // auto p = absl::MakeUnique<TensorProto>(...);
  // absl::string_view content(p->tensor_content());
  // TensorCord tc(content, TensorProtoDeleter, p.release());
  //

 public:
  static constexpr const char kTypeName[] = "tensorflow::TensorCord";

  TensorCord() : chunks_() {}

  ~TensorCord();

  // Args:
  //   `view`: should point to a location in memory that is guaranteed to remain
  //           valid until `releaser` is called.
  //   `releaser`: A callback that will be executed when there are no references
  //               left on `view`.  It will be called via `releaser(memory)`.
  //   `memory`: The argument passed to `releaser` when it is called.
  //
  // You are STRONGLY advised to provide a non-null `releaser`, and a pointer
  // to the underlying data (while ensuring that the data will not be deleted
  // until `releaser(memory)` is called).  Otherwise the TensorCord may
  // outlive the data backing `view`.
  TensorCord(absl::string_view view, CordRepReleaser releaser,
             void* memory = nullptr)
      : chunks_({new CordRep(view, releaser, memory)}) {}

  // Args:
  //   `view`: should point to a location in memory backed by `tensor`,
  //      e.g., `view` is a string_view on a tstring which is an element
  //      of `tensor`.  Furthermore, the associated tstring is not expected
  //      to be modified in such a way that the underlying memory will
  //      be changed after this TensorCord is created.
  TensorCord(absl::string_view view, Tensor* tensor)
      : chunks_({NewCordRepFromTensor(view, tensor)}) {}

  // Disallow construction with empty callback or empty tensor.
  TensorCord(absl::string_view view, std::nullptr_t, void* memory) = delete;
  TensorCord(absl::string_view view, std::nullptr_t) = delete;

  TensorCord(const TensorCord& other);

  TensorCord(TensorCord&& other) noexcept;

  TensorCord& operator=(const TensorCord& other);

  TensorCord& operator=(TensorCord&& other) noexcept;

  void Append(const TensorCord& other);

  void Append(absl::string_view view, CordRepReleaser releaser,
              void* memory = nullptr);

  void Append(absl::string_view view, Tensor* tensor);

  // Disallow Appends with empty callbacks or empty tensors.
  void Append(absl::string_view view, std::nullptr_t, void* memory) = delete;
  void Append(absl::string_view view, std::nullptr_t) = delete;

  size_t size() const;
  bool empty() const { return size() == 0; }

  // NOTE: This performs an expensive copy of the underlying data.
  explicit operator string() const;

  class ChunkIterator {
   public:
    using iterator_category = std::input_iterator_tag;
    using value_type = absl::string_view;
    using difference_type = ptrdiff_t;
    using pointer = const value_type*;
    using reference = value_type;

    ChunkIterator& operator++();

    ChunkIterator operator++(int) {
      ChunkIterator tmp(*this);
      operator++();
      return tmp;
    }

    bool operator==(const ChunkIterator& other) const {
      return (cord_ == other.cord_ && chunk_index_ == other.chunk_index_);
    }

    bool operator!=(const ChunkIterator& other) const {
      return !(*this == other);
    }
    reference operator*() const {
      assert(cord_ != nullptr);
      return view_;
    }
    pointer operator->() const {
      assert(cord_ != nullptr);
      return &view_;
    }

    friend class TensorCord;

   private:
    // Constructs a `begin()` iterator from `cord`.
    explicit ChunkIterator(const TensorCord* cord, int chunk_index);

    const TensorCord* const cord_;
    int chunk_index_;
    absl::string_view view_;
  };

  class ChunkRange {
   public:
    explicit ChunkRange(const TensorCord* cord) : cord_(cord) {}

    ChunkIterator begin() const { return ChunkIterator(cord_, 0); }

    ChunkIterator end() const {
      return ChunkIterator(cord_, cord_->chunks_.size());
    }

   private:
    const TensorCord* cord_;
  };

  // Note that the ordinary caveats of temporary lifetime extension apply:
  //
  //   void Process() {
  //     for (absl::string_view chunk : CordFactory().Chunks()) {
  //       // The temporary Cord returned by CordFactory has been destroyed!
  //     }
  //   }
  ChunkRange Chunks() const { return ChunkRange(this); }

  ChunkIterator chunk_begin() const { return ChunkIterator(this, 0); }

  ChunkIterator chunk_end() const {
    return ChunkIterator(this, chunks_.size());
  }

  static string TypeName() { return kTypeName; }

  string DebugString() const {
    return absl::StrCat("<TensorCord size=", size(), ">");
  }

  void Encode(VariantTensorData* data) const;

  bool Decode(VariantTensorData data);

 private:
  void Cleanup();

  class CordRep : public core::RefCounted {
   public:
    CordRep(absl::string_view view, CordRepReleaser releaser,
            void* arg = nullptr)
        : is_inline_(false), rep_(view, releaser, arg) {}

    // **WARNING** Only use this constructor if
    //    view.size() < CordRep::kMaxInlineSize.
    explicit CordRep(absl::string_view view) : is_inline_(true), rep_(view) {}

    ~CordRep() override;

    absl::string_view view() const {
      if (is_inline_) {
        return absl::string_view(
            rep_.internal.data() + 1,
            *reinterpret_cast<const uint8*>(rep_.internal.data()));
      } else {
        return rep_.external.view;
      }
    }

   private:
    friend class TensorCord;

    struct ExternalRep {
      absl::string_view view;
      CordRepReleaser releaser;
      void* arg;

      ExternalRep(absl::string_view view_, CordRepReleaser releaser_,
                  void* arg_)
          : view(view_), releaser(releaser_), arg(arg_) {}
    };

    // We save the size in the first byte, so subtract 1.
    static constexpr int kMaxInlineSize = sizeof(ExternalRep) - 1;
    static_assert(kMaxInlineSize < 255,
                  "Cannot store size of InlineRep in a single byte.");

    // The first byte stores the size as a uint8.  The rest of the bytes are the
    // string itself.
    using InlineRep = std::array<char, sizeof(ExternalRep)>;

    // Member variables.
    const bool is_inline_;
    const union _rep_union {
      InlineRep internal;
      ExternalRep external;

      _rep_union(absl::string_view view, CordRepReleaser releaser, void* arg)
          : external(view, releaser, arg) {}

      explicit _rep_union(absl::string_view view) {
        DCHECK_LT(view.size(), kMaxInlineSize);
        *reinterpret_cast<uint8*>(internal.data()) = view.size();
        std::memcpy(static_cast<char*>(internal.data() + 1), view.data(),
                    view.size());
      }
    } rep_;
  };

  static TensorBuffer* TensorBufWithRef(Tensor* tensor);
  static void TensorBufReleaser(void* tensor_buffer);
  static void StringReleaser(void* str_ptr);
  static CordRep* NewCordRepFromTensor(absl::string_view view, Tensor* tensor);

  absl::InlinedVector<CordRep*, 2> chunks_;
};

inline TensorCord::TensorCord(const TensorCord& other)
    : chunks_(other.chunks_) {
  for (auto* rep : chunks_) {
    rep->Ref();
  }
}

inline TensorCord::TensorCord(TensorCord&& other) noexcept
    : chunks_(std::move(other.chunks_)) {
  other.chunks_.clear();
}

inline TensorCord& TensorCord::operator=(const TensorCord& other) {
  Cleanup();
  chunks_ = other.chunks_;
  for (auto* rep : chunks_) {
    rep->Ref();
  }
  return *this;
}

inline TensorCord& TensorCord::operator=(TensorCord&& other) noexcept {
  Cleanup();
  std::swap(chunks_, other.chunks_);
  return *this;
}

inline void TensorCord::Append(const TensorCord& other) {
  for (auto* rep : other.chunks_) {
    chunks_.push_back(rep);
    rep->Ref();
  }
}

inline void TensorCord::Append(absl::string_view view, CordRepReleaser releaser,
                               void* memory) {
  chunks_.push_back(new CordRep(view, releaser, memory));
}

inline void TensorCord::Append(absl::string_view view, Tensor* tensor) {
  chunks_.push_back(NewCordRepFromTensor(view, tensor));
}

inline size_t TensorCord::size() const {
  return (chunks_.empty())
             ? 0
             : std::accumulate(chunk_begin(), chunk_end(), 0,
                               [](size_t acc, absl::string_view b) {
                                 return acc + b.size();
                               });
}

inline TensorCord::ChunkIterator& TensorCord::ChunkIterator::operator++() {
  assert(cord_ != nullptr);
  assert(chunk_index_ < cord_->chunks_.size());
  chunk_index_ += 1;
  if (chunk_index_ != cord_->chunks_.size()) {
    view_ = cord_->chunks_[chunk_index_]->view();
  }
  return *this;
}

inline TensorCord::ChunkIterator::ChunkIterator(const TensorCord* cord,
                                                int index)
    : cord_(cord), chunk_index_(index) {
  if (index < cord_->chunks_.size()) {
    view_ = cord_->chunks_[index]->view();
  }
}

inline TensorCord::CordRep* TensorCord::NewCordRepFromTensor(
    absl::string_view view, Tensor* tensor) {
  if (view.size() <= TensorCord::CordRep::kMaxInlineSize) {
    return new CordRep(view);
  } else {
    return new CordRep(view, &TensorBufReleaser, TensorBufWithRef(tensor));
  }
}

inline void TensorCord::Cleanup() {
  if (chunks_.empty()) return;
  for (auto* rep : chunks_) {
    rep->Unref();
  }
  chunks_.clear();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TENSOR_CORD_H_
