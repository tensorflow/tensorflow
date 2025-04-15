/* Copyright 2022 The OpenXLA Authors. All Rights Reserved.

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

#ifndef XLA_MLIR_TOOLS_MLIR_INTERPRETER_FRAMEWORK_TENSOR_OR_MEMREF_H_
#define XLA_MLIR_TOOLS_MLIR_INTERPRETER_FRAMEWORK_TENSOR_OR_MEMREF_H_

#include <math.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace interpreter {

template <typename T>
bool IsEqual(T a, T b) {
  return a == b;
}

// TODO(jreiffers): Replace ifndef with a command line flag.
#ifndef MLIR_INTERPRETER_COMPARE_DOUBLES_EXACT
// Compare double precision float with a small tolerance, because complex
// computations in the interpreter don't always produce the exact same result.
template <>
inline bool IsEqual(double a, double b) {
  if (isinf(a) || isinf(b)) {
    return a == b;
  }

  return fabs(a - b) < 1e-14;
}

template <>
inline bool IsEqual(std::complex<double> a, std::complex<double> b) {
  return IsEqual(a.real(), b.real()) && IsEqual(a.imag(), b.imag());
}
#endif

// Represents a view into a physical buffer.
struct BufferView {
  int64_t offset;
  llvm::SmallVector<int64_t> sizes;    // [10, 11, 12]
  llvm::SmallVector<int64_t> strides;  // [132, 12, 1]
  // Number of vector element dimensions in the tensor. nullopt if this is a
  // vector itself (is_vector is set). {0} if this is a tensor of a unit vector.
  std::optional<int64_t> num_vector_dims = std::nullopt;
  bool is_vector = false;

  int64_t num_dimensions() const {
    return sizes.size() - num_vector_dims.value_or(0);
  }

  // Removes the dimension from the view. If you need to keep it, use the
  // overload below with dim_size = 1.
  LogicalResult Slice(int64_t dim_index, int64_t dim_offset);
  LogicalResult Slice(int64_t dim_index, int64_t dim_offset, int64_t dim_size,
                      int64_t dim_stride = 1);
  LogicalResult Subview(ArrayRef<int64_t> subview_offsets,
                        ArrayRef<int64_t> subview_sizes,
                        ArrayRef<int64_t> subview_strides);
  int64_t GetNumElements(bool include_vector_dimsms = false) const;

  class LogicalIndexView {
   public:
    class Iterator {
     public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = llvm::SmallVector<int64_t>;
      using difference_type = std::ptrdiff_t;
      using pointer = llvm::SmallVector<int64_t>*;
      using reference = llvm::SmallVector<int64_t>&;

      const llvm::SmallVector<int64_t>& operator*() const {
        return view_indices_;
      }
      const llvm::SmallVector<int64_t>* operator->() const {
        return &view_indices_;
      }

      Iterator& operator++() {
        auto index_it = view_indices_.rbegin();
        auto size_it = view_->sizes.rbegin();
        if (!include_vector_dims_) {
          std::advance(size_it, view_->num_vector_dims.value_or(0));
        }

        for (auto e = view_indices_.rend(); index_it != e;
             ++index_it, ++size_it) {
          ++*index_it;
          if (*index_it < *size_it) {
            return *this;
          }
          *index_it = 0;
        }

        view_indices_.clear();
        view_indices_.push_back(-1);
        return *this;
      }

      Iterator operator++(int) {
        auto tmp = *this;
        ++(*this);
        return tmp;
      }

      bool operator==(const Iterator& other) const {
        return view_indices_ == other.view_indices_;
      }

      bool operator!=(const Iterator& other) const { return !(*this == other); }

     private:
      friend class LogicalIndexView;

      Iterator(const BufferView* view, llvm::SmallVector<int64_t> indices,
               bool include_vector_dims)
          : view_(view),
            view_indices_(std::move(indices)),
            include_vector_dims_(include_vector_dims) {}

      const BufferView* view_;
      llvm::SmallVector<int64_t> view_indices_;
      bool include_vector_dims_;
    };

    Iterator begin() const {
      if (view_->GetNumElements() == 0) return end();
      return {
          view_,
          llvm::SmallVector<int64_t>(
              view_->num_dimensions() +
              (include_vector_dims_ ? view_->num_vector_dims.value_or(0) : 0)),
          include_vector_dims_};
    }
    Iterator end() const { return {view_, {-1}, false}; }

   private:
    friend class BufferView;

    LogicalIndexView(const BufferView* view, bool include_vector_dims)
        : view_(view), include_vector_dims_(include_vector_dims) {}

    const BufferView* view_;
    bool include_vector_dims_;
  };

  // Returns nullopt if the index is out of bounds.
  std::optional<int64_t> GetPhysicalIndex(
      llvm::ArrayRef<int64_t> view_indices) const;
  LogicalIndexView Indices(bool include_vector_dims = false) const {
    return LogicalIndexView{this, include_vector_dims};
  }
  // Returns the stride resulting from collapsing the given dimensions, if
  // possible.
  std::optional<int64_t> GetCollapsedStride(llvm::ArrayRef<int64_t> dims) const;

  bool InBounds(llvm::ArrayRef<int64_t> view_indices) const;
  static SmallVector<int64_t> GetDefaultStrides(ArrayRef<int64_t> sizes);
  static SmallVector<int64_t> GetStridesForLayout(ArrayRef<int64_t> sizes,
                                                  ArrayRef<int64_t> layout);
};

// Backing for a TensorOrMemref.
class Buffer {
 private:
  struct Dummy {};

 public:
  template <typename T>
  static std::shared_ptr<Buffer> Allocate(size_t size) {
    return std::make_shared<Buffer>(Dummy{}, size, sizeof(T));
  }

  char* at(std::optional<int64_t> idx, int64_t element_size) {
    auto byte_offset = GetByteOffset(idx, element_size);
    if (!byte_offset) {
      return &storage_.data()[0];
    }
    return &storage_.data()[*byte_offset];
  }

  const char* at(std::optional<int64_t> idx, int64_t element_size) const {
    auto byte_offset = GetByteOffset(idx, element_size);
    if (!byte_offset) {
      return &storage_.data()[0];
    }
    return &storage_.data()[*byte_offset];
  }

  Buffer(Dummy, size_t num_elements, size_t element_size)
      : storage_(num_elements * element_size) {}

  int64_t GetByteSize() const { return storage_.size(); }

  void Deallocate(mlir::Operation* op) {
    if (is_alloca_) {
      SetFailure("deallocated stack buffer");
    } else if (freed_by_ != nullptr) {
      std::string failure;
      llvm::raw_string_ostream os(failure);
      os << "double-free\n";
      os << "  Note: allocated by " << *allocated_by_ << "\n";
      os << "  Note: previously freed by " << *freed_by_ << "\n";
      SetFailure(failure);
    } else {
      freed_by_ = op;
    }
  }

  bool Deallocated() const { return freed_by_ != nullptr; }
  mlir::Operation* FreedByOp() const { return freed_by_; }
  void SetAllocatedBy(mlir::Operation* allocated_by) {
    this->allocated_by_ = allocated_by;
  }

  void SetFailure(llvm::StringRef failure) const {
    this->failure_ = failure.str();
  }
  llvm::StringRef GetFailure() const { return failure_; }

  void SetIsAlloca() { is_alloca_ = true; }

 private:
  std::optional<size_t> GetByteOffset(std::optional<int64_t> idx,
                                      int64_t element_size) const {
    if (!idx) {
      SetFailure("out of bounds access");
      return std::nullopt;
    }

    if (freed_by_ != nullptr) {
      std::string failure;
      llvm::raw_string_ostream os(failure);
      os << "use-after-free\n";
      os << "  Note: allocated by " << *allocated_by_ << "\n";
      os << "  Note: previously freed by " << *freed_by_ << "\n";
      SetFailure(failure);
      return std::nullopt;
    }

    return *idx * element_size;
  }

  llvm::SmallVector<char> storage_;
  mlir::Operation* freed_by_ = nullptr;
  mlir::Operation* allocated_by_ = nullptr;
  bool is_alloca_ = false;
  mutable std::string failure_;
};

template <typename T>
struct TensorOrMemref {
  using element_type = T;

  static TensorOrMemref<T> Empty(ArrayRef<int64_t> sizes,
                                 ArrayRef<int64_t> layout = {}) {
    BufferView dummy{0, SmallVector<int64_t>(sizes), {}};
    return EmptyLike(dummy, layout);
  }

  static TensorOrMemref<T> EmptyLike(const BufferView& view,
                                     ArrayRef<int64_t> layout = {}) {
    BufferView new_view = view;
    new_view.offset = 0;
    new_view.strides = BufferView::GetStridesForLayout(view.sizes, layout);
    return {Buffer::Allocate<T>(view.GetNumElements(true)), new_view};
  }

  TensorOrMemref<T> Clone(ArrayRef<int64_t> layout = {}) const {
    auto out = EmptyLike(view, layout);
    for (auto [src_index, dst_index] :
         llvm::zip(view.Indices(true), out.view.Indices(true))) {
      out.at(dst_index) = at(src_index);
    }
    return out;
  }

  const T& at(ArrayRef<int64_t> indices) const {
    return *reinterpret_cast<const T*>(
        buffer->at(view.GetPhysicalIndex(indices), sizeof(T)));
  }

  T& at(ArrayRef<int64_t> indices) {
    return *reinterpret_cast<T*>(
        buffer->at(view.GetPhysicalIndex(indices), sizeof(T)));
  }

  TensorOrMemref VectorAt(ArrayRef<int64_t> indices) const {
    auto offset = view.GetPhysicalIndex(indices);
    BufferView subview;
    subview.strides = {view.strides.begin() + view.num_dimensions(),
                       view.strides.end()};
    subview.sizes = {view.sizes.begin() + view.num_dimensions(),
                     view.sizes.end()};
    if (offset) {
      subview.offset = *offset;
    } else {
      buffer->SetFailure("out of bounds access");
    }
    subview.is_vector = true;
    subview.num_vector_dims = std::nullopt;
    return {buffer, subview};
  }

  bool operator==(const TensorOrMemref& other) const {
    if (buffer->Deallocated() || other.buffer->Deallocated()) return false;
    if (other.view.sizes != view.sizes) return false;
    if (other.view.num_vector_dims != view.num_vector_dims) return false;
    for (const auto& indices : view.Indices(true)) {
      // Treat NaNs as equal.
      if constexpr (std::is_floating_point_v<T>) {
        bool thisnan = isnan(at(indices));
        bool othernan = isnan(other.at(indices));
        if (thisnan || othernan) {
          if (thisnan && othernan) continue;
          return false;
        }
      }
      if (!IsEqual(at(indices), other.at(indices))) return false;
    }
    return true;
  }

  std::shared_ptr<Buffer> buffer;
  BufferView view;
};

template <typename T>
struct is_tensor_or_memref : std::false_type {};  // NOLINT

template <typename T>
struct is_tensor_or_memref<TensorOrMemref<T>> : std::true_type {};  // NOLINT

template <typename T>
inline constexpr bool is_tensor_or_memref_v =  // NOLINT
    is_tensor_or_memref<std::decay_t<T>>::value;

}  // namespace interpreter
}  // namespace mlir

#endif  // XLA_MLIR_TOOLS_MLIR_INTERPRETER_FRAMEWORK_TENSOR_OR_MEMREF_H_
