/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/transforms/resource_elementwise_fold.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "Eigen/Core"  // from @eigen_archive
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/DialectResourceBlobManager.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/resource_constant_fold_utils.h"

namespace mlir::TFL {
namespace {

// Counts, for every resource buffer, how many attributes in the module refer to
// it. A buffer is only mutated in place when its count is one, and only
// released once its count drops to zero.
class ResourceRefCounts {
 public:
  explicit ResourceRefCounts(ModuleOp module) {
    module.walk([&](mlir::Operation* op) {
      for (NamedAttribute named_attr : op->getAttrs()) {
        Visit(named_attr.getValue(), /*delta=*/1, /*dead=*/nullptr);
      }
    });
  }

  int64_t Get(DenseResourceElementsAttr attr) const {
    auto it = counts_.find(GetResourceKey(attr));
    return it == counts_.end() ? 0 : it->second;
  }

  void Add(DenseResourceElementsAttr attr) { ++counts_[GetResourceKey(attr)]; }

  // Drops the references held by `op`'s attributes. Resources that are no
  // longer referenced by any attribute are appended to `dead`.
  void Remove(mlir::Operation* op,
              llvm::SmallVectorImpl<DenseResourceElementsAttr>& dead) {
    for (NamedAttribute named_attr : op->getAttrs()) {
      Visit(named_attr.getValue(), /*delta=*/-1, &dead);
    }
  }

 private:
  void Visit(Attribute attr, int64_t delta,
             llvm::SmallVectorImpl<DenseResourceElementsAttr>* dead) {
    if (auto res_attr = mlir::dyn_cast<DenseResourceElementsAttr>(attr)) {
      int64_t& count = counts_[GetResourceKey(res_attr)];
      count += delta;
      if (count == 0 && dead) dead->push_back(res_attr);
    } else if (auto array_attr = mlir::dyn_cast<ArrayAttr>(attr)) {
      for (Attribute element : array_attr) Visit(element, delta, dead);
    } else if (auto dict_attr = mlir::dyn_cast<DictionaryAttr>(attr)) {
      for (NamedAttribute element : dict_attr) {
        Visit(element.getValue(), delta, dead);
      }
    }
  }

  absl::flat_hash_map<std::string, int64_t> counts_;
};

// Read-only view of the data of a constant operand.
struct ConstantView {
  mlir::Operation* const_op = nullptr;
  mlir::ShapedType type;
  // Null for DenseElementsAttr constants.
  DenseResourceElementsAttr resource;
  AsmResourceBlob* blob = nullptr;
  // Bytes per element in the view. Booleans use one byte per element.
  int64_t element_bytes = 0;
  // Whether all elements share the single element at `Data()`.
  bool splat = false;
  // Number of bytes stored for this constant.
  int64_t stored_bytes = 0;

  const char* Data() const { return owned.empty() ? data : owned.data(); }

  const char* data = nullptr;
  // Unpacked storage for DenseElementsAttr booleans, which are bit-packed.
  std::vector<char> owned;
};

std::optional<ConstantView> GetConstantView(mlir::Value value) {
  ElementsAttr attr;
  if (!matchPattern(value, m_Constant(&attr))) return std::nullopt;
  auto type = mlir::dyn_cast<mlir::ShapedType>(attr.getType());
  // Rejects e.g. quantized constants whose storage type differs from the value
  // type.
  if (!type || !type.hasStaticShape() || type != value.getType()) {
    return std::nullopt;
  }
  Type element_type = type.getElementType();
  if (!element_type.isIntOrFloat()) return std::nullopt;
  const bool is_bool = element_type.isInteger(1);
  const int64_t bit_width = element_type.getIntOrFloatBitWidth();
  if (!is_bool && bit_width % 8 != 0) return std::nullopt;

  ConstantView view;
  view.const_op = value.getDefiningOp();
  if (!view.const_op || view.const_op->getNumResults() != 1) {
    return std::nullopt;
  }
  view.type = type;
  view.element_bytes = is_bool ? 1 : bit_width / 8;
  const int64_t num_elements = type.getNumElements();
  const int64_t full_bytes = num_elements * view.element_bytes;

  if (auto resource = mlir::dyn_cast<DenseResourceElementsAttr>(attr)) {
    AsmResourceBlob* blob = GetResourceBlob(resource);
    if (!blob) return std::nullopt;
    llvm::ArrayRef<char> data = blob->getData();
    if (static_cast<int64_t>(data.size()) != full_bytes) return std::nullopt;
    view.resource = resource;
    view.blob = blob;
    view.data = data.data();
    view.stored_bytes = full_bytes;
    return view;
  }

  auto dense = mlir::dyn_cast<DenseElementsAttr>(attr);
  if (!dense) return std::nullopt;
  view.splat = dense.isSplat();
  if (is_bool) {
    if (view.splat) {
      view.owned.push_back(dense.getSplatValue<bool>() ? 1 : 0);
    } else {
      view.owned.reserve(num_elements);
      for (bool b : dense.getValues<bool>()) view.owned.push_back(b ? 1 : 0);
    }
  } else {
    llvm::ArrayRef<char> raw = dense.getRawData();
    const int64_t expected = view.splat ? view.element_bytes : full_bytes;
    if (static_cast<int64_t>(raw.size()) != expected) return std::nullopt;
    view.data = raw.data();
  }
  view.stored_bytes = view.splat ? view.element_bytes : full_bytes;
  return view;
}

//===----------------------------------------------------------------------===//
// Element access helpers.
//===----------------------------------------------------------------------===//

enum class FloatKind { kF32, kF16, kBF16 };

std::optional<FloatKind> GetFloatKind(Type type) {
  if (type.isF32()) return FloatKind::kF32;
  if (type.isF16()) return FloatKind::kF16;
  if (type.isBF16()) return FloatKind::kBF16;
  return std::nullopt;
}

// Compile-time element access for each supported float type. All arithmetic is
// carried out in f32.
template <FloatKind K>
struct FloatTraits;

template <>
struct FloatTraits<FloatKind::kF32> {
  static constexpr int64_t kBytes = 4;
  static float Load(const char* ptr) {
    float value;
    std::memcpy(&value, ptr, sizeof(value));
    return value;
  }
  static void Store(char* ptr, float value) {
    std::memcpy(ptr, &value, sizeof(value));
  }
};

template <>
struct FloatTraits<FloatKind::kF16> {
  static constexpr int64_t kBytes = 2;
  static float Load(const char* ptr) {
    uint16_t bits;
    std::memcpy(&bits, ptr, sizeof(bits));
    return static_cast<float>(Eigen::numext::bit_cast<Eigen::half>(bits));
  }
  static void Store(char* ptr, float value) {
    const uint16_t bits = Eigen::numext::bit_cast<uint16_t>(Eigen::half(value));
    std::memcpy(ptr, &bits, sizeof(bits));
  }
};

template <>
struct FloatTraits<FloatKind::kBF16> {
  static constexpr int64_t kBytes = 2;
  static float Load(const char* ptr) {
    uint16_t bits;
    std::memcpy(&bits, ptr, sizeof(bits));
    return static_cast<float>(Eigen::numext::bit_cast<Eigen::bfloat16>(bits));
  }
  static void Store(char* ptr, float value) {
    const uint16_t bits =
        Eigen::numext::bit_cast<uint16_t>(Eigen::bfloat16(value));
    std::memcpy(ptr, &bits, sizeof(bits));
  }
};

// Invokes `fn(FloatTraits<K>{})` for the runtime `kind`, so that the body is
// instantiated once per element type and the per-element accesses have no
// type dispatch.
template <typename Fn>
void DispatchFloatKind(FloatKind kind, Fn&& fn) {
  switch (kind) {
    case FloatKind::kF32:
      fn(FloatTraits<FloatKind::kF32>{});
      return;
    case FloatKind::kF16:
      fn(FloatTraits<FloatKind::kF16>{});
      return;
    case FloatKind::kBF16:
      fn(FloatTraits<FloatKind::kBF16>{});
      return;
  }
}

// Returns the element strides that map an index of `out_shape` to an index of
// an operand of `shape` under numpy-style broadcasting.
std::optional<SmallVector<int64_t>> GetBroadcastStrides(
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> out_shape) {
  if (shape.size() > out_shape.size()) return std::nullopt;
  SmallVector<int64_t> strides(out_shape.size(), 0);
  const int64_t rank_offset = out_shape.size() - shape.size();
  int64_t stride = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    const int64_t out_dim = out_shape[i + rank_offset];
    if (shape[i] == out_dim) {
      strides[i + rank_offset] = stride;
    } else if (shape[i] != 1) {
      return std::nullopt;
    }
    stride *= shape[i];
  }
  return strides;
}

std::optional<SmallVector<int64_t>> GetOperandStrides(
    const ConstantView& view, llvm::ArrayRef<int64_t> out_shape) {
  if (view.splat) return SmallVector<int64_t>(out_shape.size(), 0);
  return GetBroadcastStrides(view.type.getShape(), out_shape);
}

template <size_t N>
using OperandStrides = std::array<SmallVector<int64_t>, N>;

template <size_t N>
std::optional<OperandStrides<N>> GetAllOperandStrides(
    llvm::ArrayRef<ConstantView> views, llvm::ArrayRef<int64_t> out_shape) {
  if (views.size() != N) return std::nullopt;
  OperandStrides<N> strides;
  for (size_t k = 0; k < N; ++k) {
    std::optional<SmallVector<int64_t>> s =
        GetOperandStrides(views[k], out_shape);
    if (!s) return std::nullopt;
    strides[k] = std::move(*s);
  }
  return strides;
}

// Calls `fn(out_index, operand_indices)` for every element of `out_shape` in
// row-major order, where `operand_indices[k]` is the element index of operand
// `k` under `strides[k]`.
template <size_t N, typename Fn>
void ForEachElement(llvm::ArrayRef<int64_t> out_shape,
                    const OperandStrides<N>& strides, Fn&& fn) {
  const int64_t rank = out_shape.size();
  int64_t num_elements = 1;
  for (int64_t dim : out_shape) num_elements *= dim;
  if (num_elements == 0) return;

  std::array<int64_t, N> indices{};
  if (rank == 0) {
    fn(int64_t{0}, indices);
    return;
  }

  // Fast path: every operand is either laid out like the output (step 1) or a
  // scalar (step 0), so the whole output is a single flat row.
  SmallVector<int64_t> out_strides(rank, 1);
  for (int64_t d = rank - 2; d >= 0; --d) {
    out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
  }
  std::array<int64_t, N> steps{};
  bool flat = true;
  for (size_t k = 0; k < N && flat; ++k) {
    bool contiguous = true;
    bool scalar = true;
    for (int64_t d = 0; d < rank; ++d) {
      if (out_shape[d] == 1) continue;
      if (strides[k][d] != 0) scalar = false;
      if (strides[k][d] != out_strides[d]) contiguous = false;
    }
    steps[k] = contiguous ? 1 : 0;
    flat = contiguous || scalar;
  }
  if (flat) {
    for (int64_t i = 0; i < num_elements; ++i) {
      for (size_t k = 0; k < N; ++k) indices[k] = i * steps[k];
      fn(i, indices);
    }
    return;
  }

  // General broadcasting: iterate the innermost dimension in a tight loop and
  // advance the outer dimensions once per row.
  const int64_t inner = out_shape[rank - 1];
  std::array<int64_t, N> inner_strides;
  for (size_t k = 0; k < N; ++k) inner_strides[k] = strides[k][rank - 1];
  std::array<int64_t, N> row_base{};
  SmallVector<int64_t> counter(rank - 1, 0);
  const int64_t num_rows = num_elements / inner;
  for (int64_t row = 0; row < num_rows; ++row) {
    indices = row_base;
    const int64_t row_offset = row * inner;
    for (int64_t j = 0; j < inner; ++j) {
      fn(row_offset + j, indices);
      for (size_t k = 0; k < N; ++k) indices[k] += inner_strides[k];
    }
    for (int64_t d = rank - 2; d >= 0; --d) {
      ++counter[d];
      for (size_t k = 0; k < N; ++k) row_base[k] += strides[k][d];
      if (counter[d] < out_shape[d]) break;
      for (size_t k = 0; k < N; ++k) {
        row_base[k] -= strides[k][d] * out_shape[d];
      }
      counter[d] = 0;
    }
  }
}

enum class Activation { kNone, kRelu, kRelu6, kReluN1To1 };

std::optional<Activation> ParseActivation(StringRef name) {
  if (name == "NONE") return Activation::kNone;
  if (name == "RELU") return Activation::kRelu;
  if (name == "RELU6") return Activation::kRelu6;
  if (name == "RELU_N1_TO_1") return Activation::kReluN1To1;
  return std::nullopt;
}

// Clamps the `num_elements` values at `out` according to `activation`. The
// clamp bounds are exactly representable in every supported type, so clamping
// after rounding equals clamping before rounding.
template <typename T>
void ApplyActivation(Activation activation, char* out, int64_t num_elements) {
  float lo, hi;
  switch (activation) {
    case Activation::kNone:
      return;
    case Activation::kRelu:
      lo = 0.0f;
      hi = std::numeric_limits<float>::infinity();
      break;
    case Activation::kRelu6:
      lo = 0.0f;
      hi = 6.0f;
      break;
    case Activation::kReluN1To1:
      lo = -1.0f;
      hi = 1.0f;
      break;
  }
  for (int64_t i = 0; i < num_elements; ++i) {
    char* ptr = out + i * T::kBytes;
    const float value = T::Load(ptr);
    T::Store(ptr, value < lo ? lo : (value > hi ? hi : value));
  }
}

//===----------------------------------------------------------------------===//
// Per-op folding plans.
//===----------------------------------------------------------------------===//

// Describes how to compute the folded result of an op into a raw buffer.
struct FoldPlan {
  // Whether output element `i` only depends on element `i` of every operand
  // whose type equals the result type. Such an operand's buffer may be reused
  // for the result.
  bool elementwise = false;
  // Invoked once per fold; the per-element work inside is fully specialized.
  std::function<void(char* out)> compute;
};

// The `Visit*Fn` helpers call `visitor(fn)` with a concrete (non type-erased)
// functor implementing `op`, and return whether `op` is of the visited kind.

template <typename Visitor>
bool VisitUnaryFn(mlir::Operation* op, Visitor&& visitor) {
  if (isa<ExpOp>(op)) {
    visitor([](float x) { return std::exp(x); });
  } else if (isa<LogOp>(op)) {
    visitor([](float x) { return std::log(x); });
  } else if (isa<SinOp>(op)) {
    visitor([](float x) { return std::sin(x); });
  } else if (isa<CosOp>(op)) {
    visitor([](float x) { return std::cos(x); });
  } else if (isa<NegOp>(op)) {
    visitor([](float x) { return -x; });
  } else if (isa<AbsOp>(op)) {
    visitor([](float x) { return std::fabs(x); });
  } else if (isa<SqrtOp>(op)) {
    visitor([](float x) { return std::sqrt(x); });
  } else if (isa<RsqrtOp>(op)) {
    visitor([](float x) { return 1.0f / std::sqrt(x); });
  } else if (isa<TanhOp>(op)) {
    visitor([](float x) { return std::tanh(x); });
  } else if (isa<LogisticOp>(op)) {
    visitor([](float x) { return 1.0f / (1.0f + std::exp(-x)); });
  } else {
    return false;
  }
  return true;
}

template <typename Visitor>
bool VisitBinaryFn(mlir::Operation* op, Visitor&& visitor) {
  if (isa<AddOp>(op)) {
    visitor([](float a, float b) { return a + b; });
  } else if (isa<SubOp>(op)) {
    visitor([](float a, float b) { return a - b; });
  } else if (isa<MulOp>(op)) {
    visitor([](float a, float b) { return a * b; });
  } else if (isa<DivOp>(op)) {
    visitor([](float a, float b) { return a / b; });
  } else if (isa<MaximumOp>(op)) {
    // Matches the TFLite reference kernels.
    visitor([](float a, float b) { return a > b ? a : b; });
  } else if (isa<MinimumOp>(op)) {
    visitor([](float a, float b) { return a < b ? a : b; });
  } else {
    return false;
  }
  return true;
}

template <typename Visitor>
bool VisitCompareFn(mlir::Operation* op, Visitor&& visitor) {
  if (isa<GreaterOp>(op)) {
    visitor([](float a, float b) { return a > b; });
  } else if (isa<GreaterEqualOp>(op)) {
    visitor([](float a, float b) { return a >= b; });
  } else if (isa<LessOp>(op)) {
    visitor([](float a, float b) { return a < b; });
  } else if (isa<LessEqualOp>(op)) {
    visitor([](float a, float b) { return a <= b; });
  } else if (isa<EqualOp>(op)) {
    visitor([](float a, float b) { return a == b; });
  } else if (isa<NotEqualOp>(op)) {
    visitor([](float a, float b) { return a != b; });
  } else {
    return false;
  }
  return true;
}

std::optional<StringRef> GetFusedActivation(mlir::Operation* op) {
  if (auto add = dyn_cast<AddOp>(op)) return add.getFusedActivationFunction();
  if (auto sub = dyn_cast<SubOp>(op)) return sub.getFusedActivationFunction();
  if (auto mul = dyn_cast<MulOp>(op)) return mul.getFusedActivationFunction();
  if (auto div = dyn_cast<DivOp>(op)) return div.getFusedActivationFunction();
  return std::nullopt;
}

bool IsSupportedOp(mlir::Operation* op) {
  auto ignore = [](auto) {};
  return VisitUnaryFn(op, ignore) || VisitBinaryFn(op, ignore) ||
         VisitCompareFn(op, ignore) ||
         isa<SelectOp, SelectV2Op, ConcatenationOp>(op);
}

template <typename Fn>
std::optional<FoldPlan> PlanUnary(llvm::ArrayRef<ConstantView> views,
                                  mlir::ShapedType result_type, Fn fn) {
  if (views.size() != 1 || views[0].type != result_type) return std::nullopt;
  std::optional<FloatKind> kind = GetFloatKind(result_type.getElementType());
  if (!kind) return std::nullopt;
  const ConstantView* in = &views[0];
  const int64_t num_elements = result_type.getNumElements();
  FoldPlan plan;
  plan.elementwise = true;
  plan.compute = [=, kind = *kind](char* out) {
    DispatchFloatKind(kind, [&](auto traits) {
      using T = decltype(traits);
      // The single operand of a resource fold is never a splat.
      const char* src = in->Data();
      for (int64_t i = 0; i < num_elements; ++i) {
        T::Store(out + i * T::kBytes, fn(T::Load(src + i * T::kBytes)));
      }
    });
  };
  return plan;
}

template <typename Fn>
std::optional<FoldPlan> PlanBinary(mlir::Operation* op,
                                   llvm::ArrayRef<ConstantView> views,
                                   mlir::ShapedType result_type, Fn fn) {
  if (views.size() != 2) return std::nullopt;
  Type element_type = result_type.getElementType();
  std::optional<FloatKind> kind = GetFloatKind(element_type);
  if (!kind || views[0].type.getElementType() != element_type ||
      views[1].type.getElementType() != element_type) {
    return std::nullopt;
  }
  Activation activation = Activation::kNone;
  if (std::optional<StringRef> name = GetFusedActivation(op)) {
    std::optional<Activation> parsed = ParseActivation(*name);
    if (!parsed) return std::nullopt;
    activation = *parsed;
  }
  SmallVector<int64_t> out_shape(result_type.getShape());
  std::optional<OperandStrides<2>> strides =
      GetAllOperandStrides<2>(views, out_shape);
  if (!strides) return std::nullopt;
  const ConstantView* lhs = &views[0];
  const ConstantView* rhs = &views[1];
  const int64_t num_elements = result_type.getNumElements();
  FoldPlan plan;
  plan.elementwise = true;
  plan.compute = [=, kind = *kind, strides = std::move(*strides)](char* out) {
    DispatchFloatKind(kind, [&](auto traits) {
      using T = decltype(traits);
      const char* a = lhs->Data();
      const char* b = rhs->Data();
      ForEachElement<2>(out_shape, strides,
                        [&](int64_t i, const std::array<int64_t, 2>& idx) {
                          T::Store(out + i * T::kBytes,
                                   fn(T::Load(a + idx[0] * T::kBytes),
                                      T::Load(b + idx[1] * T::kBytes)));
                        });
      ApplyActivation<T>(activation, out, num_elements);
    });
  };
  return plan;
}

template <typename Fn>
std::optional<FoldPlan> PlanCompare(llvm::ArrayRef<ConstantView> views,
                                    mlir::ShapedType result_type, Fn fn) {
  if (views.size() != 2 || !result_type.getElementType().isInteger(1)) {
    return std::nullopt;
  }
  Type element_type = views[0].type.getElementType();
  std::optional<FloatKind> kind = GetFloatKind(element_type);
  if (!kind || views[1].type.getElementType() != element_type) {
    return std::nullopt;
  }
  SmallVector<int64_t> out_shape(result_type.getShape());
  std::optional<OperandStrides<2>> strides =
      GetAllOperandStrides<2>(views, out_shape);
  if (!strides) return std::nullopt;
  const ConstantView* lhs = &views[0];
  const ConstantView* rhs = &views[1];
  FoldPlan plan;
  // The result type differs from the operand types, so no operand buffer can
  // be reused.
  plan.elementwise = false;
  plan.compute = [=, kind = *kind, strides = std::move(*strides)](char* out) {
    DispatchFloatKind(kind, [&](auto traits) {
      using T = decltype(traits);
      const char* a = lhs->Data();
      const char* b = rhs->Data();
      ForEachElement<2>(out_shape, strides,
                        [&](int64_t i, const std::array<int64_t, 2>& idx) {
                          out[i] = fn(T::Load(a + idx[0] * T::kBytes),
                                      T::Load(b + idx[1] * T::kBytes))
                                       ? 1
                                       : 0;
                        });
    });
  };
  return plan;
}

// Copies `size` bytes from `src` to `dst`, which may be the same address when
// folding in place.
template <int64_t kSize>
void CopyElement(char* dst, const char* src, int64_t /*size*/) {
  char tmp[kSize];
  std::memcpy(tmp, src, kSize);
  std::memcpy(dst, tmp, kSize);
}

template <>
void CopyElement<0>(char* dst, const char* src, int64_t size) {
  std::memmove(dst, src, size);
}

// Runs the select loop with a copy specialized for `kSize`-byte elements, or a
// generic copy when `kSize` is 0.
template <int64_t kSize>
void SelectLoop(llvm::ArrayRef<int64_t> out_shape,
                const OperandStrides<3>& strides, const char* cond,
                const char* x, const char* y, int64_t element_bytes,
                char* out) {
  ForEachElement<3>(
      out_shape, strides, [&](int64_t i, const std::array<int64_t, 3>& idx) {
        const char* src = cond[idx[0]] ? x + idx[1] * element_bytes
                                       : y + idx[2] * element_bytes;
        CopyElement<kSize>(out + i * element_bytes, src, element_bytes);
      });
}

std::optional<FoldPlan> PlanSelect(mlir::Operation* op,
                                   llvm::ArrayRef<ConstantView> views,
                                   mlir::ShapedType result_type) {
  if (views.size() != 3) return std::nullopt;
  const ConstantView& cond = views[0];
  if (!cond.type.getElementType().isInteger(1)) return std::nullopt;
  Type element_type = result_type.getElementType();
  if (views[1].type.getElementType() != element_type ||
      views[2].type.getElementType() != element_type) {
    return std::nullopt;
  }
  SmallVector<int64_t> out_shape(result_type.getShape());
  OperandStrides<3> strides;
  if (isa<SelectV2Op>(op)) {
    std::optional<OperandStrides<3>> s =
        GetAllOperandStrides<3>(views, out_shape);
    if (!s) return std::nullopt;
    strides = std::move(*s);
  } else {
    // tfl.select requires `x` and `y` to have the result shape. `condition`
    // either has the result shape, is a scalar, or is a rank-1 tensor that
    // selects along the first dimension.
    for (int k = 1; k < 3; ++k) {
      if (views[k].type.getShape() != llvm::ArrayRef<int64_t>(out_shape)) {
        return std::nullopt;
      }
    }
    llvm::ArrayRef<int64_t> cond_shape = cond.type.getShape();
    SmallVector<int64_t> cond_strides(out_shape.size(), 0);
    if (cond.splat || cond_shape.empty()) {
      // All zero strides.
    } else if (cond_shape == llvm::ArrayRef<int64_t>(out_shape)) {
      cond_strides = *GetBroadcastStrides(cond_shape, out_shape);
    } else if (cond_shape.size() == 1 && !out_shape.empty() &&
               cond_shape[0] == out_shape[0]) {
      cond_strides[0] = 1;
    } else {
      return std::nullopt;
    }
    strides[0] = std::move(cond_strides);
    for (int k = 1; k < 3; ++k) {
      strides[k] = *GetOperandStrides(views[k], out_shape);
    }
  }
  const ConstantView* c = &views[0];
  const ConstantView* x = &views[1];
  const ConstantView* y = &views[2];
  const int64_t element_bytes = x->element_bytes;
  FoldPlan plan;
  plan.elementwise = true;
  plan.compute = [=, strides = std::move(strides)](char* out) {
    const char* cond_data = c->Data();
    const char* x_data = x->Data();
    const char* y_data = y->Data();
    switch (element_bytes) {
      case 1:
        SelectLoop<1>(out_shape, strides, cond_data, x_data, y_data,
                      element_bytes, out);
        break;
      case 2:
        SelectLoop<2>(out_shape, strides, cond_data, x_data, y_data,
                      element_bytes, out);
        break;
      case 4:
        SelectLoop<4>(out_shape, strides, cond_data, x_data, y_data,
                      element_bytes, out);
        break;
      case 8:
        SelectLoop<8>(out_shape, strides, cond_data, x_data, y_data,
                      element_bytes, out);
        break;
      default:
        SelectLoop<0>(out_shape, strides, cond_data, x_data, y_data,
                      element_bytes, out);
        break;
    }
  };
  return plan;
}

std::optional<FoldPlan> PlanConcatenation(ConcatenationOp op,
                                          llvm::ArrayRef<ConstantView> views,
                                          mlir::ShapedType result_type) {
  if (op.getFusedActivationFunction() != "NONE" || views.empty()) {
    return std::nullopt;
  }
  llvm::ArrayRef<int64_t> out_shape = result_type.getShape();
  const int64_t rank = out_shape.size();
  int64_t axis = static_cast<int32_t>(op.getAxis());
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) return std::nullopt;

  Type element_type = result_type.getElementType();
  int64_t axis_total = 0;
  for (const ConstantView& view : views) {
    if (view.type.getElementType() != element_type) return std::nullopt;
    llvm::ArrayRef<int64_t> shape = view.type.getShape();
    if (static_cast<int64_t>(shape.size()) != rank) return std::nullopt;
    for (int64_t d = 0; d < rank; ++d) {
      if (d != axis && shape[d] != out_shape[d]) return std::nullopt;
    }
    axis_total += shape[axis];
  }
  if (axis_total != out_shape[axis]) return std::nullopt;

  int64_t outer = 1;
  for (int64_t d = 0; d < axis; ++d) outer *= out_shape[d];
  int64_t inner = 1;
  for (int64_t d = axis + 1; d < rank; ++d) inner *= out_shape[d];

  SmallVector<const ConstantView*> inputs;
  for (const ConstantView& view : views) inputs.push_back(&view);
  const int64_t element_bytes = views[0].element_bytes;
  FoldPlan plan;
  plan.elementwise = false;
  plan.compute = [=](char* out) {
    char* dst = out;
    for (int64_t o = 0; o < outer; ++o) {
      for (const ConstantView* in : inputs) {
        const int64_t chunk_bytes =
            in->type.getDimSize(axis) * inner * element_bytes;
        if (in->splat) {
          for (int64_t j = 0; j < chunk_bytes; j += element_bytes) {
            std::memcpy(dst + j, in->Data(), element_bytes);
          }
        } else {
          std::memcpy(dst, in->Data() + o * chunk_bytes, chunk_bytes);
        }
        dst += chunk_bytes;
      }
    }
  };
  return plan;
}

std::optional<FoldPlan> PlanFold(mlir::Operation* op,
                                 llvm::ArrayRef<ConstantView> views,
                                 mlir::ShapedType result_type) {
  std::optional<FoldPlan> plan;
  if (VisitUnaryFn(
          op, [&](auto fn) { plan = PlanUnary(views, result_type, fn); })) {
    return plan;
  }
  if (VisitBinaryFn(op, [&](auto fn) {
        plan = PlanBinary(op, views, result_type, fn);
      })) {
    return plan;
  }
  if (VisitCompareFn(
          op, [&](auto fn) { plan = PlanCompare(views, result_type, fn); })) {
    return plan;
  }
  if (isa<SelectOp, SelectV2Op>(op)) {
    return PlanSelect(op, views, result_type);
  }
  if (auto concat = dyn_cast<ConcatenationOp>(op)) {
    return PlanConcatenation(concat, views, result_type);
  }
  return std::nullopt;
}

// Returns whether the buffer behind `view` can hold the result of `op` without
// any other part of the module observing the mutation.
bool CanReuseBuffer(const ConstantView& view, mlir::Operation* op,
                    mlir::ShapedType result_type,
                    const ResourceRefCounts& refs) {
  if (!view.resource || !view.blob || !view.blob->isMutable()) return false;
  if (view.type != result_type) return false;
  if (refs.Get(view.resource) != 1) return false;
  return llvm::all_of(view.const_op->getUsers(),
                      [&](mlir::Operation* user) { return user == op; });
}

void TryFold(mlir::Operation* op, ResourceRefCounts& refs,
             int64_t max_growth_bytes) {
  if (op->getNumResults() != 1) return;
  auto result_type =
      mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!result_type || !result_type.hasStaticShape()) return;

  SmallVector<ConstantView> views;
  views.reserve(op->getNumOperands());
  bool has_resource = false;
  int64_t input_bytes = 0;
  for (mlir::Value operand : op->getOperands()) {
    std::optional<ConstantView> view = GetConstantView(operand);
    if (!view) return;
    has_resource |= static_cast<bool>(view->resource);
    input_bytes += view->stored_bytes;
    views.push_back(std::move(*view));
  }
  // Ops on DenseElementsAttr only are left to the regular folders.
  if (!has_resource) return;

  // `views` must not be resized after this point: the plan keeps pointers into
  // it.
  std::optional<FoldPlan> plan = PlanFold(op, views, result_type);
  if (!plan) return;

  Type element_type = result_type.getElementType();
  const int64_t element_bytes =
      element_type.isInteger(1) ? 1 : element_type.getIntOrFloatBitWidth() / 8;
  const int64_t output_bytes = result_type.getNumElements() * element_bytes;
  if (output_bytes > input_bytes && output_bytes > max_growth_bytes) return;

  const ConstantView* reused = nullptr;
  if (plan->elementwise) {
    for (const ConstantView& view : views) {
      if (CanReuseBuffer(view, op, result_type, refs)) {
        reused = &view;
        break;
      }
    }
  }

  mlir::Value replacement;
  if (reused) {
    plan->compute(reused->blob->getMutableData().data());
    replacement = reused->const_op->getResult(0);
  } else {
    AsmResourceBlob blob = HeapAsmResourceBlob::allocate(
        output_bytes, /*align=*/64, /*dataIsMutable=*/true);
    if (output_bytes > 0) plan->compute(blob.getMutableData().data());
    std::string base_key;
    for (const ConstantView& view : views) {
      if (view.resource) {
        base_key = view.resource.getRawHandle().getKey().str();
        break;
      }
    }
    auto new_attr = DenseResourceElementsAttr::get(
        result_type,
        absl::StrCat(base_key, "_", op->getName().stripDialect().str()),
        std::move(blob));
    refs.Add(new_attr);
    OpBuilder builder(op);
    replacement =
        arith::ConstantOp::create(builder, op->getLoc(), result_type, new_attr);
  }

  llvm::SmallSetVector<mlir::Operation*, 3> const_ops;
  for (const ConstantView& view : views) const_ops.insert(view.const_op);
  views.clear();

  op->getResult(0).replaceAllUsesWith(replacement);
  op->erase();

  // Erase constants that became dead and release buffers that are no longer
  // referenced by any attribute in the module.
  for (mlir::Operation* const_op : const_ops) {
    if (!const_op->use_empty()) continue;
    SmallVector<DenseResourceElementsAttr> dead;
    refs.Remove(const_op, dead);
    const_op->erase();
    for (DenseResourceElementsAttr attr : dead) {
      if (AsmResourceBlob* blob = GetResourceBlob(attr))
        *blob = AsmResourceBlob();
    }
  }
}

}  // namespace

LogicalResult FoldResourceElementwiseOps(ModuleOp module,
                                         int64_t max_growth_bytes) {
  ResourceRefCounts refs(module);
  // Pre-order visits producers before their consumers within a block, so a
  // chain of foldable ops collapses in a single sweep.
  SmallVector<mlir::Operation*> worklist;
  module.walk<WalkOrder::PreOrder>([&](mlir::Operation* op) {
    if (IsSupportedOp(op)) worklist.push_back(op);
  });
  for (mlir::Operation* op : worklist) TryFold(op, refs, max_growth_bytes);
  return success();
}

}  // namespace mlir::TFL
