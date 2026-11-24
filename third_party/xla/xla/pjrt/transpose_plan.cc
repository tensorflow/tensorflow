/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/pjrt/transpose_plan.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/permutation_util.h"
#include "xla/pjrt/transpose_kernels.h"
#include "xla/pjrt/transpose_ops.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"

namespace xla {
namespace transpose_internal {

namespace {

// Tree representation of a chunk's execution plan used during plan building
// before flattening into `std::vector<TransposeOp>`.
//
// Each node describes an action parameterized by the current input pointer `a`
// and output pointer `b`. The `Make*` constructors fold trivial cases on the
// fly (zero-work leaves, empty or single-trip loops, zero or nested offsets,
// inner offsets inside loops, contiguous nested `Loop`/`Fill1D`/`Copy1D`/
// `ConvertEf57`/`PackSubbyte` runs, and `Noop` children inside `Seq`).
struct Plan {
  // Does nothing. Removed by `Make*` everywhere except at the root of an empty
  // chunk.
  struct Noop {};

  // Runs each child plan in `parts` in order on the same `(a, b)` pointers.
  struct Seq {
    std::vector<Plan> parts;
  };

  // Adds `da` bytes to `a` and `db` bytes to `b`, then executes `body`.
  struct Offset {
    int64_t da;
    int64_t db;
    std::unique_ptr<Plan> body;
  };

  // Executes `body` `count` times; iteration `i` runs with input `a + i * da`
  // and output `b + i * db` (`da` and `db` are byte strides).
  struct Loop {
    int64_t count;
    int64_t da;
    int64_t db;
    std::unique_ptr<Plan> body;
  };

  // Pipes `step1` into `step2` through a `bytes`-byte temporary buffer carved
  // from scratch memory: runs `step1` from `a` into that buffer, then runs
  // `step2` from that buffer into `b`.
  struct Pipe {
    int64_t bytes;
    std::unique_ptr<Plan> step1;
    std::unique_ptr<Plan> step2;
  };

  // Transposes a `[b_elems, a_elems]` tile in `a` (row stride `lda` bytes,
  // contiguous along `a_elems`) into `[a_elems, b_elems]` in `b` (row stride
  // `ldb` bytes, contiguous along `b_elems`).
  struct Transpose2D {
    int a_elems;
    int b_elems;
    int64_t lda;
    int64_t ldb;
  };

  // Copies `bytes` contiguous bytes from `a` to `b`.
  struct Copy1D {
    int64_t bytes;
  };

  // Fills `bytes` contiguous bytes at `b` with byte `value`.
  struct Fill1D {
    int64_t bytes;
    uint8_t value;
  };

  // Packs `elems` 1-byte elements from `a` into `bits_per_element`-bit packed
  // values at `b`.
  struct PackSubbyte {
    int bits_per_element;
    int64_t elems;
  };

  // Converts `doubles` contiguous `double` values at `a` into `2 * doubles`
  // contiguous `float` values in TPU ef57 format at `b`.
  struct ConvertEf57 {
    int64_t doubles;
  };

  std::variant<Noop, Seq, Offset, Loop, Pipe, Transpose2D, Copy1D, Fill1D,
               PackSubbyte, ConvertEf57>
      node;

  bool IsNoop() const { return std::holds_alternative<Noop>(node); }

  static Plan MakeNoop();
  static Plan MakeSeq(std::vector<Plan> parts);
  static Plan MakeSeq(Plan first, Plan second);
  static Plan MakeOffset(int64_t da, int64_t db, Plan body);
  static Plan MakeLoop(int64_t count, int64_t da, int64_t db, Plan body);
  static Plan MakePipe(int64_t bytes, Plan step1, Plan step2);
  static Plan MakeTranspose2D(int a_elems, int b_elems, int64_t lda,
                              int64_t ldb);
  static Plan MakeCopy1D(int64_t bytes);
  static Plan MakeFill1D(int64_t bytes, uint8_t value);
  static Plan MakePackSubbyte(int bits_per_element, int64_t elems);
  static Plan MakeConvertEf57(int64_t doubles);
};

Plan Plan::MakeNoop() {
  return Plan{Noop{}};
}

Plan Plan::MakeSeq(std::vector<Plan> parts) {
  std::vector<Plan> non_empty;
  non_empty.reserve(parts.size());
  for (Plan& p : parts) {
    if (p.IsNoop()) {
      continue;
    }
    if (auto* sub = std::get_if<Seq>(&p.node)) {
      for (Plan& sp : sub->parts) {
        non_empty.push_back(std::move(sp));
      }
    } else {
      non_empty.push_back(std::move(p));
    }
  }
  if (non_empty.empty()) {
    return MakeNoop();
  }
  if (non_empty.size() == 1) {
    return std::move(non_empty[0]);
  }
  return Plan{Seq{std::move(non_empty)}};
}

Plan Plan::MakeSeq(Plan first, Plan second) {
  std::vector<Plan> parts;
  parts.reserve(2);
  parts.push_back(std::move(first));
  parts.push_back(std::move(second));
  return MakeSeq(std::move(parts));
}

Plan Plan::MakeOffset(int64_t da, int64_t db, Plan body) {
  if (body.IsNoop()) {
    return body;
  }
  if (auto* inner = std::get_if<Offset>(&body.node)) {
    da += inner->da;
    db += inner->db;
    std::unique_ptr<Plan> inner_body = std::move(inner->body);
    body = std::move(*inner_body);
  }
  if (da == 0 && db == 0) {
    return body;
  }
  return Plan{Offset{da, db, std::make_unique<Plan>(std::move(body))}};
}

Plan Plan::MakeLoop(int64_t count, int64_t da, int64_t db, Plan body) {
  if (count <= 0 || body.IsNoop()) {
    return MakeNoop();
  }
  if (count == 1) {
    return body;
  }
  if (auto* off = std::get_if<Offset>(&body.node)) {
    const int64_t oda = off->da;
    const int64_t odb = off->db;
    std::unique_ptr<Plan> off_body = std::move(off->body);
    return MakeOffset(oda, odb, MakeLoop(count, da, db, std::move(*off_body)));
  }
  if (auto* inner = std::get_if<Loop>(&body.node)) {
    if (da == inner->count * inner->da && db == inner->count * inner->db) {
      count *= inner->count;
      da = inner->da;
      db = inner->db;
      std::unique_ptr<Plan> inner_body = std::move(inner->body);
      body = std::move(*inner_body);
    }
  } else if (const auto* f = std::get_if<Fill1D>(&body.node)) {
    if (da == 0 && db == f->bytes) {
      return MakeFill1D(count * f->bytes, f->value);
    }
  } else if (const auto* c = std::get_if<Copy1D>(&body.node)) {
    if (da == c->bytes && db == c->bytes) {
      return MakeCopy1D(count * c->bytes);
    }
  } else if (const auto* e = std::get_if<ConvertEf57>(&body.node)) {
    if (da == e->doubles * static_cast<int64_t>(sizeof(double)) &&
        db == e->doubles * 2 * static_cast<int64_t>(sizeof(float))) {
      return MakeConvertEf57(count * e->doubles);
    }
  } else if (const auto* p = std::get_if<PackSubbyte>(&body.node)) {
    const int64_t epb = 8 / p->bits_per_element;
    if (p->elems % epb == 0 && da == p->elems && db == p->elems / epb) {
      return MakePackSubbyte(p->bits_per_element, count * p->elems);
    }
  }
  return Plan{Loop{count, da, db, std::make_unique<Plan>(std::move(body))}};
}

Plan Plan::MakePipe(int64_t bytes, Plan step1, Plan step2) {
  if (bytes <= 0 || step1.IsNoop() || step2.IsNoop()) {
    DCHECK_EQ(step1.IsNoop(), step2.IsNoop());
    return MakeNoop();
  }
  return Plan{Pipe{bytes, std::make_unique<Plan>(std::move(step1)),
                   std::make_unique<Plan>(std::move(step2))}};
}

Plan Plan::MakeTranspose2D(int a_elems, int b_elems, int64_t lda, int64_t ldb) {
  if (a_elems <= 0 || b_elems <= 0) {
    return MakeNoop();
  }
  return Plan{Transpose2D{a_elems, b_elems, lda, ldb}};
}

Plan Plan::MakeCopy1D(int64_t bytes) {
  if (bytes <= 0) {
    return MakeNoop();
  }
  return Plan{Copy1D{bytes}};
}

Plan Plan::MakeFill1D(int64_t bytes, uint8_t value) {
  if (bytes <= 0) {
    return MakeNoop();
  }
  return Plan{Fill1D{bytes, value}};
}

Plan Plan::MakePackSubbyte(int bits_per_element, int64_t elems) {
  if (elems <= 0) {
    return MakeNoop();
  }
  return Plan{PackSubbyte{bits_per_element, elems}};
}

Plan Plan::MakeConvertEf57(int64_t doubles) {
  if (doubles <= 0) {
    return MakeNoop();
  }
  return Plan{ConvertEf57{doubles}};
}

// Returns whether to use the `kind` interleave/deinterleave kernel for `k` and
// `elem_size`-byte elements instead of the default 2D transpose (see
// `InterleaveKernelIsFaster` and `DeinterleaveKernelIsFaster` in
// transpose_kernels.h). Among the selected cases, 16-byte interleave at k = 6
// and 7 is up to 6% slower on some x86 CPUs.
constexpr bool SmallDimKernelIsFaster(InnerKernelKind kind, int64_t k,
                                      int64_t elem_size) {
  if (kind == InnerKernelKind::kInterleave) {
    return InterleaveKernelIsFaster(elem_size, k);
  }
  return DeinterleaveKernelIsFaster(elem_size, k);
}

// Appends a human-readable indented tree representation of `plan` to `*out`
// (used by `Planner::ToString`).
void FormatPlan(const Plan& plan, int indent, int64_t elem_size_in_bytes,
                InnerKernelKind inner_kernel_kind, int interleave_size,
                std::string* out) {
  auto pad = [&](int n) { out->append(n, ' '); };
  std::visit(
      [&](const auto& node) {
        using NodeT = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<NodeT, Plan::Noop>) {
          pad(indent);
          absl::StrAppend(out, "Noop");
        } else if constexpr (std::is_same_v<NodeT, Plan::Seq>) {
          pad(indent);
          absl::StrAppend(out, "Seq(\n");
          for (size_t i = 0; i < node.parts.size(); ++i) {
            if (i > 0) {
              absl::StrAppend(out, ",\n");
            }
            FormatPlan(node.parts[i], indent + 2, elem_size_in_bytes,
                       inner_kernel_kind, interleave_size, out);
          }
          absl::StrAppend(out, ")");
        } else if constexpr (std::is_same_v<NodeT, Plan::Offset>) {
          pad(indent);
          absl::StrAppendFormat(out, "Offset(da=%d, db=%d,\n", node.da,
                                node.db);
          FormatPlan(*node.body, indent + 2, elem_size_in_bytes,
                     inner_kernel_kind, interleave_size, out);
          absl::StrAppend(out, ")");
        } else if constexpr (std::is_same_v<NodeT, Plan::Loop>) {
          pad(indent);
          absl::StrAppendFormat(out, "Loop(count=%d, da=%d, db=%d,\n",
                                node.count, node.da, node.db);
          FormatPlan(*node.body, indent + 2, elem_size_in_bytes,
                     inner_kernel_kind, interleave_size, out);
          absl::StrAppend(out, ")");
        } else if constexpr (std::is_same_v<NodeT, Plan::Pipe>) {
          pad(indent);
          absl::StrAppendFormat(out, "Pipe(bytes=%d,\n", node.bytes);
          FormatPlan(*node.step1, indent + 2, elem_size_in_bytes,
                     inner_kernel_kind, interleave_size, out);
          absl::StrAppend(out, ",\n");
          FormatPlan(*node.step2, indent + 2, elem_size_in_bytes,
                     inner_kernel_kind, interleave_size, out);
          absl::StrAppend(out, ")");
        } else if constexpr (std::is_same_v<NodeT, Plan::Copy1D>) {
          pad(indent);
          absl::StrAppendFormat(out, "Copy1D(bytes=%d)", node.bytes);
        } else if constexpr (std::is_same_v<NodeT, Plan::Fill1D>) {
          pad(indent);
          absl::StrAppendFormat(out, "Fill1D(bytes=%d, value=0x%02x)",
                                node.bytes, node.value);
        } else if constexpr (std::is_same_v<NodeT, Plan::PackSubbyte>) {
          pad(indent);
          absl::StrAppendFormat(out, "PackSubbyte<%d>(elems=%d)",
                                node.bits_per_element, node.elems);
        } else if constexpr (std::is_same_v<NodeT, Plan::ConvertEf57>) {
          pad(indent);
          absl::StrAppendFormat(out, "ConvertEf57(doubles=%d)", node.doubles);
        } else {
          static_assert(std::is_same_v<NodeT, Plan::Transpose2D>);
          pad(indent);
          if (inner_kernel_kind == InnerKernelKind::kInterleave) {
            absl::StrAppendFormat(
                out, "Interleave<elem_size=%d,K=%d>(a_elems=%d, lda=%d)",
                elem_size_in_bytes, interleave_size, node.a_elems, node.lda);
          } else if (inner_kernel_kind == InnerKernelKind::kDeinterleave) {
            absl::StrAppendFormat(
                out, "Deinterleave<elem_size=%d,K=%d>(b_elems=%d, ldb=%d)",
                elem_size_in_bytes, interleave_size, node.b_elems, node.ldb);
          } else {
            absl::StrAppendFormat(
                out,
                "Transpose2D<elem_size=%d>(a_elems=%d, b_elems=%d, lda=%d, "
                "ldb=%d)",
                elem_size_in_bytes, node.a_elems, node.b_elems, node.lda,
                node.ldb);
          }
        }
      },
      plan.node);
}

// Returns true if a tile or remainder strip may use a `bs x bs` SIMD kernel
// given a plan's primary `inner_block_elems` and `elem_size_in_bytes`.
// Secondary (halved) remainder blocks require at least 4 elements and 16 byte
// rows so the extra kernel pass beats a single scalar loop.
constexpr bool IsSupportedSimdBlockSize(int bs, int inner_block_elems,
                                        int64_t elem_size_in_bytes) {
  return bs > 1 && (bs == inner_block_elems ||
                    (bs >= 4 && bs * elem_size_in_bytes >= 16));
}

// Picks the execution function for a 2D transpose of `elem_size_in_bytes`
// bytes with extents (`a_elems`, `b_elems`), given the plan's kernel kind,
// vectorized microkernel block size `inner_block_elems`, and interleave factor
// `interleave_size`. Partially divisible tiles are decomposed into power of two
// SIMD blocks and scalar remainder strips by `BuildChunkPlan` before reaching
// here.
//
// Examples (with `elem_size_in_bytes = 4`):
//   - `kind = kDefault, inner_block_elems = 8`:
//     - `(16, 16)` -> `GetTransposeMacroKernelOp(4, 8)`
//     - `(16, 4)`  -> `GetTransposeMacroKernelOp(4, 4)`
//     - `(2, 16)`  -> `GetScalarTranspose2DOp(4)` (below 16 byte row)
//   - `kind = kInterleave, interleave_size = 3`:
//     - `(64, 3)`  -> `GetInterleaveOp(4, 3)`
//   - `kind = kDeinterleave, interleave_size = 3`:
//     - `(3, 64)`  -> `GetDeinterleaveOp(4, 3)`
template <int kElemSize>
TransposeOp::Fn SelectTranspose2DOpTyped(InnerKernelKind kind,
                                         int inner_block_elems,
                                         int interleave_size, int a_elems,
                                         int b_elems) {
  if (kind == InnerKernelKind::kInterleave) {
    DCHECK_EQ(b_elems, interleave_size);
    TransposeOp::Fn fn = GetInterleaveOp(kElemSize, interleave_size);
    CHECK(fn != nullptr) << "Unsupported interleave kernel: elem_size="
                         << kElemSize << " k=" << interleave_size;
    return fn;
  }
  if (kind == InnerKernelKind::kDeinterleave) {
    DCHECK_EQ(a_elems, interleave_size);
    TransposeOp::Fn fn = GetDeinterleaveOp(kElemSize, interleave_size);
    CHECK(fn != nullptr) << "Unsupported deinterleave kernel: elem_size="
                         << kElemSize << " k=" << interleave_size;
    return fn;
  }
  DCHECK(kind == InnerKernelKind::kDefault);
  int bs = inner_block_elems;
  while (bs > 1 && (a_elems % bs != 0 || b_elems % bs != 0)) {
    bs /= 2;
  }
  if (!IsSupportedSimdBlockSize(bs, inner_block_elems, kElemSize)) {
    return GetScalarTranspose2DOp(kElemSize);
  }
  TransposeOp::Fn fn = GetTransposeMacroKernelOp(kElemSize, bs);
  CHECK(fn != nullptr) << "Invalid block size " << bs
                       << " for elem_size=" << kElemSize;
  return fn;
}

// Dispatches `SelectTranspose2DOpTyped` on the runtime `elem_size_in_bytes`
// (1, 2, 4, 8, or 16 bytes).
TransposeOp::Fn SelectTranspose2DOp(int64_t elem_size_in_bytes,
                                    InnerKernelKind kind, int inner_block_elems,
                                    int interleave_size, int a_elems,
                                    int b_elems) {
  switch (elem_size_in_bytes) {
    case 1:
      return SelectTranspose2DOpTyped<1>(kind, inner_block_elems,
                                         interleave_size, a_elems, b_elems);
    case 2:
      return SelectTranspose2DOpTyped<2>(kind, inner_block_elems,
                                         interleave_size, a_elems, b_elems);
    case 4:
      return SelectTranspose2DOpTyped<4>(kind, inner_block_elems,
                                         interleave_size, a_elems, b_elems);
    case 8:
      return SelectTranspose2DOpTyped<8>(kind, inner_block_elems,
                                         interleave_size, a_elems, b_elems);
    case 16:
      return SelectTranspose2DOpTyped<16>(kind, inner_block_elems,
                                          interleave_size, a_elems, b_elems);
    default:
      LOG(FATAL) << "Unimplemented element size " << elem_size_in_bytes;
  }
}

// Returns the `OpPackSubbyteHwy` specialization for `bits_per_element` (1, 2,
// or 4 bits per element).
TransposeOp::Fn SelectPackSubbyteOp(int bits_per_element) {
  switch (bits_per_element) {
    case 1:
      return GetPackSubbyteOp<1>();
    case 2:
      return GetPackSubbyteOp<2>();
    case 4:
      return GetPackSubbyteOp<4>();
    default:
      LOG(FATAL) << "Invalid bits_per_element: " << bits_per_element;
  }
}

struct CopyRowMatch {
  int64_t count = 0;
  int64_t da = 0;
  int64_t db = 0;
  int64_t bytes = 0;
  int64_t tail_bytes = 0;
};

// Matches a 2-part `Seq` representing a row copy with a trailing partial chunk:
//  - `Seq(Loop(n1, da1, db1, Copy1D(bytes)),
//         Offset(n1 * da1, n1 * db1, Copy1D(tail_bytes)))`, or
//  - `Seq(Copy1D(bytes), Offset(da1, db1, Copy1D(tail_bytes)))` (`n1 = 1`).
std::optional<CopyRowMatch> MatchCopySeqRow(const Plan::Seq& seq) {
  if (seq.parts.size() != 2) {
    return std::nullopt;
  }
  const auto* off = std::get_if<Plan::Offset>(&seq.parts[1].node);
  if (off == nullptr) {
    return std::nullopt;
  }
  const auto* tail = std::get_if<Plan::Copy1D>(&off->body->node);
  if (tail == nullptr) {
    return std::nullopt;
  }
  if (const auto* loop = std::get_if<Plan::Loop>(&seq.parts[0].node)) {
    const auto* copy = std::get_if<Plan::Copy1D>(&loop->body->node);
    if (copy != nullptr && off->da == loop->count * loop->da &&
        off->db == loop->count * loop->db) {
      return CopyRowMatch{loop->count, loop->da, loop->db, copy->bytes,
                          tail->bytes};
    }
    return std::nullopt;
  }
  if (const auto* copy = std::get_if<Plan::Copy1D>(&seq.parts[0].node)) {
    return CopyRowMatch{/*count=*/1, off->da, off->db, copy->bytes,
                        tail->bytes};
  }
  return std::nullopt;
}

struct PlanFlattener {
  int64_t elem_size_in_bytes;
  InnerKernelKind inner_kernel_kind;
  int inner_block_elems;
  int interleave_size;
  std::vector<TransposeOp>& ops;

  void EmitPendingLoop(int64_t count, int64_t da, int64_t db) {
    if (count > 1) {
      TransposeOp op;
      op.fn = &OpLoop;
      op.count = count;
      op.da = da;
      op.db = db;
      ops.push_back(op);
    }
  }

  void EmitCopy2D(int64_t count0, int64_t da0, int64_t db0,
                  const CopyRowMatch& inner) {
    TransposeOp outer_op;
    outer_op.fn = &OpCopy2D;
    outer_op.count = count0;
    outer_op.da = da0;
    outer_op.db = db0;
    TransposeOp inner_op;
    inner_op.count = inner.count;
    inner_op.da = inner.da;
    inner_op.db = inner.db;
    inner_op.lda = inner.bytes;
    inner_op.ldb = inner.tail_bytes;
    ops.push_back(outer_op);
    ops.push_back(inner_op);
  }

  // Emits `plan` with an optional pending outer loop `(count, da, db)` from an
  // enclosing `Plan::Loop` (`count == 1` when unlooped). Ops with a built-in
  // loop (all 5 leaves, `Pipe`, and `Copy2D` patterns) absorb it directly;
  // other ops flush an `OpLoop` first. Returns the peak scratch memory in bytes
  // needed by this subtree.
  int64_t Emit(const Plan& plan, int64_t count = 1, int64_t da = 0,
               int64_t db = 0) {
    return std::visit(
        [&](const auto& node) -> int64_t {
          using NodeT = std::decay_t<decltype(node)>;
          if constexpr (std::is_same_v<NodeT, Plan::Noop>) {
            return 0;
          } else if constexpr (std::is_same_v<NodeT, Plan::Transpose2D>) {
            DCHECK_LE(node.a_elems, std::numeric_limits<uint16_t>::max());
            DCHECK_LE(node.b_elems, std::numeric_limits<uint16_t>::max());
            TransposeOp op;
            op.fn = SelectTranspose2DOp(elem_size_in_bytes, inner_kernel_kind,
                                        inner_block_elems, interleave_size,
                                        node.a_elems, node.b_elems);
            op.count = count;
            op.da = da;
            op.db = db;
            op.lda = node.lda;
            op.ldb = node.ldb;
            op.a_elems = static_cast<uint16_t>(node.a_elems);
            op.b_elems = static_cast<uint16_t>(node.b_elems);
            ops.push_back(op);
            return 0;
          } else if constexpr (std::is_same_v<NodeT, Plan::Copy1D>) {
            TransposeOp op;
            op.fn = &OpCopy;
            op.count = count;
            op.da = da;
            op.db = db;
            op.lda = node.bytes;
            ops.push_back(op);
            return 0;
          } else if constexpr (std::is_same_v<NodeT, Plan::Fill1D>) {
            TransposeOp op;
            op.fn = &OpFill;
            op.count = count;
            op.db = db;
            op.lda = node.bytes;
            op.a_elems = node.value;
            ops.push_back(op);
            return 0;
          } else if constexpr (std::is_same_v<NodeT, Plan::PackSubbyte>) {
            TransposeOp op;
            op.fn = SelectPackSubbyteOp(node.bits_per_element);
            op.count = count;
            op.da = da;
            op.db = db;
            op.lda = node.elems;
            ops.push_back(op);
            return 0;
          } else if constexpr (std::is_same_v<NodeT, Plan::ConvertEf57>) {
            TransposeOp op;
            op.fn = &OpConvertEf57;
            op.count = count;
            op.da = da;
            op.db = db;
            op.lda = node.doubles;
            ops.push_back(op);
            return 0;
          } else if constexpr (std::is_same_v<NodeT, Plan::Pipe>) {
            DCHECK(!node.step1->IsNoop() && !node.step2->IsNoop());
            const int64_t aligned_bytes = RoundUpTo<int64_t>(node.bytes, 64);
            const size_t idx = ops.size();
            TransposeOp op;
            op.fn = &OpPipe;
            op.count = count;
            op.da = da;
            op.db = db;
            op.lda = aligned_bytes;
            ops.push_back(op);
            const int64_t t1 = Emit(*node.step1);
            const size_t right_off = ops.size() - idx;
            DCHECK_LE(right_off, std::numeric_limits<uint32_t>::max());
            ops[idx].right_off = static_cast<uint32_t>(right_off);
            const int64_t t2 = Emit(*node.step2);
            return aligned_bytes + std::max(t1, t2);
          } else if constexpr (std::is_same_v<NodeT, Plan::Seq>) {
            if (std::optional<CopyRowMatch> row = MatchCopySeqRow(node)) {
              EmitCopy2D(count, da, db, *row);
              return 0;
            }
            EmitPendingLoop(count, da, db);
            DCHECK_GE(node.parts.size(), 2);
            // Emit binary OpSeq nodes right-associatively so OpSeq tail-calls
            // down the chain in O(1) stack space.
            int64_t max_temp = 0;
            for (size_t i = 0; i + 1 < node.parts.size(); ++i) {
              const size_t seq_idx = ops.size();
              TransposeOp op;
              op.fn = &OpSeq;
              ops.push_back(op);
              max_temp = std::max(max_temp, Emit(node.parts[i]));
              const size_t right_off = ops.size() - seq_idx;
              DCHECK_LE(right_off, std::numeric_limits<uint32_t>::max());
              ops[seq_idx].right_off = static_cast<uint32_t>(right_off);
            }
            return std::max(max_temp, Emit(node.parts.back()));
          } else if constexpr (std::is_same_v<NodeT, Plan::Offset>) {
            EmitPendingLoop(count, da, db);
            TransposeOp op;
            op.fn = &OpOffset;
            op.da = node.da;
            op.db = node.db;
            ops.push_back(op);
            return Emit(*node.body);
          } else {
            static_assert(std::is_same_v<NodeT, Plan::Loop>);
            if (count > 1) {
              if (const auto* copy =
                      std::get_if<Plan::Copy1D>(&node.body->node)) {
                EmitCopy2D(count, da, db,
                           CopyRowMatch{node.count, node.da, node.db,
                                        copy->bytes, /*tail_bytes=*/0});
                return 0;
              }
              EmitPendingLoop(count, da, db);
            }
            return Emit(*node.body, node.count, node.da, node.db);
          }
        },
        plan.node);
  }
};

// Appends a flattened depth-first encoding of `plan` into `ops`, binding each
// leaf node to its concrete execution function (fusing immediately enclosing
// `Loop` nodes into leaves, `Pipe`, and 2D `Copy1D` patterns with optional
// trailing partial tiles), and returns the peak scratch memory in bytes needed
// to execute `plan` (with each `Pipe` reservation rounded up to 64 bytes to
// keep nested scratch pointers cache-line aligned).
int64_t FlattenPlan(const Plan& plan, int64_t elem_size_in_bytes,
                    InnerKernelKind inner_kernel_kind, int inner_block_elems,
                    int interleave_size, std::vector<TransposeOp>& ops) {
  PlanFlattener flattener{elem_size_in_bytes, inner_kernel_kind,
                          inner_block_elems, interleave_size, ops};
  return flattener.Emit(plan);
}

// Scales output byte offsets, strides, and fill sizes in a padding `Plan` from
// unpacked 1-byte element units to packed sub-byte output bytes.
void ScalePlanOutputBytes(Plan& plan, int elements_per_byte) {
  std::visit(
      [&](auto& node) {
        using NodeT = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<NodeT, Plan::Noop>) {
          return;
        } else if constexpr (std::is_same_v<NodeT, Plan::Seq>) {
          for (Plan& p : node.parts) {
            ScalePlanOutputBytes(p, elements_per_byte);
          }
        } else if constexpr (std::is_same_v<NodeT, Plan::Offset>) {
          DCHECK_EQ(node.db % elements_per_byte, 0);
          node.db /= elements_per_byte;
          ScalePlanOutputBytes(*node.body, elements_per_byte);
        } else if constexpr (std::is_same_v<NodeT, Plan::Loop>) {
          DCHECK_EQ(node.db % elements_per_byte, 0);
          node.db /= elements_per_byte;
          ScalePlanOutputBytes(*node.body, elements_per_byte);
        } else if constexpr (std::is_same_v<NodeT, Plan::Fill1D>) {
          DCHECK_EQ(node.bytes % elements_per_byte, 0);
          node.bytes /= elements_per_byte;
        } else {
          LOG(FATAL) << "Unexpected node in padding plan";
        }
      },
      plan.node);
}

// Internal planning state used during `BuildTransposePlan`.
struct Planner {
  // Number of chunks requested.
  int num_chunks_requested = 1;

  // Size of each element in bytes.
  int64_t elem_size_in_bytes = 0;

  // Number of elements in the input array.
  int64_t num_elems = 0;

  // Dimensions of the input array A.
  absl::InlinedVector<int64_t, 4> a_dims;
  absl::InlinedVector<int64_t, 4> a_strides;

  // Dimensions of the output array B.
  std::vector<int64_t> b_dims;

  // Dimension permutation to apply to form B. For each dimension of B, what is
  // the corresponding dimension of A?
  absl::InlinedVector<int64_t, 4> permutation;

  // Leading-dimension sizes (byte strides) of each dimension.
  absl::InlinedVector<int64_t, 4> lda;       // Strides for tiles
  absl::InlinedVector<int64_t, 4> lda_tile;  // Strides for tile interiors
  absl::InlinedVector<int64_t, 4> ldb;       // Strides for tiles
  absl::InlinedVector<int64_t, 4> ldb_tile;  // Strides for tile interiors

  // Tile sizes in each dimension. Has size equal to the number of dimensions.
  // A 1 entry means that dimension is not tiled.
  absl::InlinedVector<int64_t, 4> a_tiling;
  absl::InlinedVector<int64_t, 4> b_tiling;
  bool a_is_tiled = false;
  bool b_is_tiled = false;
  bool is_identity_permutation = false;
  // K for kInterleave and kDeinterleave. Placed here to fill padding.
  int interleave_size = 0;

  // Per-chunk loop nests. Each loop nest has its own start/end bounds
  // representing one chunk of the work.
  std::vector<std::vector<Loop>> chunk_loops;

  // Per-chunk byte offsets and sizes into the input and output arrays.
  // `offset_bytes` are the physical offsets of the start of each chunk (minimum
  // byte addresses touched by that chunk) relative to the logical start
  // (index 0) of the array. Note: if the input strides are negative, these
  // offsets can also be negative.
  std::vector<int64_t> input_chunk_offset_bytes;
  std::vector<int64_t> output_chunk_offset_bytes;

  // `iteration_offsets` are the byte offsets of the first logical element of
  // each chunk's iteration space, relative to the physical start of that chunk
  // (i.e. relative to `input_chunk_offset_bytes`). These are non zero for
  // negative strides, where the chunk may start before the logical start of the
  // array.
  std::vector<int64_t> input_chunk_iteration_offsets;
  std::vector<int64_t> output_chunk_iteration_offsets;
  std::vector<int64_t> input_chunk_size_bytes;
  std::vector<int64_t> output_chunk_size_bytes;
  int64_t max_input_chunk_size_bytes = 0;
  int64_t max_output_chunk_size_bytes = 0;

  // Flattened preorder plan ops per chunk (indexed by chunk_id).
  absl::InlinedVector<std::vector<TransposeOp>, 1> chunk_ops;

  // Maximum per-chunk scratch buffer size in bytes (64-byte aligned). 0 means
  // no scratch buffer is required.
  int64_t scratch_size = 0;

  // Which kernel handles the innermost (stride-1) dimensions.
  InnerKernelKind inner_kernel_kind = InnerKernelKind::kDefault;

  // Size of the inner (microkernel) block size. This is the unit of work for
  // our vectorized kernels.
  int inner_block_elems = 1;
  // Size of the outer (macrokernel) block size. This is the unit of work for
  // cache blocking and need not be equal between input and output.
  int outer_block_elems_a = 4;
  int outer_block_elems_b = 4;

  // Strides used by an inner transpose kernel. Unused for memcpy kernels.
  int64_t sentinel_lda = -1;
  int64_t sentinel_ldb = -1;

  // Transformations to apply to the input before transposition.
  // Currently the only supported transformation is EF57 conversion, which is
  // a pair-of-floats extended precision representation used on TPU. We
  // support fusing transformations with the transpose for two reasons:
  // (a) it makes sense to fuse cheap computations with a memory-bandwidth
  //     bound transformation, and
  // (b) it allows us to support non-trivial striding.
  Transformation transformation = Transformation::kNone;
  int bits_per_element = 0;
  // If the inner dim is not divisible by the packing ratio, we cannot
  // pack into the destination in parallel, so we fallback to transposing
  // into a temporary buffer and then packing into the destination.
  bool use_fallback_pack = false;

  ChunkContiguity chunk_contiguity = ChunkContiguity::kNone;
  std::optional<uint8_t> tile_padding;

  bool inner_kernel_is_memcpy() const {
    return inner_kernel_kind == InnerKernelKind::kMemcpy;
  }

  // Padded input and output sizes in number of elements, accounting for tiling
  // (ignoring input striding).
  int64_t InputNumElems() const;
  int64_t OutputNumElems() const;

  // Formats a human-readable dump of the planner state for VLOG debugging,
  // optionally appending pre-formatted per-chunk plan trees.
  std::string ToString(absl::Span<const std::string> plan_strings = {}) const;

  // Builds the loop nest, chooses loop ordering and parallelization, partitions
  // loops into chunks, and builds + flattens each chunk's execution plan.
  absl::Status Initialize();

  // Builds the unflattened execution `Plan` tree (data transfer plus any
  // partial-tile padding) for `chunk_id`.
  Plan BuildChunkPlan(int chunk_id) const;

  // Assigns `parallelism` to each loop in `loop_order` and returns the total
  // number of chunks.
  int ChooseParallelizationStrategy(std::vector<Loop>& loop_order) const;

  // Marks loops in `loop_order` with `contiguity` (1 = boundary, 2 = deep
  // inner) to preserve the requested `chunk_contiguity`.
  void IdentifyContiguousLoops(std::vector<Loop>& loop_order) const;

  // Computes per-chunk physical byte offsets, iteration offsets, and byte sizes
  // from `chunk_loops`.
  void ComputeChunkSizes();

  // Moves the finished execution state and flattened ops out into a
  // `FlattenedPlan`.
  FlattenedPlan ToFlattenedPlan();
};

}  // namespace

static void ComputeStrides(
    int64_t elem_size_in_bytes, absl::Span<const int64_t> dims,
    absl::Span<const int64_t> tiling,
    absl::InlinedVector<int64_t, 4>& outer_tile_strides,
    absl::InlinedVector<int64_t, 4>& inner_tile_strides) {
  inner_tile_strides.resize(dims.size());
  int64_t acc = elem_size_in_bytes;
  for (int d = static_cast<int>(dims.size()) - 1; d >= 0; --d) {
    inner_tile_strides[d] = acc;
    acc *= tiling[d];
  }
  outer_tile_strides.resize(dims.size());
  for (int d = static_cast<int>(dims.size()) - 1; d >= 0; --d) {
    outer_tile_strides[d] = acc;
    acc *= CeilOfRatio(dims[d], tiling[d]);
  }
}

int64_t Planner::InputNumElems() const {
  int64_t size = 1;
  for (size_t i = 0; i < a_dims.size(); ++i) {
    size *= RoundUpTo(a_dims[i], a_tiling[i]);
  }
  return size;
}

int64_t Planner::OutputNumElems() const {
  int64_t size = 1;
  for (size_t i = 0; i < a_dims.size(); ++i) {
    size *= RoundUpTo(a_dims[permutation[i]], b_tiling[i]);
  }
  return size;
}

// Parses and validates a tiling specification, and populates `tiling`.
static absl::Status ParseTilingSpecification(
    int ndim, const std::optional<absl::Span<int64_t const>>& tiling_opt,
    bool nonstandard_layout, absl::InlinedVector<int64_t, 4>& tiling) {
  tiling.resize(ndim, 1);
  if (!tiling_opt) {
    return absl::OkStatus();
  }
  absl::Span<int64_t const> tiling_spec = *tiling_opt;
  if (tiling_spec.size() > ndim) {
    return InvalidArgument(
        "Tiling (%s) must have at most as many dimensions as the array (%d)",
        absl::StrJoin(tiling_spec, ","), ndim);
  }
  if (absl::c_find_if(tiling_spec, [](int64_t d) { return d < 1; }) !=
      tiling_spec.end()) {
    return InvalidArgument("Tiling sizes (%s) must be >= 1",
                           absl::StrJoin(tiling_spec, ","));
  }
  if (ndim == 1 && !nonstandard_layout) {
    // Tiling doesn't do anything for a rank-1 array with default strides,
    // except add padding. Since we're not going to touch any padding elements,
    // we can ignore it.
    // TODO(phawkins): this seems like it should be a loop optimization, not
    // done here.
    return absl::OkStatus();
  }
  int offset = ndim;
  offset -= tiling_spec.size();
  absl::c_copy(tiling_spec, tiling.begin() + offset);

  return absl::OkStatus();
}

std::string Loop::ToString() const {
  return absl::StrFormat(
      "%d%s[dim_size=%d,tile_size=%d,start=%d,end=%d,is_inner_dim_in_a=%d,is_"
      "inner_dim_in_b=%d,lda=%d,ldb=%d,parallelism=%d,contiguity=%d,has_"
      "partial_tile=%d)",
      dim_in_a, tile_interior ? "[tile]" : "", dim_size, tile_size, start, end,
      is_inner_dim_in_a, is_inner_dim_in_b, lda, ldb, parallelism, contiguity,
      has_partial_tile);
}

bool Loop::operator==(const Loop& other) const {
  return dim_in_a == other.dim_in_a && tile_interior == other.tile_interior &&
         dim_size == other.dim_size && tile_size == other.tile_size &&
         lda == other.lda && ldb == other.ldb &&
         is_inner_dim_in_a == other.is_inner_dim_in_a &&
         is_inner_dim_in_b == other.is_inner_dim_in_b &&
         parallelism == other.parallelism && start == other.start &&
         end == other.end;
}

struct ChunkPlanBuilder {
  const Planner& plan;
  absl::Span<const Loop> chunk_loops;
  int elements_per_byte;

  ChunkPlanBuilder(const Planner& plan, int chunk_id)
      : plan(plan),
        chunk_loops(plan.chunk_loops[chunk_id]),
        elements_per_byte((plan.transformation == Transformation::kPackSubbyte)
                              ? (8 / plan.bits_per_element)
                              : 1) {}

  int64_t StepB(const Loop& l, int64_t n) const {
    DCHECK_EQ((n * l.ldb) % elements_per_byte, 0);
    return (n * l.ldb) / elements_per_byte;
  }

  // Decomposes default kernel tiles into full power of two SIMD blocks (down
  // to 16 byte rows for remainder tiles) plus a trailing scalar remainder strip
  // so every `Plan::Transpose2D` leaf maps directly to a single `TransposeOp`.
  Plan MakeTranspose2D(int a_elems, int b_elems, int64_t lda,
                       int64_t ldb) const {
    if (a_elems <= 0 || b_elems <= 0) {
      return Plan::MakeNoop();
    }
    if (plan.inner_kernel_kind == InnerKernelKind::kDefault &&
        plan.inner_block_elems > 1) {
      int bs = plan.inner_block_elems;
      while (bs > 1 && (a_elems < bs || b_elems < bs)) {
        bs /= 2;
      }
      if (IsSupportedSimdBlockSize(bs, plan.inner_block_elems,
                                   plan.elem_size_in_bytes) &&
          (a_elems % bs != 0 || b_elems % bs != 0)) {
        const int done_a = (a_elems / bs) * bs;
        const int done_b = (b_elems / bs) * bs;
        std::vector<Plan> pieces;
        pieces.reserve(3);
        pieces.push_back(Plan::MakeTranspose2D(done_a, done_b, lda, ldb));
        pieces.push_back(Plan::MakeOffset(
            done_b * lda, done_b * plan.elem_size_in_bytes,
            MakeTranspose2D(done_a, b_elems - done_b, lda, ldb)));
        pieces.push_back(Plan::MakeOffset(
            done_a * plan.elem_size_in_bytes, done_a * ldb,
            MakeTranspose2D(a_elems - done_a, b_elems, lda, ldb)));
        return Plan::MakeSeq(std::move(pieces));
      }
    }
    return Plan::MakeTranspose2D(a_elems, b_elems, lda, ldb);
  }

  Plan BuildLeaf(int a_elems, int b_elems) const {
    switch (plan.transformation) {
      case Transformation::kNone:
        return MakeTranspose2D(a_elems, b_elems, plan.sentinel_lda,
                               plan.sentinel_ldb);
      case Transformation::kF64ToEf57: {
        DCHECK_EQ(a_elems % 2, 0);
        const int64_t tmp_lda = int64_t{a_elems} * sizeof(float);
        Plan step1 = Plan::MakeLoop(b_elems, plan.sentinel_lda, tmp_lda,
                                    Plan::MakeConvertEf57(a_elems / 2));
        Plan step2 =
            MakeTranspose2D(a_elems, b_elems, tmp_lda, plan.sentinel_ldb);
        return Plan::MakePipe(int64_t{b_elems} * tmp_lda, std::move(step1),
                              std::move(step2));
      }
      case Transformation::kPackSubbyte: {
        DCHECK_EQ(plan.elem_size_in_bytes, 1);
        DCHECK_EQ(b_elems % elements_per_byte, 0);
        DCHECK_EQ(plan.sentinel_ldb % elements_per_byte, 0);
        Plan step1 = MakeTranspose2D(a_elems, b_elems, plan.sentinel_lda,
                                     /*ldb=*/b_elems);
        Plan step2 = Plan::MakeLoop(
            a_elems, /*da=*/b_elems, plan.sentinel_ldb / elements_per_byte,
            Plan::MakePackSubbyte(plan.bits_per_element, b_elems));
        return Plan::MakePipe(int64_t{a_elems} * b_elems, std::move(step1),
                              std::move(step2));
      }
    }
    LOG(FATAL) << "Unhandled transformation "
               << static_cast<int>(plan.transformation);
  }

  Plan BuildTransfer(size_t idx, int a_elems, int b_elems,
                     absl::InlinedVector<bool, 4>& partial) const {
    if (idx == chunk_loops.size()) {
      DCHECK(!plan.inner_kernel_is_memcpy());
      DCHECK_GT(a_elems, 0);
      DCHECK_GT(b_elems, 0);
      return BuildLeaf(a_elems, b_elems);
    }

    const Loop& loop = chunk_loops[idx];
    const int a_dim = loop.dim_in_a;

    if (!loop.tile_interior && loop.tile_size > 1) {
      // Exterior loop over tiles of a dimension that has a trailing partial
      // tile. Split into complete tiles and (if covered by this chunk) the
      // trailing partial tile.
      DCHECK_NE(loop.dim_size % loop.tile_size, 0);
      DCHECK_EQ(loop.inc, 1);
      DCHECK(!loop.is_inner_dim_in_a && !loop.is_inner_dim_in_b);
      const int64_t num_complete = loop.dim_size / loop.tile_size;
      const int64_t complete_count =
          std::min<int64_t>(num_complete, loop.end) - loop.start;
      const bool saved_partial = partial[a_dim];
      partial[a_dim] = false;
      Plan full_plan =
          (complete_count > 0)
              ? Plan::MakeLoop(
                    complete_count, loop.lda, StepB(loop, 1),
                    BuildTransfer(idx + 1, a_elems, b_elems, partial))
              : Plan::MakeNoop();
      Plan rem_plan = Plan::MakeNoop();
      if (loop.start <= num_complete && num_complete < loop.end) {
        partial[a_dim] = true;
        const int64_t offset = num_complete - loop.start;
        rem_plan =
            Plan::MakeOffset(offset * loop.lda, StepB(loop, offset),
                             BuildTransfer(idx + 1, a_elems, b_elems, partial));
      }
      partial[a_dim] = saved_partial;
      return Plan::MakeSeq(std::move(full_plan), std::move(rem_plan));
    }

    if (loop.tile_interior) {
      DCHECK_EQ(loop.start, 0);
      DCHECK_EQ(loop.end, loop.tile_size);
    }
    const int64_t count =
        loop.tile_interior
            ? (partial[a_dim] ? (loop.dim_size % loop.tile_size)
                              : loop.tile_size)
            : std::max<int64_t>(
                  0, std::min<int64_t>(loop.dim_size, loop.end) - loop.start);
    if (count <= 0) {
      return Plan::MakeNoop();
    }

    if (plan.inner_kernel_is_memcpy() && loop.is_inner_dim_in_a) {
      DCHECK_EQ(idx + 1, chunk_loops.size());
      if (plan.transformation == Transformation::kPackSubbyte) {
        DCHECK_EQ(count % elements_per_byte, 0);
        return Plan::MakePackSubbyte(plan.bits_per_element, count);
      }
      return Plan::MakeCopy1D(count * plan.elem_size_in_bytes);
    }

    if (!plan.inner_kernel_is_memcpy() &&
        (loop.is_inner_dim_in_a || loop.is_inner_dim_in_b)) {
      const int64_t block = loop.inc;
      DCHECK_GT(block, 0);
      const int64_t full_blocks = count / block;
      const int64_t rem = count % block;
      auto build_child = [&](int width) -> Plan {
        const int next_a = loop.is_inner_dim_in_a ? width : a_elems;
        const int next_b = loop.is_inner_dim_in_b ? width : b_elems;
        return BuildTransfer(idx + 1, next_a, next_b, partial);
      };
      Plan full_plan =
          (full_blocks > 0)
              ? Plan::MakeLoop(full_blocks, block * loop.lda,
                               StepB(loop, block),
                               build_child(static_cast<int>(block)))
              : Plan::MakeNoop();
      const int64_t rem_start = full_blocks * block;
      Plan rem_plan =
          (rem > 0)
              ? Plan::MakeOffset(rem_start * loop.lda, StepB(loop, rem_start),
                                 build_child(static_cast<int>(rem)))
              : Plan::MakeNoop();
      return Plan::MakeSeq(std::move(full_plan), std::move(rem_plan));
    }

    DCHECK_EQ(loop.inc, 1);
    return Plan::MakeLoop(count, loop.lda, StepB(loop, 1),
                          BuildTransfer(idx + 1, a_elems, b_elems, partial));
  }

  Plan PadFrom(absl::Span<const Loop> loops, size_t idx, bool is_padding,
               absl::InlinedVector<bool, 4>& in_partial) const {
    if (idx == loops.size()) {
      return is_padding
                 ? Plan::MakeFill1D(plan.elem_size_in_bytes, *plan.tile_padding)
                 : Plan::MakeNoop();
    }
    const Loop& loop = loops[idx];
    auto sub_loop = [&](int64_t count, bool sub_padding) {
      if (count <= 0) {
        return Plan::MakeNoop();
      }
      return Plan::MakeLoop(count, /*da=*/0, loop.ldb,
                            PadFrom(loops, idx + 1, sub_padding, in_partial));
    };
    if (!loop.tile_interior) {
      const int64_t full = loop.dim_size / loop.tile_size;
      const bool saved = in_partial[loop.dim_in_a];
      in_partial[loop.dim_in_a] = false;
      Plan full_plan =
          sub_loop(std::min(loop.end, full) - loop.start, is_padding);
      Plan rem_plan = Plan::MakeNoop();
      if (loop.start <= full && full < loop.end) {
        in_partial[loop.dim_in_a] = true;
        rem_plan =
            Plan::MakeOffset(/*da=*/0, (full - loop.start) * loop.ldb,
                             PadFrom(loops, idx + 1, is_padding, in_partial));
      }
      in_partial[loop.dim_in_a] = saved;
      return Plan::MakeSeq(std::move(full_plan), std::move(rem_plan));
    }
    DCHECK_EQ(loop.start, 0);
    DCHECK_EQ(loop.end, loop.tile_size);
    if (!in_partial[loop.dim_in_a] || is_padding) {
      return sub_loop(loop.tile_size, is_padding);
    }
    const int64_t rem = loop.dim_size % loop.tile_size;
    return Plan::MakeSeq(
        sub_loop(rem, /*sub_padding=*/false),
        Plan::MakeOffset(
            /*da=*/0, rem * loop.ldb,
            sub_loop(loop.tile_size - rem, /*sub_padding=*/true)));
  }

  // Builds the padding fills for any partial output tiles in `chunk_id`.
  //
  // For example, a partial row-major 8x128 tile with 3 valid rows and 5 valid
  // columns has an L-shaped padding region. We cut the L horizontally so the
  // bottom rectangle spans full rows and coalesces into a single Fill1D:
  //
  //        col 0    col 5              col 128
  //          +--------+-------------------+
  //    row 0 | valid  |    padding (1)    |  Loop(3, Fill1D(123))
  //          |  3x5   |       3x123       |
  //    row 3 +--------+-------------------+
  //          |        padding (2)         |  Fill1D(5*128)
  //          |           5x128            |
  //    row 8 +----------------------------+
  //
  // Sorting the chunk's loops from largest to smallest output stride gives this
  // cut in N dimensions: at each partial-tile interior loop, `[0, rem)`
  // recurses with `is_padding=false` to pad any inner partial dimensions while
  // `[rem, tile_size)` recurses with `is_padding=true`. Recursing through inner
  // loops ensures that any inner loop partitioned across chunks only fills its
  // `[loop.start, loop.end)` slice, while `Plan::MakeLoop` coalesces unsplit
  // inner loops into a single `Fill1D`.
  Plan BuildPadding() const {
    if (!plan.tile_padding.has_value() || !plan.b_is_tiled ||
        !absl::c_any_of(chunk_loops,
                        [](const Loop& l) { return l.has_partial_tile; })) {
      return Plan::MakeNoop();
    }

    std::vector<Loop> loops;
    loops.reserve(chunk_loops.size());
    for (const Loop& l : chunk_loops) {
      if (l.ldb != 0) {
        loops.push_back(l);
      }
    }
    absl::c_sort(loops, [](const Loop& a, const Loop& b) {
      return std::make_tuple(a.ldb, !a.tile_interior) >
             std::make_tuple(b.ldb, !b.tile_interior);
    });

    absl::InlinedVector<bool, 4> in_partial(plan.a_dims.size(), false);
    Plan pad_plan = PadFrom(loops, /*idx=*/0, /*is_padding=*/false, in_partial);
    if (elements_per_byte > 1) {
      ScalePlanOutputBytes(pad_plan, elements_per_byte);
    }
    return pad_plan;
  }
};

// Builds the execution `Plan` for `chunk_id`.
Plan Planner::BuildChunkPlan(int chunk_id) const {
  const int ndim = static_cast<int>(a_dims.size());
  DCHECK_GT(ndim, 0);
  ChunkPlanBuilder builder(*this, chunk_id);
  absl::InlinedVector<bool, 4> partial(ndim, false);
  return Plan::MakeSeq(
      builder.BuildTransfer(/*idx=*/0, /*a_elems=*/0, /*b_elems=*/0, partial),
      builder.BuildPadding());
}

FlattenedPlan Planner::ToFlattenedPlan() {
  FlattenedPlan out;
  out.elem_size_in_bytes = elem_size_in_bytes;
  out.num_elems = num_elems;
  out.input_num_elems = InputNumElems();
  out.output_num_elems = OutputNumElems();
  out.input_chunk_offset_bytes = std::move(input_chunk_offset_bytes);
  out.output_chunk_offset_bytes = std::move(output_chunk_offset_bytes);
  out.input_chunk_iteration_offsets = std::move(input_chunk_iteration_offsets);
  out.output_chunk_iteration_offsets =
      std::move(output_chunk_iteration_offsets);
  out.input_chunk_size_bytes = std::move(input_chunk_size_bytes);
  out.output_chunk_size_bytes = std::move(output_chunk_size_bytes);
  out.max_input_chunk_size_bytes = max_input_chunk_size_bytes;
  out.max_output_chunk_size_bytes = max_output_chunk_size_bytes;
  out.chunk_ops = std::move(chunk_ops);
  out.scratch_size = scratch_size;
  out.bits_per_element = bits_per_element;
  out.use_fallback_pack = use_fallback_pack;
  out.inner_kernel_is_memcpy = inner_kernel_is_memcpy();
  out.tile_padding = tile_padding;
  out.inner_kernel_kind = inner_kernel_kind;
  out.interleave_size = interleave_size;
  return out;
}

absl::StatusOr<FlattenedPlan> BuildTransposePlan(const PlanOptions& o) {
  auto is_negative = [](int64_t d) { return d < 0; };
  if (absl::c_find_if(o.dims, is_negative) != o.dims.end()) {
    return InvalidArgument("dims must be non-negative, got %s",
                           absl::StrJoin(o.dims, ","));
  }
  if (o.permutation.size() != o.dims.size()) {
    return InvalidArgument(
        "dims and permutation must have equal sizes, got %d and %d",
        o.dims.size(), o.permutation.size());
  }
  if (!IsPermutation(o.permutation)) {
    return InvalidArgument("permutation argument is not valid, got: %s",
                           absl::StrJoin(o.permutation, ","));
  }
  if (o.num_threads < 1) {
    return InvalidArgument("num_threads argument must be >= 1, got: %d",
                           o.num_threads);
  }

  int ndim = o.dims.size();

  Planner plan;
  plan.num_chunks_requested = o.num_threads;
  plan.elem_size_in_bytes = o.elem_size_in_bytes;
  switch (o.elem_size_in_bytes) {
    case 1:
    case 2:
    case 4:
    case 8:
    case 16:
      break;
    default:
      return InvalidArgument("Unsupported elem_size_in_bytes=%d",
                             o.elem_size_in_bytes);
  }
  plan.num_elems = std::accumulate(o.dims.begin(), o.dims.end(), int64_t{1},
                                   std::multiplies<int64_t>());

  bool output_contiguity = o.chunk_contiguity == ChunkContiguity::kOutput;
  bool input_contiguity = o.chunk_contiguity == ChunkContiguity::kInput;

  ABSL_RETURN_IF_ERROR(ParseTilingSpecification(
      ndim, o.output_tiling,
      /*nonstandard_layout=*/output_contiguity || o.tile_padding.has_value(),
      plan.b_tiling));

  // Temporary vectors to hold un-permuted attributes
  absl::InlinedVector<int64_t, 4> temp_lda, temp_lda_tile, temp_a_tiling;

  // Parse the tile and stride specifications.
  ABSL_RETURN_IF_ERROR(ParseTilingSpecification(
      ndim, o.input_tiling,
      /*nonstandard_layout=*/o.input_striding.has_value() || input_contiguity,
      temp_a_tiling));
  ComputeStrides(plan.elem_size_in_bytes, o.dims, temp_a_tiling, temp_lda,
                 temp_lda_tile);

  // Determine tile (outer) strides
  absl::InlinedVector<int64_t, 4> input_outer_strides;
  if (o.input_striding) {
    absl::Span<int64_t const> input_strides_in_bytes = *o.input_striding;
    if (input_strides_in_bytes.size() != o.dims.size()) {
      return InvalidArgument(
          "dims and input_striding must have equal sizes, "
          "got %d and %d",
          o.dims.size(), input_strides_in_bytes.size());
    }
    input_outer_strides.assign(input_strides_in_bytes.begin(),
                               input_strides_in_bytes.end());
  } else {
    input_outer_strides = temp_lda;
  }

  // Sort the dimensions from slowest-varying (largest strides) to
  // fastest-varying (smallest strides).
  // Maps new input dim -> old input dim
  std::vector<int64_t> dim_order(ndim);
  absl::c_iota(dim_order, 0);

  auto cost = [&](int k) {
    int64_t stride = input_outer_strides.at(k);
    // If there is a dimension with size equal to the element size, sort it
    // last. This ensures that we place any stride-1 dimension last.
    bool is_stride1 = stride == o.elem_size_in_bytes;
    // If there are multiple stride-1 dimensions, we'd prefer the one that
    // matches the stride-1 dimension of the output.
    // Failing that, we'd just prefer the largest stride-1 dimension last.
    bool is_trailing_dim_in_b = o.permutation.back() == k;

    // If we are applying ef57 conversion, we want a size-2 stride-1
    // dimension last.
    bool ef57_even =
        (is_stride1 && o.transformation == Transformation::kF64ToEf57 &&
         o.dims[k] == 2);

    return std::make_tuple(is_stride1, -std::abs(stride), ef57_even,
                           is_trailing_dim_in_b, o.dims[k]);
  };
  absl::c_stable_sort(dim_order,
                      [&cost](int i, int j) { return cost(i) < cost(j); });

  // Apply permutation to all plan attributes
  // dim_order maps new input dim -> old input dim, we need its inverse to
  // compute the new permutation.
  auto inv_dim_order = InversePermutation(dim_order);
  plan.lda.reserve(ndim);
  plan.lda_tile.reserve(ndim);
  plan.a_dims.reserve(ndim);
  plan.permutation.reserve(ndim);
  plan.a_tiling.reserve(ndim);

  for (int i = 0; i < ndim; ++i) {
    int old_idx = dim_order[i];
    plan.lda.push_back(input_outer_strides.at(old_idx));
    plan.lda_tile.push_back(temp_lda_tile.at(old_idx));
    plan.a_dims.push_back(o.dims[old_idx]);
    plan.permutation.push_back(inv_dim_order[o.permutation[i]]);
    plan.a_tiling.push_back(temp_a_tiling[old_idx]);
  }

  auto is_not_one = [](int64_t x) { return x != 1; };
  plan.a_is_tiled =
      (absl::c_find_if(plan.a_tiling, is_not_one) != plan.a_tiling.end());
  plan.b_is_tiled =
      (absl::c_find_if(plan.b_tiling, is_not_one) != plan.b_tiling.end());
  if (plan.a_is_tiled && plan.b_is_tiled) {
    return Unimplemented(
        "Only one of the input and output may have a non-trivial tiling, "
        "got tilings: %s and %s",
        absl::StrJoin(plan.a_tiling, ","), absl::StrJoin(plan.b_tiling, ","));
  }

  plan.bits_per_element = o.elem_size_in_bytes * 8;
  Transformation transformation = o.transformation;
  if (o.dest_bits_per_element.has_value() &&
      transformation == Transformation::kNone) {
    transformation = Transformation::kPackSubbyte;
  }
  plan.transformation = transformation;

  plan.chunk_contiguity = o.chunk_contiguity;
  plan.tile_padding = o.tile_padding;

  switch (plan.transformation) {
    case Transformation::kNone:
      break;
    case Transformation::kF64ToEf57:
      if (o.dest_bits_per_element.has_value()) {
        return InvalidArgument(
            "EF57 conversion cannot be combined with dest_bits_per_element");
      }
      if (o.elem_size_in_bytes != sizeof(float)) {
        return InvalidArgument(
            "EF57 conversion requires a element size of %d bytes, got %d",
            sizeof(float), o.elem_size_in_bytes);
      }
      if (plan.a_dims.empty() || plan.a_dims.back() % 2 != 0 ||
          plan.lda.back() != sizeof(float)) {
        return InvalidArgument(
            "EF57 conversion requires a stride-%d dimension whose size is a "
            "multiple of 2",
            sizeof(float));
      }
      break;
    case Transformation::kPackSubbyte:
      if (o.elem_size_in_bytes != 1) {
        return InvalidArgument(
            "Pack conversion requires an element size of 1 byte, "
            "got %d",
            o.elem_size_in_bytes);
      }
      if (!o.dest_bits_per_element.has_value()) {
        return InvalidArgument(
            "Pack conversion requires dest_bits_per_element to be set");
      }
      {
        int bits = *o.dest_bits_per_element;
        if (bits != 1 && bits != 2 && bits != 4) {
          return InvalidArgument(
              "Pack conversion requires dest_bits_per_element to be 1, 2, or "
              "4, "
              "got %d",
              bits);
        }
        if (plan.tile_padding.has_value()) {
          const uint8_t mask = (1u << bits) - 1u;
          plan.tile_padding = static_cast<uint8_t>((*plan.tile_padding & mask) *
                                                   (0xffu / mask));
        }
        std::vector<int64_t> output_dims = Permute(o.dims, o.permutation);
        int elements_per_byte = 8 / bits;
        plan.bits_per_element = bits;
        if (output_dims.empty() ||
            output_dims.back() % elements_per_byte != 0 ||
            (plan.b_is_tiled &&
             plan.b_tiling.back() % elements_per_byte != 0)) {
          plan.transformation = Transformation::kNone;
          plan.use_fallback_pack = true;
        }
      }
      break;
  }

  ABSL_RETURN_IF_ERROR(plan.Initialize());
  return plan.ToFlattenedPlan();
}

namespace {

// Creates per-chunk loop vectors by splitting loop_order into per-chunk loops.
// Returns a vector of loop vectors, one per chunk. Each chunk's loops have
// their start/end bounds narrowed to represent that chunk's work.
void PartitionLoops(
    int num_dims, int num_chunks, const std::vector<Loop>& loop_order,
    std::vector<std::vector<Loop>>& result,
    std::vector<int64_t>& input_chunk_iteration_offsets,
    std::vector<int64_t>& output_chunk_iteration_offsets) {
  // Copy the base loop order for each chunk.
  result.resize(num_chunks, loop_order);
  input_chunk_iteration_offsets.resize(num_chunks);
  output_chunk_iteration_offsets.resize(num_chunks);
  for (int chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
    // For each loop, narrow the start/end bounds to this chunk's portion.
    int task_id_remaining = chunk_id;
    int num_tasks_remaining = num_chunks;

    for (size_t i = 0; i < loop_order.size(); ++i) {
      Loop& chunk_loop = result[chunk_id][i];
      const Loop& base_loop = loop_order[i];

      num_tasks_remaining /= base_loop.parallelism;
      int task_id = task_id_remaining / num_tasks_remaining;
      task_id_remaining = task_id_remaining % num_tasks_remaining;

      // Divide this loop's iterations (in element units) among parallelism
      // tasks.
      int64_t iterations = base_loop.end - base_loop.start;
      int64_t blocks = CeilOfRatio(iterations, base_loop.inc);
      int64_t blocks_per_task =
          CeilOfRatio<int64_t>(blocks, base_loop.parallelism);

      chunk_loop.start =
          base_loop.start +
          std::min(iterations, task_id * blocks_per_task * base_loop.inc);
      chunk_loop.end =
          base_loop.start +
          std::min(iterations, (task_id + 1) * blocks_per_task * base_loop.inc);
      input_chunk_iteration_offsets[chunk_id] +=
          chunk_loop.start * chunk_loop.lda;
      output_chunk_iteration_offsets[chunk_id] +=
          chunk_loop.start * chunk_loop.ldb;
    }

    // Set has_partial_tile for tile interior loops.
    // First pass: find which dimensions' exterior loops include the last tile.
    std::vector<bool> dims_with_partial(num_dims, false);
    for (const Loop& loop : result[chunk_id]) {
      if (!loop.tile_interior && loop.dim_size % loop.tile_size != 0) {
        int64_t full_tiles = CeilOfRatio(loop.dim_size, loop.tile_size);
        if (loop.end >= full_tiles) {
          dims_with_partial[loop.dim_in_a] = true;
        }
      }
    }
    // Second pass: set has_partial_tile on interior loops.
    for (Loop& loop : result[chunk_id]) {
      if (loop.tile_interior && dims_with_partial[loop.dim_in_a]) {
        loop.has_partial_tile = true;
      }
    }
  }
}

}  // namespace

absl::Status Planner::Initialize() {
  if (num_elems == 0) {
    return absl::OkStatus();
  }
  // permutation maps dimensions of b to a
  // inverse_permutation maps dimensions of a to b
  std::vector<int64_t> inverse_permutation = InversePermutation(permutation);

  int ndim = a_dims.size();

  // Returns the inner of the strides for dimension `d`. If the dimension is
  // tiled, that's the tile stride, and if it is not, that is the untiled
  // stride.
  auto inner_stride = [&](int d, absl::Span<int64_t const> strides,
                          absl::Span<int64_t const> tile_strides,
                          absl::Span<int64_t const> tiling) {
    return tiling[d] > 1 ? tile_strides[d] : strides[d];
  };

  // If the plan is 0-dimensional, or the innermost dimension of A is not of
  // stride 1, adds a trivial size 1 dimension. The transpose kernels rely on
  // the presence of a stride-1 innermost dimension in the input.
  int pos_stride1a_in_a = ndim - 1;
  if (lda.empty() ||
      inner_stride(ndim - 1, lda, lda_tile, a_tiling) != elem_size_in_bytes) {
    int dim = static_cast<int>(a_dims.size());
    permutation.push_back(dim);
    inverse_permutation.push_back(dim);
    a_dims.push_back(1);
    lda.push_back(elem_size_in_bytes);
    lda_tile.push_back(1);
    a_tiling.push_back(1);
    b_tiling.push_back(1);
    ++ndim;
    ++pos_stride1a_in_a;
  }

  is_identity_permutation = true;
  for (int i = 0; i < permutation.size(); ++i) {
    if (permutation[i] != i) {
      is_identity_permutation = false;
      break;
    }
  }

  b_dims = Permute(a_dims, permutation);

  ComputeStrides(elem_size_in_bytes, b_dims, b_tiling, ldb, ldb_tile);

  // Find the innermost dimension of B that is stride 1 element. We know such a
  // dimension exists, because we do not accept arbitrary stridings for B, but
  // it may not be the last dimension in the presence of tiling.
  int pos_stride1b_in_b = -1;
  for (int i = ndim - 1; i >= 0; --i) {
    if (inner_stride(i, ldb, ldb_tile, b_tiling) == elem_size_in_bytes) {
      pos_stride1b_in_b = i;
      break;
    }
  }
  CHECK_GE(pos_stride1b_in_b, 0);

  int pos_stride1b_in_a = permutation[pos_stride1b_in_b];
  if (transformation == Transformation::kPackSubbyte) {
    int elements_per_byte = 8 / bits_per_element;
    int64_t b_stride1_tile_size =
        std::max(a_tiling[pos_stride1b_in_a], b_tiling[pos_stride1b_in_b]);
    if (b_dims[pos_stride1b_in_b] % elements_per_byte != 0 ||
        (b_stride1_tile_size > 1 &&
         b_stride1_tile_size % elements_per_byte != 0)) {
      transformation = Transformation::kNone;
      use_fallback_pack = true;
    }
  }

  int64_t stride_pos1a =
      inner_stride(pos_stride1a_in_a, lda, lda_tile, a_tiling);
  int64_t stride_pos1b =
      inner_stride(pos_stride1b_in_b, ldb, ldb_tile, b_tiling);

  inner_kernel_kind = (pos_stride1b_in_a == pos_stride1a_in_a) &&
                              (stride_pos1a == elem_size_in_bytes) &&
                              (stride_pos1b == elem_size_in_bytes)
                          ? InnerKernelKind::kMemcpy
                          : InnerKernelKind::kDefault;

  if (inner_kernel_is_memcpy() && transformation != Transformation::kNone &&
      transformation != Transformation::kPackSubbyte) {
    return InvalidArgument(
        "Memcpy-based plans cannot be used with transformations.");
  }

  // Calculate sentinel strides.
  if (!inner_kernel_is_memcpy()) {
    int pos_stride1a_in_b = inverse_permutation[pos_stride1a_in_a];
    sentinel_lda = inner_stride(pos_stride1b_in_a, lda, lda_tile, a_tiling);
    sentinel_ldb = inner_stride(pos_stride1a_in_b, ldb, ldb_tile, b_tiling);
  }

  // Order to traverse dimensions, from slowest-varying to fastest-varying.
  std::vector<Loop> loop_order;

  loop_order.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    Loop loop;
    loop.dim_in_a = i;
    loop.tile_interior = false;
    int64_t dim_size = a_dims[i];
    int64_t tile_size = std::max(a_tiling[i], b_tiling[inverse_permutation[i]]);
    bool has_partial_tile = (dim_size % tile_size != 0);
    if (has_partial_tile) {
      // We only need to track tile interiors correctly if there are partial
      // tiles.
      loop.dim_size = dim_size;
      loop.tile_size = tile_size;
    } else {
      loop.dim_size = dim_size / tile_size;
      loop.tile_size = 1;
    }

    loop.lda = lda[i];
    if (a_tiling[i] == 1) {
      loop.lda *= tile_size;
    }
    loop.ldb = ldb[inverse_permutation[i]];
    if (b_tiling[inverse_permutation[i]] == 1) {
      loop.ldb *= tile_size;
    }
    loop.is_inner_dim_in_a = (tile_size == 1) && (i == pos_stride1a_in_a);
    loop.is_inner_dim_in_b = (tile_size == 1) && (i == pos_stride1b_in_a);
    loop_order.push_back(loop);

    if (tile_size > 1) {
      loop.tile_interior = has_partial_tile;
      if (!has_partial_tile) {
        loop.dim_size = tile_size;
        loop.tile_size = 1;
      }
      loop.lda = a_is_tiled ? lda_tile[i] : lda[i];
      loop.ldb = b_is_tiled ? ldb_tile[inverse_permutation[i]]
                            : ldb[inverse_permutation[i]];
      loop.is_inner_dim_in_a = (i == pos_stride1a_in_a);
      loop.is_inner_dim_in_b = (i == pos_stride1b_in_a);
      loop_order.push_back(loop);
    }
  }

  auto loops_to_string = [](absl::Span<const Loop> loops) {
    return absl::StrJoin(loops, "\n  ", [](std::string* out, const Loop& l) {
      absl::StrAppend(out, l.ToString());
    });
  };

  VLOG(5) << "Before RemoveTrivialLoops: " << loops_to_string(loop_order);
  RemoveTrivialLoops(loop_order);
  VLOG(5) << "After RemoveTrivialLoops: " << loops_to_string(loop_order);
  CoalesceLoops(loop_order);
  VLOG(5) << "After CoalesceLoops: " << loops_to_string(loop_order);

  // Bound the block sizes so they are smaller than the stride-1 dimension
  // size.
  int64_t a_stride1_size =
      std::max(a_tiling[pos_stride1a_in_a],
               b_tiling[inverse_permutation[pos_stride1a_in_a]]);
  if (a_stride1_size == 1) {
    a_stride1_size = a_dims[pos_stride1a_in_a];
  } else {
    // If there's only one tile, we should use the dimension size.
    a_stride1_size = std::min(a_dims[pos_stride1a_in_a], a_stride1_size);
  }
  int64_t b_stride1_size = std::max(a_tiling[permutation[pos_stride1b_in_b]],
                                    b_tiling[pos_stride1b_in_b]);
  if (b_stride1_size == 1) {
    b_stride1_size = b_dims[pos_stride1b_in_b];
  } else {
    b_stride1_size = std::min(b_stride1_size, b_dims[pos_stride1b_in_b]);
  }

  // For kInterleave and kDeinterleave, the trip count of the large side's
  // stride-1 loop.
  int64_t large_loop_size = 0;
  if (!inner_kernel_is_memcpy() && transformation == Transformation::kNone) {
    const auto loop_a_it = absl::c_find_if(
        loop_order, [](const Loop& l) { return l.is_inner_dim_in_a; });
    const auto loop_b_it = absl::c_find_if(
        loop_order, [](const Loop& l) { return l.is_inner_dim_in_b; });
    DCHECK(loop_a_it != loop_order.end());
    DCHECK(loop_b_it != loop_order.end());
    // loop_a and loop_b iterate over A's and B's stride-1 dimensions (or their
    // tile interiors).
    const Loop& loop_a = *loop_a_it;
    const Loop& loop_b = *loop_b_it;
    DCHECK_EQ(loop_a.lda, elem_size_in_bytes);
    DCHECK_EQ(loop_b.ldb, elem_size_in_bytes);
    DCHECK_EQ(loop_a.ldb, sentinel_ldb);
    DCHECK_EQ(loop_b.lda, sentinel_lda);
    const auto trip_count = [](const Loop& l) {
      return l.tile_interior ? std::min(l.tile_size, l.dim_size) : l.dim_size;
    };
    const int64_t loop_a_size = trip_count(loop_a);
    const int64_t loop_b_size = trip_count(loop_b);
    // Returns whether to use the `kind` kernel with K = small.dim_size, where
    // `small` is the stride-1 loop of the small side.
    const auto use_kernel = [&](InnerKernelKind kind, const Loop& small,
                                int64_t group_stride, int64_t large_size) {
      DCHECK_EQ(small.tile_size == 1, !small.tile_interior);
      const int64_t k = small.dim_size;
      // With a partial tile, the trip count is not a constant K.
      const bool no_partial_tile = !small.tile_interior;
      // The K-groups are back to back, so the small side is one contiguous run.
      const bool contiguous = group_stride == k * elem_size_in_bytes;
      // Faster than the default kernel for this K and element size.
      const bool faster = SmallDimKernelIsFaster(kind, k, elem_size_in_bytes);
      const bool large_side_is_long = large_size >= 8;
      return no_partial_tile && contiguous && faster && large_side_is_long;
    };
    if (use_kernel(InnerKernelKind::kInterleave, loop_b, sentinel_ldb,
                   loop_a_size)) {
      inner_kernel_kind = InnerKernelKind::kInterleave;
      interleave_size = loop_b.dim_size;
      large_loop_size = loop_a_size;
    } else if (use_kernel(InnerKernelKind::kDeinterleave, loop_a, sentinel_lda,
                          loop_b_size)) {
      inner_kernel_kind = InnerKernelKind::kDeinterleave;
      interleave_size = loop_a.dim_size;
      large_loop_size = loop_b_size;
    }
  }

  if (inner_kernel_is_memcpy()) {
    inner_block_elems = -1;
    outer_block_elems_a = -1;
    outer_block_elems_b = -1;
  } else {
    // What are the smallest and largest block sizes for which we have a
    // vectorized kernel for this element size?
    const int max_outer_block_elems =
        kMaxOuterBlockSizeBytes / elem_size_in_bytes;
    const int max_inner_block_bytes =
        GetMaxTransposeInnerBlockSizeBytes(elem_size_in_bytes);
    const int min_inner_block_elems = 1;
    const int max_inner_block_elems = std::min<int>(
        max_outer_block_elems, max_inner_block_bytes / elem_size_in_bytes);
    if (inner_kernel_kind == InnerKernelKind::kInterleave ||
        inner_kernel_kind == InnerKernelKind::kDeinterleave) {
      // Cap the auto vectorized interleave strip size at the 32 byte block
      // scale (at most 2048 elements for 1 byte types, or 28 KiB across input
      // and output at K = 7) so strips fit in L1D and kF64ToEf57 scratch stays
      // within kStackScratchBytes (4 KiB).
      constexpr int kMaxInterleaveBlockBytes = 32;
      const int max_interleave_block_elems =
          std::min(max_inner_block_bytes, kMaxInterleaveBlockBytes) /
          elem_size_in_bytes;
      const int64_t strip_elems = std::min<int64_t>(
          large_loop_size, max_interleave_block_elems * max_outer_block_elems);
      inner_block_elems = 1;
      outer_block_elems_a =
          (inner_kernel_kind == InnerKernelKind::kInterleave) ? strip_elems : 1;
      outer_block_elems_b =
          (inner_kernel_kind == InnerKernelKind::kDeinterleave) ? strip_elems
                                                                : 1;
    } else {
      inner_block_elems = max_inner_block_elems;
      while (inner_block_elems > std::min(a_stride1_size, b_stride1_size)) {
        inner_block_elems /= 2;
      }
      if (inner_block_elems < min_inner_block_elems) {
        // Size is smaller than our smallest vectorized kernel. Use the scalar
        // path.
        inner_block_elems = 1;
      }
      outer_block_elems_a = FloorOfRatio<int64_t>(
          std::min<int64_t>(max_outer_block_elems, a_stride1_size),
          inner_block_elems);
      outer_block_elems_a = std::max<int64_t>(outer_block_elems_a, 1);
      outer_block_elems_b = FloorOfRatio<int64_t>(
          std::min<int64_t>(max_outer_block_elems, b_stride1_size),
          inner_block_elems);
      outer_block_elems_b = std::max<int64_t>(outer_block_elems_b, 1);
    }
  }

  // Identify contiguous loops for chunk scheduling.
  // Loops with smallest strides in the contiguous buffer form the contiguous
  // region. Accumulate loops until their combined iteration count reaches
  // the target chunk size.
  IdentifyContiguousLoops(loop_order);
  ChooseLoopOrder(loop_order, inner_kernel_is_memcpy(), chunk_contiguity,
                  is_identity_permutation, inner_block_elems,
                  outer_block_elems_a, outer_block_elems_b);

  for (Loop& loop : loop_order) {
    if (loop.tile_interior || loop.tile_size == 1) {
      if (inner_kernel_is_memcpy()) {
        if (transformation == Transformation::kPackSubbyte &&
            loop.is_inner_dim_in_a) {
          loop.inc = 8 / bits_per_element;
        }
      } else if (loop.is_inner_dim_in_a) {
        loop.inc = (inner_kernel_kind == InnerKernelKind::kDeinterleave)
                       ? loop.dim_size
                       : inner_block_elems * outer_block_elems_a;
      } else if (loop.is_inner_dim_in_b) {
        loop.inc = (inner_kernel_kind == InnerKernelKind::kInterleave)
                       ? loop.dim_size
                       : inner_block_elems * outer_block_elems_b;
      }
    }
  }

  // It is a required invariant of the loop order that tile interiors always
  // appear after the corresponding tile exterior. This is a consequence of the
  // heuristic above, because the tile interior must have smaller strides in
  // both input and output.

  // The stride-1 loop must be innermost for a memcpy loop.
  DCHECK(!inner_kernel_is_memcpy() || loop_order.back().is_inner_dim_in_a)
      << ToString();

  int num_chunks = ChooseParallelizationStrategy(loop_order);
  VLOG(5) << "After ChooseParallelizationStrategy num_chunks=" << num_chunks
          << " loops: " << loops_to_string(loop_order);
  PartitionLoops(a_dims.size(), num_chunks, loop_order, chunk_loops,
                 input_chunk_iteration_offsets, output_chunk_iteration_offsets);
  ComputeChunkSizes();
  VLOG(8) << "Before plan build: " << ToString();
  chunk_ops.resize(num_chunks);
  scratch_size = 0;
  std::vector<std::string> vlog_plans;
  const bool vlog_plan = VLOG_IS_ON(5);
  if (vlog_plan) {
    vlog_plans.reserve(num_chunks);
  }
  for (int chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
    Plan chunk_plan = BuildChunkPlan(chunk_id);
    if (vlog_plan) {
      std::string s = "thread:\n";
      FormatPlan(chunk_plan, /*indent=*/4, elem_size_in_bytes,
                 inner_kernel_kind, interleave_size, &s);
      vlog_plans.push_back(std::move(s));
    }
    const int64_t chunk_scratch =
        FlattenPlan(chunk_plan, elem_size_in_bytes, inner_kernel_kind,
                    inner_block_elems, interleave_size, chunk_ops[chunk_id]);
    scratch_size = std::max(scratch_size, chunk_scratch);
  }

  if (vlog_plan) {
    VLOG(5) << "Final plan: " << ToString(vlog_plans);
  }
  return absl::OkStatus();
}

void ChooseLoopOrder(std::vector<Loop>& loop_order, bool is_memcpy,
                     ChunkContiguity chunk_contiguity,
                     bool is_identity_permutation, int inner_block_elems,
                     int outer_block_elems_a, int outer_block_elems_b) {
  // Loop order heuristic: try to make loops with small strides innermost.
  // We use a greedy selection to pick loops from slowest-varying to
  // fastest-varying, subject to hard constraints:
  // 1. If the inner kernel is a memcpy, the stride-1 dimension MUST be
  //    innermost.
  // 2. A tile interior MUST come after its corresponding tile exterior.
  std::vector<Loop> remaining = std::move(loop_order);
  loop_order.clear();
  loop_order.reserve(remaining.size());

  // Computes a sort key for each candidate loop, returning a 3-tuple:
  //   <contiguity, -miss_cost, -stride_tie_breaker>
  // Lexicographical ordering selects the smallest tuple first, placing that
  // loop into an outer position (from slowest-varying to fastest-varying):
  // 1. Contiguity category (non-contiguous loops placed in outer loops).
  // 2. Primary cost: cache-line misses (higher miss cost placed in outer
  // loops).
  // 3. Secondary tie-breaker: when cache misses tie (e.g. both loops have
  //    strides >= 128B), the loop with larger minimum stride goes outer to
  //    favor better locality in at least one buffer for the inner loop.
  auto soft_cost = [&](const Loop& l) -> std::tuple<int, double, double> {
    int64_t a_stride = std::abs(l.lda);
    if (!is_memcpy && l.is_inner_dim_in_a) {
      a_stride *= inner_block_elems * outer_block_elems_a;
    }
    int64_t b_stride = std::abs(l.ldb);
    if (!is_memcpy && l.is_inner_dim_in_b) {
      b_stride *= inner_block_elems * outer_block_elems_b;
    }

    switch (chunk_contiguity) {
      case ChunkContiguity::kOutput:
        return std::make_tuple(0, -b_stride, 0.0);
      case ChunkContiguity::kInput:
        return std::make_tuple(0, -a_stride, 0.0);
      case ChunkContiguity::kNone: {
        constexpr double kOutputPenalty = 1.01;
        const double min_stride =
            std::min<double>(a_stride * kOutputPenalty, b_stride);

        // Pure tile-packing identity linearizers/delinearizers preserve the
        // baseline std::min cost to keep 2D hardware tile pairings intact.
        if (is_identity_permutation) {
          return std::make_tuple(l.contiguity, -min_stride, 0.0);
        }

        // Model the memory traffic of placing this loop in an inner position
        // based on the count of cache lines missed (bytes pulled from L2 to
        // L1). Memory transfers occur at cache-line granularity (128-byte cache
        // line pairs on modern CPUs like AMD Zen 3/4).
        // - Strides < 128 bytes share cache lines across iterations (spatial
        //   reuse).
        // - Strides >= 128 bytes incur a full cache-line miss per iteration.
        constexpr double kCacheLineBytes = 128.0;

        const double misses_a = std::min(1.0, a_stride / kCacheLineBytes);
        const double misses_b = std::min(1.0, b_stride / kCacheLineBytes);
        const double base_cost = misses_a + misses_b * kOutputPenalty;

        // Break ties among loops with full cache-line misses using the minimum
        // stride across both buffers (favoring loops with better locality in
        // at least one buffer, such as tile interior loops over tile exterior
        // loops).
        return std::make_tuple(l.contiguity, -base_cost, -min_stride);
      }
    }
  };

  while (!remaining.empty()) {
    int best_idx = -1;
    for (int i = 0; i < remaining.size(); ++i) {
      const Loop& l = remaining[i];

      // Hard constraint 1: memcpy kernel requirement.
      if (is_memcpy && l.is_inner_dim_in_a && remaining.size() > 1) {
        continue;
      }

      // Hard constraint 2: tile ordering.
      // A tile interior MUST come after its corresponding tile exterior.
      if (l.tile_interior) {
        auto is_exterior = [&](const Loop& r) {
          return !r.tile_interior && r.dim_in_a == l.dim_in_a;
        };
        if (absl::c_any_of(remaining, is_exterior)) {
          continue;
        }
      }

      if (best_idx == -1 || soft_cost(l) < soft_cost(remaining[best_idx])) {
        best_idx = i;
      }
    }
    CHECK_NE(best_idx, -1);
    loop_order.push_back(std::move(remaining[best_idx]));
    remaining.erase(remaining.begin() + best_idx);
  }

  VLOG(5) << "After loop ordering sort: "
          << absl::StrJoin(loop_order, ", ",
                           [](std::string* out, const Loop& l) {
                             absl::StrAppend(out, l.ToString());
                           });
}
int Planner::ChooseParallelizationStrategy(
    std::vector<Loop>& loop_order) const {
  int available_parallelism = num_chunks_requested;

  // Compute the number of iterations in `loop`.
  auto loop_iterations = [&](const Loop& loop) {
    int64_t size = loop.tile_interior
                       ? ((tile_padding.has_value() && b_is_tiled)
                              ? loop.tile_size
                              : std::min(loop.tile_size, loop.dim_size))
                       : (CeilOfRatio(loop.dim_size, loop.tile_size));
    return CeilOfRatio<int64_t>(size, loop.inc);
  };

  // Estimate the number of bytes each iteration of each loop processes.
  absl::InlinedVector<int64_t, 4> work_in_bytes(loop_order.size());
  int64_t acc = elem_size_in_bytes;
  if (inner_kernel_kind == InnerKernelKind::kInterleave ||
      inner_kernel_kind == InnerKernelKind::kDeinterleave) {
    acc *= interleave_size * inner_block_elems * outer_block_elems_a *
           outer_block_elems_b;
  } else if (!inner_kernel_is_memcpy()) {
    acc *= inner_block_elems * inner_block_elems * outer_block_elems_a *
           outer_block_elems_b;
  } else if (transformation == Transformation::kPackSubbyte) {
    acc *= 8 / bits_per_element;
  }
  auto work_it = work_in_bytes.rbegin();
  for (auto it = loop_order.rbegin(); it != loop_order.rend(); ++it) {
    *work_it++ = acc;
    acc *= loop_iterations(*it);
  }
  VLOG(7) << "Per-loop iteration work in bytes: "
          << absl::StrJoin(work_in_bytes, ",");

  // Heuristic that attempts to parallelize the outermost loops, down to a
  // minimum per-thread number of bytes processed.
  int num_chunks = 1;
  for (size_t i = 0; i < loop_order.size(); ++i) {
    Loop& loop = loop_order[i];
    CHECK_GE(available_parallelism, 1);

    // Initialize loop iteration bounds to full range in element units.
    loop.start = 0;
    loop.end = loop.tile_interior ? loop.tile_size
                                  : CeilOfRatio(loop.dim_size, loop.tile_size);

    int64_t iterations_without_blocking = loop.end - loop.start;
    int64_t iterations = loop_iterations(loop);

    // Contiguous loops must not be parallelized to maintain chunk contiguity.
    // Tile interiors with partial tiles must not be parallelized since
    // otherwise we will not account for things like padding correctly when
    // calculating chunk sizes.
    if (loop.contiguity > 1 || loop.tile_interior) {
      loop.parallelism = 1;
      if (chunk_contiguity != ChunkContiguity::kNone &&
          loop.parallelism < iterations_without_blocking) {
        available_parallelism = 1;
      }
      VLOG(8) << "loop " << i << " restricted to parallelism=1"
              << " iterations=" << iterations_without_blocking
              << " available_parallelism=" << available_parallelism;
      continue;
    }

    int kMinBytesPerThread =
        chunk_contiguity != ChunkContiguity::kNone ? 1 : (1 << 26);
    int64_t min_iterations_per_thread =
        CeilOfRatio<int64_t>(kMinBytesPerThread, work_in_bytes[i]);
    int64_t parallel_work = CeilOfRatio(iterations, min_iterations_per_thread);

    VLOG(8) << "loop " << i << ": iterations=" << iterations
            << " parallel_work=" << parallel_work
            << " available_parallelism=" << available_parallelism;
    int parallelism = std::min<int64_t>(available_parallelism, parallel_work);
    if (parallelism > 1) {
      // If we use CeilOfRatio(iterations, parallelism) as the chunk size, we
      // might end up with fewer chunks than parallelism if the chunk size is
      // large. For example, if iterations=17 and parallelism=16,
      // chunk_size=2. Then useful_tasks=9. We should reduce parallelism to 9.
      int64_t chunk_size =
          CeilOfRatio(iterations, static_cast<int64_t>(parallelism));
      int64_t useful_tasks = CeilOfRatio(iterations, chunk_size);
      parallelism = useful_tasks;
    }
    loop.parallelism = parallelism;
    num_chunks *= loop.parallelism;

    // Once a loop is sequential (parallelism < iterations), we must not
    // parallelize any inner loops if we want to maintain chunk contiguity.
    // This is because parallelizing an inner loop while a sequential outer
    // loop has multiple iterations would interleave the inner splits across
    // the outer iterations.
    if (chunk_contiguity != ChunkContiguity::kNone &&
        loop.parallelism < iterations_without_blocking) {
      available_parallelism = 1;
    } else {
      available_parallelism /= loop.parallelism;
    }
  }
  return num_chunks;
}

void Planner::ComputeChunkSizes() {
  input_chunk_size_bytes.resize(chunk_loops.size(), 0);
  output_chunk_size_bytes.resize(chunk_loops.size(), 0);
  input_chunk_offset_bytes.resize(chunk_loops.size(), 0);
  output_chunk_offset_bytes.resize(chunk_loops.size(), 0);
  max_input_chunk_size_bytes = 0;
  max_output_chunk_size_bytes = 0;
  // Compute chunk sizes from the pre-computed chunk loop bounds.
  // For each loop, accumulate the offset range based on the iteration bounds
  // and strides. Strides can be negative, so track both min and max.
  for (size_t chunk_id = 0; chunk_id < chunk_loops.size(); ++chunk_id) {
    absl::Span<const Loop> loops = chunk_loops[chunk_id];

    int64_t input_min = 0, input_max = 0;
    int64_t output_min = 0, output_max = 0;

    for (const Loop& loop : loops) {
      if (loop.start >= loop.end) {
        continue;  // Empty iteration range
      }

      int64_t first = loop.start;
      int64_t last = loop.end - 1;  // Last iteration (exclusive end)

      // For tile interior loops with partial tiles, use the full tile size.
      int64_t input_last = last;
      int64_t output_last = last;
      if (loop.tile_interior && loop.has_partial_tile) {
        int64_t partial_size = loop.dim_size % loop.tile_size;
        if (a_is_tiled) {
          input_last = loop.tile_size - 1;
        } else {
          input_last = partial_size - 1;
        }
        // Output: if tiled, use full tile extent (no adjustment)
        // If untiled, use partial size
        if (b_is_tiled) {
          output_last = loop.tile_size - 1;
        } else {
          output_last = partial_size - 1;
        }
      }

      // Accumulate min/max (strides can be negative)
      int64_t input_first_offset = first * loop.lda;
      int64_t input_last_offset = input_last * loop.lda;
      input_min += std::min(input_first_offset, input_last_offset);
      input_max += std::max(input_first_offset, input_last_offset);

      int64_t output_first_offset = first * loop.ldb;
      int64_t output_last_offset = output_last * loop.ldb;
      output_min += std::min(output_first_offset, output_last_offset);
      output_max += std::max(output_first_offset, output_last_offset);
    }

    input_chunk_offset_bytes[chunk_id] = input_min;
    output_chunk_offset_bytes[chunk_id] = output_min;

    // Adjust iteration offsets to be relative to the physical start of the
    // chunk.
    input_chunk_iteration_offsets[chunk_id] -= input_min;
    output_chunk_iteration_offsets[chunk_id] -= output_min;

    // Extent = range of offsets + element size at the max position.
    input_chunk_size_bytes[chunk_id] =
        input_max - input_min + elem_size_in_bytes;
    output_chunk_size_bytes[chunk_id] =
        output_max - output_min + elem_size_in_bytes;
    if (transformation == Transformation::kPackSubbyte) {
      const int elements_per_byte = 8 / bits_per_element;
      DCHECK_EQ(output_chunk_offset_bytes[chunk_id] % elements_per_byte, 0);
      DCHECK_EQ(output_chunk_iteration_offsets[chunk_id] % elements_per_byte,
                0);
      DCHECK_EQ(output_chunk_size_bytes[chunk_id] % elements_per_byte, 0);
      output_chunk_offset_bytes[chunk_id] /= elements_per_byte;
      output_chunk_iteration_offsets[chunk_id] /= elements_per_byte;
      output_chunk_size_bytes[chunk_id] /= elements_per_byte;
    }

    max_input_chunk_size_bytes =
        std::max(max_input_chunk_size_bytes, input_chunk_size_bytes[chunk_id]);
    max_output_chunk_size_bytes = std::max(max_output_chunk_size_bytes,
                                           output_chunk_size_bytes[chunk_id]);
  }
}

std::string Planner::ToString(
    absl::Span<const std::string> plan_strings) const {
  std::string plans_str = absl::StrJoin(plan_strings, "\n");
  auto format_loop = [](std::string* out, const Loop& loop) {
    absl::StrAppend(out, loop.ToString());
  };
  std::vector<std::string> chunk_strings;
  chunk_strings.reserve(chunk_loops.size());
  for (int i = 0; i < chunk_loops.size(); ++i) {
    chunk_strings.push_back(absl::StrFormat(
        "    chunk %d: physical_input_offset=%d physical_output_offset=%d "
        "input_size=%d output_size=%d "
        "logical_input_offset=%d logical_output_offset=%d loops:\n      %s",
        i, input_chunk_offset_bytes[i], output_chunk_offset_bytes[i],
        input_chunk_size_bytes[i], output_chunk_size_bytes[i],
        input_chunk_iteration_offsets[i], output_chunk_iteration_offsets[i],
        absl::StrJoin(chunk_loops[i], "\n      ", format_loop)));
  }
  std::string chunk_loops_str = absl::StrJoin(chunk_strings, "\n");
  std::string transformation_str;
  switch (transformation) {
    case Transformation::kNone:
      transformation_str = "none";
      break;
    case Transformation::kF64ToEf57:
      transformation_str = "ef57";
      break;
    case Transformation::kPackSubbyte:
      transformation_str = absl::StrCat("pack(", bits_per_element, ")");
      break;
  }
  std::string inner_kernel_str;
  switch (inner_kernel_kind) {
    case InnerKernelKind::kDefault:
      inner_kernel_str = "default";
      break;
    case InnerKernelKind::kMemcpy:
      inner_kernel_str = "memcpy";
      break;
    case InnerKernelKind::kInterleave:
      inner_kernel_str = absl::StrCat("interleave(", interleave_size, ")");
      break;
    case InnerKernelKind::kDeinterleave:
      inner_kernel_str = absl::StrCat("deinterleave(", interleave_size, ")");
      break;
  }
  std::string tile_padding_str = tile_padding.has_value()
                                     ? absl::StrFormat("0x%02x", *tile_padding)
                                     : "none";
  return absl::StrFormat(
      "elem_size=%d a_dims=%s b_dims=%s permutation=%s a_tiling=%s b_tiling=%s "
      "lda=%s lda_tile=%s ldb=%s ldb_tile=%s "
      "outer_bs=[%d,%d] inner_bs=%d inner_kernel=%s "
      "transformation=%s tile_padding=%s scratch_size=%d "
      "num_chunks_requested=%d\n"
      "chunk_loops:\n%s\n"
      "plans:\n%s",
      elem_size_in_bytes, absl::StrJoin(a_dims, ","),
      absl::StrJoin(Permute(a_dims, permutation), ","),
      absl::StrJoin(permutation, ","), absl::StrJoin(a_tiling, ","),
      absl::StrJoin(b_tiling, ","), absl::StrJoin(lda, ","),
      absl::StrJoin(lda_tile, ","), absl::StrJoin(ldb, ","),
      absl::StrJoin(ldb_tile, ","), outer_block_elems_a, outer_block_elems_b,
      inner_block_elems, inner_kernel_str, transformation_str, tile_padding_str,
      scratch_size, num_chunks_requested, chunk_loops_str, plans_str);
}

void Planner::IdentifyContiguousLoops(std::vector<Loop>& loop_order) const {
  if (chunk_contiguity != ChunkContiguity::kNone) {
    int64_t target_chunk_bytes = CeilOfRatio<int64_t>(
        num_elems * elem_size_in_bytes, num_chunks_requested);

    // Sort loops by stride in the contiguous buffer (ascending).
    auto contiguous_stride = [&](const Loop& l) {
      return chunk_contiguity == ChunkContiguity::kInput ? std::abs(l.lda)
                                                         : std::abs(l.ldb);
    };

    // Create index vector and sort by contiguous stride.
    std::vector<size_t> indices(loop_order.size());
    std::iota(indices.begin(), indices.end(), 0);
    auto cost = [&](const Loop& l) {
      int64_t stride = contiguous_stride(l);
      return std::make_tuple(!l.tile_interior, stride);
    };
    absl::c_stable_sort(indices, [&](size_t a, size_t b) {
      return cost(loop_order[a]) < cost(loop_order[b]);
    });

    int prev = -1;
    for (size_t idx : indices) {
      Loop& loop = loop_order[idx];
      int64_t stride = contiguous_stride(loop);
      if (stride == 0 ||
          (!loop.tile_interior && loop.dim_size == loop.tile_size)) {
        continue;
      }
      if (stride >= target_chunk_bytes) {
        break;
      }
      loop.contiguity = 1;
      if (prev != -1) {
        loop_order[prev].contiguity = 2;
      }
      prev = idx;
    }
  }
}

void RemoveTrivialLoops(std::vector<Loop>& loops) {
  auto it = std::remove_if(loops.begin(), loops.end(), [](const Loop& loop) {
    // We must preserve the loop if it corresponds to the innermost dimension
    // of the layout, because the kernels (especially TransposeConstStride1)
    // rely on finding a node with is_inner_dim_in_a/b set to true.
    if (loop.is_inner_dim_in_a || loop.is_inner_dim_in_b) {
      return false;
    }
    if (loop.tile_interior) {
      return loop.tile_size == 1;
    }
    // Exterior loop.
    // Trivial if dim_size == tile_size (1 complete tile, no partials). This
    // also accounts for the case where the dimension is of size 1, since in
    // that case the tile size is also 1.
    return loop.dim_size == loop.tile_size;
  });
  loops.erase(it, loops.end());
}

void CoalesceLoops(std::vector<Loop>& loops) {
  if (loops.empty()) {
    return;
  }

  // Coalesce from slow-varying to fast-varying (outer to inner).
  // loops[0] is slowest.
  int write_pos = 0;
  for (int read_pos = 1; read_pos < loops.size(); ++read_pos) {
    Loop& outer = loops[write_pos];
    const Loop& inner = loops[read_pos];

    int64_t inner_iter_size = inner.tile_interior
                                  ? inner.tile_size
                                  : (inner.dim_size / inner.tile_size);

    // Two loops can be coalesced if:
    // * neither has a partial tile
    // * the inner loop is a multiple of the outer loop.
    bool coalescable = (outer.dim_size % outer.tile_size == 0) &&
                       (inner.dim_size % inner.tile_size == 0) &&
                       (outer.lda == inner.lda * inner_iter_size) &&
                       (outer.ldb == inner.ldb * inner_iter_size);
    if (coalescable) {
      if (outer.tile_interior) {
        outer.tile_size *= inner.tile_size;
        outer.dim_size *= inner.dim_size;
      } else {
        outer.dim_size *= inner_iter_size;
      }

      outer.lda = inner.lda;
      outer.ldb = inner.ldb;

      outer.is_inner_dim_in_a =
          inner.is_inner_dim_in_a || outer.is_inner_dim_in_a;
      outer.is_inner_dim_in_b =
          inner.is_inner_dim_in_b || outer.is_inner_dim_in_b;

      // Don't advance write_pos, so we can merge more into 'outer'.
    } else {
      ++write_pos;
      if (write_pos != read_pos) {
        loops[write_pos] = inner;
      }
    }
  }
  loops.resize(write_pos + 1);
}

}  // namespace transpose_internal
}  // namespace xla
