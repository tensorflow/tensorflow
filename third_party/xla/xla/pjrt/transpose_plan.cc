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

  // Transposes a `[b_elems, a_elems]` tile of `elem_size`-byte elements in `a`
  // (row stride `lda` bytes, contiguous along `a_elems`) into `[a_elems,
  // b_elems]` in `b` (row stride `ldb` bytes, contiguous along `b_elems`).
  // `inner_block_elems` is the SIMD microkernel block size (1 for scalar).
  struct Transpose2D {
    int elem_size;
    int inner_block_elems;
    int a_elems;
    int b_elems;
    int64_t lda;
    int64_t ldb;
  };

  // Merges `k` rows of `a_elems` `elem_size`-byte elements in `a` (`lda` bytes
  // apart) element by element into one contiguous run of `k * a_elems` elements
  // in `b`, i.e., `b[k * i + j] = a[j][i]`.
  struct Interleave {
    int elem_size;
    int k;
    int a_elems;
    int64_t lda;
  };

  // Splits one contiguous run of `k * b_elems` `elem_size`-byte elements in `a`
  // into `k` rows of `b_elems` elements in `b` (`ldb` bytes apart), i.e.,
  // `b[j][i] = a[k * i + j]`.
  struct Deinterleave {
    int elem_size;
    int k;
    int b_elems;
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

  std::variant<Noop, Seq, Offset, Loop, Pipe, Transpose2D, Interleave,
               Deinterleave, Copy1D, Fill1D, PackSubbyte, ConvertEf57>
      node;

  bool IsNoop() const { return std::holds_alternative<Noop>(node); }

  // Appends a human-readable indented tree representation of `*this` to `*out`.
  void Format(int indent, std::string* out) const;

  static Plan MakeNoop();
  static Plan MakeSeq(std::vector<Plan> parts);
  static Plan MakeSeq(Plan first, Plan second);
  static Plan MakeOffset(int64_t da, int64_t db, Plan body);
  static Plan MakeLoop(int64_t count, int64_t da, int64_t db, Plan body);
  static Plan MakePipe(int64_t bytes, Plan step1, Plan step2);
  // Decomposes tiles into full power-of-two SIMD blocks (halving
  // `inner_block_elems` down to 16-byte rows and `bs >= 4` for smaller or
  // remainder tiles) plus scalar remainder strips so every `Transpose2D` leaf
  // carries its resolved block size and maps 1-to-1 to a single `TransposeOp`.
  static Plan MakeTranspose2D(int elem_size, int inner_block_elems, int a_elems,
                              int b_elems, int64_t lda, int64_t ldb);
  static Plan MakeInterleave(int elem_size, int k, int a_elems, int64_t lda);
  static Plan MakeDeinterleave(int elem_size, int k, int b_elems, int64_t ldb);
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

// Returns true if a tile or remainder strip may use a `bs x bs` SIMD kernel
// given a plan's primary `inner_block_elems` and `elem_size`.
// Secondary (halved) remainder blocks require at least 4 elements and 16 byte
// rows so the extra kernel pass beats a single scalar loop.
constexpr bool IsSupportedSimdBlockSize(int bs, int inner_block_elems,
                                        int elem_size) {
  return bs > 1 &&
         (bs == inner_block_elems || (bs >= 4 && bs * elem_size >= 16));
}

Plan Plan::MakeTranspose2D(int elem_size, int inner_block_elems, int a_elems,
                           int b_elems, int64_t lda, int64_t ldb) {
  if (a_elems <= 0 || b_elems <= 0) {
    return MakeNoop();
  }
  if (inner_block_elems > 1) {
    int bs = inner_block_elems;
    while (bs > 1 && (a_elems < bs || b_elems < bs)) {
      bs /= 2;
    }
    if (IsSupportedSimdBlockSize(bs, inner_block_elems, elem_size)) {
      if (a_elems % bs != 0 || b_elems % bs != 0) {
        const int done_a = (a_elems / bs) * bs;
        const int done_b = (b_elems / bs) * bs;
        std::vector<Plan> pieces;
        pieces.reserve(3);
        pieces.push_back(
            Plan{Transpose2D{elem_size, bs, done_a, done_b, lda, ldb}});
        pieces.push_back(
            MakeOffset(done_b * lda, int64_t{done_b} * elem_size,
                       MakeTranspose2D(elem_size, inner_block_elems, done_a,
                                       b_elems - done_b, lda, ldb)));
        pieces.push_back(
            MakeOffset(int64_t{done_a} * elem_size, done_a * ldb,
                       MakeTranspose2D(elem_size, inner_block_elems,
                                       a_elems - done_a, b_elems, lda, ldb)));
        return MakeSeq(std::move(pieces));
      }
      return Plan{Transpose2D{elem_size, bs, a_elems, b_elems, lda, ldb}};
    }
  }
  return Plan{Transpose2D{elem_size, /*inner_block_elems=*/1, a_elems, b_elems,
                          lda, ldb}};
}

Plan Plan::MakeInterleave(int elem_size, int k, int a_elems, int64_t lda) {
  if (a_elems <= 0 || k <= 0) {
    return MakeNoop();
  }
  return Plan{Interleave{elem_size, k, a_elems, lda}};
}

Plan Plan::MakeDeinterleave(int elem_size, int k, int b_elems, int64_t ldb) {
  if (b_elems <= 0 || k <= 0) {
    return MakeNoop();
  }
  return Plan{Deinterleave{elem_size, k, b_elems, ldb}};
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

void Plan::Format(int indent, std::string* out) const {
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
            node.parts[i].Format(indent + 2, out);
          }
          absl::StrAppend(out, ")");
        } else if constexpr (std::is_same_v<NodeT, Plan::Offset>) {
          pad(indent);
          absl::StrAppendFormat(out, "Offset(da=%d, db=%d,\n", node.da,
                                node.db);
          node.body->Format(indent + 2, out);
          absl::StrAppend(out, ")");
        } else if constexpr (std::is_same_v<NodeT, Plan::Loop>) {
          pad(indent);
          absl::StrAppendFormat(out, "Loop(count=%d, da=%d, db=%d,\n",
                                node.count, node.da, node.db);
          node.body->Format(indent + 2, out);
          absl::StrAppend(out, ")");
        } else if constexpr (std::is_same_v<NodeT, Plan::Pipe>) {
          pad(indent);
          absl::StrAppendFormat(out, "Pipe(bytes=%d,\n", node.bytes);
          node.step1->Format(indent + 2, out);
          absl::StrAppend(out, ",\n");
          node.step2->Format(indent + 2, out);
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
        } else if constexpr (std::is_same_v<NodeT, Plan::Interleave>) {
          pad(indent);
          absl::StrAppendFormat(
              out, "Interleave<elem_size=%d,K=%d>(a_elems=%d, lda=%d)",
              node.elem_size, node.k, node.a_elems, node.lda);
        } else if constexpr (std::is_same_v<NodeT, Plan::Deinterleave>) {
          pad(indent);
          absl::StrAppendFormat(
              out, "Deinterleave<elem_size=%d,K=%d>(b_elems=%d, ldb=%d)",
              node.elem_size, node.k, node.b_elems, node.ldb);
        } else {
          static_assert(std::is_same_v<NodeT, Plan::Transpose2D>);
          pad(indent);
          absl::StrAppendFormat(
              out,
              "Transpose2D<elem_size=%d,bs=%d>(a_elems=%d, b_elems=%d, lda=%d, "
              "ldb=%d)",
              node.elem_size, node.inner_block_elems, node.a_elems,
              node.b_elems, node.lda, node.ldb);
        }
      },
      node);
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

// Picks the execution function for a `Plan::Transpose2D` leaf whose
// `inner_block_elems` has already been resolved by `Plan::MakeTranspose2D`
// (`1` for scalar, or the SIMD microkernel block size dividing both `a_elems`
// and `b_elems`).
TransposeOp::Fn SelectTranspose2DOp(int elem_size, int inner_block_elems,
                                    int a_elems, int b_elems) {
  if (inner_block_elems <= 1) {
    return GetScalarTranspose2DOp(elem_size);
  }
  DCHECK_EQ(a_elems % inner_block_elems, 0);
  DCHECK_EQ(b_elems % inner_block_elems, 0);
  TransposeOp::Fn fn = GetTransposeMacroKernelOp(elem_size, inner_block_elems);
  CHECK(fn != nullptr) << "Invalid block size " << inner_block_elems
                       << " for elem_size=" << elem_size;
  return fn;
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
  // loop (all leaves, `Pipe`, and `Copy2D` patterns) absorb it directly; other
  // ops flush an `OpLoop` first. Returns the peak scratch memory in bytes
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
            op.fn = SelectTranspose2DOp(node.elem_size, node.inner_block_elems,
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
          } else if constexpr (std::is_same_v<NodeT, Plan::Interleave>) {
            DCHECK_LE(node.a_elems, std::numeric_limits<uint16_t>::max());
            DCHECK_LE(node.k, std::numeric_limits<uint16_t>::max());
            TransposeOp::Fn fn = GetInterleaveOp(node.elem_size, node.k);
            CHECK(fn != nullptr)
                << "Unsupported interleave kernel: elem_size=" << node.elem_size
                << " k=" << node.k;
            TransposeOp op;
            op.fn = fn;
            op.count = count;
            op.da = da;
            op.db = db;
            op.lda = node.lda;
            op.ldb = int64_t{node.k} * node.elem_size;
            op.a_elems = static_cast<uint16_t>(node.a_elems);
            op.b_elems = static_cast<uint16_t>(node.k);
            ops.push_back(op);
            return 0;
          } else if constexpr (std::is_same_v<NodeT, Plan::Deinterleave>) {
            DCHECK_LE(node.k, std::numeric_limits<uint16_t>::max());
            DCHECK_LE(node.b_elems, std::numeric_limits<uint16_t>::max());
            TransposeOp::Fn fn = GetDeinterleaveOp(node.elem_size, node.k);
            CHECK(fn != nullptr)
                << "Unsupported deinterleave kernel: elem_size="
                << node.elem_size << " k=" << node.k;
            TransposeOp op;
            op.fn = fn;
            op.count = count;
            op.da = da;
            op.db = db;
            op.lda = int64_t{node.k} * node.elem_size;
            op.ldb = node.ldb;
            op.a_elems = static_cast<uint16_t>(node.k);
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
int64_t FlattenPlan(const Plan& plan, std::vector<TransposeOp>& ops) {
  PlanFlattener flattener{ops};
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

  // Dimension permutation to apply to form B. For each dimension of B, what is
  // the corresponding dimension of A?
  absl::InlinedVector<int64_t, 4> permutation;

  // Leading-dimension sizes (byte strides) of each dimension at each tiling
  // level: `lda[0]` / `ldb[0]` are strides across outermost tiles, and
  // `lda[1..L]` / `ldb[1..L]` are strides for tile interiors (outermost to
  // innermost). Size is always `tilings.size() + 1` (1 when untiled).
  //
  // Multi-level tile sizes in each dimension, outermost (level 0) to innermost
  // (level L-1). Each level has size equal to the number of dimensions; a 1
  // entry means that dimension is not tiled at that level. Empty means untiled.
  //
  // Running example: a 2D `[M, N]` array of 2-byte values with tiling
  // `(8, 128)(2, 1)` (outer 8x128 tiles, each split into a 4x128 grid of 2x1
  // pairs):
  //   - `a_tilings` = `{{8, 128}, {2, 1}}`
  //   - `lda[0]`    = `{ceil(N / 128) * 2048, 2048}` (across 8x128 tiles)
  //   - `lda[1]`    = `{512, 4}`                     (across 2x1 pairs)
  //   - `lda[2]`    = `{2, 2}`                       (within a 2x1 pair)
  // If untiled, `a_tilings` is empty and `lda` has 1 level: `{{2 * N, 2}}`.
  absl::InlinedVector<absl::InlinedVector<int64_t, 4>, 3> lda;
  absl::InlinedVector<absl::InlinedVector<int64_t, 4>, 3> ldb;
  absl::InlinedVector<absl::InlinedVector<int64_t, 4>, 2> a_tilings;
  absl::InlinedVector<absl::InlinedVector<int64_t, 4>, 2> b_tilings;
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

// Computes byte strides for a tiled array (with zero, one, or multiple tiling
// levels), populating `strides` with `tilings.size() + 1` levels from
// outermost (`strides[0]`) to innermost (`strides[L]`):
//   - `strides[L]`: stride of 1 element along dimension `d` inside an
//     innermost `tilings[L - 1]` tile.
//   - `strides[l]` (`1 <= l < L`): stride from one `tilings[l]` block to the
//     next along dimension `d` inside a `tilings[l - 1]` tile.
//   - `strides[0]`: stride from one top-level `tilings[0]` tile to the next
//     along dimension `d` (`ceil(dims[d] / tilings[0][d])` tiles).
//
// For example, on a 2D array `[M, N]` of 2-byte values with tiling `(8, 128)`
// and inner tiling `(2, 1)`:
//   - `strides[2]` = `{2, 2}`   (inside a `2 x 1` pair)
//   - `strides[1]` = `{512, 4}` (across the `4 x 128` grid of `2 x 1` pairs
//                                inside an `8 x 128` tile)
//   - `strides[0]` = `{ceil(N / 128) * 2048, 2048}`
static void ComputeStrides(
    int64_t elem_size_in_bytes, absl::Span<const int64_t> dims,
    absl::Span<const absl::InlinedVector<int64_t, 4>> tilings,
    absl::InlinedVector<absl::InlinedVector<int64_t, 4>, 3>& strides) {
  const int ndim = static_cast<int>(dims.size());
  const int num_levels = static_cast<int>(tilings.size());
  strides.resize(num_levels + 1);
  int64_t acc = elem_size_in_bytes;
  for (int l = num_levels; l >= 1; --l) {
    strides[l].resize(ndim);
    for (int d = ndim - 1; d >= 0; --d) {
      strides[l][d] = acc;
      const int64_t next_t = (l < num_levels) ? tilings[l][d] : 1;
      acc *= tilings[l - 1][d] / next_t;
    }
  }
  strides[0].resize(ndim);
  for (int d = ndim - 1; d >= 0; --d) {
    strides[0][d] = acc;
    const int64_t outer_t = (num_levels > 0) ? tilings[0][d] : 1;
    acc *= CeilOfRatio(dims[d], outer_t);
  }
}

// Returns the byte stride between two adjacent elements along dimension `d`.
//
// A dimension can have tile size > 1 at an outer level and tile size 1 at
// deeper levels. For example, with `(8, 128)` and `(2, 1)` on 2-byte values,
// dimension 1 has tile size 128 at level 0 and tile size 1 at level 1, so
// moving by 1 element along dimension 1 steps to the next `2 x 1` pair at
// level 1 (`strides[1][1] = 4` bytes), not `strides[2][1]`. We therefore
// return `strides[l + 1][d]` from the deepest level `l` where
// `tilings[l][d] > 1`, or `strides[0][d]` if dimension `d` is not tiled.
static int64_t InnermostStride(
    int d, absl::Span<const absl::InlinedVector<int64_t, 4>> strides,
    absl::Span<const absl::InlinedVector<int64_t, 4>> tilings) {
  for (int l = static_cast<int>(tilings.size()) - 1; l >= 0; --l) {
    if (tilings[l][d] > 1) {
      return strides[l + 1][d];
    }
  }
  return strides[0][d];
}

static bool HasNonTrivialTiling(
    absl::Span<const absl::InlinedVector<int64_t, 4>> tilings) {
  return absl::c_any_of(tilings, [](const auto& level) {
    return absl::c_any_of(level, [](int64_t x) { return x != 1; });
  });
}

static std::string FormatLevels(
    absl::Span<const absl::InlinedVector<int64_t, 4>> levels) {
  if (levels.empty()) {
    return "none";
  }
  return absl::StrJoin(
      levels, "",
      [](std::string* out, const absl::InlinedVector<int64_t, 4>& v) {
        absl::StrAppend(out, "(", absl::StrJoin(v, ","), ")");
      });
}

// Maximum number of valid (unpadded) iterations along `l` in a single pass.
static int64_t MaxValidTripCount(const Loop& l) {
  if (l.is_partial() && l.tile_level > 0) {
    return std::min(l.full_extent, CeilOfRatio(l.dim_size, l.unit_size));
  }
  return l.full_extent;
}

// Only the outermost tiling level affects the padded element count because
// each deeper level evenly divides its enclosing level (rearranging elements
// within an outer tile without adding further padding).
int64_t Planner::InputNumElems() const {
  int64_t size = 1;
  for (size_t i = 0; i < a_dims.size(); ++i) {
    const int64_t t0 = a_tilings.empty() ? 1 : a_tilings[0][i];
    size *= RoundUpTo(a_dims[i], t0);
  }
  return size;
}

int64_t Planner::OutputNumElems() const {
  int64_t size = 1;
  for (size_t i = 0; i < a_dims.size(); ++i) {
    const int64_t t0 = b_tilings.empty() ? 1 : b_tilings[0][i];
    size *= RoundUpTo(a_dims[permutation[i]], t0);
  }
  return size;
}

// Parses and validates a multi-level tiling specification, and populates
// `tilings` (each level padded on the left with 1s to length `ndim`).
static absl::Status ParseTilingSpecification(
    int ndim, absl::Span<const absl::Span<int64_t const>> tilings_spec,
    bool nonstandard_layout,
    absl::InlinedVector<absl::InlinedVector<int64_t, 4>, 2>& tilings) {
  tilings.clear();
  if (tilings_spec.empty()) {
    return absl::OkStatus();
  }
  absl::InlinedVector<absl::InlinedVector<int64_t, 4>, 2> parsed;
  parsed.reserve(tilings_spec.size());
  for (size_t level = 0; level < tilings_spec.size(); ++level) {
    absl::Span<int64_t const> tiling_spec = tilings_spec[level];
    if (tiling_spec.size() > static_cast<size_t>(ndim)) {
      return InvalidArgument(
          "Tiling (%s) must have at most as many dimensions as the array (%d)",
          absl::StrJoin(tiling_spec, ","), ndim);
    }
    absl::InlinedVector<int64_t, 4> t(ndim, 1);
    int offset = ndim - static_cast<int>(tiling_spec.size());
    for (size_t i = 0; i < tiling_spec.size(); ++i) {
      int64_t d = tiling_spec[i];
      if (d < 1) {
        return InvalidArgument("Tiling sizes (%s) must be >= 1",
                               absl::StrJoin(tiling_spec, ","));
      }
      if (!parsed.empty() && parsed.back()[offset + i] % d != 0) {
        return InvalidArgument(
            "Subtiling (%s) must divide enclosing tiling (%s) at dimension %d",
            absl::StrJoin(tiling_spec, ","),
            absl::StrJoin(tilings_spec[level - 1], ","), offset + i);
      }
      t[offset + i] = d;
    }
    parsed.push_back(std::move(t));
  }
  if (ndim == 1 && !nonstandard_layout) {
    // Tiling doesn't do anything for a rank-1 array with default strides,
    // except add padding. Since we're not going to touch any padding elements,
    // we can ignore it.
    // TODO(phawkins): this seems like it should be a loop optimization, not
    // done here.
    return absl::OkStatus();
  }
  tilings = std::move(parsed);
  return absl::OkStatus();
}

std::string Loop::ToString() const {
  std::string level_tag;
  if (tile_level > 0) {
    level_tag = absl::StrFormat("[tile:%d]", tile_level);
  }
  return absl::StrFormat(
      "%d%s[dim_size=%d,unit=%d,ext=%d,start=%d,end=%d,is_inner_dim_in_a=%d,"
      "is_inner_dim_in_b=%d,lda=%d,ldb=%d,parallelism=%d,contiguity=%d,"
      "has_partial_tile=%d)",
      dim_in_a, level_tag, dim_size, unit_size, full_extent, start, end,
      is_inner_dim_in_a, is_inner_dim_in_b, lda, ldb, parallelism, contiguity,
      has_partial_tile);
}

bool Loop::operator==(const Loop& other) const {
  return dim_in_a == other.dim_in_a && tile_level == other.tile_level &&
         dim_size == other.dim_size && unit_size == other.unit_size &&
         full_extent == other.full_extent && lda == other.lda &&
         ldb == other.ldb && is_inner_dim_in_a == other.is_inner_dim_in_a &&
         is_inner_dim_in_b == other.is_inner_dim_in_b &&
         parallelism == other.parallelism && start == other.start &&
         end == other.end;
}

// Builds the execution `Plan` for a single chunk
// (`plan.chunk_loops[chunk_id]`): `BuildTransfer` copies/transposes/transforms
// the valid (unpadded) array elements from `a` into `b`, and invokes
// `BuildPadding` at `*pad_idx` (the first sub-tile loop) to fill trailing
// padding slots in partial output tiles while each tile is still hot in cache.
struct ChunkPlanBuilder {
  const Planner& plan;
  absl::Span<const Loop> chunk_loops;
  int elements_per_byte;
  // Loop index in `chunk_loops` where tile padding is emitted (immediately
  // after the transfer of `chunk_loops[*pad_idx..]`), or `std::nullopt` when
  // this chunk needs no padding.
  //
  // Padding is fused at the first loop that writes inside a single output tile
  // (`ldb < tile_bytes`) rather than as a second pass over the whole chunk.
  // For example, on `float[64, 513, 513]` with `{8, 128}` output tiling, the
  // outer batch and tile-exterior loops stay outside `*pad_idx`, so each 4 KiB
  // partial tile is padded right after its data is written while still hot in
  // L1, while `PadFrom` still sees all inner loops of the tile and coalesces
  // full-row `Fill1D` calls identically to whole-chunk padding.
  std::optional<size_t> pad_idx;

  ChunkPlanBuilder(const Planner& plan, int chunk_id)
      : plan(plan),
        chunk_loops(plan.chunk_loops[chunk_id]),
        elements_per_byte((plan.transformation == Transformation::kPackSubbyte)
                              ? (8 / plan.bits_per_element)
                              : 1) {
    if (!plan.tile_padding.has_value() || !plan.b_is_tiled ||
        !absl::c_any_of(chunk_loops,
                        [](const Loop& l) { return l.has_partial_tile; })) {
      return;
    }
    int64_t tile_bytes = plan.elem_size_in_bytes;
    if (!plan.b_tilings.empty()) {
      for (int64_t t : plan.b_tilings[0]) {
        tile_bytes *= t;
      }
    }
    // Stop at the first loop whose output stride is smaller than one full tile
    // (for example, on `float[4096, 130]` tiled `{8, 128}`, `tile_bytes = 4096`
    // stops at `row_int` with `ldb = 512` rather than descending into `col_int`
    // and issuing per-row padding), or at a 2D-kernel axis whose `Plan::Loop`
    // steps by `inc > 1` so `PadFrom` does not run inside a blocked loop.
    for (size_t i = 0; i < chunk_loops.size(); ++i) {
      const Loop& l = chunk_loops[i];
      const bool is_kernel_axis = !plan.inner_kernel_is_memcpy() &&
                                  (l.is_inner_dim_in_a || l.is_inner_dim_in_b);
      if (is_kernel_axis || (l.ldb > 0 && l.ldb < tile_bytes)) {
        pad_idx = i;
        break;
      }
    }
  }

  int64_t StepB(const Loop& l, int64_t n) const {
    DCHECK_EQ((n * l.ldb) % elements_per_byte, 0);
    return (n * l.ldb) / elements_per_byte;
  }

  Plan MakeInnerLeaf(int a_elems, int b_elems, int64_t lda, int64_t ldb) const {
    switch (plan.inner_kernel_kind) {
      case InnerKernelKind::kInterleave:
        DCHECK_EQ(b_elems, plan.interleave_size);
        DCHECK_EQ(ldb, int64_t{plan.interleave_size} * plan.elem_size_in_bytes);
        return Plan::MakeInterleave(plan.elem_size_in_bytes,
                                    plan.interleave_size, a_elems, lda);
      case InnerKernelKind::kDeinterleave:
        DCHECK_EQ(a_elems, plan.interleave_size);
        DCHECK_EQ(lda, int64_t{plan.interleave_size} * plan.elem_size_in_bytes);
        return Plan::MakeDeinterleave(plan.elem_size_in_bytes,
                                      plan.interleave_size, b_elems, ldb);
      case InnerKernelKind::kDefault:
        return Plan::MakeTranspose2D(plan.elem_size_in_bytes,
                                     plan.inner_block_elems, a_elems, b_elems,
                                     lda, ldb);
      case InnerKernelKind::kMemcpy:
        LOG(FATAL) << "Unexpected kMemcpy in MakeInnerLeaf";
    }
    LOG(FATAL) << "Unhandled InnerKernelKind "
               << static_cast<int>(plan.inner_kernel_kind);
  }

  Plan BuildLeaf(int a_elems, int b_elems) const {
    switch (plan.transformation) {
      case Transformation::kNone:
        return MakeInnerLeaf(a_elems, b_elems, plan.sentinel_lda,
                             plan.sentinel_ldb);
      case Transformation::kF64ToEf57: {
        DCHECK_EQ(a_elems % 2, 0);
        const int64_t tmp_lda = int64_t{a_elems} * sizeof(float);
        Plan step1 = Plan::MakeLoop(b_elems, plan.sentinel_lda, tmp_lda,
                                    Plan::MakeConvertEf57(a_elems / 2));
        Plan step2 =
            MakeInnerLeaf(a_elems, b_elems, tmp_lda, plan.sentinel_ldb);
        return Plan::MakePipe(int64_t{b_elems} * tmp_lda, std::move(step1),
                              std::move(step2));
      }
      case Transformation::kPackSubbyte: {
        DCHECK_EQ(plan.elem_size_in_bytes, 1);
        DCHECK_EQ(b_elems % elements_per_byte, 0);
        DCHECK_EQ(plan.sentinel_ldb % elements_per_byte, 0);
        Plan step1 = MakeInnerLeaf(a_elems, b_elems, plan.sentinel_lda,
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

  // First pass: builds the `Plan` that transfers (copies, transposes, or
  // transforms) the valid array elements from `a` to `b` for
  // `chunk_loops[idx..]`. Unused padding slots in partial output tiles are
  // left for `BuildPadding`.
  //
  // `rem_elems[d]` is the number of valid (unpadded) elements along input
  // dimension `d` inside the enclosing tile currently being visited. It starts
  // at `plan.a_dims[d]`. When a partial tiled loop splits into tiles of size
  // `loop.unit_size`, full tiles recurse with `loop.unit_size` elements and any
  // trailing partial tile recurses with `rem_elems[d] % loop.unit_size`.
  Plan BuildTransfer(size_t idx, int a_elems, int b_elems,
                     absl::InlinedVector<int64_t, 4>& rem_elems) const {
    if (idx == chunk_loops.size()) {
      DCHECK(!plan.inner_kernel_is_memcpy());
      DCHECK_GT(a_elems, 0);
      DCHECK_GT(b_elems, 0);
      return BuildLeaf(a_elems, b_elems);
    }

    const Loop& loop = chunk_loops[idx];

    // Emits `count` full (non-partial) tiles/elements of `loop`:
    //   - Innermost memcpy loop -> single `Copy1D` or `PackSubbyte` leaf.
    //   - Inner 2D-kernel axis  -> `loop.inc`-wide full blocks + remainder.
    //   - Any other loop        -> plain `Plan::Loop` (`loop.inc == 1`).
    auto emit_full_tiles = [&](int64_t count) -> Plan {
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
          return BuildTransfer(idx + 1, next_a, next_b, rem_elems);
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
      return Plan::MakeLoop(
          count, loop.lda, StepB(loop, 1),
          BuildTransfer(idx + 1, a_elems, b_elems, rem_elems));
    };

    Plan transfer;
    if (!loop.is_partial()) {
      transfer =
          emit_full_tiles(std::min(loop.full_extent, loop.end) - loop.start);
    } else {
      const int a_dim = loop.dim_in_a;
      DCHECK_LT(a_dim, static_cast<int>(plan.a_dims.size()));
      DCHECK_GE(loop.start, 0);
      const int64_t avail = rem_elems[a_dim];
      const int64_t unit = loop.unit_size;
      const int64_t num_full = avail / unit;
      const int64_t partial_rem = avail % unit;

      rem_elems[a_dim] = unit;
      Plan full_plan =
          emit_full_tiles(std::min(loop.end, num_full) - loop.start);

      Plan partial_plan = Plan::MakeNoop();
      if (partial_rem > 0 && loop.start <= num_full && num_full < loop.end) {
        DCHECK(!loop.is_inner_dim_in_a && !loop.is_inner_dim_in_b);
        rem_elems[a_dim] = partial_rem;
        const int64_t offset = num_full - loop.start;
        partial_plan = Plan::MakeOffset(
            offset * loop.lda, StepB(loop, offset),
            BuildTransfer(idx + 1, a_elems, b_elems, rem_elems));
      }

      rem_elems[a_dim] = avail;
      transfer = Plan::MakeSeq(std::move(full_plan), std::move(partial_plan));
    }

    if (pad_idx == idx) {
      return Plan::MakeSeq(std::move(transfer), BuildPadding(idx, rem_elems));
    }
    return transfer;
  }

  Plan PadFrom(absl::Span<const Loop> loops, size_t idx, bool is_padding,
               absl::InlinedVector<int64_t, 4>& rem_pad) const {
    if (idx == loops.size()) {
      return is_padding
                 ? Plan::MakeFill1D(plan.elem_size_in_bytes, *plan.tile_padding)
                 : Plan::MakeNoop();
    }
    const Loop& loop = loops[idx];
    const int64_t chunk_count =
        std::max<int64_t>(0, std::min(loop.end, loop.full_extent) - loop.start);
    if (is_padding) {
      return Plan::MakeLoop(
          chunk_count, /*da=*/0, loop.ldb,
          PadFrom(loops, idx + 1, /*is_padding=*/true, rem_pad));
    }
    if (!loop.is_partial()) {
      if (idx + 1 == loops.size()) {
        return Plan::MakeNoop();
      }
      return Plan::MakeLoop(
          chunk_count, /*da=*/0, loop.ldb,
          PadFrom(loops, idx + 1, /*is_padding=*/false, rem_pad));
    }

    const int a_dim = loop.dim_in_a;
    const int64_t saved_avail = rem_pad[a_dim];
    const int64_t unit = loop.unit_size;
    const int64_t num_full = saved_avail / unit;
    const int64_t rem = saved_avail % unit;
    const int64_t full_in_chunk = std::max<int64_t>(
        0, std::min(loop.end, num_full) - std::max<int64_t>(loop.start, 0));

    std::vector<Plan> parts;
    if (full_in_chunk > 0 && idx + 1 < loops.size()) {
      rem_pad[a_dim] = unit;
      parts.push_back(Plan::MakeLoop(
          full_in_chunk, /*da=*/0, loop.ldb,
          PadFrom(loops, idx + 1, /*is_padding=*/false, rem_pad)));
    }
    if (rem > 0 && loop.start <= num_full && num_full < loop.end &&
        idx + 1 < loops.size()) {
      rem_pad[a_dim] = rem;
      parts.push_back(Plan::MakeOffset(
          /*da=*/0, (num_full - loop.start) * loop.ldb,
          PadFrom(loops, idx + 1, /*is_padding=*/false, rem_pad)));
    }
    rem_pad[a_dim] = saved_avail;

    const int64_t used = num_full + (rem > 0 ? 1 : 0);
    if (loop.tile_level > 0 && used < loop.full_extent) {
      const int64_t pad_start = std::max(loop.start, used);
      const int64_t pad_end = std::min(loop.end, loop.full_extent);
      if (pad_start < pad_end) {
        parts.push_back(Plan::MakeOffset(
            /*da=*/0, (pad_start - loop.start) * loop.ldb,
            Plan::MakeLoop(
                pad_end - pad_start, /*da=*/0, loop.ldb,
                PadFrom(loops, idx + 1, /*is_padding=*/true, rem_pad))));
      }
    }
    return Plan::MakeSeq(std::move(parts));
  }

  // Builds the `Fill1D` padding plan for unused slots inside partial output
  // tiles of `chunk_loops[start_idx..]`, using `rem_pad` to track remaining
  // valid elements from enclosing loops in `chunk_loops[0..start_idx)`.
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
  // `[rem, full_extent)` recurses with `is_padding=true`. Recursing through
  // inner loops ensures that any inner loop partitioned across chunks only
  // fills its `[loop.start, loop.end)` slice, while `Plan::MakeLoop` coalesces
  // unsplit inner loops into a single `Fill1D`.
  Plan BuildPadding(size_t start_idx,
                    absl::InlinedVector<int64_t, 4>& rem_pad) const {
    std::vector<Loop> loops;
    loops.reserve(chunk_loops.size() - start_idx);
    for (size_t i = start_idx; i < chunk_loops.size(); ++i) {
      const Loop& l = chunk_loops[i];
      if (l.ldb != 0) {
        loops.push_back(l);
      }
    }
    absl::c_sort(loops, [](const Loop& a, const Loop& b) {
      return std::make_tuple(a.ldb, -a.tile_level) >
             std::make_tuple(b.ldb, -b.tile_level);
    });

    Plan pad_plan = PadFrom(loops, /*idx=*/0, /*is_padding=*/false, rem_pad);
    if (elements_per_byte > 1) {
      ScalePlanOutputBytes(pad_plan, elements_per_byte);
    }
    return pad_plan;
  }
};

// Builds the execution `Plan` for `chunk_id`.
Plan Planner::BuildChunkPlan(int chunk_id) const {
  DCHECK(!a_dims.empty());
  ChunkPlanBuilder builder(*this, chunk_id);
  absl::InlinedVector<int64_t, 4> rem_elems(a_dims.begin(), a_dims.end());
  return builder.BuildTransfer(/*idx=*/0, /*a_elems=*/0, /*b_elems=*/0,
                               rem_elems);
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
      plan.b_tilings));

  // Temporary vectors to hold un-permuted attributes
  absl::InlinedVector<absl::InlinedVector<int64_t, 4>, 3> temp_lda;
  absl::InlinedVector<absl::InlinedVector<int64_t, 4>, 2> temp_a_tilings;

  // Parse the tile and stride specifications.
  ABSL_RETURN_IF_ERROR(ParseTilingSpecification(
      ndim, o.input_tiling,
      /*nonstandard_layout=*/o.input_striding.has_value() || input_contiguity,
      temp_a_tilings));
  ComputeStrides(plan.elem_size_in_bytes, o.dims, temp_a_tilings, temp_lda);

  // Override outer tile strides if explicit input_striding is provided.
  if (o.input_striding) {
    absl::Span<int64_t const> input_strides_in_bytes = *o.input_striding;
    if (input_strides_in_bytes.size() != o.dims.size()) {
      return InvalidArgument(
          "dims and input_striding must have equal sizes, "
          "got %d and %d",
          o.dims.size(), input_strides_in_bytes.size());
    }
    temp_lda[0].assign(input_strides_in_bytes.begin(),
                       input_strides_in_bytes.end());
  }

  // Sort the dimensions from slowest-varying (largest strides) to
  // fastest-varying (smallest strides).
  // Maps new input dim -> old input dim
  std::vector<int64_t> dim_order(ndim);
  absl::c_iota(dim_order, 0);

  auto cost = [&](int k) {
    const int64_t outer_stride = temp_lda[0].at(k);
    const int64_t inner_stride = InnermostStride(k, temp_lda, temp_a_tilings);
    // If there is a dimension with size equal to the element size, sort it
    // last. This ensures that we place any stride-1 dimension last.
    const bool is_stride1 = inner_stride == plan.elem_size_in_bytes;
    // If there are multiple stride-1 dimensions, we'd prefer the one that
    // matches the stride-1 dimension of the output.
    // Failing that, we'd just prefer the largest stride-1 dimension last.
    const bool is_trailing_dim_in_b = o.permutation.back() == k;

    // If we are applying ef57 conversion, we want an even-sized stride-1
    // dimension last.
    const bool ef57_even =
        (is_stride1 && o.transformation == Transformation::kF64ToEf57 &&
         o.dims[k] % 2 == 0);

    return std::make_tuple(is_stride1, -std::abs(outer_stride),
                           -std::abs(inner_stride), ef57_even,
                           is_trailing_dim_in_b, o.dims[k]);
  };
  absl::c_stable_sort(dim_order,
                      [&cost](int i, int j) { return cost(i) < cost(j); });

  // Apply permutation to all plan attributes
  // dim_order maps new input dim -> old input dim, we need its inverse to
  // compute the new permutation.
  auto inv_dim_order = InversePermutation(dim_order);
  plan.a_dims.reserve(ndim);
  plan.permutation.reserve(ndim);
  plan.lda.resize(temp_lda.size());
  for (auto& level : plan.lda) {
    level.reserve(ndim);
  }
  plan.a_tilings.resize(temp_a_tilings.size());
  for (auto& level : plan.a_tilings) {
    level.reserve(ndim);
  }

  for (int i = 0; i < ndim; ++i) {
    int old_idx = dim_order[i];
    plan.a_dims.push_back(o.dims[old_idx]);
    plan.permutation.push_back(inv_dim_order[o.permutation[i]]);
    for (size_t l = 0; l < temp_lda.size(); ++l) {
      plan.lda[l].push_back(temp_lda[l][old_idx]);
    }
    for (size_t l = 0; l < temp_a_tilings.size(); ++l) {
      plan.a_tilings[l].push_back(temp_a_tilings[l][old_idx]);
    }
  }

  plan.a_is_tiled = HasNonTrivialTiling(plan.a_tilings);
  plan.b_is_tiled = HasNonTrivialTiling(plan.b_tilings);
  if (plan.a_is_tiled && plan.b_is_tiled) {
    return Unimplemented(
        "Only one of the input and output may have a non-trivial tiling, "
        "got tilings: %s and %s",
        FormatLevels(plan.a_tilings), FormatLevels(plan.b_tilings));
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
          InnermostStride(ndim - 1, plan.lda, plan.a_tilings) !=
              sizeof(float)) {
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
        plan.bits_per_element = bits;
        if (plan.tile_padding.has_value()) {
          const uint8_t mask = (1u << bits) - 1u;
          plan.tile_padding = static_cast<uint8_t>((*plan.tile_padding & mask) *
                                                   (0xffu / mask));
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

    // Set has_partial_tile on exterior and interior loops touching the trailing
    // partial tile of any dimension.
    std::vector<bool> has_exterior_loop(num_dims, false);
    std::vector<bool> exterior_hits_partial(num_dims, false);
    for (Loop& loop : result[chunk_id]) {
      if (loop.tile_level == 0) {
        has_exterior_loop[loop.dim_in_a] = true;
        if (loop.is_partial() && loop.end >= loop.full_extent) {
          exterior_hits_partial[loop.dim_in_a] = true;
          loop.has_partial_tile = true;
        }
      }
    }
    for (Loop& loop : result[chunk_id]) {
      if (loop.tile_level > 0 && loop.is_partial()) {
        if (!has_exterior_loop[loop.dim_in_a] ||
            exterior_hits_partial[loop.dim_in_a]) {
          loop.has_partial_tile = true;
        }
      }
    }
  }
}

}  // namespace

absl::Status Planner::Initialize() {
  if (num_elems == 0) {
    return absl::OkStatus();
  }
  if (a_dims.empty()) {
    a_dims.push_back(1);
    permutation.push_back(0);
    for (auto& level : lda) {
      level.push_back(elem_size_in_bytes);
    }
    for (auto& level : a_tilings) {
      level.push_back(1);
    }
    for (auto& level : b_tilings) {
      level.push_back(1);
    }
  }

  // permutation maps dimensions of b to a
  // inverse_permutation maps dimensions of a to b
  std::vector<int64_t> inverse_permutation = InversePermutation(permutation);
  int ndim = static_cast<int>(a_dims.size());

  is_identity_permutation = true;
  for (int i = 0; i < ndim; ++i) {
    if (permutation[i] != i) {
      is_identity_permutation = false;
      break;
    }
  }

  std::vector<int64_t> b_dims = Permute(a_dims, permutation);
  ComputeStrides(elem_size_in_bytes, b_dims, b_tilings, ldb);

  // 1. Turn each dimension into its outer tile loop (level 0) plus any inner
  //    tile loops (levels 1..L). At most one of A and B is tiled, so at each
  //    level `l` at most one of `a_tilings[l]` and `b_tilings[l]` is > 1.
  std::vector<Loop> loop_order;
  const int num_a_levels = static_cast<int>(a_tilings.size());
  const int num_b_levels = static_cast<int>(b_tilings.size());
  const int num_levels = std::max(num_a_levels, num_b_levels);

  for (int i = 0; i < ndim; ++i) {
    const int b_dim = inverse_permutation[i];
    const int64_t dim_size = a_dims[i];

    auto level_tile_size = [&](int l) -> int64_t {
      if (l >= num_levels) {
        return 1;
      }
      const int64_t ta = (l < num_a_levels) ? a_tilings[l][i] : 1;
      const int64_t tb = (l < num_b_levels) ? b_tilings[l][b_dim] : 1;
      return std::max(ta, tb);
    };

    const int64_t t0 = level_tile_size(0);
    loop_order.push_back(Loop{
        /*dim_in_a=*/i,
        /*tile_level=*/0,
        /*dim_size=*/dim_size,
        /*unit_size=*/t0,
        /*full_extent=*/CeilOfRatio(dim_size, t0),
        /*lda=*/lda[0][i] * (a_is_tiled ? 1 : t0),
        /*ldb=*/ldb[0][b_dim] * (b_is_tiled ? 1 : t0),
    });

    for (int l = 0; l < num_levels; ++l) {
      const int64_t parent_u = level_tile_size(l);
      const int64_t child_u = level_tile_size(l + 1);
      if (parent_u == child_u) {
        continue;
      }
      loop_order.push_back(Loop{
          /*dim_in_a=*/i,
          /*tile_level=*/l + 1,
          /*dim_size=*/dim_size,
          /*unit_size=*/child_u,
          /*full_extent=*/parent_u / child_u,
          /*lda=*/a_is_tiled ? lda[l + 1][i] : (lda[0][i] * child_u),
          /*ldb=*/b_is_tiled ? ldb[l + 1][b_dim] : (ldb[0][b_dim] * child_u),
      });
    }
  }

  auto loops_to_string = [](absl::Span<const Loop> loops) {
    return absl::StrJoin(loops, "\n  ", [](std::string* out, const Loop& l) {
      absl::StrAppend(out, l.ToString());
    });
  };

  // 2. Drop size-1 loops and merge back-to-back contiguous loops BEFORE
  //    choosing the inner leaf operation, so leaf selection sees the real
  //    contiguous run lengths (for example, two full tiles of 3 on a size-6
  //    dimension merge back into a single loop of 6).
  VLOG(5) << "Before RemoveTrivialLoops: " << loops_to_string(loop_order);
  RemoveTrivialLoops(loop_order);
  if (loop_order.empty()) {
    loop_order.push_back(Loop{/*dim_in_a=*/0, /*tile_level=*/0, /*dim_size=*/1,
                              /*unit_size=*/1, /*full_extent=*/1,
                              /*lda=*/elem_size_in_bytes,
                              /*ldb=*/elem_size_in_bytes});
  }
  VLOG(5) << "After RemoveTrivialLoops: " << loops_to_string(loop_order);
  // TODO(phawkins): Use the coalesced stride-1 extents for default 2D block
  // sizing once loop cache blocking lands. Without blocking, picking larger 2D
  // blocks on coalesced shapes (e.g. [64, 224, 224, 3] perm [1, 2, 3, 0])
  // regresses Arm.
  int64_t a_stride1_bound = 1;
  int64_t b_stride1_bound = 1;
  for (const Loop& l : loop_order) {
    if (l.unit_size == 1 && l.lda == elem_size_in_bytes) {
      a_stride1_bound = MaxValidTripCount(l);
    }
    if (l.unit_size == 1 && l.ldb == elem_size_in_bytes) {
      b_stride1_bound = MaxValidTripCount(l);
    }
  }
  CoalesceLoops(loop_order);
  VLOG(5) << "After CoalesceLoops: " << loops_to_string(loop_order);

  // Locate the stride-1 loops in A and B.
  int loop_b_idx = -1;
  for (int i = static_cast<int>(loop_order.size()) - 1; i >= 0; --i) {
    if (loop_order[i].unit_size == 1 &&
        loop_order[i].ldb == elem_size_in_bytes) {
      loop_b_idx = i;
      break;
    }
  }
  CHECK_GE(loop_b_idx, 0);

  int loop_a_idx = -1;
  if (loop_order[loop_b_idx].lda == elem_size_in_bytes) {
    loop_a_idx = loop_b_idx;
  } else {
    for (int i = static_cast<int>(loop_order.size()) - 1; i >= 0; --i) {
      if (loop_order[i].unit_size == 1 &&
          loop_order[i].lda == elem_size_in_bytes) {
        loop_a_idx = i;
        break;
      }
    }
  }

  // Fallback to whole-array PackIntN when sub-byte output strides or extents
  // are not byte-aligned.
  if (transformation == Transformation::kPackSubbyte) {
    const int elements_per_byte = 8 / bits_per_element;
    const Loop& inner = loop_order[loop_b_idx];
    bool aligned = inner.full_extent % elements_per_byte == 0 &&
                   inner.dim_size % elements_per_byte == 0;
    if (aligned) {
      for (int i = 0; i < static_cast<int>(loop_order.size()); ++i) {
        if (i != loop_b_idx && loop_order[i].ldb != 0 &&
            loop_order[i].ldb % elements_per_byte != 0) {
          aligned = false;
          break;
        }
      }
    }
    if (!aligned) {
      transformation = Transformation::kNone;
      use_fallback_pack = true;
    }
  }

  // If A has no stride-1 loop (e.g. strided or negative-stride input), append a
  // synthetic size-1 stride-1 loop for A.
  if (loop_a_idx < 0) {
    loop_a_idx = static_cast<int>(loop_order.size());
    loop_order.push_back(Loop{/*dim_in_a=*/ndim, /*tile_level=*/0,
                              /*dim_size=*/1, /*unit_size=*/1,
                              /*full_extent=*/1, /*lda=*/elem_size_in_bytes,
                              /*ldb=*/0});
    a_dims.push_back(1);
    permutation.push_back(ndim);
    for (auto& level : lda) {
      level.push_back(elem_size_in_bytes);
    }
    for (auto& level : ldb) {
      level.push_back(0);
    }
    for (auto& level : a_tilings) {
      level.push_back(1);
    }
    for (auto& level : b_tilings) {
      level.push_back(1);
    }
    ++ndim;
  }

  loop_order[loop_a_idx].is_inner_dim_in_a = true;
  loop_order[loop_b_idx].is_inner_dim_in_b = true;

  const bool is_memcpy = (loop_a_idx == loop_b_idx);
  const int output_elements_per_byte =
      (transformation == Transformation::kPackSubbyte) ? (8 / bits_per_element)
                                                       : 1;
  if (is_memcpy) {
    if (transformation == Transformation::kF64ToEf57) {
      return InvalidArgument(
          "Memcpy-based plans cannot be used with transformations.");
    }
    inner_kernel_kind = InnerKernelKind::kMemcpy;
    inner_block_elems = -1;
    outer_block_elems_a = -1;
    outer_block_elems_b = -1;
    if (transformation == Transformation::kPackSubbyte) {
      loop_order[loop_a_idx].inc = output_elements_per_byte;
    }
  } else {
    sentinel_lda = loop_order[loop_b_idx].lda;
    sentinel_ldb = loop_order[loop_a_idx].ldb;

    const Loop& la = loop_order[loop_a_idx];
    const Loop& lb = loop_order[loop_b_idx];
    const int64_t a_full = la.full_extent;
    const int64_t b_full = lb.full_extent;
    const int64_t a_trip = MaxValidTripCount(la);
    const int64_t b_trip = MaxValidTripCount(lb);

    const int max_outer_block_elems =
        std::max<int>(1, kMaxOuterBlockSizeBytes / elem_size_in_bytes);
    const int max_inner_block_bytes =
        GetMaxTransposeInnerBlockSizeBytes(elem_size_in_bytes);
    const int max_inner_block_elems = std::min<int>(
        max_outer_block_elems, max_inner_block_bytes / elem_size_in_bytes);
    // Cap the auto vectorized interleave strip size at the 32 byte block
    // scale (at most 2048 elements for 1 byte types, or 28 KiB across input
    // and output at K = 7) so strips fit in L1D and kF64ToEf57 scratch stays
    // within kStackScratchBytes (4 KiB).
    constexpr int kMaxInterleaveBlockBytes = 32;
    const int max_interleave_block_elems =
        std::min(max_inner_block_bytes, kMaxInterleaveBlockBytes) /
        elem_size_in_bytes;

    const int64_t interleave_stride = b_full * elem_size_in_bytes;
    const int64_t deinterleave_stride = a_full * elem_size_in_bytes;

    // Direct 2D interleave or deinterleave when one array's small innermost
    // loop (`K`) sits right next to the other array's innermost loop in memory.
    inner_kernel_kind = InnerKernelKind::kDefault;
    if (transformation == Transformation::kNone) {
      if (!lb.is_partial() && sentinel_ldb == interleave_stride &&
          SmallDimKernelIsFaster(InnerKernelKind::kInterleave, b_full,
                                 elem_size_in_bytes) &&
          a_trip >= 8) {
        const int k = static_cast<int>(b_full);
        inner_kernel_kind = InnerKernelKind::kInterleave;
        interleave_size = k;
        inner_block_elems = 1;
        outer_block_elems_a = std::min<int64_t>(
            a_trip, max_interleave_block_elems * max_outer_block_elems);
        outer_block_elems_b = 1;
        loop_order[loop_a_idx].inc = outer_block_elems_a;
        loop_order[loop_b_idx].inc = k;
      } else if (!la.is_partial() && sentinel_lda == deinterleave_stride &&
                 SmallDimKernelIsFaster(InnerKernelKind::kDeinterleave, a_full,
                                        elem_size_in_bytes) &&
                 b_trip >= 8) {
        const int k = static_cast<int>(a_full);
        inner_kernel_kind = InnerKernelKind::kDeinterleave;
        interleave_size = k;
        inner_block_elems = 1;
        outer_block_elems_a = 1;
        outer_block_elems_b = std::min<int64_t>(
            b_trip, max_interleave_block_elems * max_outer_block_elems);
        loop_order[loop_a_idx].inc = k;
        loop_order[loop_b_idx].inc = outer_block_elems_b;
      }
    }

    if (inner_kernel_kind == InnerKernelKind::kDefault) {
      const int64_t a_bound = std::min(a_trip, a_stride1_bound);
      const int64_t b_bound = std::min(b_trip, b_stride1_bound);
      int inner_bs = max_inner_block_elems;
      while (inner_bs > 1 && inner_bs > std::min(a_bound, b_bound)) {
        inner_bs /= 2;
      }
      inner_bs = std::max(inner_bs, 1);
      inner_block_elems = inner_bs;
      outer_block_elems_a = std::max<int64_t>(
          1, FloorOfRatio<int64_t>(
                 std::min<int64_t>(max_outer_block_elems, a_bound), inner_bs));
      outer_block_elems_b = std::max<int64_t>(
          1, FloorOfRatio<int64_t>(
                 std::min<int64_t>(max_outer_block_elems, b_bound), inner_bs));
      const int64_t default_a_align =
          (transformation == Transformation::kF64ToEf57) ? 2 : 1;
      const int64_t default_b_align =
          (transformation == Transformation::kPackSubbyte)
              ? output_elements_per_byte
              : 1;
      loop_order[loop_a_idx].inc =
          RoundUpTo<int64_t>(inner_bs * outer_block_elems_a, default_a_align);
      loop_order[loop_b_idx].inc =
          RoundUpTo<int64_t>(inner_bs * outer_block_elems_b, default_b_align);
    }
  }

  // Identify contiguous loops for chunk scheduling.
  // Loops with smallest strides in the contiguous buffer form the contiguous
  // region. Accumulate loops until their combined iteration count reaches
  // the target chunk size.
  IdentifyContiguousLoops(loop_order);
  ChooseLoopOrder(loop_order, inner_kernel_is_memcpy(), chunk_contiguity,
                  is_identity_permutation, output_elements_per_byte);

  // It is a required invariant of the loop order that inner tile levels always
  // appear after any enclosing loop of the same dimension. This is enforced
  // by the hard constraint in ChooseLoopOrder.

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
      chunk_plan.Format(/*indent=*/4, &s);
      vlog_plans.push_back(std::move(s));
    }
    const int64_t chunk_scratch = FlattenPlan(chunk_plan, chunk_ops[chunk_id]);
    scratch_size = std::max(scratch_size, chunk_scratch);
  }

  if (vlog_plan) {
    VLOG(5) << "Final plan: " << ToString(vlog_plans);
  }
  return absl::OkStatus();
}

void ChooseLoopOrder(std::vector<Loop>& loop_order, bool is_memcpy,
                     ChunkContiguity chunk_contiguity,
                     bool is_identity_permutation,
                     int output_elements_per_byte) {
  // Loop order heuristic: try to make loops with small strides innermost.
  // We use a greedy selection to pick loops from slowest-varying to
  // fastest-varying, subject to hard constraints:
  // 1. If the inner kernel is a memcpy, the stride-1 dimension MUST be
  //    innermost.
  // 2. Every inner tile level MUST come after any enclosing loop of the same
  //    dimension.
  std::vector<Loop> remaining = std::move(loop_order);
  loop_order.clear();
  loop_order.reserve(remaining.size());

  // Computes a sort key for each candidate loop, returning a 4-tuple:
  //   <contiguity, -miss_cost, tile_level, -stride_tie_breaker>
  // Lexicographical ordering selects the smallest tuple first, placing that
  // loop into an outer position (from slowest-varying to fastest-varying):
  // 1. Contiguity category (non-contiguous loops placed in outer loops).
  // 2. Primary cost: cache-line misses (higher miss cost placed in outer
  //    loops).
  // 3. Tile level: prefer outer tile levels before inner tile levels on ties.
  // 4. Secondary tie-breaker: when cache misses and tile level tie, the loop
  //    with larger minimum stride goes outer to favor better locality in at
  //    least one buffer for the inner loop.
  auto soft_cost = [&](const Loop& l) -> std::tuple<int, double, int, double> {
    int64_t a_stride = std::abs(l.lda);
    if (!is_memcpy && l.is_inner_dim_in_a) {
      a_stride *= l.inc;
    }
    int64_t b_stride = std::abs(l.ldb);
    if (!is_memcpy && l.is_inner_dim_in_b) {
      b_stride *= l.inc;
    }

    const double a_bytes = static_cast<double>(a_stride);
    const double b_bytes =
        static_cast<double>(b_stride) / output_elements_per_byte;

    switch (chunk_contiguity) {
      case ChunkContiguity::kOutput:
        return std::make_tuple(
            0, -static_cast<double>(std::abs(l.ldb)) / output_elements_per_byte,
            l.tile_level, 0.0);
      case ChunkContiguity::kInput:
        return std::make_tuple(0, -static_cast<double>(std::abs(l.lda)),
                               l.tile_level, 0.0);
      case ChunkContiguity::kNone: {
        constexpr double kOutputPenalty = 1.01;
        const double min_stride =
            std::min<double>(a_bytes * kOutputPenalty, b_bytes);

        // For memcpy-based identity linearizers/delinearizers, preserve the
        // baseline std::min cost to keep 2D hardware tile pairings intact.
        if (is_identity_permutation && is_memcpy) {
          return std::make_tuple(l.contiguity, -min_stride, 0, 0.0);
        }

        // Model the memory traffic of placing this loop in an inner position
        // based on the count of cache lines missed (bytes pulled from L2 to
        // L1). Memory transfers occur at cache-line granularity (128-byte cache
        // line pairs on modern CPUs like AMD Zen 3/4).
        // - Strides < 128 bytes share cache lines across iterations (spatial
        //   reuse).
        // - Strides >= 128 bytes incur a full cache-line miss per iteration.
        constexpr double kCacheLineBytes = 128.0;

        const double misses_a = std::min(1.0, a_bytes / kCacheLineBytes);
        const double misses_b = std::min(1.0, b_bytes / kCacheLineBytes);
        const double base_cost = misses_a + misses_b * kOutputPenalty;

        // Break ties among loops with full cache-line misses using tile level
        // and minimum stride across both buffers.
        return std::make_tuple(l.contiguity, -base_cost, l.tile_level,
                               -min_stride);
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

      // Hard constraint 2: keep outer tile levels before inner tile levels of
      // the same dimension.
      if (l.tile_level > 0) {
        auto is_enclosing = [&](const Loop& r) {
          return r.dim_in_a == l.dim_in_a &&
                 std::make_pair(r.tile_level, -r.unit_size) <
                     std::make_pair(l.tile_level, -l.unit_size);
        };
        if (absl::c_any_of(remaining, is_enclosing)) {
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
    int64_t size = (tile_padding.has_value() && b_is_tiled)
                       ? loop.full_extent
                       : MaxValidTripCount(loop);
    return CeilOfRatio<int64_t>(size, loop.inc);
  };

  // Estimate the number of bytes each iteration of each loop processes.
  absl::InlinedVector<int64_t, 4> work_in_bytes(loop_order.size());
  int64_t acc = elem_size_in_bytes;
  for (const Loop& l : loop_order) {
    acc *= l.inc;
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
    loop.end = loop.full_extent;

    int64_t iterations_without_blocking = loop.end - loop.start;
    int64_t iterations = loop_iterations(loop);

    // Contiguous loops must not be parallelized to maintain chunk contiguity.
    // Tile interiors with partial tiles must not be parallelized since
    // otherwise we will not account for things like padding correctly when
    // calculating chunk sizes.
    if (loop.contiguity > 1 || (loop.tile_level > 0 && loop.is_partial())) {
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

    // Accumulates `loop`'s min/max byte-offset contribution for one buffer
    // (input or output).
    auto accumulate_byte_range = [&](const Loop& loop, int64_t stride,
                                     bool is_tiled, int64_t& min_offset,
                                     int64_t& max_offset) {
      const int64_t first = loop.start;
      int64_t last = loop.end - 1;  // Last iteration (exclusive end)
      if (loop.tile_level > 0 && loop.has_partial_tile) {
        if (is_tiled) {
          // On the tiled side, an incomplete tile still takes up its full
          // padded tile size in memory.
          last = loop.full_extent - 1;
        } else {
          // On the untiled side, when this chunk includes the final incomplete
          // tile, an inner tile loop only reaches the last real element inside
          // that tile.
          last = ((loop.dim_size - 1) % (loop.full_extent * loop.unit_size)) /
                 loop.unit_size;
        }
      }

      // Accumulate min/max (strides can be negative)
      const int64_t first_off = first * stride;
      const int64_t last_off = last * stride;
      min_offset += std::min(first_off, last_off);
      max_offset += std::max(first_off, last_off);
    };

    for (const Loop& loop : loops) {
      if (loop.start >= loop.end) {
        continue;  // Empty iteration range
      }
      accumulate_byte_range(loop, loop.lda, a_is_tiled, input_min, input_max);
      accumulate_byte_range(loop, loop.ldb, b_is_tiled, output_min, output_max);
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
      "elem_size=%d a_dims=%s b_dims=%s permutation=%s a_tilings=%s "
      "b_tilings=%s lda=%s ldb=%s "
      "outer_bs=[%d,%d] inner_bs=%d inner_kernel=%s "
      "transformation=%s tile_padding=%s scratch_size=%d "
      "num_chunks_requested=%d\n"
      "chunk_loops:\n%s\n"
      "plans:\n%s",
      elem_size_in_bytes, absl::StrJoin(a_dims, ","),
      absl::StrJoin(Permute(a_dims, permutation), ","),
      absl::StrJoin(permutation, ","), FormatLevels(a_tilings),
      FormatLevels(b_tilings), FormatLevels(lda), FormatLevels(ldb),
      outer_block_elems_a, outer_block_elems_b, inner_block_elems,
      inner_kernel_str, transformation_str, tile_padding_str, scratch_size,
      num_chunks_requested, chunk_loops_str, plans_str);
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
      const bool is_partial_interior = l.tile_level > 0 && l.is_partial();
      int64_t stride = contiguous_stride(l);
      return std::make_tuple(!is_partial_interior,
                             is_partial_interior ? -l.tile_level : 0, stride);
    };
    absl::c_stable_sort(indices, [&](size_t a, size_t b) {
      return cost(loop_order[a]) < cost(loop_order[b]);
    });

    int prev = -1;
    for (size_t idx : indices) {
      Loop& loop = loop_order[idx];
      int64_t stride = contiguous_stride(loop);
      if (stride == 0 || (loop.full_extent == 1 && !loop.is_partial())) {
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
    // Preserve loops explicitly flagged as stride-1 inner dimensions (used by
    // direct unit-test callers; `Planner::Initialize` assigns those flags after
    // `RemoveTrivialLoops` and synthesizes a size-1 loop if needed).
    if (loop.is_inner_dim_in_a || loop.is_inner_dim_in_b) {
      return false;
    }
    return loop.full_extent == 1;
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

    const int64_t outer_iter_size = outer.full_extent;
    const int64_t inner_iter_size = inner.full_extent;

    // Two loops can be coalesced if:
    // * neither has a partial tile
    // * the inner loop is a multiple of the outer loop.
    const bool coalescable = !outer.is_partial() && !inner.is_partial() &&
                             (outer.lda == inner.lda * inner_iter_size) &&
                             (outer.ldb == inner.ldb * inner_iter_size);
    if (coalescable) {
      const int64_t merged_ext = outer_iter_size * inner_iter_size;
      if (outer.dim_in_a == inner.dim_in_a) {
        outer.unit_size = inner.unit_size;
        outer.tile_level = std::min(outer.tile_level, inner.tile_level);
      } else {
        outer.unit_size = 1;
      }
      outer.full_extent = merged_ext;
      outer.dim_size = merged_ext * outer.unit_size;

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
