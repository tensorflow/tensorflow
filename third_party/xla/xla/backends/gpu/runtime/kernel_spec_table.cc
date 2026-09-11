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
#include "xla/backends/gpu/runtime/kernel_spec_table.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>

#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/kernel_spec_table.pb.h"
#include "xla/stream_executor/kernel_args_packing_spec.h"
#include "xla/stream_executor/kernel_spec.h"

namespace xla::gpu {
namespace {

namespace se = ::stream_executor;

bool AreKernelLoaderSpecsEqual(const se::KernelLoaderSpec& a,
                               const se::KernelLoaderSpec& b) {
  if (a.arity() != b.arity() || a.kernel_name() != b.kernel_name()) {
    return false;
  }
  if (a.has_in_process_symbol() != b.has_in_process_symbol() ||
      a.has_cuda_cubin_in_memory() != b.has_cuda_cubin_in_memory() ||
      a.has_cuda_ptx_in_memory() != b.has_cuda_ptx_in_memory()) {
    return false;
  }
  if (a.has_in_process_symbol()) {
    if (a.in_process_symbol()->persistent_name !=
            b.in_process_symbol()->persistent_name ||
        a.in_process_symbol()->symbol != b.in_process_symbol()->symbol) {
      return false;
    }
  }
  if (a.has_cuda_cubin_in_memory()) {
    absl::Span<const uint8_t> bytes_a = a.cuda_cubin_in_memory()->cubin_bytes;
    absl::Span<const uint8_t> bytes_b = b.cuda_cubin_in_memory()->cubin_bytes;
    if (bytes_a.data() != bytes_b.data() || bytes_a.size() != bytes_b.size()) {
      if (bytes_a != bytes_b) {
        return false;
      }
    }
  }
  if (a.has_cuda_ptx_in_memory()) {
    absl::string_view ptx_a = a.cuda_ptx_in_memory()->ptx;
    absl::string_view ptx_b = b.cuda_ptx_in_memory()->ptx;
    if (ptx_a.data() != ptx_b.data() || ptx_a.size() != ptx_b.size()) {
      if (ptx_a != ptx_b) {
        return false;
      }
    }
  }
  if (a.kernel_args_packing().index() != b.kernel_args_packing().index()) {
    return false;
  }
  using PackingFunc = se::KernelLoaderSpec::KernelArgsPackingFunc;
  if (std::holds_alternative<se::KernelArgsPackingSpec>(
          a.kernel_args_packing())) {
    if (std::get<se::KernelArgsPackingSpec>(a.kernel_args_packing()) !=
        std::get<se::KernelArgsPackingSpec>(b.kernel_args_packing())) {
      return false;
    }
  } else if (std::holds_alternative<PackingFunc>(a.kernel_args_packing())) {
    const auto& fn_a = std::get<PackingFunc>(a.kernel_args_packing());
    const auto& fn_b = std::get<PackingFunc>(b.kernel_args_packing());
    if ((fn_a == nullptr) != (fn_b == nullptr)) {
      return false;
    }
    if (fn_a != nullptr) {
      return false;
    }
  }
  return true;
}

uint64_t HashKernelLoaderSpec(const se::KernelLoaderSpec& spec) {
  uint64_t h = absl::HashOf(spec.arity(), spec.kernel_name());
  if (spec.has_in_process_symbol()) {
    h = absl::HashOf(h, 0, spec.in_process_symbol()->persistent_name);
  } else if (spec.has_cuda_cubin_in_memory()) {
    h = absl::HashOf(h, 1, spec.cuda_cubin_in_memory()->cubin_bytes);
  } else if (spec.has_cuda_ptx_in_memory()) {
    h = absl::HashOf(h, 2, spec.cuda_ptx_in_memory()->ptx);
  }
  if (std::holds_alternative<se::KernelArgsPackingSpec>(
          spec.kernel_args_packing())) {
    h = absl::HashOf(
        h, true,
        std::get<se::KernelArgsPackingSpec>(spec.kernel_args_packing()));
  } else {
    h = absl::HashOf(h, false);
  }
  return h;
}

}  // namespace

int32_t KernelSpecTable::Append(se::KernelLoaderSpec spec) {
  const uint64_t hash = HashKernelLoaderSpec(spec);
  const int32_t index = static_cast<int32_t>(specs_.size());
  specs_.push_back(std::move(spec).ToShared());
  index_[hash].push_back(index);
  return index;
}

int32_t KernelSpecTable::Intern(se::KernelLoaderSpec spec) {
  const uint64_t hash = HashKernelLoaderSpec(spec);
  const absl::InlinedVector<int32_t, 1>* bucket = nullptr;
  if (auto it = index_.find(hash); it != index_.end()) {
    bucket = &it->second;
  }
  if (bucket != nullptr) {
    for (const int32_t candidate : *bucket) {
      if (AreKernelLoaderSpecsEqual(specs_[candidate], spec)) {
        return candidate;
      }
    }
  }
  return Append(std::move(spec));
}

absl::StatusOr<const se::KernelLoaderSpec*> KernelSpecTable::Get(
    int32_t index) const {
  if (index < 0 || index >= static_cast<int64_t>(specs_.size())) {
    return absl::OutOfRangeError(absl::StrFormat(
        "Kernel spec index %d is out of range; the table holds %d specs.",
        index, specs_.size()));
  }
  return &specs_[index];
}

absl::StatusOr<KernelSpecTableProto> KernelSpecTable::ToProto() const {
  KernelSpecTableProto proto;
  proto.mutable_kernel_specs()->Reserve(specs_.size());
  for (const se::KernelLoaderSpec& spec : specs_) {
    ABSL_ASSIGN_OR_RETURN(*proto.add_kernel_specs(), spec.ToProto());
  }
  return proto;
}

absl::StatusOr<KernelSpecTable> KernelSpecTable::FromProto(
    const KernelSpecTableProto& proto,
    const std::optional<se::KernelLoaderSpec::SymbolResolver>&
        symbol_resolver) {
  KernelSpecTable table;
  for (const se::KernelLoaderSpecProto& spec_proto : proto.kernel_specs()) {
    ABSL_ASSIGN_OR_RETURN(
        se::KernelLoaderSpec spec,
        se::KernelLoaderSpec::FromProto(spec_proto, symbol_resolver));
    table.Append(std::move(spec));
  }
  return table;
}

}  // namespace xla::gpu
