// Copyright 2023 The TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/compiler/xla/service/cpu/runtime/rng.h"

#include <array>
#include <cstdint>

#include "absl/status/status.h"
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/custom_call_registry.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace cpu {

using ::xla::runtime::CustomCall;
using ::xla::runtime::Executable;
using ::xla::runtime::FlatMemrefView;

// Disable all CustomCall checks in optimized build.
static constexpr CustomCall::RuntimeChecks RuntimeChecks() {
#if defined(NDEBUG)
  return CustomCall::RuntimeChecks::kNone;
#else
  return CustomCall::RuntimeChecks::kDefault;
#endif
}

namespace {
struct XlaThreeFry {
  absl::Status operator()(const ExecutableRunOptions*,
                          FlatMemrefView state_buffer,
                          FlatMemrefView state_out_buffer,
                          FlatMemrefView values_buffer) const;
};
struct XlaPhilox {
  absl::Status operator()(const ExecutableRunOptions*,
                          FlatMemrefView state_buffer,
                          FlatMemrefView state_out_buffer,
                          FlatMemrefView values_buffer) const;
};
}  // namespace

static std::array<uint32_t, 2> threefry2x32(std::array<uint32_t, 2> key,
                                            std::array<uint32_t, 2> ctr) {
  constexpr std::array<std::array<int, 4>, 2> rotations{
      std::array<int, 4>{13, 15, 26, 6}, std::array<int, 4>{17, 29, 16, 24}};

  std::array<uint32_t, 3> ks{key[0], key[1], key[0] ^ key[1] ^ 0x1BD11BDAu};
  ctr[0] += ks[0];
  ctr[1] += ks[1];

  auto apply_round = [&](int r, int i0, int i1, int b) {
    for (int64_t rot : rotations[r]) {
      ctr[0] += ctr[1];
      ctr[1] = (ctr[1] << rot) | (ctr[1] >> (32 - rot));
      ctr[1] ^= ctr[0];
    }
    ctr[0] += ks[i0];
    ctr[1] += ks[i1] + b;
  };

  apply_round(0, 1, 2, 1);
  apply_round(1, 2, 0, 2);
  apply_round(0, 0, 1, 3);
  apply_round(1, 1, 2, 4);
  apply_round(0, 2, 0, 5);
  return ctr;
}

static std::array<uint32_t, 4> philox4x32(std::array<uint32_t, 2> key,
                                          std::array<uint32_t, 4> ctr) {
  auto mulhilo = [](uint64_t a, uint64_t b) -> std::array<uint32_t, 2> {
    return {static_cast<uint32_t>((a * b) >> 32), static_cast<uint32_t>(a * b)};
  };
  for (int i = 0; i < 10; ++i) {
    auto [hi0, lo0] = mulhilo(0xD2511F53, ctr[0]);
    auto [hi1, lo1] = mulhilo(0xCD9E8D57, ctr[2]);
    ctr = {{hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0}};
    key[0] += 0x9E3779B9u;
    key[1] += 0xBB67AE85u;
  }
  return ctr;
}

template <typename E, typename T, typename C>
void FillBuffer(void* buffer, void* state_buffer, int64_t size_bytes, T fn,
                C ctr, std::array<uint32_t, 2> key) {
  E* out = static_cast<E*>(buffer);
  int64_t i = 0;
  int64_t num = size_bytes / sizeof(E);
  while (i < num) {
    auto val = fn(key, ctr);
    for (int64_t j = 0; j < val.size() && i < num; ++i, ++j) {
      out[i] = val[j];
    }
    if (!++ctr[0]) {
      ++ctr[1];
    }
  }

  auto state_out = static_cast<uint32_t*>(state_buffer);
  state_out[0] = key[0];
  state_out[1] = key[1];
  state_out[2] = ctr[0];
  state_out[3] = ctr[1];
}

static absl::Status ValidateStateBuffers(FlatMemrefView state_buffer,
                                         FlatMemrefView state_out_buffer,
                                         bool allow_24 = false) {
  if (state_buffer.size_in_bytes != 16 &&
      !(allow_24 && state_buffer.size_in_bytes == 24)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unexpected state size: ", state_buffer.size_in_bytes));
  }
  if (state_out_buffer.size_in_bytes != state_buffer.size_in_bytes) {
    return absl::InvalidArgumentError(
        "Expected state output to have the same size as input.");
  }
  return absl::OkStatus();
}

absl::Status XlaThreeFry::operator()(const ExecutableRunOptions*,
                                     FlatMemrefView state_buffer,
                                     FlatMemrefView state_out_buffer,
                                     FlatMemrefView values_buffer) const {
  auto status = ValidateStateBuffers(state_buffer, state_out_buffer);
  if (!status.ok()) {
    return status;
  }

  auto* state_vals = static_cast<uint32_t*>(state_buffer.data);
  std::array<uint32_t, 2> key{state_vals[0], state_vals[1]};
  std::array<uint32_t, 2> ctr{state_vals[2], state_vals[3]};

  switch (values_buffer.dtype) {
    case S8:
    case U8:
    case F16:
    case U16:
    case S16:
      // XLA's RngBitGeneratorExpander has a corner case for bit widths less
      // than 32 where it discards half the bits. We don't really need that, but
      // some TF tests depend on it, somehow.
      FillBuffer<uint16_t>(values_buffer.data, state_out_buffer.data,
                           values_buffer.size_in_bytes, threefry2x32, ctr, key);
      break;
    case F32:
    case U32:
    case S32:
    case F64:
    case U64:
    case S64:
      FillBuffer<uint32_t>(values_buffer.data, state_out_buffer.data,
                           values_buffer.size_in_bytes, threefry2x32, ctr, key);
      break;
    default:
      return absl::UnimplementedError(
          "Type not implemented by ThreeFryBitGenerator");
  }

  return absl::OkStatus();
}

absl::Status XlaPhilox::operator()(const ExecutableRunOptions*,
                                   FlatMemrefView state_buffer,
                                   FlatMemrefView state_out_buffer,
                                   FlatMemrefView values_buffer) const {
  auto status = ValidateStateBuffers(state_buffer, state_out_buffer, true);
  if (!status.ok()) {
    return status;
  }

  auto* state_vals = static_cast<uint32_t*>(state_buffer.data);
  std::array<uint32_t, 2> key{state_vals[0], state_vals[1]};
  bool is_24 = state_buffer.size_in_bytes == 24;
  std::array<uint32_t, 4> ctr{state_vals[2], state_vals[3],
                              state_vals[is_24 ? 4 : 0],
                              state_vals[is_24 ? 5 : 1]};

  switch (values_buffer.dtype) {
    case S8:
    case U8:
    case F16:
    case U16:
    case S16:
      FillBuffer<uint16_t>(values_buffer.data, state_out_buffer.data,
                           values_buffer.size_in_bytes, philox4x32, ctr, key);
      break;
    case F32:
    case U32:
    case S32:
    case F64:
    case U64:
    case S64:
      FillBuffer<uint32_t>(values_buffer.data, state_out_buffer.data,
                           values_buffer.size_in_bytes, philox4x32, ctr, key);
      break;
    default:
      return absl::UnimplementedError(
          "Type not implemented by PhiloxBitGenerator");
  }
  return absl::OkStatus();
}

static bool ThreeFry(xla::runtime::ExecutionContext* ctx, void** args,
                     void** attrs, void** rets) {
  static auto* handler = CustomCall::Bind("xla.cpu.rng.three_fry")
                             .UserData<const ExecutableRunOptions*>()
                             .Arg<FlatMemrefView>()
                             .Arg<FlatMemrefView>()
                             .Arg<FlatMemrefView>()
                             .To<RuntimeChecks()>(XlaThreeFry())
                             .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

static bool Philox(xla::runtime::ExecutionContext* ctx, void** args,
                   void** attrs, void** rets) {
  static auto* handler = CustomCall::Bind("xla.cpu.rng.philox")
                             .UserData<const ExecutableRunOptions*>()
                             .Arg<FlatMemrefView>()
                             .Arg<FlatMemrefView>()
                             .Arg<FlatMemrefView>()
                             .To<RuntimeChecks()>(XlaPhilox())
                             .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

void PopulateXlaCpuRngCall(xla::runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.cpu.rng.three_fry", &ThreeFry);
  registry.Register("xla.cpu.rng.philox", &Philox);
}

}  // namespace cpu
}  // namespace xla
