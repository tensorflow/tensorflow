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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_outputbuffer.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"  // IWYU pragma: keep
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

class WritableStringFile : public WritableFile {
 public:
  explicit WritableStringFile(std::string* data) : data_(data) {}
  ~WritableStringFile() override = default;

  absl::Status Append(absl::string_view data) override {
    data_->append(data.data(), data.size());
    return absl::OkStatus();
  }

  absl::Status Close() override { return absl::OkStatus(); }
  absl::Status Flush() override { return absl::OkStatus(); }
  absl::Status Sync() override { return absl::OkStatus(); }

  absl::Status Tell(int64_t* position) override {
    *position = data_->size();
    return absl::OkStatus();
  }

 private:
  std::string* data_;
};

std::string GeneratePayload(size_t num_bytes, bool compressible) {
  std::string payload(num_bytes, '\0');
  uint32_t state = 0x12345678u;
  if (compressible) {
    // Semi-compressible payload (~1.85x compression ratio): mixes a 32-byte
    // structured template with 32 bytes of deterministic pseudo-random data per
    // 64-byte block so 512 KB uncompressed still exceeds the 256 KB kBufferSize
    // after compression (spanning 2 chunks).
    constexpr absl::string_view kPattern = "tensorflow_decode_compressed_op_";
    static_assert(kPattern.size() == 32);
    for (size_t i = 0; i < num_bytes; ++i) {
      if ((i & 63) < kPattern.size()) {
        payload[i] = kPattern[i & 63];
      } else {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        payload[i] = static_cast<char>((state & 0xFF) ^ (i >> 6));
      }
    }
  } else {
    // High-entropy pseudo-random payload (~1.0x compression ratio): ensures
    // compressed size >= uncompressed size so multi-chunk (>256 KB) inputs
    // exercise repeated MemoryInputStream::ReadNBytes calls with minimal
    // LZ77 back-reference work.
    for (size_t i = 0; i < num_bytes; ++i) {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      payload[i] = static_cast<char>(state & 0xFF);
    }
  }
  return payload;
}

std::string CompressZlibOrGzip(absl::string_view raw,
                               absl::string_view compression_type) {
  std::string compressed;
  WritableStringFile file(&compressed);
  io::ZlibCompressionOptions opts = compression_type == "ZLIB"
                                        ? io::ZlibCompressionOptions::DEFAULT()
                                        : io::ZlibCompressionOptions::GZIP();
  io::ZlibOutputBuffer buffer(&file, opts.input_buffer_size,
                              opts.output_buffer_size, opts);
  CHECK_OK(buffer.Init());
  CHECK_OK(buffer.Append(raw));
  CHECK_OK(buffer.Close());
  return compressed;
}

// Computes a power-of-two ring buffer size such that the total working set is
// at least 64 MB (exceeding L3 cache), clamped to [1, 4096].
size_t ComputePoolSize(size_t bytes_per_tensor) {
  constexpr size_t kTargetWorkingSetBytes = 64ULL << 20;  // 64 MB
  constexpr size_t kMaxPoolSize = 4096;
  const size_t effective_bytes = std::max<size_t>(bytes_per_tensor, 64);
  const size_t min_tensors =
      std::min(kMaxPoolSize,
               std::max<size_t>(1, kTargetWorkingSetBytes / effective_bytes));
  size_t pool_size = 1;
  while (pool_size < min_tensors) {
    pool_size <<= 1;
  }
  return pool_size;
}

class DecodeCompressedOpTest : public OpsTestBase {};

TEST_F(DecodeCompressedOpTest, UncompressedForwardsInputBuffer) {
  ASSERT_TRUE(NodeDefBuilder("decode_none", "DecodeCompressed")
                  .Input(FakeInput(DT_STRING))
                  .Attr("compression_type", "")
                  .Finalize(node_def())
                  .ok());
  ASSERT_TRUE(InitOp().ok());
  const std::string payload = GeneratePayload(1024, /*compressible=*/false);
  AddInputFromArray<tstring>(TensorShape({2}),
                             {tstring(payload), tstring(payload)});
  // Hold an extra reference so !RefCountIsOne(), verifying read-only set_output
  // forwarding rather than mutable-only forward_input_or_allocate_output.
  Tensor shared_ref = *inputs_[0].tensor;
  ASSERT_FALSE(inputs_[0].tensor->RefCountIsOne());
  ASSERT_TRUE(RunOpKernel().ok());
  Tensor* output = GetOutput(0);
  ASSERT_EQ(output->NumElements(), 2);
  EXPECT_EQ(output->flat<tstring>()(0), payload);
  EXPECT_EQ(output->flat<tstring>()(1), payload);
  EXPECT_EQ(output->flat<tstring>().data(),
            inputs_[0].tensor->flat<tstring>().data());
}

TEST_F(DecodeCompressedOpTest, MultiChunkZlibAndGzipRoundTrip) {
  for (bool compressible : {false, true}) {
    const std::string payload = GeneratePayload(512 << 10, compressible);
    for (absl::string_view comp_type : {"ZLIB", "GZIP"}) {
      const std::string compressed = CompressZlibOrGzip(payload, comp_type);
      ASSERT_GT(compressed.size(), 256 << 10);
      node_def()->Clear();
      ASSERT_TRUE(NodeDefBuilder("decode", "DecodeCompressed")
                      .Input(FakeInput(DT_STRING))
                      .Attr("compression_type", comp_type)
                      .Finalize(node_def())
                      .ok());
      ASSERT_TRUE(InitOp().ok());
      inputs_.clear();
      for (Tensor* tensor : tensors_) delete tensor;
      tensors_.clear();
      AddInputFromArray<tstring>(TensorShape({1}), {tstring(compressed)});
      ASSERT_TRUE(RunOpKernel().ok());
      EXPECT_EQ(GetOutput(0)->scalar<tstring>()(), payload);
    }
  }
}

// `test_benchmark.h` provides `BENCHMARK` and `::benchmark::State` without
// re-exporting the internal headers that define them (`registration.h`,
// `state.h`), which do not exist in the benchmark release that open-source
// TensorFlow builds with.
// NOLINTBEGIN(misc-include-cleaner)
void BM_DecodeCompressed_None(::benchmark::State& state) {
  const int64_t num_elements = state.range(0);
  const int64_t string_bytes = state.range(1);

  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));

  const std::string sample = GeneratePayload(static_cast<size_t>(string_bytes),
                                             /*compressible=*/false);
  const size_t bytes_per_tensor =
      static_cast<size_t>(num_elements) *
      (static_cast<size_t>(string_bytes) + sizeof(tstring));
  const size_t pool_size = ComputePoolSize(bytes_per_tensor);
  const size_t pool_mask = pool_size - 1;

  std::vector<Tensor> input_pool;
  input_pool.reserve(pool_size);
  for (size_t pool_idx = 0; pool_idx < pool_size; ++pool_idx) {
    Tensor input_tensor(DT_STRING, TensorShape({num_elements}));
    auto flat = input_tensor.flat<tstring>();
    for (int64_t i = 0; i < num_elements; ++i) {
      flat(i).assign(sample.data(), sample.size());
      flat(i).mdata()[0] = static_cast<char>((pool_idx + i) & 0xFF);
    }
    input_pool.push_back(std::move(input_tensor));
  }

  NodeDef node_def;
  CHECK_OK(NodeDefBuilder("decode_compressed", "DecodeCompressed")
               .Input(FakeInput(DT_STRING))
               .Attr("compression_type", "")
               .Finalize(&node_def));
  absl::Status status;
  std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, device.get(),
                                              cpu_allocator(), node_def,
                                              TF_GRAPH_DEF_VERSION, &status));
  CHECK_OK(status);

  absl::InlinedVector<TensorValue, 4> inputs(1, {nullptr, &input_pool[0]});
  OpKernelContext::Params params;
  params.device = device.get();
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = inputs;
  params.op_kernel = op.get();
  std::vector<AllocatorAttributes> attrs;
  test::SetOutputAttrs(&params, &attrs);

  OpKernelContext ctx(&params);
  // Warmup iteration outside timed loop.
  op->Compute(&ctx);
  CHECK_OK(ctx.status());
  delete ctx.release_output(0).tensor;

  size_t iter = 0;
  for (auto _ : state) {
    inputs[0].tensor = &input_pool[iter++ & pool_mask];
    op->Compute(&ctx);
    DCHECK_OK(ctx.status());
    Tensor* out = ctx.release_output(0).tensor;
    const char* out_data = out->flat<tstring>()(0).c_str();
    tensorflow::testing::DoNotOptimize(out_data);
    delete out;
  }

  const int64_t total_iters = static_cast<int64_t>(state.iterations());
  state.SetItemsProcessed(num_elements * total_iters);
  state.SetBytesProcessed(num_elements * string_bytes * total_iters);
  state.SetLabel(absl::StrFormat("elems:%d,str_bytes:%d,pool:%zu", num_elements,
                                 string_bytes, pool_size));
}

void RunCompressedBenchmark(::benchmark::State& state,
                            absl::string_view compression_type) {
  const int64_t uncompressed_bytes = state.range(0);
  const bool compressible = state.range(1) != 0;

  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));

  const std::string raw =
      GeneratePayload(static_cast<size_t>(uncompressed_bytes), compressible);
  const std::string compressed = CompressZlibOrGzip(raw, compression_type);

  const size_t pool_size = ComputePoolSize(compressed.size() + sizeof(tstring));
  const size_t pool_mask = pool_size - 1;

  std::vector<Tensor> input_pool;
  input_pool.reserve(pool_size);
  for (size_t pool_idx = 0; pool_idx < pool_size; ++pool_idx) {
    Tensor input_tensor(DT_STRING, TensorShape({1}));
    tstring& elem = input_tensor.scalar<tstring>()();
    elem.assign(compressed.data(), compressed.size());
    input_pool.push_back(std::move(input_tensor));
  }

  NodeDef node_def;
  CHECK_OK(NodeDefBuilder("decode_compressed", "DecodeCompressed")
               .Input(FakeInput(DT_STRING))
               .Attr("compression_type", compression_type)
               .Finalize(&node_def));
  absl::Status status;
  std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, device.get(),
                                              cpu_allocator(), node_def,
                                              TF_GRAPH_DEF_VERSION, &status));
  CHECK_OK(status);

  absl::InlinedVector<TensorValue, 4> inputs(1, {nullptr, &input_pool[0]});
  OpKernelContext::Params params;
  params.device = device.get();
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = inputs;
  params.op_kernel = op.get();
  std::vector<AllocatorAttributes> attrs;
  test::SetOutputAttrs(&params, &attrs);

  OpKernelContext ctx(&params);
  // Warmup and correctness check outside timed loop.
  op->Compute(&ctx);
  CHECK_OK(ctx.status());
  Tensor* warmup_out = ctx.release_output(0).tensor;
  CHECK_EQ(warmup_out->scalar<tstring>()().size(), raw.size());
  delete warmup_out;

  size_t iter = 0;
  for (auto _ : state) {
    inputs[0].tensor = &input_pool[iter++ & pool_mask];
    op->Compute(&ctx);
    DCHECK_OK(ctx.status());
    Tensor* out = ctx.release_output(0).tensor;
    const char* out_data = out->scalar<tstring>()().c_str();
    tensorflow::testing::DoNotOptimize(out_data);
    delete out;
  }

  constexpr size_t kChunkSize = 256 << 10;
  const size_t chunks = (compressed.size() + kChunkSize - 1) / kChunkSize;
  const int64_t total_iters = static_cast<int64_t>(state.iterations());
  state.SetItemsProcessed(total_iters);
  state.SetBytesProcessed(uncompressed_bytes * total_iters);
  state.SetLabel(absl::StrFormat("comp:%s,comp_bytes:%zu,chunks:%zu,pool:%zu",
                                 compressible ? "semi" : "incomp",
                                 compressed.size(), chunks, pool_size));
}

BENCHMARK(BM_DecodeCompressed_None)
    ->ArgPair(1, 16)
    ->ArgPair(1, 1024)
    ->ArgPair(1, 65536)
    ->ArgPair(64, 4096)
    ->ArgPair(1024, 256)
    ->ArgPair(256, 65536);

void BM_DecodeCompressed_Zlib(::benchmark::State& state) {
  RunCompressedBenchmark(state, "ZLIB");
}
BENCHMARK(BM_DecodeCompressed_Zlib)
    ->ArgPair(4 << 10, 1)
    ->ArgPair(64 << 10, 1)
    ->ArgPair(256 << 10, 0)
    ->ArgPair(512 << 10, 1)
    ->ArgPair(512 << 10, 0)
    ->ArgPair(4 << 20, 1)
    ->ArgPair(4 << 20, 0);

void BM_DecodeCompressed_Gzip(::benchmark::State& state) {
  RunCompressedBenchmark(state, "GZIP");
}
BENCHMARK(BM_DecodeCompressed_Gzip)
    ->ArgPair(4 << 10, 1)
    ->ArgPair(64 << 10, 1)
    ->ArgPair(256 << 10, 0)
    ->ArgPair(512 << 10, 1)
    ->ArgPair(512 << 10, 0)
    ->ArgPair(4 << 20, 1)
    ->ArgPair(4 << 20, 0);
// NOLINTEND(misc-include-cleaner)

}  // namespace
}  // namespace tensorflow
