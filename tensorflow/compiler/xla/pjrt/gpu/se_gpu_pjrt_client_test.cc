/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/gpu/se_gpu_pjrt_client.h"

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"

namespace xla {
namespace {

StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> CompileExecutable(
    absl::string_view program, xla::PjRtClient& client,
    xla::CompileOptions compile_options = xla::CompileOptions()) {
  TF_ASSIGN_OR_RETURN(auto hlo_module,
                      ParseAndReturnUnverifiedModule(program, {}));

  xla::XlaComputation xla_computation(hlo_module->ToProto());
  return client.Compile(xla_computation, compile_options);
}

// Given the result of a PjrtExecutable::Execute call (TF-status of vectors of
// vectors), extract the zeroth result from the zeroth device.
StatusOr<std::shared_ptr<xla::Literal>> ExtractSingleResult(
    xla::StatusOr<std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>>&
        result) {
  TF_RETURN_IF_ERROR(result.status());
  TF_RET_CHECK(result->size() == 1);
  std::vector<std::unique_ptr<xla::PjRtBuffer>>& result_buffers = (*result)[0];
  TF_RET_CHECK(result_buffers.size() == 1);
  auto literal_or = result_buffers[0]->ToLiteralSync();
  if (!literal_or.status().ok()) return literal_or.status();
  return *literal_or;
}

TEST(StreamExecutorGpuClientTest, SendRecvSynchronous) {
  constexpr absl::string_view kProgram = R"(HloModule HostTransfer
    ENTRY SendRecvSynchronous() -> f32[] {
      in_chain = token[] after-all()

      data = f32[] constant(2)
      send = (f32[], u32[], token[]) send(data, in_chain),
        channel_id=1,
        is_host_transfer=true,
        frontend_attributes={
          _xla_host_transfer_handler_name="undef",
          _xla_host_transfer_original_type="f32",
          _xla_host_transfer_rendezvous="undef"
        }
      send-done = token[] send-done(send),
        channel_id=1, is_host_transfer=true

      recv = (f32[], u32[], token[]) recv(send-done),
        channel_id=2,
        is_host_transfer=true,
        frontend_attributes={
          _xla_host_transfer_handler_name="undef",
          _xla_host_transfer_original_type="f32",
          _xla_host_transfer_rendezvous="undef"
        }
      recv-done = (f32[], token[]) recv-done(recv),
        channel_id=2, is_host_transfer=true

      ROOT result = f32[] get-tuple-element(recv-done), index=0
    })";

  TF_ASSERT_OK_AND_ASSIGN(
      auto client, GetStreamExecutorGpuClient(true, /*allocator_config=*/{},
                                              /*distributed_client=*/nullptr,
                                              /*node_id=*/0));

  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kProgram, *client));

  float sent_value = 0.0f;

  // Send buffer to host.
  SendCallback send_callback = {
      /*channel_id=*/1, [&](const PjRtTransferMetadata& m, PjRtChunk chunk,
                            int64_t total_size_in_bytes, bool done) {
        sent_value = *reinterpret_cast<float*>(chunk.data());
        return OkStatus();
      }};

  // Recv buffer from host.
  RecvCallback recv_callback = {
      /*channel_id=*/2, [&](const PjRtTransferMetadata& m,
                            std::unique_ptr<CopyToDeviceStream> stream) {
        auto chunk = PjRtChunk::AllocateDefault(stream->total_bytes());
        *reinterpret_cast<float*>(chunk.data()) = 5.0f;
        TF_CHECK_OK(stream->AddChunk(std::move(chunk)).Await());
        return OkStatus();
      }};

  // Callbacks for point-to-point communication ops.
  std::vector<std::vector<SendCallback>> send_callbacks = {{send_callback}};
  std::vector<std::vector<RecvCallback>> recv_callbacks = {{recv_callback}};

  ExecuteOptions opts;
  opts.send_callbacks = send_callbacks;
  opts.recv_callbacks = recv_callbacks;

  auto result = executable->Execute(/*argument_handles=*/{{}}, opts);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> result_literal,
                          ExtractSingleResult(result));
  EXPECT_EQ(sent_value, 2.0f);
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR0(5.0f), *result_literal));
}

}  // namespace
}  // namespace xla
