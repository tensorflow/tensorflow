/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <thread>

#include <grpc++/grpc++.h>

#include "monster_test.grpc.fb.h"
#include "monster_test_generated.h"
#include "test_assert.h"

using namespace MyGame::Example;
using flatbuffers::grpc::MessageBuilder;
using flatbuffers::FlatBufferBuilder;

void message_builder_tests();

// The callback implementation of our server, that derives from the generated
// code. It implements all rpcs specified in the FlatBuffers schema.
class ServiceImpl final : public MyGame::Example::MonsterStorage::Service {
  virtual ::grpc::Status Store(
      ::grpc::ServerContext *context,
      const flatbuffers::grpc::Message<Monster> *request,
      flatbuffers::grpc::Message<Stat> *response) override {
    // Create a response from the incoming request name.
    fbb_.Clear();
    auto stat_offset = CreateStat(
        fbb_, fbb_.CreateString("Hello, " + request->GetRoot()->name()->str()));
    fbb_.Finish(stat_offset);
    // Transfer ownership of the message to gRPC
    *response = fbb_.ReleaseMessage<Stat>();
    return grpc::Status::OK;
  }
  virtual ::grpc::Status Retrieve(
      ::grpc::ServerContext *context,
      const flatbuffers::grpc::Message<Stat> *request,
      ::grpc::ServerWriter<flatbuffers::grpc::Message<Monster>> *writer)
      override {
    for (int i = 0; i < 5; i++) {
      fbb_.Clear();
      // Create 5 monsters for resposne.
      auto monster_offset =
          CreateMonster(fbb_, 0, 0, 0,
                        fbb_.CreateString(request->GetRoot()->id()->str() +
                                          " No." + std::to_string(i)));
      fbb_.Finish(monster_offset);

      flatbuffers::grpc::Message<Monster> monster =
          fbb_.ReleaseMessage<Monster>();

      // Send monster to client using streaming.
      writer->Write(monster);
    }
    return grpc::Status::OK;
  }

 private:
  flatbuffers::grpc::MessageBuilder fbb_;
};

// Track the server instance, so we can terminate it later.
grpc::Server *server_instance = nullptr;
// Mutex to protec this variable.
std::mutex wait_for_server;
std::condition_variable server_instance_cv;

// This function implements the server thread.
void RunServer() {
  auto server_address = "0.0.0.0:50051";
  // Callback interface we implemented above.
  ServiceImpl service;
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  // Start the server. Lock to change the variable we're changing.
  wait_for_server.lock();
  server_instance = builder.BuildAndStart().release();
  wait_for_server.unlock();
  server_instance_cv.notify_one();

  std::cout << "Server listening on " << server_address << std::endl;
  // This will block the thread and serve requests.
  server_instance->Wait();
}

template <class Builder>
void StoreRPC(MonsterStorage::Stub *stub) {
  Builder fbb;
  grpc::ClientContext context;
  // Build a request with the name set.
  auto monster_offset = CreateMonster(fbb, 0, 0, 0, fbb.CreateString("Fred"));
  MessageBuilder mb(std::move(fbb));
  mb.Finish(monster_offset);
  auto request = mb.ReleaseMessage<Monster>();
  flatbuffers::grpc::Message<Stat> response;

  // The actual RPC.
  auto status = stub->Store(&context, request, &response);

  if (status.ok()) {
    auto resp = response.GetRoot()->id();
    std::cout << "RPC response: " << resp->str() << std::endl;
  } else {
    std::cout << "RPC failed" << std::endl;
  }
}

template <class Builder>
void RetrieveRPC(MonsterStorage::Stub *stub) {
  Builder fbb;
  grpc::ClientContext context;
  fbb.Clear();
  auto stat_offset = CreateStat(fbb, fbb.CreateString("Fred"));
  fbb.Finish(stat_offset);
  auto request = MessageBuilder(std::move(fbb)).ReleaseMessage<Stat>();

  flatbuffers::grpc::Message<Monster> response;
  auto stream = stub->Retrieve(&context, request);
  while (stream->Read(&response)) {
    auto resp = response.GetRoot()->name();
    std::cout << "RPC Streaming response: " << resp->str() << std::endl;
  }
}

int grpc_server_test() {
  // Launch server.
  std::thread server_thread(RunServer);

  // wait for server to spin up.
  std::unique_lock<std::mutex> lock(wait_for_server);
  while (!server_instance) server_instance_cv.wait(lock);

  // Now connect the client.
  auto channel = grpc::CreateChannel("localhost:50051",
                                     grpc::InsecureChannelCredentials());
  auto stub = MyGame::Example::MonsterStorage::NewStub(channel);

  StoreRPC<MessageBuilder>(stub.get());
  StoreRPC<FlatBufferBuilder>(stub.get());

  RetrieveRPC<MessageBuilder>(stub.get());
  RetrieveRPC<FlatBufferBuilder>(stub.get());


#if !FLATBUFFERS_GRPC_DISABLE_AUTO_VERIFICATION
  {
    // Test that an invalid request errors out correctly
    grpc::ClientContext context;
    flatbuffers::grpc::Message<Monster> request;  // simulate invalid message
    flatbuffers::grpc::Message<Stat> response;
    auto status = stub->Store(&context, request, &response);
    // The rpc status should be INTERNAL to indicate a verification error. This
    // matches the protobuf gRPC status code for an unparseable message.
    assert(!status.ok());
    assert(status.error_code() == ::grpc::StatusCode::INTERNAL);
    assert(strcmp(status.error_message().c_str(),
                  "Message verification failed") == 0);
  }
#endif

  server_instance->Shutdown();

  server_thread.join();

  delete server_instance;

  return 0;
}

int main(int /*argc*/, const char * /*argv*/ []) {
  message_builder_tests();
  grpc_server_test();

  if (!testing_fails) {
    TEST_OUTPUT_LINE("ALL TESTS PASSED");
    return 0;
  } else {
    TEST_OUTPUT_LINE("%d FAILED TESTS", testing_fails);
    return 1;
  }
}

