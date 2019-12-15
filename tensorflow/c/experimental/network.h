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
#ifndef TENSORFLOW_C_EXPERIMENTAL_NETWORK_H_
#define TENSORFLOW_C_EXPERIMENTAL_NETWORK_H_

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/experimental/rendezvous.h"

#ifdef __cplusplus
extern "C" {
#endif

// --------------------------------------------------------------------------
// C API for TensorFlow Networking.
// NOTE: This API is unstable and almost certainly will change in the near
// future.
//
// Users wishing to register a custom GrpcServer should call
// TF_NewServerFactory and then TF_RegisterGrpcServerFactory.
//
// Example:
// ```c++
// auto* rendezvous_builder = TF_NewRemoteRendezvousBuilder(
//     rendezvous_init_function,
//     receive_from_remote_async_function,
//     rendezvous_delete_function);
//
// TF_GrpcServerFactory* factory = TF_NewGrpcServerFactory(
//     accept_function,
//     init_function,
//     start_function,
//     stop_function,
//     join_function,
//     delete_function,
//     rendezvous_builder);
// TF_RegisterGrpcServerFactory("customfactory", factory);
// ...
// TF_DeleteGrpcServerFactory(factory);
// ```

typedef struct TF_GrpcServerFactory TF_GrpcServerFactory;
typedef struct TF_GrpcServerOptions TF_GrpcServerOptions;
typedef struct TF_GrpcServer TF_GrpcServer;
typedef struct TF_ServerContext {
  TF_GrpcServer* const server;
  void* context;
} TF_ServerContext;

// Creates a new TF_GrpcServerFactory instance. Caller takes ownership
// of TF_GrpcServerFactory instance and should deallocate it by calling
// TF_GrpcDeleteServerFactory.
// accept_function should return true if this ServerFactory can create
// server instances for the given protocol name (for e.g. grpc+verbs).
// GRPC servers created by this factory will call provided
// init_function, start_function, stop_function, join_function and
// delete_function.
//
// Note that clean shutdown is currently not implemented for GrpcServer.
// So, stop_function will never be called now but may be in the future
// when stop mechanism is supported.
TF_CAPI_EXPORT extern TF_GrpcServerFactory* TF_NewGrpcServerFactory(
    bool (*accept_function)(const char*),
    void* (*init_function)(const TF_GrpcServer*, TF_Status*),
    void (*start_function)(const TF_GrpcServer*, void*, TF_Status*),
    void (*stop_function)(const TF_GrpcServer*, void*, TF_Status*),
    void (*join_function)(const TF_GrpcServer*, void*, TF_Status*),
    void (*delete_function)(void*),
    TF_RemoteRendezvousBuilder* rendezvous_builder);

// Deletes TF_GrpcServerFactory instances.
// Note that this function only deletes TF_GrpcServerFactory wrapper.
// Actual underlying server factory would not be deleted and will
// remain registered.
TF_CAPI_EXPORT extern void TF_DeleteGrpcServerFactory(
    TF_GrpcServerFactory* server_factory);

// Registers provided server_factory for the given server_type.
// server_type must be unique to the server factory.
TF_CAPI_EXPORT extern void TF_RegisterGrpcServerFactory(
    const char* server_type, TF_GrpcServerFactory* server_factory);

#ifdef __cplusplus
} /* end extern "C" */
#endif
#endif  // TENSORFLOW_C_EXPERIMENTAL_NETWORK_H_
