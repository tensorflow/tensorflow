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
#ifndef TENSORFLOW_C_EXPERIMENTAL_RENDEZVOUS_H_
#define TENSORFLOW_C_EXPERIMENTAL_RENDEZVOUS_H_

#include "tensorflow/c/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// --------------------------------------------------------------------------
// C API for Rendezvous.
// NOTE: This API is unstable and almost certainly will change in the near
// future.
//
// Custom rendezvous allows for custom implementations of Recv call.
//
// Users wishing to create custom rendezvous objects should call
// TF_NewRemoteRendezvousBuilder and pass returned TF_RemoteRendezvousBuilder
// to to TF_NewServerFactory.

typedef struct TF_RemoteRendezvousBuilder TF_RemoteRendezvousBuilder;
typedef struct TF_ParsedKey TF_ParsedKey;
typedef struct TF_RendezvousArgs TF_RendezvousArgs;
typedef struct TF_RendezvousDoneCallback TF_RendezvousDoneCallback;

// Creates a new TF_RemoteRendezvousBuilder instance.
// Rendezvous instances will forward calls to init_function,
// receive_from_remote_async_function and delete_function passed here.
//
// Note that receive_from_remote_async_function implementation must call
// TF_Done with the TF_DoneCallback passed as an argument.
TF_CAPI_EXPORT extern TF_RemoteRendezvousBuilder* TF_NewRemoteRendezvousBuilder(
    void* (*init_function)(void* server_context),
    void (*receive_from_remote_async_function)(TF_ParsedKey*,
                                               TF_RendezvousArgs*,
                                               TF_RendezvousDoneCallback*,
                                               void* context),
    void (*delete_function)(void* context));

// Deletes TF_RemoteRendezvousBuilder instances.
TF_CAPI_EXPORT extern void TF_DeleteRemoteRendezvousBuilder(
    TF_RemoteRendezvousBuilder* rendezvous_builder);

// Calls TF_DoneCallback and destroys callback instance and
// TF_DoneCallback members except `tensor` and `status`. Caller is
// responsible for deleting `tensor` and `status` after TF_Done returns.
TF_CAPI_EXPORT extern void TF_RendezvousDone(
    TF_RendezvousDoneCallback* callback);

#ifdef __cplusplus
} /* end extern "C" */
#endif
#endif  // TENSORFLOW_C_EXPERIMENTAL_RENDEZVOUS_H_
