/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_LOGGER_H_
#define TENSORFLOW_CORE_PLATFORM_LOGGER_H_

#include "google/protobuf/any.pb.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

// Abstract logging interface. Contrary to logging.h, this class describes an
// interface, not a concrete logging mechanism. This is useful when we want to
// log anything to a non-local place, e.g. a database.
class Logger {
 public:
  // The singleton is supposed to be used in the following steps:
  // * At program start time, REGISTER_MODULE_INITIALIZER calls
  //   SetSingletonFactory.
  // * At some point in the program execution, Singleton() is called for the
  //   first time, initializing the logger.
  // * Succeeding calls to Singleton() return the initialized logger.
  using FactoryFunc = Logger* (*)();

  static void SetSingletonFactory(FactoryFunc factory) {
    singleton_factory_ = factory;
  }

  // Returns the per-process Logger instance, constructing synchronously it if
  // necessary.
  static Logger* GetSingleton();

  // Like GetSingleton, except that this does not wait for the construction of
  // Logger to finish before returning.
  //
  // Returns the constructed instance of Logger if it has been constructed,
  // otherwise returns nullptr (if the logger is not ready yet).
  static Logger* GetSingletonAsync();

  virtual ~Logger() = default;

  // Logs a typed proto.
  template <typename ProtoType>
  void LogProto(const ProtoType& proto) {
    google::protobuf::Any any;
    any.PackFrom(proto);
    DoLogProto(&any);
  }

  // Flushes any pending log. Blocks until everything is flushed.
  void Flush() { DoFlush(); }

 private:
  virtual void DoLogProto(google::protobuf::Any* proto) = 0;
  virtual void DoFlush() = 0;

  static FactoryFunc singleton_factory_;

  friend struct AsyncSingletonImpl;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_LOGGER_H_
