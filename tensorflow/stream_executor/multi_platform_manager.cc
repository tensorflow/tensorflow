/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/multi_platform_manager.h"

#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/str_util.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"

namespace stream_executor {

/* static */ mutex MultiPlatformManager::platforms_mutex_{LINKER_INITIALIZED};

/* static */ port::StatusOr<Platform*> MultiPlatformManager::LookupByNameLocked(
    const string& target) {
  PlatformMap* platform_map = GetPlatformMap();
  auto it = platform_map->find(port::Lowercase(target));
  if (it == platform_map->end()) {
    return port::Status(
        port::error::NOT_FOUND,
        "could not find registered platform with name: \"" + target + "\"");
  }
  return it->second;
}

/* static */ port::StatusOr<Platform*> MultiPlatformManager::LookupByIdLocked(
    const Platform::Id& id) {
  PlatformIdMap* platform_map = GetPlatformByIdMap();
  auto it = platform_map->find(id);
  if (it == platform_map->end()) {
    return port::Status(
        port::error::NOT_FOUND,
        port::Printf("could not find registered platform with id: 0x%p", id));
  }
  return it->second;
}

/* static */ port::Status MultiPlatformManager::RegisterPlatform(
    std::unique_ptr<Platform> platform) {
  CHECK(platform != nullptr);
  string key = port::Lowercase(platform->Name());
  mutex_lock lock(platforms_mutex_);
  if (GetPlatformMap()->find(key) != GetPlatformMap()->end()) {
    return port::Status(port::error::INTERNAL,
                        "platform is already registered with name: \"" +
                            platform->Name() + "\"");
  }
  GetPlatformByIdMap()->insert(std::make_pair(platform->id(), platform.get()));
  // Release ownership/uniqueness to prevent destruction on program exit.
  // This avoids Platforms "cleaning up" on program exit, because otherwise,
  // there are _very_ tricky races between StreamExecutor and underlying
  // platforms (CUDA, OpenCL) during exit. Since these are fixed-size and 1x per
  // program, these are deemed acceptable.
  (*GetPlatformMap())[key] = platform.release();
  return port::Status::OK();
}

/* static */ port::StatusOr<Platform*> MultiPlatformManager::PlatformWithName(
    const string& target) {
  mutex_lock lock(platforms_mutex_);

  SE_ASSIGN_OR_RETURN(Platform * platform, LookupByNameLocked(target));
  if (!platform->Initialized()) {
    SE_RETURN_IF_ERROR(platform->Initialize({}));
  }

  return platform;
}

/* static */ port::StatusOr<Platform*> MultiPlatformManager::PlatformWithId(
    const Platform::Id& id) {
  mutex_lock lock(platforms_mutex_);

  SE_ASSIGN_OR_RETURN(Platform * platform, LookupByIdLocked(id));
  if (!platform->Initialized()) {
    SE_RETURN_IF_ERROR(platform->Initialize({}));
  }

  return platform;
}

/* static */ port::StatusOr<Platform*>
MultiPlatformManager::InitializePlatformWithName(
    const string& target, const std::map<string, string>& options) {
  mutex_lock lock(platforms_mutex_);

  SE_ASSIGN_OR_RETURN(Platform * platform, LookupByNameLocked(target));
  if (platform->Initialized()) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "platform \"" + target + "\" is already initialized");
  }

  SE_RETURN_IF_ERROR(platform->Initialize(options));

  return platform;
}

/* static */ port::StatusOr<Platform*>
MultiPlatformManager::InitializePlatformWithId(
    const Platform::Id& id, const std::map<string, string>& options) {
  mutex_lock lock(platforms_mutex_);

  SE_ASSIGN_OR_RETURN(Platform * platform, LookupByIdLocked(id));
  if (platform->Initialized()) {
    return port::Status(
        port::error::FAILED_PRECONDITION,
        port::Printf("platform with id 0x%p is already initialized", id));
  }

  SE_RETURN_IF_ERROR(platform->Initialize(options));

  return platform;
}

/* static */ void MultiPlatformManager::ClearPlatformRegistry() {
  mutex_lock lock(platforms_mutex_);
  GetPlatformMap()->clear();
  GetPlatformByIdMap()->clear();
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(
    multi_platform_manager,
    {
        // Nothing -- this is just a module initializer
        // definition to reference for sequencing
        // purposes from Platform subclasses that register
        // themselves with the MultiPlatformManager.
    });
