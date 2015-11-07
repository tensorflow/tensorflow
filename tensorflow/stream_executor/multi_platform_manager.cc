#include "tensorflow/stream_executor/multi_platform_manager.h"

#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/str_util.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"

namespace perftools {
namespace gputools {

/* static */ mutex MultiPlatformManager::platforms_mutex_(LINKER_INITIALIZED);

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
  auto it = GetPlatformMap()->find(port::Lowercase(target));

  if (it == GetPlatformMap()->end()) {
    return port::Status(
        port::error::NOT_FOUND,
        "could not find registered platform with name: \"" + target + "\"");
  }

  return it->second;
}

/* static */ port::StatusOr<Platform*> MultiPlatformManager::PlatformWithId(
    const Platform::Id& id) {
  mutex_lock lock(platforms_mutex_);
  auto it = GetPlatformByIdMap()->find(id);
  if (it == GetPlatformByIdMap()->end()) {
    return port::Status(
        port::error::NOT_FOUND,
        port::Printf("could not find registered platform with id: 0x%p", id));
  }

  return it->second;
}

/* static */ void MultiPlatformManager::ClearPlatformRegistry() {
  mutex_lock lock(platforms_mutex_);
  GetPlatformMap()->clear();
  GetPlatformByIdMap()->clear();
}

}  // namespace gputools
}  // namespace perftools
