#include "tensorflow/stream_executor/executor_cache.h"

#include "tensorflow/stream_executor/lib/stringprintf.h"

namespace perftools {
namespace gputools {

port::Status ExecutorCache::Insert(const StreamExecutorConfig& config,
                                   std::unique_ptr<StreamExecutor> entry) {
  if (Get(config).ok()) {
    return port::Status(port::error::ALREADY_EXISTS,
                        "An executor with a matching config already exists.");
  }

  cache_[config.ordinal].emplace_back(Entry(config, std::move(entry)));

  return port::Status::OK();
}

port::StatusOr<StreamExecutor*> ExecutorCache::Get(
    const StreamExecutorConfig& config) {
  auto entries = cache_.find(config.ordinal);
  if (entries == cache_.end()) {
    return port::Status(
        port::error::NOT_FOUND,
        port::Printf("No executors registered for ordinal %d", config.ordinal));
  }

  for (const auto& iter : entries->second) {
    if (iter.first.plugin_config == config.plugin_config &&
        iter.first.device_options == config.device_options) {
      return iter.second.get();
    }
  }

  return port::Status(port::error::NOT_FOUND,
                      "No executor found with a matching config.");
}

void ExecutorCache::DestroyAllExecutors() { cache_.clear(); }

}  // namespace gputools
}  // namespace perftools
