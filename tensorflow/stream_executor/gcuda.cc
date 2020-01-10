#include "tensorflow/stream_executor/gcuda.h"

namespace perftools {
namespace gputools {

// Returns the mapping of gcudacc kernel stub to preferred cache
// configuration. C++ static singleton pattern.
std::map<void *, KernelCacheConfig> &GetGcudaccStubToCacheConfigMap() {
  static std::map<void *, KernelCacheConfig> cache_config_by_stub;
  return cache_config_by_stub;
}

shared_mem_config::SharedMemConfig DeviceGetSharedMemConfig(
    StreamExecutor *stream_exec) {
  SharedMemoryConfig config = stream_exec->GetDeviceSharedMemoryConfig();

  switch (config) {
    case SharedMemoryConfig::kDefault:
      return shared_mem_config::kDefaultBankSize;
    case SharedMemoryConfig::kFourByte:
      return shared_mem_config::kFourByteBankSize;
    case SharedMemoryConfig::kEightByte:
      return shared_mem_config::kEightByteBankSize;
    default:
      LOG(FATAL) << "Impossible shared memory config returned: "
                 << static_cast<int>(config);
  }
}

void DeviceSetSharedMemConfig(StreamExecutor *stream_exec,
                              shared_mem_config::SharedMemConfig config) {
  SharedMemoryConfig executor_config;
  switch (config) {
    case shared_mem_config::kDefaultBankSize:
      executor_config = SharedMemoryConfig::kDefault;
      break;
    case shared_mem_config::kFourByteBankSize:
      executor_config = SharedMemoryConfig::kFourByte;
      break;
    case shared_mem_config::kEightByteBankSize:
      executor_config = SharedMemoryConfig::kEightByte;
      break;
    default:
      LOG(FATAL) << "Impossible shared memory config specified: "
                 << static_cast<int>(config);
  }

  if (!stream_exec->SetDeviceSharedMemoryConfig(executor_config).ok()) {
    // The message is logged at a higher level.
    LOG(INFO) << "Unable to set cache configuration; proceeding.";
  }
}

template <>
void FuncSetCacheConfig<void *>(Stream *stream, void *fptr,
                                cache_config::CacheConfig cache_config) {
  // Map from the legacy to the C++11 type.
  KernelCacheConfig kernel_cache_config;
  switch (cache_config) {
    case cache_config::kPreferShared:
      kernel_cache_config = KernelCacheConfig::kPreferShared;
      break;
    case cache_config::kPreferL1:
      kernel_cache_config = KernelCacheConfig::kPreferL1;
      break;
    case cache_config::kPreferEqual:
      kernel_cache_config = KernelCacheConfig::kPreferEqual;
      break;
    default:
      kernel_cache_config = KernelCacheConfig::kNoPreference;
  }
  auto cache_config_map = GetGcudaccStubToCacheConfigMap();
  cache_config_map[fptr] = kernel_cache_config;
}

template <>
KernelCacheConfig FuncGetCacheConfig<void *>(void *fptr) {
  auto cache_config_map = GetGcudaccStubToCacheConfigMap();
  auto iter = cache_config_map.find(fptr);
  if (iter == cache_config_map.end()) {
    return KernelCacheConfig::kNoPreference;
  }
  return cache_config_map[fptr];
}

}  // namespace gputools
}  // namespace perftools
