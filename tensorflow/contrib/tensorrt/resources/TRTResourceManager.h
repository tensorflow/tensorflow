//
// Created by skama on 1/23/18.
//

#ifndef TENSORFLOW_CONTRIB_TENSORRT_RESOURCES_TRTRESOURCEMANAGER_H_

#define TENSORFLOW_CONTRIB_TENSORRT_RESOURCE_TRTRESOURCEMANAGER_H_
#include <memory>

#include <string>
#include <unordered_map>
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace trt {
class TRTResourceManager {
  TRTResourceManager() = default;

 public:
  static std::shared_ptr<TRTResourceManager> instance() {
    static std::shared_ptr<TRTResourceManager> instance_(
        new TRTResourceManager);
    return instance_;
  }
  // returns a manager for given op, if it doesn't exists it creates one
  std::shared_ptr<tensorflow::ResourceMgr> getManager(
      const std::string& op_name);

 private:
  std::unordered_map<std::string, std::shared_ptr<tensorflow::ResourceMgr>>
      managers_;
  tensorflow::mutex map_mutex_;
};
}  // namespace trt
}  // namespace tensorflow
#endif  // TENSORFLOW_CONTRIB_TENSORRT_RESOURCES_TRTRESOURCEMANAGER_H_
