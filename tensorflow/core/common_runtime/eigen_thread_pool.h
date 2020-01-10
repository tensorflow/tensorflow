#ifndef TENSORFLOW_COMMON_RUNTIME_EIGEN_THREAD_POOL_H_
#define TENSORFLOW_COMMON_RUNTIME_EIGEN_THREAD_POOL_H_

#include "tensorflow/core/lib/core/threadpool.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

class EigenThreadPoolWrapper : public Eigen::ThreadPoolInterface {
 public:
  explicit EigenThreadPoolWrapper(thread::ThreadPool* pool) : pool_(pool) {}
  ~EigenThreadPoolWrapper() override {}

  void Schedule(std::function<void()> fn) override { pool_->Schedule(fn); }

 private:
  thread::ThreadPool* pool_ = nullptr;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_EIGEN_THREAD_POOL_H_
