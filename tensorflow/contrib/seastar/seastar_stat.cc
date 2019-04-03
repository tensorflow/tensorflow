#include "tensorflow/contrib/seastar/seastar_stat.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

SeastarStat::SeastarStat() {
  Setup();
}

void SeastarStat::Setup() {
  _counter = 0;
  _total_tensor_size = 0;
  gettimeofday(&_start, nullptr);
}

void SeastarStat::TensorResponse(size_t tensor_size) {
  _total_tensor_size += tensor_size;
  ++_counter;

  if (_counter == COUNTER_INTERVAL) {
    timeval stop;
    gettimeofday(&stop, nullptr);
    double latency = (stop.tv_sec-_start.tv_sec) * 1000 + (stop.tv_usec-_start.tv_usec) / 1000;
    double qps = (COUNTER_INTERVAL * 1000 / latency);
    double average_tensor_size = _total_tensor_size / COUNTER_INTERVAL;
    LOG(INFO) << "Seastar TensorResponse QPS is:" << qps 
              << ", average tensor size:" << average_tensor_size;

    Setup();
  }
}

void SeastarStat::Request() {
  ++_counter;
  if (_counter == COUNTER_INTERVAL) {
    timeval stop;
    gettimeofday(&stop, nullptr);
    double latency = (stop.tv_sec-_start.tv_sec) * 1000 + (stop.tv_usec-_start.tv_usec) / 1000;
    double qps = (COUNTER_INTERVAL * 1000 / latency);
    LOG(INFO) << "Seastar Request QPS is:" << qps;

    Setup();
  }
}

}
