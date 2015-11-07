#include "tensorflow/core/util/util.h"

#include "tensorflow/core/platform/logging.h"
namespace tensorflow {

StringPiece NodeNamePrefix(const StringPiece& op_name) {
  StringPiece sp(op_name);
  auto p = sp.find('/');
  if (p == StringPiece::npos || p == 0) {
    return "";
  } else {
    return StringPiece(sp.data(), p);
  }
}

StringPiece NodeNameFullPrefix(const StringPiece& op_name) {
  StringPiece sp(op_name);
  auto p = sp.rfind('/');
  if (p == StringPiece::npos || p == 0) {
    return "";
  } else {
    return StringPiece(sp.data(), p);
  }
}

MovingAverage::MovingAverage(int window)
    : window_(window),
      sum_(0.0),
      data_(new double[window_]),
      head_(0),
      count_(0) {
  CHECK_GE(window, 1);
}

MovingAverage::~MovingAverage() { delete[] data_; }

void MovingAverage::Clear() {
  count_ = 0;
  head_ = 0;
  sum_ = 0;
}

double MovingAverage::GetAverage() const {
  if (count_ == 0) {
    return 0;
  } else {
    return static_cast<double>(sum_) / count_;
  }
}

void MovingAverage::AddValue(double v) {
  if (count_ < window_) {
    // This is the warmup phase. We don't have a full window's worth of data.
    head_ = count_;
    data_[count_++] = v;
  } else {
    if (window_ == ++head_) {
      head_ = 0;
    }
    // Toss the oldest element
    sum_ -= data_[head_];
    // Add the newest element
    data_[head_] = v;
  }
  sum_ += v;
}

static char hex_char[] = "0123456789abcdef";

string PrintMemory(const char* ptr, int n) {
  string ret;
  ret.resize(n * 3);
  for (int i = 0; i < n; ++i) {
    ret[i * 3] = ' ';
    ret[i * 3 + 1] = hex_char[ptr[i] >> 4];
    ret[i * 3 + 2] = hex_char[ptr[i] & 0xf];
  }
  return ret;
}

}  // namespace tensorflow
