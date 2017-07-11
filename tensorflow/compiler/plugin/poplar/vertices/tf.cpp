#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>

#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>


using namespace poplar;

// Random

template<typename T>
class RandomBernoulli : public Vertex {
public:
  Output<Vector<T>> out;
  Input<float> mean;

  std::default_random_engine random_engine;
  std::uniform_real_distribution<float> uniform_real_distribution;

  bool compute() {
    for (unsigned i = 0; i < out.size(); ++i) {
      T val = (T)uniform_real_distribution(random_engine);
      out[i] = (val > mean) ? (T)0 : (T)1;
    }
    return true;
  }

  int getCycleEstimate() const { return 1; }
};

template class RandomBernoulli<unsigned int>;


template<typename T>
class RandomUniform : public Vertex {
public:
  Output<Vector<T>> out;
  Input<T> lower;
  Input<T> upper;

  std::default_random_engine random_engine;
  std::uniform_real_distribution<float> uniform_real_distribution;

  bool compute() {
    T l = lower;
    T u = upper;
    float range = (float)(u - l);

    for (unsigned i = 0; i < out.size(); ++i) {
      T val = (T)(uniform_real_distribution(random_engine) * range);
      out[i] = val + l;
    }
    return true;
  }

  int getCycleEstimate() const { return 1; }
};

template class RandomUniform<int>;
template class RandomUniform<unsigned int>;
template class RandomUniform<float>;
template class RandomUniform<half>;

template<typename T>
class RandomNormal : public Vertex {
public:
  Output<Vector<T>> out;
  Input<T> mean;
  Input<T> sd;

  std::default_random_engine random_engine;
  std::normal_distribution<float> normal_distribution;

  bool compute() {
    T m = mean;
    T s = sd;
    for (unsigned i = 0; i < out.size(); ++i) {
      T val = (T)normal_distribution(random_engine);
      out[i] = val * s + m;
    }
    return true;
  }

  int getCycleEstimate() const { return 1; }
};

template class RandomNormal<float>;
template class RandomNormal<half>;

template<typename T>
class RandomTruncatedNormal : public Vertex {
public:
  Output<Vector<T>> out;
  Input<T> mean;
  Input<T> sd;

  std::default_random_engine random_engine;
  std::normal_distribution<float> normal_distribution;

  bool compute() {
    T m = mean;
    T s = sd;
    for (unsigned i = 0; i < out.size(); ++i) {
      T val = (T)normal_distribution(random_engine);
      while (val < (T)(-2.0) || val > (T)(2.0)) {
        val = (T)normal_distribution(random_engine);
      }
      out[i] = val * s + m;
    }
    return true;
  }

  int getCycleEstimate() const { return 1; }
};

template class RandomTruncatedNormal<float>;
template class RandomTruncatedNormal<half>;

template<typename T>
class ScalarSelect : public Vertex {
public:
  Input<bool> pred;
  Input<Vector<T>> a;
  Input<Vector<T>> b;
  Output<Vector<T>> out;

  bool compute() {
    for (unsigned i = 0; i < a.size(); ++i) {
      out[i] = pred ? a[i] : b[i];
    }
    return true;
  }

  int getCycleEstimate() const { return 1; }
};

template class ScalarSelect<float>;
template class ScalarSelect<half>;
template class ScalarSelect<int>;
template class ScalarSelect<bool>;

template<typename T>
class Select : public Vertex {
public:
  Input<Vector<bool>> pred;
  Input<Vector<T>> a;
  Input<Vector<T>> b;
  Output<Vector<T>> out;

  bool compute() {
    for (unsigned i = 0; i < a.size(); ++i) {
      out[i] = pred[i] ? a[i] : b[i];
    }
    return true;
  }

  int getCycleEstimate() const { return 1; }
};

template class Select<float>;
template class Select<half>;
template class Select<int>;
template class Select<bool>;


// Simple reductions

#define REDUCTION_ELEMENTWISE(NAME, INIT, EXP) \
template<typename T> \
class NAME : public Vertex { \
public: \
  Input<Vector<T>> a; \
  Output<Vector<T>> out; \
\
  bool compute() { \
    T v = (INIT); \
\
    for (unsigned i = 0; i < a.size(); ++i) { \
      v = (EXP); \
    } \
\
    out[0] = v; \
    return true; \
  } \
\
  int getCycleEstimate() const { return 1; } \
}; \
\
template class NAME<float>; \
template class NAME<half>; \
template class NAME<int>;

REDUCTION_ELEMENTWISE(ReductionMax, std::numeric_limits<T>::lowest(), std::max(a[i], v))
REDUCTION_ELEMENTWISE(ReductionMin, std::numeric_limits<T>::max(),    std::min(a[i], v))
REDUCTION_ELEMENTWISE(ReductionAdd, 0.0, v + a[i])
REDUCTION_ELEMENTWISE(ReductionMul, 1.0, v * a[i])

#define LOGICAL_REDUCTION_ELEMENTWISE(NAME, INIT, EXP) \
template<typename T> \
class NAME : public Vertex { \
public: \
  Input<Vector<T>> a; \
  Output<Vector<T>> out; \
\
  bool compute() { \
    T v = (INIT); \
\
    for (unsigned i = 0; i < a.size(); ++i) { \
      v = (EXP); \
    } \
\
    out[0] = v; \
    return out[0]; \
  } \
\
  int getCycleEstimate() const { return 1; } \
}; \
\
template class NAME<bool>;

LOGICAL_REDUCTION_ELEMENTWISE(ReductionAnd, true,  v && a[i])
LOGICAL_REDUCTION_ELEMENTWISE(ReductionOr,  false, v || a[i])

#define WINDOWED_SELECTION(NAME, OP) \
template<typename T> \
class NAME : public Vertex { \
public: \
  Input<Vector<T>> a; \
  Input<Vector<T>> b; \
  InOut<Vector<T>> out; \
\
  bool compute() { \
    unsigned selected = 0; \
    for (unsigned i = 1; i < a.size(); ++i) { \
      if (a[i] OP a[selected]) { \
        selected = i; \
      }; \
    } \
    out[selected] = b[0]; \
    return true; \
  } \
\
  int getCycleEstimate() const { return 1; } \
}; \
\
template class NAME<float>; \
template class NAME<half>; \
template class NAME<int>;

WINDOWED_SELECTION(SelectionGe, >=)
WINDOWED_SELECTION(SelectionGt, >)
WINDOWED_SELECTION(SelectionLe, <=)
WINDOWED_SELECTION(SelectionLt, <)
