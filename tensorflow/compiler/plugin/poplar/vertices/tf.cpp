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

// Dynamic slicing

int calculateIndex(const std::vector<int>& index,
                   const Vector<int>& base,
                   const Vector<int>& shape) {
  int v=0;
  for (int i=0; i<shape.size(); i++) {
    v *= shape[i];
    v += (index[i] + base[i]);
  }
  return v;
}

template<typename T>
class DynamicSlice : public Vertex {
public:
  Input<Vector<T>> in;
  Input<Vector<int>> index_base;
  Output<Vector<T>> out;

  Vector<int> in_shape;
  Vector<int> out_shape;

  bool compute() {

    std::vector<int> pos(in_shape.size());
    for (int i=0; i<out.size(); i++) {

      int input_index = calculateIndex(pos, index_base, out_shape);
      out[i] = in[input_index];

      // Advance the element
      for (int d=in_shape.size()-1; d>=0; d--) {
        pos[d]++;
        if (pos[d] < in_shape[d]) break;
        pos[d] = 0;
      }
    }
    return true;
  }

  int getCycleEstimate() const { return 1; }
};

template class DynamicSlice<float>;
template class DynamicSlice<half>;
template class DynamicSlice<int>;


template<typename T>
class DynamicUpdateSlice : public Vertex {
public:
  InOut<Vector<T>> in;
  Input<Vector<T>> update;
  Input<Vector<int>> index_base;

  Vector<int> in_shape;
  Vector<int> update_shape;

  bool compute() {
    std::vector<int> pos(in_shape.size());
    for (int i=0; i<update.size(); i++) {

      int output_index = calculateIndex(pos, index_base, update_shape);
      in[output_index] = update[i];

      // Advance the element
      for (int d=in_shape.size()-1; d>=0; d--) {
        pos[d]++;
        if (pos[d] < in_shape[d]) break;
        pos[d] = 0;
      }
    }
    return true;
  }

  int getCycleEstimate() const { return 1; }
};

template class DynamicUpdateSlice<float>;
template class DynamicUpdateSlice<half>;
template class DynamicUpdateSlice<int>;
