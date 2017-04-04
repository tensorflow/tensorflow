#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>

#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>


using namespace poplar;


#define BINARY_ELEMENTWISE(NAME, EXP) \
template<typename T> \
class NAME : public Vertex { \
public: \
  Input<Vector<T>> a; \
  Input<Vector<T>> b; \
  Output<Vector<T>> out; \
\
  bool compute() { \
    for (unsigned i = 0; i < a.size(); ++i) { \
      out[i] = (EXP); \
    } \
    return true; \
  } \
\
  int getCycleEstimate() const { return 1; } \
}; \
\
template class NAME<float>; \
template class NAME<half>; \
template class NAME<int>;

// General math
BINARY_ELEMENTWISE(Add, a[i] + b[i])
BINARY_ELEMENTWISE(Div, a[i] / b[i])
BINARY_ELEMENTWISE(Maximum, a[i] > b[i] ? a[i] : b[i])
BINARY_ELEMENTWISE(Minimum, a[i] < b[i] ? a[i] : b[i])
BINARY_ELEMENTWISE(Mul, a[i] * b[i])
BINARY_ELEMENTWISE(Pow, pow(a[i], b[i]))
BINARY_ELEMENTWISE(Remainder, fmod(a[i], b[i]))
BINARY_ELEMENTWISE(Sub, a[i] - b[i])

// Predicates
#define PREDICATE_ELEMENTWISE(NAME, EXP) \
template<typename T> \
class NAME : public Vertex { \
public: \
  Input<Vector<T>> a; \
  Input<Vector<T>> b; \
  Output<Vector<bool>> out; \
\
  bool compute() { \
    bool res=true; \
    for (unsigned i = 0; i < a.size(); ++i) { \
      out[i] = (EXP); \
      res = res & out[i]; \
    } \
    return res; \
  } \
\
  int getCycleEstimate() const { return 1; } \
};  \
\
template class NAME<float>; \
template class NAME<half>; \
template class NAME<int>; \
template class NAME<bool>;


// Predicates
PREDICATE_ELEMENTWISE(EqualTo, a[i] == b[i])
PREDICATE_ELEMENTWISE(NotEqual, a[i] != b[i])
PREDICATE_ELEMENTWISE(LessEqual, a[i] <= b[i])
PREDICATE_ELEMENTWISE(LessThan, a[i] < b[i])
PREDICATE_ELEMENTWISE(GreaterEqual, a[i] >= b[i])
PREDICATE_ELEMENTWISE(GreaterThan, a[i] > b[i])

// Logical binary
#define LOGICAL_BINARY_ELEMENTWISE(NAME, EXP) \
template<typename T> \
class NAME : public Vertex { \
public: \
  Input<Vector<bool>> a; \
  Input<Vector<bool>> b; \
  Output<Vector<bool>> out; \
\
  bool compute() { \
    bool res=true; \
    for (unsigned i = 0; i < a.size(); ++i) { \
      out[i] = (EXP); \
      res = res & out[i]; \
    } \
    return res; \
  } \
\
  int getCycleEstimate() const { return 1; } \
}; \
\
template class NAME<bool>;

// Predicates
LOGICAL_BINARY_ELEMENTWISE(LogicalAnd, a[i] && b[i])
LOGICAL_BINARY_ELEMENTWISE(LogicalOr, a[i] || b[i])

// Cast
template<typename Tin, typename Tout>
class Cast : public Vertex {
public:
  Input<Vector<Tin>> a;
  Output<Vector<Tout>> out;

  bool compute() {
    for (unsigned i = 0; i < a.size(); ++i) {
      out[i] = (Tout)(a[i]);
    }
    return true;
  }

  int getCycleEstimate() const { return 1; }
};

template class Cast<float,half>;
template class Cast<float,int>;
template class Cast<float,bool>;

template class Cast<int,float>;
template class Cast<int,half>;
template class Cast<int,bool>;

template class Cast<half,float>;
template class Cast<half,int>;
template class Cast<half,bool>;

template class Cast<bool,float>;
template class Cast<bool,half>;
template class Cast<bool,int>;

// Clamp
template <typename T>
class Clamp : public Vertex {
public:
  Input<Vector<T>> a;
  Input<Vector<T>> b;
  Input<Vector<T>> c;
  Output<Vector<T>> out;

  bool compute() {
    for (unsigned i = 0; i < a.size(); ++i) {
      T val = b[i];
      if (val < a[i]) val = a[i];
      if (val > c[i]) val = c[i];
      out[i] = val;
    }
    return true;
  }

  int getCycleEstimate() const { return 1; }
};

template class Clamp<float>;
template class Clamp<half>;
template class Clamp<int>;

// Dot product
template <typename T>
class Dot : public Vertex {
public:
  Input<Vector<T>> a;
  Input<Vector<T>> b;
  Output<Vector<T>> out;

  bool compute() {
    T sum = 0;
    for (unsigned i = 0; i < a.size(); ++i)
      sum += a[i] * b[i];
    out[0] = sum;
    return true;
  }

  int getCycleEstimate() const { return 1; }
};

template class Dot<float>;
template class Dot<half>;
template class Dot<int>;

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



#define UNARY_ELEMENTWISE(NAME, EXP) \
template<typename T> \
class NAME : public Vertex { \
public: \
  Input<Vector<T>> a; \
  Output<Vector<T>> out; \
\
  bool compute() { \
    for (unsigned i = 0; i < a.size(); ++i) { \
      out[i] = (EXP); \
    } \
    return true; \
  } \
\
  int getCycleEstimate() const { return 1; } \
}; \
\
template class NAME<float>; \
template class NAME<half>; \
template class NAME<int>;

UNARY_ELEMENTWISE(Abs, std::abs(a[i]))
UNARY_ELEMENTWISE(Ceil, ceil(a[i]))
UNARY_ELEMENTWISE(Exp, exp(a[i]))
UNARY_ELEMENTWISE(Floor, floor(a[i]))
UNARY_ELEMENTWISE(Log, log(a[i]))
UNARY_ELEMENTWISE(Neg, -a[i])
UNARY_ELEMENTWISE(Sign, (a[i] > (T)0) ? (T)1 : ((a[i] < (T)0) ? (T)-1 : (T)0))
UNARY_ELEMENTWISE(Tanh, tanh(a[i]))

// Logical unary
#define LOGICAL_UNNARY_ELEMENTWISE(NAME, EXP) \
template<typename T> \
class NAME : public Vertex { \
public: \
  Input<Vector<bool>> a; \
  Output<Vector<bool>> out; \
\
  bool compute() { \
    bool res=true; \
    for (unsigned i = 0; i < a.size(); ++i) { \
      out[i] = (EXP); \
      res = res & out[i]; \
    } \
    return res; \
  } \
\
  int getCycleEstimate() const { return 1; } \
}; \
\
template class NAME<bool>;

// Predicates
LOGICAL_UNNARY_ELEMENTWISE(LogicalNot, !a[i])

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
REDUCTION_ELEMENTWISE(ReductionSub, 0.0, v - a[i])
REDUCTION_ELEMENTWISE(ReductionMul, 1.0, v * a[i])
REDUCTION_ELEMENTWISE(ReductionDiv, 1.0, v / a[i])

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
