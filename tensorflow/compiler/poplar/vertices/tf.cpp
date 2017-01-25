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
    for (unsigned i = 0; i != a.size(); ++i) { \
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
    for (unsigned i = 0; i != a.size(); ++i) { \
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
    for (unsigned i = 0; i != a.size(); ++i) { \
      out[i] = (EXP); \
    } \
    return true; \
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
    for (unsigned i = 0; i != a.size(); ++i) {
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
/*
template<typename T>
class RandomUniform : public Vertex {
public:
  Output<Vector<T>> out;

  std::default_random_engine engine;
  std::uniform_real_distribution<float> dis;

  bool compute() {
    for (unsigned i = 0; i != out.size(); ++i) {
      out[i] = (T)dis(engine);
    }
    return true;
  }

  int getCycleEstimate() const { return 1; }
};

template class RandomUniform<float>;
template class RandomUniform<half>;

template<typename T>
class RandomStandardNormal : public Vertex {
public:
  Output<Vector<T>> out;

  std::default_random_engine engine;
  std::normal_distribution<float> dis;

  bool compute() {
    for (unsigned i = 0; i != out.size(); ++i) {
      out[i] = (T)dis(engine);
    }
    return true;
  }

  int getCycleEstimate() const { return 1; }
};

template class RandomStandardNormal<float>;
template class RandomStandardNormal<half>;

template<typename T>
class TruncatedNormal : public Vertex {
public:
  Output<Vector<T>> out;

  std::default_random_engine engine;
  std::normal_distribution<float> dis;

  bool compute() {
    for (unsigned i = 0; i != out.size(); ++i) {
      T val = (T)dis(engine);
      while (val >= (T)2 || val <= (T)-2) {
        val = (T)dis(engine);
      }
      out[i] = val;
    }
    return true;
  }

  int getCycleEstimate() const { return 1; }
};

template class TruncatedNormal<float>;
template class TruncatedNormal<half>;
*/

#define UNARY_ELEMENTWISE(NAME, EXP) \
template<typename T> \
class NAME : public Vertex { \
public: \
  Input<Vector<T>> a; \
  Output<Vector<T>> out; \
\
  bool compute() { \
    for (unsigned i = 0; i != a.size(); ++i) { \
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
    for (unsigned i = 0; i != a.size(); ++i) { \
      out[i] = (EXP); \
    } \
    return true; \
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
    for (unsigned i = 0; i != a.size(); ++i) {
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
    for (unsigned i = 0; i != a.size(); ++i) {
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

template<typename T>
class ReductionMax : public Vertex {
public:
  Input<Vector<T>> a;
  Output<Vector<T>> out;

  bool compute() {
    T max = std::numeric_limits<T>::lowest();

    for (unsigned i = 0; i != a.size(); ++i) {
      max = std::max(a[i], max);
    }

    out[0] = max;
    return true;
  }

  int getCycleEstimate() const { return 1; }
};

template class ReductionMax<float>;
template class ReductionMax<half>;
template class ReductionMax<int>;

template<typename T>
class ReductionMin : public Vertex {
public:
  Input<Vector<T>> a;
  Output<Vector<T>> out;

  bool compute() {
    T min = std::numeric_limits<T>::max();

    for (unsigned i = 0; i != a.size(); ++i) {
      min = std::min(a[i], min);
    }

    out[0] = min;
    return true;
  }

  int getCycleEstimate() const { return 1; }
};

template class ReductionMin<float>;
template class ReductionMin<half>;
template class ReductionMin<int>;

template<typename T>
class ReductionAdd : public Vertex {
public:
  Input<Vector<T>> a;
  Output<Vector<T>> out;

  bool compute() {
    T sum = 0.0;

    for (unsigned i = 0; i != a.size(); ++i) {
      sum += a[i];
    }

    out[0] = sum;
    return true;
  }

  int getCycleEstimate() const { return 1; }
};

template class ReductionAdd<float>;
template class ReductionAdd<half>;
template class ReductionAdd<int>;

template<typename T>
class ReductionSub : public Vertex {
public:
  Input<Vector<T>> a;
  Output<Vector<T>> out;

  bool compute() {
    T diff = 0.0;

    for (unsigned i = 0; i != a.size(); ++i) {
      diff -= a[i];
    }

    out[0] = diff;
    return true;
  }

  int getCycleEstimate() const { return 1; }
};

template class ReductionSub<float>;
template class ReductionSub<half>;
template class ReductionSub<int>;


template<typename T>
class ReductionMul : public Vertex {
public:
  Input<Vector<T>> a;
  Output<Vector<T>> out;

  bool compute() {
    T product = 1.0;

    for (unsigned i = 0; i != a.size(); ++i) {
      product *= a[i];
    }

    out[0] = product;
    return true;
  }

  int getCycleEstimate() const { return 1; }
};

template class ReductionMul<float>;
template class ReductionMul<half>;
template class ReductionMul<int>;

template<typename T>
class ReductionDiv : public Vertex {
public:
  Input<Vector<T>> a;
  Output<Vector<T>> out;

  bool compute() {
    T quot = 1.0;

    for (unsigned i = 0; i != a.size(); ++i) {
      quot /= a[i];
    }

    out[0] = quot;
    return true;
  }

  int getCycleEstimate() const { return 1; }
};

template class ReductionDiv<float>;
template class ReductionDiv<half>;
template class ReductionDiv<int>;

template<typename T>
class ReductionAnd : public Vertex {
public:
  Input<Vector<T>> a;
  Output<Vector<T>> out;

  bool compute() {
    T res = true;

    for (unsigned i = 0; i != a.size(); ++i) {
      res = res && a[i];
    }

    out[0] = res;
    return true;
  }

  int getCycleEstimate() const { return 1; }
};

template class ReductionAnd<bool>;

template<typename T>
class ReductionOr : public Vertex {
public:
  Input<Vector<T>> a;
  Output<Vector<T>> out;

  bool compute() {
    T res = false;

    for (unsigned i = 0; i != a.size(); ++i) {
      res = res || a[i];
    }

    out[0] = res;
    return true;
  }

  int getCycleEstimate() const { return 1; }
};

template class ReductionOr<bool>;

