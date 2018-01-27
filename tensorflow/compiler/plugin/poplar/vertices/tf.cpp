#include <cmath>
#include <limits>

#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>


using namespace poplar;

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

REDUCTION_ELEMENTWISE(ReductionMax, std::numeric_limits<T>::lowest(), ((a[i] > v) ? a[i] : v))
REDUCTION_ELEMENTWISE(ReductionMin, std::numeric_limits<T>::max(),    ((a[i] < v) ? a[i] : v))
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
  Input<T> b; \
  Output<Vector<T>> out; \
  T initval; \
\
  bool compute() { \
    unsigned selected = 0; \
    for (unsigned i = 1; i < a.size(); ++i) { \
      if (a[i] OP a[selected]) { \
        selected = i; \
      } \
    } \
    for (unsigned i = 0; i < out.size(); ++i) { \
      out[i] = (i == selected) ? b : initval; \
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

WINDOWED_SELECTION(SelectionGe, >=)
WINDOWED_SELECTION(SelectionGt, >)
WINDOWED_SELECTION(SelectionLe, <=)
WINDOWED_SELECTION(SelectionLt, <)

