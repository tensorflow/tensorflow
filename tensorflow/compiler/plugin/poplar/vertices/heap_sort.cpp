// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#include <limits>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <utility>

template <typename BIter>
static void reverse(BIter begin, BIter end) {
  while (begin < end) {
    end--;
    std::swap(*begin, *end);
    begin++;
  }
}

template <typename Iter>
static void rotate(Iter begin, Iter new_begin, Iter end) {
  reverse(begin, new_begin);
  reverse(new_begin, end);
  reverse(begin, end);
}

static std::uint32_t root() { return 0; }

static std::uint32_t parent(std::uint32_t index) { return (index - 1) / 2; }

static std::uint32_t left(std::uint32_t index) { return (index * 2) + 1; }

static std::uint32_t right(std::uint32_t index) { return (index * 2) + 2; }

template <typename ValueType>
class HeapSortVertex : public poplar::Vertex {
 public:
  poplar::InOut<poplar::Vector<ValueType>> out;

  bool compute() {
    tail = 0;

    rotate(out.begin(), out.begin() + 1, out.begin() + out.size());
    reverse(out.begin(), out.begin() + out.size() - 2);

    for (std::size_t i = 0; i < out.size(); ++i) {
      push(out[i]);
    }

    while (tail > 0) {
      pop();
    }

    return true;
  }

 private:
  std::uint32_t tail;

  void swap_elem(std::uint32_t a, std::uint32_t b) {
    if (a < tail && b < tail) {
      std::swap(out[a], out[b]);
    }
  }

  ValueType elem(std::uint32_t index) {
    if (index < tail) {
      return out[index];
    } else {
      return std::numeric_limits<ValueType>::lowest();
    }
  }

  std::uint32_t max_child(std::uint32_t index) {
    if (elem(left(index)) < elem(right(index))) {
      return right(index);
    } else {
      return left(index);
    }
  }

  void push(ValueType value) {
    std::uint32_t index = tail++;
    out[index] = value;

    for (std::uint32_t i = index; i != root() && elem(i) > elem(parent(i));
         i = parent(i)) {
      swap_elem(parent(i), i);
    }
  }

  ValueType pop() {
    swap_elem(root(), tail - 1);
    tail--;

    std::uint32_t i = root();
    while (elem(i) < elem(max_child(i))) {
      const auto dst = max_child(i);
      swap_elem(i, dst);
      i = dst;
    }

    return out[tail];
  }
};

template class HeapSortVertex<float>;
template class HeapSortVertex<int>;
template class HeapSortVertex<half>;

template <typename KeyType, typename ValueType>
class HeapSortVertexKV : public poplar::Vertex {
 public:
  poplar::InOut<poplar::Vector<KeyType>> key;
  poplar::InOut<poplar::Vector<ValueType>> value;

  bool compute() {
    tail = 0;

    rotate(key.begin(), key.begin() + 1, key.begin() + key.size());
    reverse(key.begin(), key.begin() + key.size() - 2);

    rotate(value.begin(), value.begin() + 1, value.begin() + value.size());
    reverse(value.begin(), value.begin() + value.size() - 2);

    for (std::size_t i = 0; i < key.size(); ++i) {
      push(key[i], value[i]);
    }

    while (tail > 0) {
      pop();
    }

    return true;
  }

 private:
  std::uint32_t tail;

  void swap_elem(std::uint32_t a, std::uint32_t b) {
    if (a < tail && b < tail) {
      std::swap(key[a], key[b]);
      std::swap(value[a], value[b]);
    }
  }

  KeyType elem(std::uint32_t index) {
    if (index < tail) {
      return key[index];
    } else {
      return std::numeric_limits<KeyType>::lowest();
    }
  }

  std::uint32_t max_child(std::uint32_t index) {
    if (elem(left(index)) < elem(right(index))) {
      return right(index);
    } else {
      return left(index);
    }
  }

  void push(KeyType k, ValueType v) {
    std::uint32_t index = tail++;
    key[index] = k;
    value[index] = v;

    for (std::uint32_t i = index; i != root() && elem(i) > elem(parent(i));
         i = parent(i)) {
      swap_elem(parent(i), i);
    }
  }

  float pop() {
    swap_elem(root(), tail - 1);
    tail--;

    std::uint32_t i = root();
    while (elem(i) < elem(max_child(i))) {
      const auto dst = max_child(i);
      swap_elem(i, dst);
      i = dst;
    }

    return value[tail];
  }
};

template class HeapSortVertexKV<float, float>;
template class HeapSortVertexKV<float, int>;
template class HeapSortVertexKV<float, half>;
template class HeapSortVertexKV<int, float>;
template class HeapSortVertexKV<int, int>;
template class HeapSortVertexKV<int, half>;
template class HeapSortVertexKV<half, float>;
template class HeapSortVertexKV<half, int>;
template class HeapSortVertexKV<half, half>;
