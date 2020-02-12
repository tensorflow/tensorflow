#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include "tensorflow/compiler/mlir/tensorflow/utils/runner_utils.h"


extern "C"
int64_t _global_get_unique_ids_count(
    StridedMemRefType<int64_t, 1> *ids, /*StridedMemRefType<int64_t, 0> *N*/int64_t N) {
  int64_t unique_count = 0;
  int64_t *data = ids->data;
  std::unordered_set<int64_t> m;
  int64_t real_n = N; //*((int64_t*)(N->data));
  for (int64_t i = 0; i < real_n; ++i) {
    if (m.find(*(data+i)) == m.end()) {
      ++unique_count;
      m.insert(*(data+i));
    }
  }

  return unique_count;
}

extern "C"
void _global_unique_ids(
    StridedMemRefType<int64_t, 1> *input_ids,
    StridedMemRefType<int64_t, 0> *id_count,
    StridedMemRefType<int64_t, 1> *output_ids) {
  int32_t cur_idx = -1;
  std::unordered_map<int64_t, int64_t> m;
  int64_t real_n = input_ids->sizes[0]; 
  for (int64_t i = 0; i < real_n; ++i) {
    int64_t spec_id = (*(input_ids->data + i));
    if (m.find(spec_id) == m.end()) {
      m[spec_id] = ++cur_idx;
      *(output_ids->data + cur_idx) = spec_id;
    } else {
      continue;
    }
  }
}

extern "C"
void _global_unique_index32(
    StridedMemRefType<int64_t, 1> *ids,
    StridedMemRefType<int64_t, 1> *unique_ids,
    StridedMemRefType<int32_t, 1> *ids_index) {
  std::unordered_map<int64_t, int32_t> m;
  int64_t unique_N = unique_ids->sizes[0]; 

  for (int32_t i = 0; i < unique_N; ++i) {
    m[*(unique_ids->data + i)] = i;
  }
  int64_t input_N = ids->sizes[0]; 
  for(int64_t i = 0; i < input_N; i++) {
    *(ids_index->data + i) = m[*(ids->data + i)];
  }
}

extern "C"
void _global_unique_index64(
    StridedMemRefType<int64_t, 1> *ids,
    StridedMemRefType<int64_t, 1> *unique_ids,
    StridedMemRefType<int64_t, 1> *ids_index) {
  std::unordered_map<int64_t, int64_t> m;
  int64_t unique_N = unique_ids->sizes[0]; 

  for (int64_t i = 0; i < unique_N; ++i) {
    m[*(unique_ids->data + i)] = i;
  }
  int64_t input_N = ids->sizes[0]; 
  for(int64_t i = 0; i < input_N; i++) {
    *(ids_index->data + i) = m[*(ids->data + i)];
  }
}

extern "C"
void _global_unique_i64_i64(
    StridedMemRefType<int64_t, 1> *input,
    StridedMemRefType<int64_t, 1> *unique_ids,
    StridedMemRefType<int64_t, 1> *idx) {
  std::unordered_map<int64_t, int64_t> m;
  int64_t N = input->sizes[0];
  int64_t* data = (int64_t*)(input->data);
  int64_t* unique_data = (int64_t*)(unique_ids->data);
  int64_t uniqueN = unique_ids->sizes[0];
  int64_t cur = 0;
  for (int64_t i = 0; i < N; ++i) {
    if (m.find(data[i]) == m.end()) {
      m[data[i]] = cur;
      unique_data[cur++] = data[i];
    }
  }
  if (cur != uniqueN) {
    assert(false && "_global_unique_i64_i64 failed.");
  }

  int64_t* idx_data = (int64_t*)(idx->data);
  for (int64_t i = 0; i < N; ++i) {
    idx_data[i] = m[data[i]];
  }
}

extern "C"
void _global_unique_i64_i32(
    StridedMemRefType<int64_t, 1> *input,
    StridedMemRefType<int64_t, 1> *unique_ids,
    StridedMemRefType<int32_t, 1> *idx) {
  std::unordered_map<int64_t, int32_t> m;
  int64_t N = input->sizes[0];
  int64_t* data = (int64_t*)(input->data);
  int64_t* unique_data = (int64_t*)(unique_ids->data);
  int64_t uniqueN = unique_ids->sizes[0];
  int cur = 0;
  for (int64_t i = 0; i < N; ++i) {
    if (m.find(data[i]) == m.end()) {
      m[data[i]] = cur;
      unique_data[cur++] = data[i];
    }
  }
  if (cur != uniqueN) {
    assert(false && "_global_unique_i64_i32 failed.");
  }

  int32_t* idx_data = (int32_t*)(idx->data);
  for (int64_t i = 0; i < N; ++i) {
    idx_data[i] = m[data[i]];
  }
}

extern "C"
void _global_unique_i32_i64(
    StridedMemRefType<int32_t, 1> *input,
    StridedMemRefType<int32_t, 1> *unique_ids,
    StridedMemRefType<int64_t, 1> *idx) {
  std::unordered_map<int32_t, int64_t> m;
  int64_t N = input->sizes[0];
  int32_t* data = (int32_t*)(input->data);
  int32_t* unique_data = (int32_t*)(unique_ids->data);
  int64_t uniqueN = unique_ids->sizes[0];
  int64_t cur = 0;
  for (int64_t i = 0; i < N; ++i) {
    if (m.find(data[i]) == m.end()) {
      m[data[i]] = cur;
      unique_data[cur++] = data[i];
    }
  }
  if (cur != uniqueN) {
    assert(false && "_global_unique_i32_i64 failed.");
  }

  int64_t* idx_data = (int64_t*)(idx->data);
  for (int64_t i = 0; i < N; ++i) {
    idx_data[i] = m[data[i]];
  }
}

extern "C"
void _global_unique_i32_i32(
    StridedMemRefType<int32_t, 1> *input,
    StridedMemRefType<int32_t, 1> *unique_ids,
    StridedMemRefType<int32_t, 1> *idx) {
  std::unordered_map<int32_t, int32_t> m;
  int64_t N = input->sizes[0];
  int32_t* data = (int32_t*)(input->data);
  int32_t* unique_data = (int32_t*)(unique_ids->data);
  int64_t uniqueN = unique_ids->sizes[0];
  int cur = 0;
  for (int64_t i = 0; i < N; ++i) {
    if (m.find(data[i]) == m.end()) {
      m[data[i]] = cur;
      unique_data[cur++] = data[i];
    }
  }
  if (cur != uniqueN) {
    assert(false && "_global_unique_i32_i32 failed.");
  }

  int32_t* idx_data = (int32_t*)(idx->data);
  for (int64_t i = 0; i < N; ++i) {
    idx_data[i] = m[data[i]];
  }
}
