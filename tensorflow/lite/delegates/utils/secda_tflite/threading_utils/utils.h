#ifndef SECDA_TFLITE_UTILS
#define SECDA_TFLITE_UTILS

#include <cmath>


typedef struct int_data_pointers {
  int* W1;
  int* W2;
  int* W3;
  int* W4;
  int* R1;
  int* R2;
  int* R3;
  int* R4;

  int_data_pointers(int* _W1, int* _W2, int* _W3, int* _W4, int* _R1, int* _R2,
                    int* _R3, int* _R4) {
    W1 = _W1;
    W2 = _W2;
    W3 = _W3;
    W4 = _W4;
    R1 = _R1;
    R2 = _R2;
    R3 = _R3;
    R4 = _R4;
  }

} INT_DP;

struct int8_params {
  int8_t* data;
  const int8_t* immutable_data;
  int order;
  int rows;
  int cols;
  int depth;
  int zero_point;

  void Init(int8_t* data_, int order_, int row_, int cols_, int zero_point_) {
    data = data_;
    order = order_;
    rows = row_;
    cols = cols_;
    depth = 0;
    zero_point = zero_point_;
  }

  void Init(const int8_t* data_, int order_, int row_, int cols_, int depth_,
            int zero_point_) {
    immutable_data = data_;
    order = order_;
    rows = row_;
    cols = cols_;
    depth = depth_;
    zero_point = zero_point_;
  }
};


struct int32_params {
  int32_t* data;
  const int32_t* immutable_data;
  int order;
  int rows;
  int cols;
  int depth;
  int zero_point;

  void Init(int32_t* data_, int order_, int row_, int cols_, int zero_point_) {
    data = data_;
    order = order_;
    rows = row_;
    cols = cols_;
    depth = 0;
    zero_point = zero_point_;
  }

  void Init(const int32_t* data_, int order_, int row_, int cols_, int depth_,
            int zero_point_) {
    immutable_data = data_;
    order = order_;
    rows = row_;
    cols = cols_;
    depth = depth_;
    zero_point = zero_point_;
  }
};

int roundUp(int numToRound, int multiple) {
  if (multiple == 0) return numToRound;
  int remainder = numToRound % multiple;
  if (remainder == 0) return numToRound;
  return numToRound + multiple - remainder;
}


int roundDown(int numToRound, int multiple) {
  return numToRound - (numToRound % multiple);
}

void unpad_matrix(int N_dim, int M_dim, int tN, int tM, int8_t** padded_A,
                  int8_t* A) {
  int pM = roundUp(M_dim, tM);
  for (int n = 0; n < N_dim; n++) {
    for (int m = 0; m < M_dim; m++) {
      A[n * M_dim + m] = padded_A[n][m];
    }
  }
}

void pad_matrix(int N_dim, int M_dim, int tN, int tM, const int8_t* A,
                int8_t* padded_A) {
  int pM = roundUp(M_dim, tM);
  for (int n = 0; n < N_dim; n++) {
    for (int m = 0; m < M_dim; m++) {
      padded_A[n * pM + m] = A[n * M_dim + m];
    }
  }
}

void pad_matrix(int N_dim, int M_dim, int tN, int tM, const int8_t* A,
                int8_t** padded_A) {
  int pM = roundUp(M_dim, tM);
  for (int n = 0; n < N_dim; n++) {
    for (int m = 0; m < M_dim; m++) {
      padded_A[n][m] = A[n * M_dim + m];
    }
  }
  int pN = roundUp(N_dim, tN);
}

void trans_matrix(int N_dim, int pN, int M_dim, const int8_t* A, int8_t** B) {
  for (int n = 0; n < N_dim; n++)
    for (int m = 0; m < M_dim; m++) B[m][n] = A[n * M_dim + m];
}

void padT_matrix(int N_dim, int M_dim, int tN, int tM, const int8_t* A,
                 int8_t** padded_AT) {
  int pN = roundUp(N_dim, tN);
  trans_matrix(N_dim, pN, M_dim, A, padded_AT);
}

void unpadT_matrix(int N_dim, int M_dim, int tN, int tM, int8_t** padded_A,
                   int8_t* A) {
  for (int n = 0; n < N_dim; n++) {
    for (int m = 0; m < M_dim; m++) {
      A[n * M_dim + m] = padded_A[m][n];
    }
  }
}

#endif // SECDA_TFLITE_UTILS