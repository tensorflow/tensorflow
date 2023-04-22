#include "acc.h"

void ACCNAME::scheduler() {
  schedule.write(false);
  wait();
  while (true) {
    wait();
    while (!schedule.read()) wait();

    int N = inp_block;
    int M = wgt_block;
    for (int n = 0; n < N; n += 4) {
      for (int m = 0; m < M; m += 4) {
        if (m == 4 && n == 0) {
          int k = 0;
        }

        wp_val.write(m);
        ip_val.write(n);
        gemm_wait.write(false);
        DWAIT();
        while (!gemm_wait.read()) wait();
        DWAIT();
      }
    }
    while (storing.read()) wait();
    schedule.write(false);
    DWAIT();
  }
}