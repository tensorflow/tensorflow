#include "acc.h"

sc_int<32> ACCNAME::mul_s8(sc_int<8> a, sc_int<8> b) {
  sc_int<32> c;
#pragma HLS RESOURCE variable = c core = Mul
  c = a * b;
  return c;
}

void ACCNAME::compute() {
  int acc[4][4];
  dat_t in[16][4];
  dat_t we[16][4];
  sc_int<64> wgt8x[8];
  sc_int<64> inp8x[8];
  int od[4][4][16];
  int prod[4][4][16];

  gemm_wait.write(true);
  wait();
  while (true) {
    while (gemm_wait.read()) {
      DWAIT();
    }

    int d = depth_val / 8;
    int wp = wp_val;
    int ip = ip_val;
    int wi = wp * d / 4;
    int ii = ip * d / 4;
    int N = inp_block / 2;

    for (int n = 0; n < 4; n++) {
      for (int m = 0; m < 2; m++) {
        int acc_idx = (n * N) + (wp * N) + m + (ip / 2);
        sc_int<64> acc2x = acc_mem[acc_idx];
        acc[n][m * 2] = acc2x.range(31, 0);
        acc[n][m * 2 + 1] = acc2x.range(63, 32);
      }
    }

    for (int n = 0; n < 4; n++) {
#pragma HLS unroll
      for (int m = 0; m < 4; m++) {
#pragma HLS unroll
        od[n][m][0] = 0;
        od[n][m][1] = 0;
        od[n][m][2] = 0;
        od[n][m][3] = 0;
        od[n][m][4] = 0;
        od[n][m][5] = 0;
        od[n][m][6] = 0;
        od[n][m][7] = 0;
        od[n][m][8] = 0;
        od[n][m][9] = 0;
        od[n][m][10] = 0;
        od[n][m][11] = 0;
        od[n][m][12] = 0;
        od[n][m][13] = 0;
        od[n][m][14] = 0;
        od[n][m][15] = 0;
      }
    }

    for (int rin = 0; rin < d; rin += 2) {
#pragma HLS pipeline II = 1
      wgt8x[0] = wgt_mem1[rin + wi];
      wgt8x[1] = wgt_mem2[rin + wi];
      wgt8x[2] = wgt_mem3[rin + wi];
      wgt8x[3] = wgt_mem4[rin + wi];
      wgt8x[4] = wgt_mem1[rin + wi + 1];
      wgt8x[5] = wgt_mem2[rin + wi + 1];
      wgt8x[6] = wgt_mem3[rin + wi + 1];
      wgt8x[7] = wgt_mem4[rin + wi + 1];

      inp8x[0] = inp_mem1[rin + ii];
      inp8x[1] = inp_mem2[rin + ii];
      inp8x[2] = inp_mem3[rin + ii];
      inp8x[3] = inp_mem4[rin + ii];
      inp8x[4] = inp_mem1[rin + ii + 1];
      inp8x[5] = inp_mem2[rin + ii + 1];
      inp8x[6] = inp_mem3[rin + ii + 1];
      inp8x[7] = inp_mem4[rin + ii + 1];

      for (int i = 0; i < 4; i++) {
#pragma HLS unroll
        in[0][i] = inp8x[i].range(7, 0);
        in[1][i] = inp8x[i].range(15, 8);
        in[2][i] = inp8x[i].range(23, 16);
        in[3][i] = inp8x[i].range(31, 24);
        in[4][i] = inp8x[i].range(39, 32);
        in[5][i] = inp8x[i].range(47, 40);
        in[6][i] = inp8x[i].range(55, 48);
        in[7][i] = inp8x[i].range(63, 56);
        in[8][i] = inp8x[i + 4].range(7, 0);
        in[9][i] = inp8x[i + 4].range(15, 8);
        in[10][i] = inp8x[i + 4].range(23, 16);
        in[11][i] = inp8x[i + 4].range(31, 24);
        in[12][i] = inp8x[i + 4].range(39, 32);
        in[13][i] = inp8x[i + 4].range(47, 40);
        in[14][i] = inp8x[i + 4].range(55, 48);
        in[15][i] = inp8x[i + 4].range(63, 56);

        we[0][i] = wgt8x[i].range(7, 0);
        we[1][i] = wgt8x[i].range(15, 8);
        we[2][i] = wgt8x[i].range(23, 16);
        we[3][i] = wgt8x[i].range(31, 24);
        we[4][i] = wgt8x[i].range(39, 32);
        we[5][i] = wgt8x[i].range(47, 40);
        we[6][i] = wgt8x[i].range(55, 48);
        we[7][i] = wgt8x[i].range(63, 56);
        we[8][i] = wgt8x[i + 4].range(7, 0);
        we[9][i] = wgt8x[i + 4].range(15, 8);
        we[10][i] = wgt8x[i + 4].range(23, 16);
        we[11][i] = wgt8x[i + 4].range(31, 24);
        we[12][i] = wgt8x[i + 4].range(39, 32);
        we[13][i] = wgt8x[i + 4].range(47, 40);
        we[14][i] = wgt8x[i + 4].range(55, 48);
        we[15][i] = wgt8x[i + 4].range(63, 56);
      }

      for (int m = 0; m < 4; m++) {
#pragma HLS unroll
        for (int n = 0; n < 4; n++) {
#pragma HLS unroll
          prod[m][n][0] = in[0][n] * we[0][m];
          prod[m][n][1] = in[1][n] * we[1][m];
          prod[m][n][2] = in[2][n] * we[2][m];
          prod[m][n][3] = in[3][n] * we[3][m];
          prod[m][n][4] = in[4][n] * we[4][m];
          prod[m][n][5] = in[5][n] * we[5][m];
          prod[m][n][6] = in[6][n] * we[6][m];
          prod[m][n][7] = in[7][n] * we[7][m];
          prod[m][n][8] = mul_s8(in[8][n], we[8][m]);
          prod[m][n][9] = mul_s8(in[9][n], we[9][m]);
          prod[m][n][10] = mul_s8(in[10][n], we[10][m]);
          prod[m][n][11] = mul_s8(in[11][n], we[11][m]);
          prod[m][n][12] = mul_s8(in[12][n], we[12][m]);
          prod[m][n][13] = mul_s8(in[13][n], we[13][m]);
          prod[m][n][14] = mul_s8(in[14][n], we[14][m]);
          prod[m][n][15] = mul_s8(in[15][n], we[15][m]);
        }
      }
      // Profile number of Macs
      macs->value+=256;

      for (int m = 0; m < 4; m++) {
#pragma HLS unroll
        for (int n = 0; n < 4; n++) {
#pragma HLS unroll
          for (int k = 0; k < 16; k++) {
#pragma HLS unroll
            od[m][n][k] += prod[m][n][k];
          }
        }
      }
    }
    while (storing.read()) wait();

    m_off.write(wp);
    n_off.write(ip);
    for (int n = 0; n < 4; n++) {
#pragma HLS pipeline II = 1
      for (int m = 0; m < 4; m++) {
        int od_val = 0;
        for (int k = 0; k < 16; k++) {
          od_val += od[m][n][k];
        }
        out_mem[m][n] = od_val + acc[m][n];
      }
    }
    storing.write(true);
    gemm_wait.write(true);
    DWAIT();
  }
}