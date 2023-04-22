#include <array>
#include <iomanip>
#include <iostream>
#include <string>

// int ACCNAME::Quantised_Multiplier(int x, int qm, sc_int<8> shift) {
//   int32_t reduced_multiplier =
//       (qm < 0x7FFF0000) ? ((qm + (1 << 15)) >> 16) : 0x7FFF;
//   int total_shift = 15 - shift;
//   int64_t temp =
//       (x * (int64_t)reduced_multiplier) + ((int64_t)1 << (total_shift - 1));
//   int32_t result = temp >> total_shift;
//   return result;
// }

// int ACCNAME::Quantised_Multiplier(int x, int qm, sc_int<8> shift) {
//   int nshift = shift;
//   int total_shift = 31 - shift;
//   sc_int<64> x_64 = x;
//   sc_int<64> quantized_multiplier_64(qm);
//   sc_int<64> one = 1;
//   sc_int<64> round = one << (total_shift - 1);  // ALU ADD + ALU SHLI
//   sc_int<64> result =
//       x_64 * quantized_multiplier_64 + round;  // ALU ADD + ALU MUL
//   result = result >> total_shift;              // ALU SHRI
//   int nresult = result;
//   if (result > MAX) result = MAX;  // ALU MIN
//   if (result < MIN) result = MIN;  // ALU MAX
//   sc_int<32> result_32 = result;

//   return result_32;
// }

int ACCNAME::Quantised_Multiplier(int x, int qm, sc_int<8> shift) {
  sc_int<64> pl;
  sc_int<32> pr;
  sc_int<32> msk;
  sc_int<32> sm;
  if (shift > 0) {
    pl = shift;
    // pl = (1 << shift);
    pr = 0;
    msk = 0;
    sm = 0;
  } else {
    // pl = 1;
    pl = 0;
    pr = -shift;
    msk = (1 << -shift) - 1;
    sm = msk >> 1;
  }
  // sc_int<64> val = x * pl;
  sc_int<64> val = x * (1 << pl);
  if (val > MAX) val = MAX;  // ALU MIN
  if (val < MIN) val = MIN;  // ALU MAX
  sc_int<64> val_2 = val * qm;
  sc_int<32> temp_1;
  temp_1 = (val_2 + POS) / DIVMAX;
  if (val_2 < 0) temp_1 = (val_2 + NEG) / DIVMAX;
  sc_int<32> val_3 = temp_1;
  val_3 = val_3 >> pr;
  sc_int<32> temp_2 = temp_1 & msk;
  sc_int<32> temp_3 = (temp_1 < 0) & 1;
  sc_int<32> temp_4 = sm + temp_3;
  sc_int<32> temp_5 = ((temp_2 > temp_4) & 1);
  sc_int<32> result_32 = val_3 + temp_5;
  int res = result_32;
  return result_32;
}

void ACCNAME::PPU(int* x, sc_int<32>* g1, sc_int<32>* r1) {
  for (int i = 0; i < 4; i++) {
#pragma HLS unroll
    for (int j = 0; j < 4; j++) {
#pragma HLS unroll
      int accum = g1[j * 4 + i] + x[j];
      r1[j * 4 + i] = accum;
      // cerr << (int)r1[j * 4 + i] << " : " << (j * 4 + i) << endl;
    }
  }

  for (int i = 0; i < 4; i++) {
    dst[dex_map1[dex_i]] += r1[0 + 4 * i];
    dst[dex_map2[dex_i]] += r1[1 + 4 * i];
    dst[dex_map3[dex_i]] += r1[2 + 4 * i];
    dst[dex_map4[dex_i++]] += r1[3 + 4 * i];
    DWAIT(51);
  }
}

void ACCNAME::Post1() {
  int x[4];

#pragma HLS array_partition variable = x complete dim = 0
  wait();
  while (true) {
    while (!write1.read()) wait();

    for (int i = 0; i < 4; i++) {
#pragma HLS unroll
      x[i] = WRQ1.read();
    }

    PPU(x, g1, r1);

    arrange1.write(1);
    wait();
    while (arrange1.read()) wait();

    wait();
    write1.write(0);
    wait();
  }
}

void ACCNAME::BiasAddQauntize() {
  bias_quantize.write(0);
  wait();
  while (true) {
    while (!bias_quantize.read()) wait();
    // saveMatrixCSV("aData/tconv/del_out_col2im_acc.csv", dst, out_r, out_c);
    for (int c = 0; c < out_c; c++) {
      for (int r = 0; r < out_r; r++) {
        int qm_ret = ra + Quantised_Multiplier(dst[out_c * r + c] + bias[c],
                                               crf1[c], crx[c].range(7, 0));
        if (qm_ret > MAX8)
          qm_ret = MAX8;
        else if (qm_ret < MIN8)
          qm_ret = MIN8;
        dst[out_c * r + c] = qm_ret;
      }
    }
    bias_quantize.write(0);
    wait();
  }
}

// void ACCNAME::Post2() {
//   int x[4];

// #pragma HLS array_partition variable = x complete dim = 0
//   wait();
//   while (true) {
//     while (!write2.read()) wait();

//     for (int i = 0; i < 4; i++) {
// #pragma HLS unroll
//       x[i] = WRQ2.read();
//     }

//     PPU(x, g2, r2);
//     arrange2.write(1);
//     wait();
//     while (arrange2.read()) wait();
//     write2.write(0);
//     wait();
//   }
// }

// void ACCNAME::Post3() {
//   int x[4];

// #pragma HLS array_partition variable = x complete dim = 0
//   wait();
//   while (true) {
//     while (!write3.read()) wait();

//     for (int i = 0; i < 4; i++) {
// #pragma HLS unroll
//       x[i] = WRQ3.read();
//     }

//     PPU(x, g3, r3);
//     arrange3.write(1);
//     wait();
//     while (arrange3.read()) wait();
//     write3.write(0);
//     wait();
//   }
// }

// void ACCNAME::Post4() {
//   int x[4];

// #pragma HLS array_partition variable = x complete dim = 0
//   wait();
//   while (true) {
//     while (!write4.read()) wait();

//     for (int i = 0; i < 4; i++) {
// #pragma HLS unroll
//       x[i] = WRQ4.read();
//     }

//     PPU(x, g4, r4);
//     arrange4.write(1);
//     wait();
//     while (arrange4.read()) wait();
//     write4.write(0);
//     wait();
//   }
// }