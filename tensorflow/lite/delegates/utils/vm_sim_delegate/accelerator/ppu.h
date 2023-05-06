sc_int<64> ACCNAME::mul_s64(int a, sc_int<64> b) {
  sc_int<64> c;
#pragma HLS RESOURCE variable = c core = MulnS
  c = a * b;
  return c;
}

int ACCNAME::Quantised_Multiplier(int x, int qm, sc_int<8> shift) {
  int nshift = shift;
  int total_shift = 31 - shift;
  sc_int<64> x_64 = x;
  sc_int<64> quantized_multiplier_64(qm);
  sc_int<64> one = 1;
  sc_int<64> round = one << (total_shift - 1); // ALU ADD + ALU SHLI
  sc_int<64> result =
      x_64 * quantized_multiplier_64 + round; // ALU ADD + ALU MUL
  result = result >> total_shift;             // ALU SHRI

  int nresult = result;

  if (result > MAX) result = MAX; // ALU MIN
  if (result < MIN) result = MIN; // ALU MAX
  sc_int<32> result_32 = result;

  return result_32;
}

// int ACCNAME::Quantised_Multiplier(int x, int qm, sc_int<8> shift) {
//   sc_int<64> pl;
//   sc_int<32> pr;
//   sc_int<32> msk;
//   sc_int<32> sm;
//   if (shift > 0) {
//     pl = shift;
//     pr = 0;
//     msk = 0;
//     sm = 0;
//   } else {
//     pl = 1;
//     pr = -shift;
//     msk = (1 << -shift) - 1;
//     sm = msk >> 1;
//   }
//   sc_int<64> val = x * pl;
//   if (val > MAX) val = MAX;  // ALU MIN
//   if (val < MIN) val = MIN;  // ALU MAX
//   sc_int<64> val_2 = val * qm;
//   sc_int<32> temp_1;
//   temp_1 = (val_2 + POS) / DIVMAX;
//   if (val_2 < 0) temp_1 = (val_2 + NEG) / DIVMAX;
//   sc_int<32> val_3 = temp_1;
//   val_3 = val_3 >> pr;
//   sc_int<32> temp_2 = temp_1 & msk;
//   sc_int<32> temp_3 = (temp_1 < 0) & 1;
//   sc_int<32> temp_4 = sm + temp_3;
//   sc_int<32> temp_5 = ((temp_2 > temp_4) & 1);
//   sc_int<32> result_32 = val_3 + temp_5;
//   int res = result_32;
//   return result_32;
// }

int ACCNAME::Quantised_Multiplier_v2(int x, int qm, sc_int<64> pl,
                                     sc_int<32> pr, sc_int<32> msk,
                                     sc_int<32> sm) {
  sc_int<64> val = mul_s64(x, pl);
  if (val > MAX) val = MAX; // ALU MIN
  if (val < MIN) val = MIN; // ALU MAX
  sc_int<64> val_2 = mul_s64(qm, val);
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

void ACCNAME::PPU(int *x, int *y, int *pcrf, sc_int<8> *pex, sc_int<32> *g,
                  sc_int<8> *r) {
  int accum[16];
  ACC_DTYPE<64> pls[4];
  ACC_DTYPE<32> prs[4];
  ACC_DTYPE<32> msks[4];
  ACC_DTYPE<32> sms[4];

#pragma HLS array_partition variable = accum complete dim = 0
#pragma HLS array_partition variable = pls complete dim = 0
#pragma HLS array_partition variable = prs complete dim = 0
#pragma HLS array_partition variable = msks complete dim = 0
#pragma HLS array_partition variable = sms complete dim = 0

  wait();

  for (int i = 0; i < 4; i++) {
#pragma HLS unroll
    for (int j = 0; j < 4; j++) {
#pragma HLS unroll
      accum[j * 4 + i] = g[j * 4 + i] + y[i] + x[j];
    }
  }

  for (int i = 0; i < 4; i++) {
#pragma HLS unroll
    if (pex[i] > 0) {
      pls[i] = pex[i];
      prs[i] = 0;
      msks[i] = 0;
      sms[i] = 0;
    } else {
      pls[i] = 1;
      prs[i] = -pex[i];
      msks[i] = (1 << -pex[i]) - 1;
      sms[i] = ((1 << -pex[i]) - 1) >> 1;
    }
  }

  for (int j = 0; j < 4; j++) {
    for (int i = 0; i < 4; i++) {
#pragma HLS pipeline II = 1
      int accum1 = accum[j * 4 + i];
      int ret_accum1 = Quantised_Multiplier_v2(accum1, pcrf[j], pls[j], prs[j],
                                               msks[j], sms[j]);
      sc_int<32> f1_a1 = ret_accum1 + ra;
      if (f1_a1 > MAX8) f1_a1 = MAX8;
      else if (f1_a1 < MIN8) f1_a1 = MIN8;
      r[j * 4 + i] = f1_a1.range(7, 0);
    }
  }
  DWAIT(51);
}

// void ACCNAME::PPU(int *x, int *y, int *pcrf, sc_int<8> *pex, sc_int<32> *g1,
//                   sc_int<8> *r1) {
//   for (int i = 0; i < 4; i++) {
// #pragma HLS unroll
//     for (int j = 0; j < 4; j++) {
// #pragma HLS unroll
//       int accum = g1[j * 4 + i] + y[i] + x[j];
//       int ret_accum = Quantised_Multiplier(accum, pcrf[j], pex[j]);
//       sc_int<32> f_a1 = ret_accum + ra;
//       int res = f_a1;
//       if (f_a1 > MAX8) f_a1 = MAX8;
//       else if (f_a1 < MIN8) f_a1 = MIN8;
//       r1[j * 4 + i] = f_a1.range(7, 0);
//     }
//   }
//   DWAIT(51);
// }

void ACCNAME::Post1() {
  int y[4];
  int x[4];
  int pcrf[4];
  ACC_DTYPE<8> pex[4];

#pragma HLS array_partition variable = y complete dim = 0
#pragma HLS array_partition variable = x complete dim = 0
#pragma HLS array_partition variable = pcrf complete dim = 0
#pragma HLS array_partition variable = pex complete dim = 0
  wait();
  while (true) {
    while (!write1.read())
      wait();
    for (int i = 0; i < 4; i++) {
#pragma HLS unroll
      y[i] = WRQ1.read();
    }

    for (int i = 0; i < 4; i++) {
#pragma HLS unroll
      x[i] = WRQ1.read();
    }

    for (int i = 0; i < 4; i++) {
#pragma HLS unroll
      pcrf[i] = WRQ1.read();
    }
    ACC_DTYPE<32> ex = WRQ1.read();
    pex[0] = ex.range(7, 0);
    pex[1] = ex.range(15, 8);
    pex[2] = ex.range(23, 16);
    pex[3] = ex.range(31, 24);
    PPU(x, y, pcrf, pex, g1, r1);
    arrange1.write(1);
    wait();
    while (arrange1.read())
      wait();
    write1.write(0);
    wait();
  }
}

void ACCNAME::Post2() {
  int y[4];
  int x[4];
  int pcrf[4];
  ACC_DTYPE<8> pex[4];

#pragma HLS array_partition variable = y complete dim = 0
#pragma HLS array_partition variable = x complete dim = 0
#pragma HLS array_partition variable = pcrf complete dim = 0
#pragma HLS array_partition variable = pex complete dim = 0
  wait();
  while (true) {
    while (!write2.read())
      wait();
    for (int i = 0; i < 4; i++) {
#pragma HLS unroll
      y[i] = WRQ2.read();
    }

    for (int i = 0; i < 4; i++) {
#pragma HLS unroll
      x[i] = WRQ2.read();
    }

    for (int i = 0; i < 4; i++) {
#pragma HLS unroll
      pcrf[i] = WRQ2.read();
    }
    ACC_DTYPE<32> ex = WRQ2.read();
    pex[0] = ex.range(7, 0);
    pex[1] = ex.range(15, 8);
    pex[2] = ex.range(23, 16);
    pex[3] = ex.range(31, 24);
    PPU(x, y, pcrf, pex, g2, r2);
    arrange2.write(1);
    wait();
    while (arrange2.read())
      wait();
    write2.write(0);
    wait();
  }
}

void ACCNAME::Post3() {
  int y[4];
  int x[4];
  int pcrf[4];
  ACC_DTYPE<8> pex[4];

#pragma HLS array_partition variable = y complete dim = 0
#pragma HLS array_partition variable = x complete dim = 0
#pragma HLS array_partition variable = pcrf complete dim = 0
#pragma HLS array_partition variable = pex complete dim = 0
  wait();
  while (true) {
    while (!write3.read())
      wait();
    for (int i = 0; i < 4; i++) {
#pragma HLS unroll
      y[i] = WRQ3.read();
    }

    for (int i = 0; i < 4; i++) {
#pragma HLS unroll
      x[i] = WRQ3.read();
    }

    for (int i = 0; i < 4; i++) {
#pragma HLS unroll
      pcrf[i] = WRQ3.read();
    }
    ACC_DTYPE<32> ex = WRQ3.read();
    pex[0] = ex.range(7, 0);
    pex[1] = ex.range(15, 8);
    pex[2] = ex.range(23, 16);
    pex[3] = ex.range(31, 24);
    PPU(x, y, pcrf, pex, g3, r3);
    arrange3.write(1);
    wait();
    while (arrange3.read())
      wait();
    write3.write(0);
    wait();
  }
}

void ACCNAME::Post4() {
  int y[4];
  int x[4];
  int pcrf[4];
  ACC_DTYPE<8> pex[4];

#pragma HLS array_partition variable = y complete dim = 0
#pragma HLS array_partition variable = x complete dim = 0
#pragma HLS array_partition variable = pcrf complete dim = 0
#pragma HLS array_partition variable = pex complete dim = 0
  wait();
  while (true) {
    while (!write4.read())
      wait();
    for (int i = 0; i < 4; i++) {
#pragma HLS unroll
      y[i] = WRQ4.read();
    }

    for (int i = 0; i < 4; i++) {
#pragma HLS unroll
      x[i] = WRQ4.read();
    }

    for (int i = 0; i < 4; i++) {
#pragma HLS unroll
      pcrf[i] = WRQ4.read();
    }
    ACC_DTYPE<32> ex = WRQ4.read();
    pex[0] = ex.range(7, 0);
    pex[1] = ex.range(15, 8);
    pex[2] = ex.range(23, 16);
    pex[3] = ex.range(31, 24);
    PPU(x, y, pcrf, pex, g4, r4);
    arrange4.write(1);
    wait();
    while (arrange4.read())
      wait();
    write4.write(0);
    wait();
  }
}
