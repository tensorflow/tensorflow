void ACCNAME::Post1() {
  ACC_DTYPE<32> pcrf[16];
  ACC_DTYPE<8> pex[16];
  ACC_DTYPE<32> pls[16];
  ACC_DTYPE<32> prs[16];
  ACC_DTYPE<32> msks[16];
  ACC_DTYPE<32> sms[16];

  ACC_DTYPE<32> ls[16];
  ACC_DTYPE<32> exar[4];

  int yoff[16];
  int xoff[16];
  ACC_DTYPE<32> ind[256];
  ACC_DTYPE<8> pram[256];
  ACC_DTYPE<8> r1[256];
  DATA ot[4][16];

#pragma HLS array_partition variable = yoff cyclic factor = 4
#pragma HLS array_partition variable = xoff cyclic factor = 4
#pragma HLS array_partition variable = ls cyclic factor = 4

#pragma HLS array_partition variable = pcrf cyclic factor = 4
// #pragma HLS array_partition variable = pex cyclic factor = 4
#pragma HLS array_partition variable = pls cyclic factor = 4
#pragma HLS array_partition variable = prs cyclic factor = 4
#pragma HLS array_partition variable = msks cyclic factor = 4
#pragma HLS array_partition variable = sms cyclic factor = 4

#pragma HLS array_partition variable = ind complete
#pragma HLS array_partition variable = pram cyclic factor = 4
#pragma HLS array_partition variable = r1 cyclic factor = 4
#pragma HLS array_partition variable = ot complete

  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 4; j++) {
      ot[j][i].tlast = false;
    }
  }
  wait();
  while (true) {
    while (!write1.read())
      wait();

    for (int i = 0; i < 4; i++) {
#pragma HLS pipeline II = 1
      ACC_DTYPE<32> ex = WRQ3.read();
      pex[(i * 4) + 0] = ex.range(7, 0);
      pex[(i * 4) + 1] = ex.range(15, 8);
      pex[(i * 4) + 2] = ex.range(23, 16);
      pex[(i * 4) + 3] = ex.range(31, 24);
      exar[0] = pex[(i * 4) + 0];
      exar[1] = pex[(i * 4) + 1];
      exar[2] = pex[(i * 4) + 2];
      exar[3] = pex[(i * 4) + 3];
      for (int j = 0; j < 4; j++) {
#pragma HLS unroll
        if (exar[j] > 0) {
          ls[(i * 4) + j] = exar[j];
          prs[(i * 4) + j] = 0;
        } else {
          ls[(i * 4) + j] = 0;
          prs[(i * 4) + j] = -exar[j];
        }
      }
    }
    DWAIT(6);

    for (int i = 0; i < 16; i++) {
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 4
      pls[i] = (1 << ls[i]);
      msks[i] = (1 << prs[i]) - 1;
      sms[i] = ((1 << prs[i]) - 1) >> 1;
      yoff[i] = WRQ1.read();
      xoff[i] = WRQ2.read();
      pcrf[i] = WRQ3.read();
    }
    DWAIT(16);

    for (int i = 0; i < 256; i++) {
#pragma HLS unroll
      ind[i] = g1[i];
    }

    wait();
    wait();
    DWAIT();
    for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j += 4) {
#pragma HLS pipeline II = 1
        int yf1 = yoff[i];
        int xf1 = xoff[j + 0];
        int xf2 = xoff[j + 1];
        int xf3 = xoff[j + 2];
        int xf4 = xoff[j + 3];

        sc_int<32> in1 = ind[(i * 16) + j + 0];
        sc_int<32> in2 = ind[(i * 16) + j + 1];
        sc_int<32> in3 = ind[(i * 16) + j + 2];
        sc_int<32> in4 = ind[(i * 16) + j + 3];

        int mi1 = j;
        int mi2 = j + 1;
        int mi3 = j + 2;
        int mi4 = j + 3;

        int aa1 = yf1 + xf1 + in1;
        int aa2 = yf1 + xf2 + in2;
        int aa3 = yf1 + xf3 + in3;
        int aa4 = yf1 + xf4 + in4;

        // before 64
        sc_int<32> loff1 = pls[mi1];
        sc_int<32> loff2 = pls[mi2];
        sc_int<32> loff3 = pls[mi3];
        sc_int<32> loff4 = pls[mi4];

        sc_int<32> rs1 = prs[mi1];
        sc_int<32> rs2 = prs[mi2];
        sc_int<32> rs3 = prs[mi3];
        sc_int<32> rs4 = prs[mi4];
        sc_int<32> ms1 = msks[mi1];
        sc_int<32> ms2 = msks[mi2];
        sc_int<32> ms3 = msks[mi3];
        sc_int<32> ms4 = msks[mi4];
        sc_int<32> sm1 = sms[mi1];
        sc_int<32> sm2 = sms[mi2];
        sc_int<32> sm3 = sms[mi3];
        sc_int<32> sm4 = sms[mi4];
        sc_int<64> rf1 = pcrf[mi1];
        sc_int<64> rf2 = pcrf[mi2];
        sc_int<64> rf3 = pcrf[mi3];
        sc_int<64> rf4 = pcrf[mi4];

        // before 64
        sc_int<32> a1 = (aa1)*loff1;
        sc_int<32> a2 = (aa2)*loff2;
        sc_int<32> a3 = (aa3)*loff1;
        sc_int<32> a4 = (aa4)*loff2;

        if (a1 > MAX) a1 = MAX;
        if (a2 > MAX) a2 = MAX;
        if (a3 > MAX) a3 = MAX;
        if (a4 > MAX) a4 = MAX;

        if (a1 < MIN) a1 = MIN;
        if (a2 < MIN) a2 = MIN;
        if (a3 < MIN) a3 = MIN;
        if (a4 < MIN) a4 = MIN;

        sc_int<64> r_a1 = a1 * rf1;
        sc_int<64> r_a2 = a2 * rf2;
        sc_int<64> r_a3 = a3 * rf3;
        sc_int<64> r_a4 = a4 * rf4;

        sc_int<32> bf_a1;
        sc_int<32> bf_a2;
        sc_int<32> bf_a3;
        sc_int<32> bf_a4;

        bf_a1 = (r_a1 + POS) / DIVMAX;
        bf_a2 = (r_a2 + POS) / DIVMAX;
        bf_a3 = (r_a3 + POS) / DIVMAX;
        bf_a4 = (r_a4 + POS) / DIVMAX;

        if (r_a1 < 0) bf_a1 = (r_a1 + NEG) / DIVMAX;
        if (r_a2 < 0) bf_a2 = (r_a2 + NEG) / DIVMAX;
        if (r_a3 < 0) bf_a3 = (r_a3 + NEG) / DIVMAX;
        if (r_a4 < 0) bf_a4 = (r_a4 + NEG) / DIVMAX;

        sc_int<32> f_a1 = (bf_a1);
        sc_int<32> f_a2 = (bf_a2);
        sc_int<32> f_a3 = (bf_a3);
        sc_int<32> f_a4 = (bf_a4);

        f_a1 = SHR(f_a1, rs1);
        f_a2 = SHR(f_a2, rs2);
        f_a3 = SHR(f_a3, rs3);
        f_a4 = SHR(f_a4, rs4);

        sc_int<32> rf_a1 = bf_a1 & ms1;
        sc_int<32> rf_a2 = bf_a2 & ms2;
        sc_int<32> rf_a3 = bf_a3 & ms3;
        sc_int<32> rf_a4 = bf_a4 & ms4;

        sc_int<32> lf_a1 = (bf_a1 < 0) & 1;
        sc_int<32> lf_a2 = (bf_a2 < 0) & 1;
        sc_int<32> lf_a3 = (bf_a3 < 0) & 1;
        sc_int<32> lf_a4 = (bf_a4 < 0) & 1;

        sc_int<32> tf_a1 = sm1 + lf_a1;
        sc_int<32> tf_a2 = sm2 + lf_a2;
        sc_int<32> tf_a3 = sm3 + lf_a3;
        sc_int<32> tf_a4 = sm4 + lf_a4;

        sc_int<32> af_a1 = ((rf_a1 > tf_a1) & 1) + ra;
        sc_int<32> af_a2 = ((rf_a2 > tf_a2) & 1) + ra;
        sc_int<32> af_a3 = ((rf_a3 > tf_a3) & 1) + ra;
        sc_int<32> af_a4 = ((rf_a4 > tf_a4) & 1) + ra;

        f_a1 += af_a1;
        f_a2 += af_a2;
        f_a3 += af_a3;
        f_a4 += af_a4;

        if (f_a1 > MAX8) f_a1 = MAX8;
        else if (f_a1 < MIN8) f_a1 = MIN8;
        if (f_a2 > MAX8) f_a2 = MAX8;
        else if (f_a2 < MIN8) f_a2 = MIN8;
        if (f_a3 > MAX8) f_a3 = MAX8;
        else if (f_a3 < MIN8) f_a3 = MIN8;
        if (f_a4 > MAX8) f_a4 = MAX8;
        else if (f_a4 < MIN8) f_a4 = MIN8;

        pram[(i * 16) + j + 0] = f_a1.range(7, 0);
        pram[(i * 16) + j + 1] = f_a2.range(7, 0);
        pram[(i * 16) + j + 2] = f_a3.range(7, 0);
        pram[(i * 16) + j + 3] = f_a4.range(7, 0);
      }
    }
    wait();
    DWAIT(85);

    // Rearrange
    for (int i = 0; i < 256; i++) {
#pragma HLS unroll factor = 4
      r1[i] = pram[i];
    }
    DWAIT(192);
    wait();

    for (int j = 0; j < 4; j++) {
#pragma HLS unroll
      for (int i = 0; i < 16; i++) {
#pragma HLS unroll
        ot[j][i].data.range(7, 0) = r1[(i / 4) * 64 + (i % 4) * 4 + j * 16 + 0];
        ot[j][i].data.range(15, 8) =
            r1[(i / 4) * 64 + (i % 4) * 4 + j * 16 + 1];
        ot[j][i].data.range(23, 16) =
            r1[(i / 4) * 64 + (i % 4) * 4 + j * 16 + 2];
        ot[j][i].data.range(31, 24) =
            r1[(i / 4) * 64 + (i % 4) * 4 + j * 16 + 3];
      }
    }

    int rb = WRQ1.read();
    int lb = WRQ2.read();
    bool lb12f = lb < 12;
    bool lb8f = lb < 8;
    bool lb4f = lb < 4;
    bool rb15 = rb < 15;
    bool rb14 = rb < 14;
    bool rb13 = rb < 13;
    bool rb12 = rb < 12;
    bool rb11 = rb < 11;
    bool rb10 = rb < 10;
    bool rb9 = rb < 9;
    bool rb8 = rb < 8;
    bool rb7 = rb < 7;
    bool rb6 = rb < 6;
    bool rb5 = rb < 5;
    bool rb4 = rb < 4;
    bool rb3 = rb < 3;
    bool rb2 = rb < 2;
    bool rb1 = rb < 1;
    wait();

    dout1.write(ot[0][0]);
    if ((lb12f)) dout1.write(ot[0][1]);
    if ((lb8f)) dout1.write(ot[0][2]);
    if ((lb4f)) dout1.write(ot[0][3]);

    if (rb15) dout2.write(ot[1][0]);
    if (rb15 && (lb12f)) dout2.write(ot[1][1]);
    if (rb15 && (lb8f)) dout2.write(ot[1][2]);
    if (rb15 && (lb4f)) dout2.write(ot[1][3]);

    if (rb14) dout3.write(ot[2][0]);
    if (rb14 && (lb12f)) dout3.write(ot[2][1]);
    if (rb14 && (lb8f)) dout3.write(ot[2][2]);
    if (rb14 && (lb4f)) dout3.write(ot[2][3]);

    if (rb13) dout4.write(ot[3][0]);
    if (rb13 && (lb12f)) dout4.write(ot[3][1]);
    if (rb13 && (lb8f)) dout4.write(ot[3][2]);
    if (rb13 && (lb4f)) dout4.write(ot[3][3]);

    wait();
    if (rb12) dout1.write(ot[0][4]);
    if (rb12 && (lb12f)) dout1.write(ot[0][5]);
    if (rb12 && (lb8f)) dout1.write(ot[0][6]);
    if (rb12 && (lb4f)) dout1.write(ot[0][7]);

    if (rb11) dout2.write(ot[1][4]);
    if (rb11 && (lb12f)) dout2.write(ot[1][5]);
    if (rb11 && (lb8f)) dout2.write(ot[1][6]);
    if (rb11 && (lb4f)) dout2.write(ot[1][7]);

    if (rb10) dout3.write(ot[2][4]);
    if (rb10 && (lb12f)) dout3.write(ot[2][5]);
    if (rb10 && (lb8f)) dout3.write(ot[2][6]);
    if (rb10 && (lb4f)) dout3.write(ot[2][7]);

    if (rb9) dout4.write(ot[3][4]);
    if (rb9 && (lb12f)) dout4.write(ot[3][5]);
    if (rb9 && (lb8f)) dout4.write(ot[3][6]);
    if (rb9 && (lb4f)) dout4.write(ot[3][7]);

    wait();

    if ((rb8)) dout1.write(ot[0][8]);
    if ((rb8) && (lb12f)) dout1.write(ot[0][9]);
    if (rb8 && (lb8f)) dout1.write(ot[0][10]);
    if (rb8 && (lb4f)) dout1.write(ot[0][11]);

    if ((rb7)) dout2.write(ot[1][8]);
    if ((rb7) && (lb12f)) dout2.write(ot[1][9]);
    if (rb7 && (lb8f)) dout2.write(ot[1][10]);
    if (rb7 && (lb4f)) dout2.write(ot[1][11]);

    if ((rb6)) dout3.write(ot[2][8]);
    if ((rb6) && (lb12f)) dout3.write(ot[2][9]);
    if (rb6 && (lb8f)) dout3.write(ot[2][10]);
    if (rb6 && (lb4f)) dout3.write(ot[2][11]);

    if ((rb5)) dout4.write(ot[3][8]);
    if ((rb5) && (lb12f)) dout4.write(ot[3][9]);
    if (rb5 && (lb8f)) dout4.write(ot[3][10]);
    if (rb5 && (lb4f)) dout4.write(ot[3][11]);

    wait();

    if ((rb4)) dout1.write(ot[0][12]);
    if ((rb4) && (lb12f)) dout1.write(ot[0][13]);
    if (rb4 && (lb8f)) dout1.write(ot[0][14]);
    if (rb4 && (lb4f)) dout1.write(ot[0][15]);

    if ((rb3)) dout2.write(ot[1][12]);
    if ((rb3) && (lb12f)) dout2.write(ot[1][13]);
    if (rb3 && (lb8f)) dout2.write(ot[1][14]);
    if (rb3 && (lb4f)) dout2.write(ot[1][15]);

    if ((rb2)) dout3.write(ot[2][12]);
    if ((rb2) && (lb12f)) dout3.write(ot[2][13]);
    if (rb2 && (lb8f)) dout3.write(ot[2][14]);
    if (rb2 && (lb4f)) dout3.write(ot[2][15]);

    if ((rb1)) dout4.write(ot[3][12]);
    if ((rb1) && (lb12f)) dout4.write(ot[3][13]);
    if (rb1 && (lb8f)) dout4.write(ot[3][14]);
    if (rb1 && (lb4f)) dout4.write(ot[3][15]);
    DWAIT(77);

#ifndef __SYNTHESIS__
    gouts->value += 256;
#endif

    write1.write(0);
    wait();
  }
}
