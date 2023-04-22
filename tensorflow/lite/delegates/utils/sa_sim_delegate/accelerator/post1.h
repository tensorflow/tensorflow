
int ACCNAME::Quantised_Multiplier(int x, int qm, sc_int<8> shift) {
  int nshift = shift;
  int total_shift = 31 - shift;
  sc_int<64> x_64 = x;
  sc_int<64> quantized_multiplier_64(qm);
  sc_int<64> one = 1;
  sc_int<64> round = one << (total_shift - 1); // ALU ADD + ALU SHLI
  sc_int<64> result = x_64 * quantized_multiplier_64 + round;// ALU ADD + ALU MUL
  result = result >> total_shift; // ALU SHRI

  int nresult = result;

  if (result > MAX) result = MAX; // ALU MIN
  if (result < MIN) result = MIN; // ALU MAX
  sc_int<32> result_32 = result; 

  return result_32;
}


// int ACCNAME::Quantised_Multiplier(int x, int qm, sc_int<8> shift) {
// 	sc_int<64> pl;
// 	sc_int<32> pr;
// 	sc_int<32> msk;
// 	sc_int<32> sm;
// 	if(shift>0){
// 		pl = shift;
// 		pr = 0;
// 		msk = 0;
// 		sm = 0;
// 	}else{
// 		pl = 1;
// 		pr = -shift;
// 		msk = (1 << -shift)-1;
// 		sm = msk>>1;
// 	}
// 	sc_int<64> val = x*pl;
//   if (val > MAX) val = MAX; // ALU MIN
//   if (val < MIN) val = MIN; // ALU MAX
//   sc_int<64> val_2 = val * qm;
// 	sc_int<32> temp_1;
// 	temp_1 = (val_2+POS)/DIVMAX;
// 	if(val_2<0)temp_1 = (val_2+NEG)/DIVMAX;
// 	sc_int<32> val_3 = temp_1;
// 	val_3 = val_3>>pr;
// 	sc_int<32> temp_2 = temp_1 & msk;
// 	sc_int<32> temp_3 = (temp_1 < 0) & 1;
// 	sc_int<32> temp_4 = sm + temp_3;
// 	sc_int<32> temp_5 = ((temp_2 > temp_4) & 1);
//   sc_int<32> result_32 = val_3 + temp_5;
// 	return result_32;
// }

void ACCNAME::Post1() {
  int yoff[16];
  int xoff[16];
  int pcrf[16];
  ACC_DTYPE<8> pex[16];

  ACC_DTYPE<32> ind[256];
  ACC_DTYPE<8> pram1[256];
  ACC_DTYPE<8> r1[256];
  DATA ot[4][16];

#pragma HLS array_partition variable = yoff cyclic factor = 4 dim = 0
#pragma HLS array_partition variable = xoff cyclic factor = 4 dim = 0
#pragma HLS array_partition variable = ind complete dim = 0
#pragma HLS array_partition variable = pram1 cyclic factor = 4 dim = 0
#pragma HLS array_partition variable = r1 cyclic factor = 4 dim = 0
#pragma HLS array_partition variable = ot complete dim = 0

  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 4; j++) {
      ot[j][i].tlast = false;
    }
  }
  wait();
  while (true) {
    while (!write1.read()) wait();

    for (int i = 0; i < 4; i++) {
#pragma HLS pipeline II = 1
      ACC_DTYPE<32> ex = WRQ3.read();
      pex[(i * 4) + 0] = ex.range(7, 0);
      pex[(i * 4) + 1] = ex.range(15, 8);
      pex[(i * 4) + 2] = ex.range(23, 16);
      pex[(i * 4) + 3] = ex.range(31, 24);
    }

    for (int i = 0; i < 16; i++) {
#pragma HLS unroll factor = 4
#pragma HLS pipeline II = 4
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
      for (int j = 0; j < 16; j++) {
#pragma HLS pipeline II = 1

        int yf1 = yoff[i];
        int xf1 = xoff[j + 0];
        ACC_DTYPE<32> in1 = ind[(i * 16) + j + 0];
        int value = in1;
        int accum = yf1 + xf1 + in1;



        int rf =  pcrf[j];
        int ex =  pex[j];


        int ret_accum = Quantised_Multiplier(accum, pcrf[j], pex[j]);
        sc_int<32> f_a1 = ret_accum + ra; // ALU ADD

        if (f_a1 > MAX8)
          f_a1 = MAX8;
        else if (f_a1 < MIN8)
          f_a1 = MIN8;

        // cout << value << " " << yf1+xf1  << " " << accum << "-->" << f_a1 << endl;
        // cout <<  pcrf[j] << " " <<  pex[j]  << " " << accum << "-->" << f_a1 << endl;
        int kasas = f_a1;
        pram1[(i * 16) + j + 0] = f_a1.range(7, 0);
      }
    }
    wait();
    DWAIT(92);

    // Rearrange
    for (int i = 0; i < 256; i++) {
#pragma HLS unroll factor = 4
      r1[i] = pram1[i];
    }
    DWAIT(192);
    wait();

    // Map to douts
    for (int i = 0; i < 16; i++) {
#pragma HLS unroll
      for (int j = 0; j < 4; j++) {
#pragma HLS unroll
        ot[j][i].data.range(7, 0) = r1[i * 16 + j * 4 + 0];
        ot[j][i].data.range(15, 8) = r1[i * 16 + j * 4 + 1];
        ot[j][i].data.range(23, 16) = r1[i * 16 + j * 4 + 2];
        ot[j][i].data.range(31, 24) = r1[i * 16 + j * 4 + 3];
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
    if ((lb12f)) dout1.write(ot[1][0]);
    if ((lb8f)) dout1.write(ot[2][0]);
    if ((lb4f)) dout1.write(ot[3][0]);

    if (rb15) dout2.write(ot[0][1]);
    if (rb15 && (lb12f)) dout2.write(ot[1][1]);
    if (rb15 && (lb8f)) dout2.write(ot[2][1]);
    if (rb15 && (lb4f)) dout2.write(ot[3][1]);

    if (rb14) dout3.write(ot[0][2]);
    if (rb14 && (lb12f)) dout3.write(ot[1][2]);
    if (rb14 && (lb8f)) dout3.write(ot[2][2]);
    if (rb14 && (lb4f)) dout3.write(ot[3][2]);

    if (rb13) dout4.write(ot[0][3]);
    if (rb13 && (lb12f)) dout4.write(ot[1][3]);
    if (rb13 && (lb8f)) dout4.write(ot[2][3]);
    if (rb13 && (lb4f)) dout4.write(ot[3][3]);

    wait();

    if (rb12) dout1.write(ot[0][4]);
    if (rb12 && (lb12f)) dout1.write(ot[1][4]);
    if (rb12 && (lb8f)) dout1.write(ot[2][4]);
    if (rb12 && (lb4f)) dout1.write(ot[3][4]);

    if (rb11) dout2.write(ot[0][5]);
    if (rb11 && (lb12f)) dout2.write(ot[1][5]);
    if (rb11 && (lb8f)) dout2.write(ot[2][5]);
    if (rb11 && (lb4f)) dout2.write(ot[3][5]);

    if (rb10) dout3.write(ot[0][6]);
    if (rb10 && (lb12f)) dout3.write(ot[1][6]);
    if (rb10 && (lb8f)) dout3.write(ot[2][6]);
    if (rb10 && (lb4f)) dout3.write(ot[3][6]);

    if (rb9) dout4.write(ot[0][7]);
    if (rb9 && (lb12f)) dout4.write(ot[1][7]);
    if (rb9 && (lb8f)) dout4.write(ot[2][7]);
    if (rb9 && (lb4f)) dout4.write(ot[3][7]);

    wait();

    if ((rb8)) dout1.write(ot[0][8]);
    if ((rb8) && (lb12f)) dout1.write(ot[1][8]);
    if (rb8 && (lb8f)) dout1.write(ot[2][8]);
    if (rb8 && (lb4f)) dout1.write(ot[3][8]);

    if ((rb7)) dout2.write(ot[0][9]);
    if ((rb7) && (lb12f)) dout2.write(ot[1][9]);
    if (rb7 && (lb8f)) dout2.write(ot[2][9]);
    if (rb7 && (lb4f)) dout2.write(ot[3][9]);

    if ((rb6)) dout3.write(ot[0][10]);
    if ((rb6) && (lb12f)) dout3.write(ot[1][10]);
    if (rb6 && (lb8f)) dout3.write(ot[2][10]);
    if (rb6 && (lb4f)) dout3.write(ot[3][10]);

    if ((rb5)) dout4.write(ot[0][11]);
    if ((rb5) && (lb12f)) dout4.write(ot[1][11]);
    if (rb5 && (lb8f)) dout4.write(ot[2][11]);
    if (rb5 && (lb4f)) dout4.write(ot[3][11]);

    wait();

    if ((rb4)) dout1.write(ot[0][12]);
    if ((rb4) && (lb12f)) dout1.write(ot[1][12]);
    if (rb4 && (lb8f)) dout1.write(ot[2][12]);
    if (rb4 && (lb4f)) dout1.write(ot[3][12]);

    if ((rb3)) dout2.write(ot[0][13]);
    if ((rb3) && (lb12f)) dout2.write(ot[1][13]);
    if (rb3 && (lb8f)) dout2.write(ot[2][13]);
    if (rb3 && (lb4f)) dout2.write(ot[3][13]);

    if ((rb2)) dout3.write(ot[0][14]);
    if ((rb2) && (lb12f)) dout3.write(ot[1][14]);
    if (rb2 && (lb8f)) dout3.write(ot[2][14]);
    if (rb2 && (lb4f)) dout3.write(ot[3][14]);

    if ((rb1)) dout4.write(ot[0][15]);
    if ((rb1) && (lb12f)) dout4.write(ot[1][15]);
    if (rb1 && (lb8f)) dout4.write(ot[2][15]);
    if (rb1 && (lb4f)) dout4.write(ot[3][15]);
    DWAIT(77);

#ifndef __SYNTHESIS__
    gouts->value+=256;
#endif

    write1.write(0);
    wait();
  }
}
