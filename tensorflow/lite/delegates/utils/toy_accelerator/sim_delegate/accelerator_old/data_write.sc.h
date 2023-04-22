
int ACCNAME::Quantised_Multiplier(int x, int qm, sc_int<8> shift) {
  sc_int<64> pl;
  sc_int<32> pr;
  sc_int<32> msk;
  sc_int<32> sm;
  if (shift > 0) {
    pl = shift;
    pr = 0;
    msk = 0;
    sm = 0;
  } else {
    pl = 1;
    pr = -shift;
    msk = (1 << -shift) - 1;
    sm = msk >> 1;
  }
  sc_int<64> val = x * pl;
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
  return result_32;
}

void ACCNAME::Data_Write() {
  DATA d;
  int sums[4];
  int results[4];

  wait();
  while (1) {
    while (!writeX && !writeY) wait();

    int len = 0;
    if (writeX)
      len = lenX;
    else
      len = lenY;

    for (int i = 0; i < len; i++) {
      if (writeX) {
        sums[0] = C1[i * 4 + 0];
        sums[1] = C1[i * 4 + 1];
        sums[2] = C1[i * 4 + 2];
        sums[3] = C1[i * 4 + 3];

      } else {
        sums[0] = C2[i * 4 + 0];
        sums[1] = C2[i * 4 + 1];
        sums[2] = C2[i * 4 + 2];
        sums[3] = C2[i * 4 + 3];
      }

      results[0] = Quantised_Multiplier(sums[0], qm, shift);
      results[1] = Quantised_Multiplier(sums[1], qm, shift);
      results[2] = Quantised_Multiplier(sums[2], qm, shift);
      results[3] = Quantised_Multiplier(sums[3], qm, shift);

      d.data.range(7, 0) = results[0];
      d.data.range(15, 8) = results[1];
      d.data.range(23, 16) = results[2];
      d.data.range(31, 24) = results[3];

      d.tlast = (i + 1 == len);
      dout1.write(d);
    }

    if (writeX) writeX.write(0);
    if (writeY) writeY.write(0);
  }
}
