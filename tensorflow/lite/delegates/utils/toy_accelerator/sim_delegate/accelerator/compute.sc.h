int ACCNAME::Quantised_Multiplier(int x, int qm, int shift) {
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

ACC_DTYPE<32> ACCNAME::Clamp_Combine(int i1, int i2, int i3, int i4, int qa_max,
                                     int qa_min) {
  if (i1 < qa_min) i1 = qa_min;
  if (i1 > qa_max) i1 = qa_max;
  if (i2 < qa_min) i2 = qa_min;
  if (i2 > qa_max) i2 = qa_max;
  if (i3 < qa_min) i3 = qa_min;
  if (i3 > qa_max) i3 = qa_max;
  if (i4 < qa_min) i4 = qa_min;
  if (i4 > qa_max) i4 = qa_max;

  ACC_DTYPE<32> d;
  d.range(7, 0) = i1;
  d.range(15, 8) = i2;
  d.range(23, 16) = i3;
  d.range(31, 24) = i4;

  return d;
}

void ACCNAME::Counter() {
  wait();
  while (1) {
    per_batch_cycles->value++;
    if (computeS.read()==1)
      active_cycles->value++;
    wait();
  }
}

void ACCNAME::Compute() {
  ACC_DTYPE<32> i1;
  ACC_DTYPE<32> i2;
  ACC_DTYPE<8> i1mem[4];
  ACC_DTYPE<8> i2mem[4];
  int length;

  int s_in1[4];
  int s_in2[4];
  int sum[4];
  int f_out[4];

  DATA d;

  computeS.write(0);
  wait();
  while (1) {
    computeS.write(0);
    DWAIT();

    length = din1.read().data;
    computeS.write(1);
    DWAIT();
    lshift = (1 << din1.read().data);

    in1_off = din1.read().data;
    in1_sv = din1.read().data;
    in1_mul = din1.read().data;

    in2_off = din1.read().data;
    in2_sv = din1.read().data;
    in2_mul = din1.read().data;

    out1_off = din1.read().data;
    out1_sv = din1.read().data;
    out1_mul = din1.read().data;

    qa_max = din1.read().data;
    qa_min = din1.read().data;

    for (int i = 0; i < length; i++) {
      i1 = din1.read().data;
      i2 = din1.read().data;
      i1mem[0] = i1.range(7, 0);
      i1mem[1] = i1.range(15, 8);
      i1mem[2] = i1.range(23, 16);
      i1mem[3] = i1.range(31, 24);

      i2mem[0] = i2.range(7, 0);
      i2mem[1] = i2.range(15, 8);
      i2mem[2] = i2.range(23, 16);
      i2mem[3] = i2.range(31, 24);

      // cout << i1mem[0] << endl;
      // cout << i2mem[0] << endl;

      s_in1[0] = (i1mem[0] + in1_off) * lshift;
      s_in1[1] = (i1mem[1] + in1_off) * lshift;
      s_in1[2] = (i1mem[2] + in1_off) * lshift;
      s_in1[3] = (i1mem[3] + in1_off) * lshift;

      s_in2[0] = (i2mem[0] + in2_off) * lshift;
      s_in2[1] = (i2mem[1] + in2_off) * lshift;
      s_in2[2] = (i2mem[2] + in2_off) * lshift;
      s_in2[3] = (i2mem[3] + in2_off) * lshift;

      s_in1[0] = Quantised_Multiplier(s_in1[0], in1_mul, in1_sv);
      s_in1[1] = Quantised_Multiplier(s_in1[1], in1_mul, in1_sv);
      s_in1[2] = Quantised_Multiplier(s_in1[2], in1_mul, in1_sv);
      s_in1[3] = Quantised_Multiplier(s_in1[3], in1_mul, in1_sv);

      s_in2[0] = Quantised_Multiplier(s_in2[0], in2_mul, in2_sv);
      s_in2[1] = Quantised_Multiplier(s_in2[1], in2_mul, in2_sv);
      s_in2[2] = Quantised_Multiplier(s_in2[2], in2_mul, in2_sv);
      s_in2[3] = Quantised_Multiplier(s_in2[3], in2_mul, in2_sv);

      sum[0] = s_in1[0] + s_in2[0];
      sum[1] = s_in1[1] + s_in2[1];
      sum[2] = s_in1[2] + s_in2[2];
      sum[3] = s_in1[3] + s_in2[3];

      f_out[0] = Quantised_Multiplier(sum[0], out1_mul, out1_sv) + out1_off;
      f_out[1] = Quantised_Multiplier(sum[1], out1_mul, out1_sv) + out1_off;
      f_out[2] = Quantised_Multiplier(sum[2], out1_mul, out1_sv) + out1_off;
      f_out[3] = Quantised_Multiplier(sum[3], out1_mul, out1_sv) + out1_off;

      d.data =
          Clamp_Combine(f_out[0], f_out[1], f_out[2], f_out[3], qa_max, qa_min);

      if (i + 1 == length)
        d.tlast = true;
      else
        d.tlast = false;

      dout1.write(d);
    }
    DWAIT();
  }
}