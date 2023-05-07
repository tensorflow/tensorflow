#ifndef __SYNTHESIS__
int ACCNAME::Quantised_Multiplier(int x, int qm, int shift) {
  int nshift = shift;
  int total_shift = 31 - shift;
  sc_int<64> x_64 = x;
  sc_int<64> quantized_multiplier_64(qm);
  sc_int<64> one = 1;
  sc_int<64> round = one << (total_shift - 1);
  sc_int<64> result = x_64 * quantized_multiplier_64 + round;
  result = result >> total_shift;
  int nresult = result;
  if (result > MAX32) result = MAX32;
  if (result < MIN32) result = MIN32;
  sc_int<32> result_32 = result;
  return result_32;
}
#else
int ACCNAME::Quantised_Multiplier(int x, int qm, int shift) {
  sc_int<64> val = x * pl;
  if (val > MAX32) val = MAX32;
  if (val < MIN32) val = MIN32;
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
#endif

void ACCNAME::store() {
  storing.write(false);
  unsigned int out_write[4];
  sc_int<8> out_store[4][4];
  sc_int<32> temp_out;

#pragma HLS array_partition variable = out_write complete dim = 0
#pragma HLS array_partition variable = out_store complete dim = 0

  wait();
  while (true) {
    while (!storing.read())
      wait();

    unsigned int moff = m_off.read() / 4;
    unsigned int noff = n_off.read();
    unsigned int mem_idx = store_doffset;
    unsigned int dstride = store_dstride;

    for (int n = 0; n < 4; n++) {
#pragma HLS pipeline II = 1
      for (int m = 0; m < 4; m++) {
#pragma HLS unroll factor = 4
        int cval = out_mem[m][n];
        int value = Quantised_Multiplier(cval, crf_val, crx_val);
        sc_int<32> svalue = value + ra_val;
        if (svalue > MAX8) svalue = MAX8;
        else if (svalue < MIN8) svalue = MIN8;
        out_store[m][n] = svalue.range(7, 0);
      }
      temp_out.range(7, 0) = out_store[0][n];
      temp_out.range(15, 8) = out_store[1][n];
      temp_out.range(23, 16) = out_store[2][n];
      temp_out.range(31, 24) = out_store[3][n];
      out_write[n] = temp_out;
      unsigned int ddr_idx = mem_idx + moff + (noff + n) * dstride;
      out_port->write(output_addr + ddr_idx, (unsigned int *)&out_write[n]);
    }

    storing.write(false);
    wait();
  }
}