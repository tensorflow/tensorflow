int ACCNAME::SHR(int value, int shift) { return value >> shift; }

sc_int<32> ACCNAME::mul_s8(sc_int<8> a, sc_int<8> b) {
  sc_int<32> c;
#pragma HLS RESOURCE variable = c core = Mul
  c = a * b;
  return c;
}

void ACCNAME::Output_Handler() {
  bool ready = false;
  bool resetted = true;
  DATA last = {5000, 1};
  wait();
  while (1) {
    while (out_check.read() && !ready && resetted) {
      bool w1 = w1S.read() == 10;
      bool w2 = w2S.read() == 10;
      bool w3 = w3S.read() == 10;
      bool w4 = w4S.read() == 10;

      bool wr1 = !write1.read();
      bool wr2 = !write2.read();
      bool wr3 = !write3.read();
      bool wr4 = !write4.read();

      bool block_done = !schedule.read();

      ready = block_done && w1 && w2 && w3 && w4 && wr1 && wr2 && wr3 && wr4;

      if (ready) {
        dout1.write(last);
        dout2.write(last);
        dout3.write(last);
        dout4.write(last);
        out_check.write(0);
        resetted = false;
      }
      wait();
      DWAIT(4);
    }

    if (!out_check.read()) {
      resetted = true;
      ready = false;
    }
    wait();
    DWAIT();
  }
}
