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
  DATA d1 = {0, 0};
  DATA d2 = {0, 0};
  DATA d3 = {0, 0};
  DATA d4 = {0, 0};
  send_output.write(0);
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

      // ready = block_done && w1 && w2 && w3 && w4 && wr1 && wr2 && wr3 && wr4;
      ready = block_done && w1 && wr1;

      if (send_output.read()) {
        int i = 0;
        for (; i < out_int8_lenr; i += 4) {
          d1.data = dst[i + 0];
          d2.data = dst[i + 1];
          d3.data = dst[i + 2];
          d4.data = dst[i + 3];
          dout1.write(d1);
          dout2.write(d2);
          dout3.write(d3);
          dout4.write(d4);
          dst[i + 0] = 0;
          dst[i + 1] = 0;
          dst[i + 2] = 0;
          dst[i + 3] = 0;
        }

        for (; i < out_int8_len; i++) {
          d1.data = dst[i];
          dst[i + 0] = 0;
          dout1.write(d1);
        }
        send_output.write(0);
        ready = true;
        wait();
      }

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
