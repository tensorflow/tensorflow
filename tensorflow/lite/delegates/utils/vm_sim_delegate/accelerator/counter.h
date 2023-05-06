#ifndef __SYNTHESIS__
void ACCNAME::Read_Cycle_Counter() {
  while (1) {
    while (read_inputs) {
      read_cycles->value++;
      DWAIT();
    }
    wait();
  }
}

void ACCNAME::Process_Cycle_Counter() {
  wait();
  while (1) {
    while (out_check) {
      process_cycles->value++;
      if (w1S.read() == 10) idle1->value++;
      if (w2S.read() == 10) idle2->value++;
      if (w3S.read() == 10) idle3->value++;
      if (w4S.read() == 10) idle4->value++;
      DWAIT();
    }
    wait();
  }
}

void ACCNAME::Writer_Cycle_Counter() {
  wait();
  while (1) {
    while (out_check) {
      int w1 = w1S.read();
      int w2 = w2S.read();
      int w3 = w3S.read();
      int w4 = w4S.read();
      w1SS.write(w1);
      w2SS.write(w2);
      w3SS.write(w3);
      w4SS.write(w4);

      if (write1.read()) gemmw1->value++;
      if (write2.read()) gemmw2->value++;
      if (write3.read()) gemmw3->value++;
      if (write4.read()) gemmw4->value++;
      if (w1 == 3) gemm1->value++;
      if (w2 == 3) gemm2->value++;
      if (w3 == 3) gemm3->value++;
      if (w4 == 3) gemm4->value++;
      if (w1 == 9) wstall1->value++;
      if (w2 == 9) wstall2->value++;
      if (w3 == 9) wstall3->value++;
      if (w4 == 9) wstall4->value++;
      DWAIT();
    }
    wait();
  }
}
#endif
