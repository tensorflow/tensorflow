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
      if (w1S.read() == 10) idle->value++;
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
      w1SS.write(w1);
      if (write1.read()) gemmw->value++;
      if (w1 == 3) gemm->value++;
      if (w1 == 9) wstall->value++;
      DWAIT();
    }
    wait();
  }
}
