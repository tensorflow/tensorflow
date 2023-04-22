void ACCNAME::WSync1() {
  write1_1.write(1);
  write1_2.write(1);
  write1_3.write(1);
  write1_4.write(1);
  wait();
  while (true) {
    while (write1_1.read() || write1_2.read() || write1_3.read() ||
           write1_4.read())
      wait();
    write1_1.write(1);
    write1_2.write(1);
    write1_3.write(1);
    write1_4.write(1);
    arrange1.write(0);
    DWAIT();
  }
}

void ACCNAME::Arranger1() {
  DATA d;
  d.tlast = false;
  arrange1.write(0);
  wait();
  while (true) {
    while (!arrange1.read()) wait();
    d.data = r1[0];
    dout1.write(d);
    d.data = r1[4];
    dout1.write(d);
    d.data = r1[8];
    dout1.write(d);
    d.data = r1[12];
    dout1.write(d);
    write1_1.write(0);
    DWAIT();
    while (!write1_1.read()) wait();
  }
}

void ACCNAME::Arranger2() {
  DATA d;
  d.tlast = false;
  wait();
  while (true) {
    while (!arrange1.read()) wait();
    d.data = r1[1];
    dout2.write(d);
    d.data = r1[5];
    dout2.write(d);
    d.data = r1[9];
    dout2.write(d);
    d.data = r1[13];
    dout2.write(d);
    write1_2.write(0);
    DWAIT();
    while (!write1_2.read()) wait();
  }
}

void ACCNAME::Arranger3() {
  DATA d;
  d.tlast = false;
  wait();
  while (true) {
    while (!arrange1.read()) wait();
    d.data = r1[2];
    dout3.write(d);
    d.data = r1[6];
    dout3.write(d);
    d.data = r1[10];
    dout3.write(d);
    d.data = r1[14];
    dout3.write(d);
    write1_3.write(0);
    DWAIT();
    while (!write1_3.read()) wait();
  }
}

void ACCNAME::Arranger4() {
  DATA d;
  d.tlast = false;
  wait();
  while (true) {
    while (!arrange1.read()) wait();
    d.data = r1[3];
    dout4.write(d);
    d.data = r1[7];
    dout4.write(d);
    d.data = r1[11];
    dout4.write(d);
    d.data = r1[15];
    dout4.write(d);
    write1_4.write(0);
    DWAIT();
    while (!write1_4.read()) wait();
  }
}
