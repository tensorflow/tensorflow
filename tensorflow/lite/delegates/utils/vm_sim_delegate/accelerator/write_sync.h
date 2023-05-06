void ACCNAME::WSync1() {
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

void ACCNAME::WSync2() {
  wait();
  while (true) {
    while (write2_1.read() || write2_2.read() || write2_3.read() ||
           write2_4.read())
      wait();
    write2_1.write(1);
    write2_2.write(1);
    write2_3.write(1);
    write2_4.write(1);
    arrange2.write(0);
    DWAIT();
  }
}

void ACCNAME::WSync3() {
  wait();
  while (true) {
    while (write3_1.read() || write3_2.read() || write3_3.read() ||
           write3_4.read())
      wait();
    write3_1.write(1);
    write3_2.write(1);
    write3_3.write(1);
    write3_4.write(1);
    arrange3.write(0);
    DWAIT();
  }
}

void ACCNAME::WSync4() {
  wait();
  while (true) {
    while (write4_1.read() || write4_2.read() || write4_3.read() ||
           write4_4.read())
      wait();
    write4_1.write(1);
    write4_2.write(1);
    write4_3.write(1);
    write4_4.write(1);
    arrange4.write(0);
    DWAIT();
  }
}

void ACCNAME::Arranger1() {
  DATA d;
  d.tlast = false;
  wait();
  while (true) {
    while (!arrange1.read())
      wait();
    d.data.range(7, 0) = r1[0];
    d.data.range(15, 8) = r1[4];
    d.data.range(23, 16) = r1[8];
    d.data.range(31, 24) = r1[12];
    dout1.write(d);
    DWAIT();
    write1_1.write(0);

    while (!arrange2.read())
      wait();
    d.data.range(7, 0) = r2[0];
    d.data.range(15, 8) = r2[4];
    d.data.range(23, 16) = r2[8];
    d.data.range(31, 24) = r2[12];
    dout1.write(d);
    DWAIT();
    write2_1.write(0);

    while (!arrange3.read())
      wait();
    d.data.range(7, 0) = r3[0];
    d.data.range(15, 8) = r3[4];
    d.data.range(23, 16) = r3[8];
    d.data.range(31, 24) = r3[12];
    dout1.write(d);
    DWAIT();
    write3_1.write(0);

    while (!arrange4.read())
      wait();
    d.data.range(7, 0) = r4[0];
    d.data.range(15, 8) = r4[4];
    d.data.range(23, 16) = r4[8];
    d.data.range(31, 24) = r4[12];
    dout1.write(d);
    DWAIT();
    write4_1.write(0);
  }
}

void ACCNAME::Arranger2() {
  DATA d;
  d.tlast = false;
  wait();
  while (true) {
    while (!arrange1.read())
      wait();
    d.data.range(7, 0) = r1[1];
    d.data.range(15, 8) = r1[5];
    d.data.range(23, 16) = r1[9];
    d.data.range(31, 24) = r1[13];
    dout2.write(d);
    DWAIT();
    write1_2.write(0);

    while (!arrange2.read())
      wait();
    d.data.range(7, 0) = r2[1];
    d.data.range(15, 8) = r2[5];
    d.data.range(23, 16) = r2[9];
    d.data.range(31, 24) = r2[13];
    dout2.write(d);
    DWAIT();
    write2_2.write(0);

    while (!arrange3.read())
      wait();
    d.data.range(7, 0) = r3[1];
    d.data.range(15, 8) = r3[5];
    d.data.range(23, 16) = r3[9];
    d.data.range(31, 24) = r3[13];
    dout2.write(d);
    DWAIT();
    write3_2.write(0);

    while (!arrange4.read())
      wait();
    d.data.range(7, 0) = r4[1];
    d.data.range(15, 8) = r4[5];
    d.data.range(23, 16) = r4[9];
    d.data.range(31, 24) = r4[13];
    dout2.write(d);
    DWAIT();
    write4_2.write(0);
  }
}

void ACCNAME::Arranger3() {
  DATA d;
  d.tlast = false;
  wait();
  while (true) {
    while (!arrange1.read())
      wait();
    d.data.range(7, 0) = r1[2];
    d.data.range(15, 8) = r1[6];
    d.data.range(23, 16) = r1[10];
    d.data.range(31, 24) = r1[14];
    dout3.write(d);
    DWAIT();
    write1_3.write(0);

    while (!arrange2.read())
      wait();
    d.data.range(7, 0) = r2[2];
    d.data.range(15, 8) = r2[6];
    d.data.range(23, 16) = r2[10];
    d.data.range(31, 24) = r2[14];
    dout3.write(d);
    DWAIT();
    write2_3.write(0);

    while (!arrange3.read())
      wait();
    d.data.range(7, 0) = r3[2];
    d.data.range(15, 8) = r3[6];
    d.data.range(23, 16) = r3[10];
    d.data.range(31, 24) = r3[14];
    dout3.write(d);
    DWAIT();
    write3_3.write(0);

    while (!arrange4.read())
      wait();
    d.data.range(7, 0) = r4[2];
    d.data.range(15, 8) = r4[6];
    d.data.range(23, 16) = r4[10];
    d.data.range(31, 24) = r4[14];
    dout3.write(d);
    DWAIT();
    write4_3.write(0);
  }
}

void ACCNAME::Arranger4() {
  DATA d;
  d.tlast = false;
  wait();
  while (true) {
    while (!arrange1.read())
      wait();
    d.data.range(7, 0) = r1[3];
    d.data.range(15, 8) = r1[7];
    d.data.range(23, 16) = r1[11];
    d.data.range(31, 24) = r1[15];
    dout4.write(d);
    DWAIT();
    write1_4.write(0);

    while (!arrange2.read())
      wait();
    d.data.range(7, 0) = r2[3];
    d.data.range(15, 8) = r2[7];
    d.data.range(23, 16) = r2[11];
    d.data.range(31, 24) = r2[15];
    dout4.write(d);
    DWAIT();
    write2_4.write(0);

    while (!arrange3.read())
      wait();
    d.data.range(7, 0) = r3[3];
    d.data.range(15, 8) = r3[7];
    d.data.range(23, 16) = r3[11];
    d.data.range(31, 24) = r3[15];
    dout4.write(d);
    DWAIT();
    write3_4.write(0);

    while (!arrange4.read())
      wait();
    d.data.range(7, 0) = r4[3];
    d.data.range(15, 8) = r4[7];
    d.data.range(23, 16) = r4[11];
    d.data.range(31, 24) = r4[15];
    dout4.write(d);
    DWAIT();
    write4_4.write(0);
  }
}