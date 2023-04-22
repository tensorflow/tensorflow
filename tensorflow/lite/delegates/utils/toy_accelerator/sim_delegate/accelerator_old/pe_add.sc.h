
void ACCNAME::PE_Add() {
  ACC_DTYPE<32> i1;
  ACC_DTYPE<32> i2;
  ACC_DTYPE<32> sum[4];
  wait();

  while (1) {
    while (!computeX && !computeY) wait();

    int len = 0;
    if (computeX) {
      while (writeX) wait();
      len = lenX;
    } else {
      while (writeY) wait();
      len = lenY;
    }

    for (int i = 0; i < len; i++) {
      if (computeX) {
        i1 = A1[i];
        i2 = B1[i];
      } else {
        i1 = A2[i];
        i2 = B2[i];
      }

      sum[0] = i1.range(7, 0) + i2.range(7, 0);
      sum[1] = i1.range(15, 8) + i2.range(15, 8);
      sum[2] = i1.range(23, 16) + i2.range(23, 16);
      sum[3] = i1.range(31, 24) + i2.range(31, 24);

      if (computeX) {
        C1[i * 4 + 0] = sum[0];
        C1[i * 4 + 1] = sum[1];
        C1[i * 4 + 2] = sum[2];
        C1[i * 4 + 3] = sum[3];
      } else {
        C2[i * 4 + 0] = sum[0];
        C2[i * 4 + 1] = sum[1];
        C2[i * 4 + 2] = sum[2];
        C2[i * 4 + 3] = sum[3];
      }
    }

    if (computeX) {
      computeX.write(0);
      writeX.write(1);
    }
    if (computeY) {
      computeY.write(0);
      writeY.write(1);
    }
  }
}