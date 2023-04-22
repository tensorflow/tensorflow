
void ACCNAME::Control() {
  readX.write(0);
  readY.write(0);
  writeX.write(0);
  writeY.write(0);

  bool X = true;
  wait();
  while (1) {
    ACC_DTYPE<32> length = din1.read().data;
    // sc_pause();
    // wait();

    if (X) {
      while (computeX) wait();
      readX.write(1);
      lenX.write(length);
      DWAIT();
      while (readX) wait();
      X = false;
      DWAIT();

    } else {
      while (computeY) wait();
      readY.write(1);
      lenY.write(length);
      DWAIT();
      while (readY) wait();
      X = true;
      DWAIT();
    }

    DWAIT();
  }
}