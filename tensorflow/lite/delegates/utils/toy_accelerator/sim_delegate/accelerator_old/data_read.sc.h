
void ACCNAME::Data_Read() {
  computeX.write(0);
  computeY.write(0);
  wait();
  while (1) {
    while (!readX && !readY) wait();

    int len = 0;
    if (readX)
      len = lenX;
    else
      len = lenY;

    for (int i = 0; i < len; i++) {
      if (readX)
        A1[i] = din1.read().data;
      else if (readY)
        A2[i] = din1.read().data;
    }

    for (int i = 0; i < len; i++) {
      if (readX)
        B1[i] = din1.read().data;
      else if (readY)
        B2[i] = din1.read().data;
    }

    if (readX) {
      computeX.write(1);
      readX.write(0);
    }
    if (readY) {
      computeY.write(1);
      readY.write(0);
    }
    DWAIT();
  }
}