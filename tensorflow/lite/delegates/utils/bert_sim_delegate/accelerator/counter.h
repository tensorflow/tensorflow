#ifndef __SYNTHESIS__
void ACCNAME::counter() {
  while (1) {
    per_batch_cycles->value++;
    DWAIT();
  }
}
#endif