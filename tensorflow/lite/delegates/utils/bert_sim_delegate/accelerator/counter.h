#include "acc.h"

void ACCNAME::counter() {
  while (1) {
      per_batch_cycles->value++;
      DWAIT();
  }
}
