
#include "tensorflow/lite/micro/micro_error_reporter.h"

int a = 45;
bool init_to_false = false;
bool init_to_true = true;
int* init_to_nullptr = nullptr;

int main(int argc, char** argv) {
  MicroPrintf("a: %d", a);
  MicroPrintf("init_to_false: %d", init_to_false);
  if (init_to_false == false) {
    MicroPrintf("Was initialized to false");
  } else {
    MicroPrintf("Was not initialized to false");
  }

  init_to_false = true;
  MicroPrintf("init_to_false: %d", init_to_false);
  MicroPrintf("init_to_true: %d", init_to_true);
  MicroPrintf("init_to_nullptr: %d", init_to_nullptr);
}

