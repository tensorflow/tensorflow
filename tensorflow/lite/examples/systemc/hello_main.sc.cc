// Created by Nicolas Agostini

#include <systemc/systemc.h>
//#include "tensorflow/lite/examples/systemc/logging.h"

// Simple Hello World module
SC_MODULE (hello_world) {
  SC_CTOR (hello_world) {
  }
  void say_hello() {
    cout << "Hello World.\n";
  }
};

int sc_main(int argc, char* argv[]) {
  hello_world hello("HELLO");
  hello.say_hello();
  return (0);
}
