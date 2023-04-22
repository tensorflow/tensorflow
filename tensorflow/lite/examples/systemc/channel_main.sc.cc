// Created by Nicolas Agostini

#include <systemc/systemc.h>
#include "tensorflow/lite/examples/systemc/channel/stack.sc.h"
#include "tensorflow/lite/examples/systemc/channel/producer.sc.h"
#include "tensorflow/lite/examples/systemc/channel/consumer.sc.h"


int sc_main(int argc, char* argv[]) {

  sc_clock ClkSlow("ClkSlow", 100.0, SC_NS);
  sc_clock ClkFast("ClkFast", 50.0, SC_NS);


  stack Stack1("S1");

  producer P1("P1");
  P1.out(Stack1);
  P1.Clock(ClkSlow);

  // Consumer consumes faster then producer
  consumer C1("C1");
  C1.in(Stack1);
  C1.Clock(ClkFast);

  sc_start(5000, SC_NS);

  return 0;
}
