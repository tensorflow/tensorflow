// Created by Nicolas Agostini

#ifndef STACK_H
#define STACK_H

#include <systemc/systemc.h>
#include "stack_if.sc.h"

/// Implements the virtual functions in the stack interface
class stack : public sc_module, public stack_write_if, public stack_read_if {
  private:
    char data[20];
    int top;

  public:
    stack(sc_module_name nm) : sc_module(nm) , top(0) {
    }
    
    bool nb_write(char c) {
      if (top < 20) {
        data[top++] = c;
        return true;
      }
      return false;
    }

    void reset() {
      top = 0;
    }

    bool nb_read(char& c) {
      if (top > 0) {
        c = data[--top];
        return true;
      }
      return false;
    }

    void register_port(sc_port_base & port_,
        const char * if_typename_) {
      std::cout << "binding    " << port_.name() << " to "
        << "interface: " << if_typename_ << std::endl;
    }
};

#endif // STACK_H
