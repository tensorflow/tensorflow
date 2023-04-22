// Created by Nicolas Agostini

#ifndef STACK_IF_H
#define STACK_IF_H

#include <systemc/systemc.h>

class stack_write_if: virtual public sc_interface {
  public:
    /// Write a character
    virtual bool nb_write(char)  = 0;

    /// Empty the stack
    virtual void reset() = 0;
};


class stack_read_if: virtual public sc_interface {
  public:
    /// Read a character
    virtual bool nb_read(char&)  = 0;
};

#endif // STACK_IF_H
