# Microcontrollers

## Overview

Microcontrollers are compact integrated circuits with very limited resources. Currently, they only perform simple functions.

With the onset of TensorFlow Lite, hence smaller binary sizes, these devices will be able to support machine learning applications, opening the industry up to a myriad of use cases.

## Getting started

Note: This is an experimental release aimed at microcontrollers and other devices with only kilobytes of memory. It doesn't require any operating system support, any standard C or C++ libraries, or dynamic memory allocation, so it's designed to be portable even to 'bare metal' systems.

One of the challenges of embedded software development is that there are a lot of different architectures, devices, operating systems, and build systems. We aim to support as many of the popular combinations as we can and make it as easy as possible to add support for others.

Read more about [how to get started](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro#getting-started).

## Goals

The design goals are to make the framework readable, easy to modify, well-tested, easy to integrate, and compatible (e.g. consistent file schema, interpreter, API, kernel interface).

Read more about [goals and tradeoffs](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro#goals).
