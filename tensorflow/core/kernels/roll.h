// roll.h
#ifndef KERNEL_ROLL_H_
#define KERNEL_ROLL_H_

template <typename Device, typename T>
struct RollFunctor {
    void operator()(const Device& d, int N, int D, int* dim_size, const T* input, T* output,\
                    const int* shifts, const int* strides);
};

#endif //KERNEL_ROLL_H_
