#Class tensorflow::TensorBuffer





##Member Summary

* [tensorflow::TensorBuffer::~TensorBuffer](#tensorflow_TensorBuffer_TensorBuffer)
* [virtual void* tensorflow::TensorBuffer::data](#virtual_void_tensorflow_TensorBuffer_data)
* [virtual size_t tensorflow::TensorBuffer::size](#virtual_size_t_tensorflow_TensorBuffer_size)
* [virtual TensorBuffer* tensorflow::TensorBuffer::root_buffer](#virtual_TensorBuffer_tensorflow_TensorBuffer_root_buffer)
* [virtual void tensorflow::TensorBuffer::FillAllocationDescription](#virtual_void_tensorflow_TensorBuffer_FillAllocationDescription)
* [T* tensorflow::TensorBuffer::base](#T_tensorflow_TensorBuffer_base)

##Member Details

#### tensorflow::TensorBuffer::~TensorBuffer() override {#tensorflow_TensorBuffer_TensorBuffer}





#### virtual void* tensorflow::TensorBuffer::data() const =0 {#virtual_void_tensorflow_TensorBuffer_data}





#### virtual size_t tensorflow::TensorBuffer::size() const =0 {#virtual_size_t_tensorflow_TensorBuffer_size}





#### virtual TensorBuffer* tensorflow::TensorBuffer::root_buffer()=0 {#virtual_TensorBuffer_tensorflow_TensorBuffer_root_buffer}





#### virtual void tensorflow::TensorBuffer::FillAllocationDescription(AllocationDescription *proto) const =0 {#virtual_void_tensorflow_TensorBuffer_FillAllocationDescription}





#### T* tensorflow::TensorBuffer::base() const {#T_tensorflow_TensorBuffer_base}




