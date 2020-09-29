# Generic Cortex-Mx customizations
The customization requires a definition where the debug log goes to. The purpose of the generic Cortex-Mx target is to generate a TFLM library file for use in application projects outside of this repo. As the chip HAL and the board specific layer are only defined in the application project, the TFLM library cannot write the debug log anywhere. Instead, we allow the application layer to register a callback function for writing the TFLM kernel debug log.

# Usage
See debug_log_callback.h
