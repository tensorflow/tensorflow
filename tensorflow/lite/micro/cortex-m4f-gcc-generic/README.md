# Generic Cortex-M4F customizations
The customization requires a definition where the debug log goes to. The purpose of the generic Cortex-M4F target is to generate a TFLu library file for use in application projects outside of this repo. As the chip HAL and the board specific layer are only defined in the application project, the TFLu library cannot write the debug log anywhere. Instead, we allow the application layer to register a callback function for writing the TFLu kernel debug log.

# Usage
The application layer must implement and register the callback before calling the network in a way similar to

    void debug_log_printf(const char* s)
    {
        printf(s);
    }

    int main(void)
    {
        // Register callback for printing debug log
        DebugLog_register_callback(debug_log_printf);
        
        // now call the network
        TfLiteStatus invoke_status = interpreter->Invoke();
    }
