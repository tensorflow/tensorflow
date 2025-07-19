# PJRT Plugins

This directory contains code related to PJRT plugins. These plugins enable
consumption of XLA's backends for multiple diverse hardware platforms (e.g.,
CPU, GPU, TPU).

## Directory Contents

*   **Plugin Registration Targets:** Build rules for specific hardware backends
    that can register the PJRT C API. These implementations are located in
    subdirectories such as `xla_cpu`, `xla_gpu`, and `xla_tpu`.
*   **Plugin Registration Mechanism:** Code that allows plugins to be registered
    and discovered by the XLA runtime. This mechanism relies on the
    `plugin_names.h` file for defining valid plugin names and the
    `static_registration.h` for the macro to register plugins.
*   **`GetXla...PjrtPlugin` functions**: Functions to get a single, statically
    linked plugin for a specific platform.

## Two Paths to Use PJRT Plugins

There are two primary ways to integrate a PJRT plugin into your application:

1.  **Dynamic Plugin Selection with `GetCApiPlugin`:**

    *   **Mechanism:**
        *   You provide the `GetCApiPlugin` function with a plugin name (defined
            in `plugin_names.h`).
        *   The XLA runtime dynamically loads and initializes the plugin
            corresponding to the provided name.
        *   The plugin must have been registered using a static registration
            mechanism (see `static_registration.h`).
    *   **Benefits:**
        *   **Flexibility:** Allows you to choose which plugin to use at
            runtime, without recompiling.
        *   **Modularity:** Enables you to build your application with multiple
            plugins and select them as needed.
        *   **Support for multiple plugins**: Allows to support multiple plugins
            in the same binary.
    *   **Requirements:**
        *   You need to know the valid plugin names (defined in
            `plugin_names.h`).
        *   You must link the static registration target of the desired plugin
            into your build.
    *   **Example (Conceptual):**

        ```c++
        const char* plugin_name = PJRT_PLUGIN_NAME_CPU;
        // Or another name from plugin_names.h

        absl::StatusOr<std::unique_ptr<PjrtClient>> client =
            GetCApiClient(plugin_name, {}, nullptr);
        if (!client.ok()) {
            // Handle error
        } else {
            // Use the client: client.value()
        }
        ```

2.  **Static Plugin Linking with `GetXla...PjrtPlugin`:**

    *   **Mechanism:**
        *   You call a specific function like `GetXlaCpuPjrtPlugin()` (or
            similar for GPU and TPU).
        *   This function returns a pointer to a statically linked PJRT plugin.
    *   **Benefits:**
        * **Explicit:** You can explicitly initialize the type of plugin that
            you intend to.
    *   **Requirements:**
        *   You must choose the correct `GetXla...PjrtPlugin` function for your
            target platform.
    *   **Example (Conceptual):**

        ```c++
        absl::StatusOr<std::unique_ptr<PjrtClient>> client =
            GetXlaCpuPjrtPlugin();
        if (!client.ok()) {
            // Handle error
        } else {
            // Use the client: client.value()
        }
        ```

In summary, this `plugin` directory provides the infrastructure for consuming
XLA with hardware-specific plugins. You can choose between a flexible but more
complex dynamic approach using `GetCApiPlugin` or a simpler but less flexible
static approach using `GetXla...PjrtPlugin`.
