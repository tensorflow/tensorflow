# Audio "frontend" library for feature generation

A feature generation library (also called frontend) that receives raw audio
input, and produces filter banks (a vector of values).

The raw audio input is expected to be 16-bit PCM features, with a configurable
sample rate. More specifically the audio signal goes through a pre-emphasis
filter (optionally); then gets sliced into (potentially overlapping) frames and
a window function is applied to each frame; afterwards, we do a Fourier
transform on each frame (or more specifically a Short-Time Fourier Transform)
and calculate the power spectrum; and subsequently compute the filter banks.

By default the library is configured with a set of defaults to perform the
different processing tasks. This takes place with the frontend_util.c function:

```c++
void FrontendFillConfigWithDefaults(struct FrontendConfig* config)
```

A single invocation looks like:

```c++
struct FrontendConfig frontend_config;
FrontendFillConfigWithDefaults(&frontend_config);
int sample_rate = 16000;
FrontendPopulateState(&frontend_config, &frontend_state, sample_rate);
int16_t* audio_data = ;  // PCM audio samples at 16KHz.
size_t audio_size = ;  // Number of audio samples.
size_t num_samples_read;  // How many samples were processed.
struct FrontendOutput output =
    FrontendProcessSamples(
        &frontend_state, audio_data, audio_size, &num_samples_read);
for (i = 0; i < output.size; ++i) {
  printf("%d ", output.values[i]);  // Print the feature vector.
}
```

Something to note in the above example is that the frontend consumes as many
samples needed from the audio data to produce a single feature vector (according
to the frontend configuration). If not enough samples were available to generate
a feature vector, the returned size will be 0 and the values pointer will be
`NULL`.

An example of how to use the frontend is provided in frontend_main.cc and its
binary frontend_main. This example, expects a path to a file containing `int16`
PCM features at a sample rate of 16KHz, and upon execution will printing out
the coefficients according to the frontend default configuration.

## Extra features
Extra features of this frontend library include a noise reduction module, as
well as a gain control module.

**Noise cancellation**. Removes stationary noise from each channel of the signal
using a low pass filter.

**Gain control**. A novel automatic gain control based dynamic compression to
replace the widely used static (such as log or root) compression. Disabled
by default.

## Memory map
The binary frontend_memmap_main shows a sample usage of how to avoid all the
initialization code in your application, by first running
"frontend_generate_memmap" to create a header/source file that uses a baked in
frontend state. This command could be automated as part of your build process,
or you can just use the output directly.
