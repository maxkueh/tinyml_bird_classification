/* This app is based on the micro speech example created by the TensorFlow team.
==============================================================================*/

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MAIN_FUNCTIONS_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MAIN_FUNCTIONS_H_

// Expose a C friendly interface for main functions.
#ifdef __cplusplus
extern "C" {
#endif

void setup();

void loop();

#ifdef __cplusplus
}
#endif

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MAIN_FUNCTIONS_H_
