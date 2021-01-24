/* This app is based on the micro speech example created by the TensorFlow team.
==============================================================================*/

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_FEATURE_PROVIDER_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_FEATURE_PROVIDER_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

class FeatureProvider {
 public:
  // Create the provider, and bind it to an area of memory. This memory should
  // remain accessible for the lifetime of the provider object, since subsequent
  // calls will fill it with feature data. The provider does no memory
  // management of this data.
  FeatureProvider(int feature_size, int8_t* feature_data);
  ~FeatureProvider();

  // Fills the feature data with information from audio inputs, and returns how
  // many feature slices were updated.
  TfLiteStatus PopulateFeatureData(tflite::ErrorReporter* error_reporter,
                                   int32_t last_time_in_ms, int32_t time_in_ms,
                                   int* how_many_new_slices);

 private:
  int feature_size_;
  int8_t* feature_data_;
  // Make sure we don't try to use cached information if this is the first call
  // into the provider.
  bool is_first_run_;
};

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_FEATURE_PROVIDER_H_
