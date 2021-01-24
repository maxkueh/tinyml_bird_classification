/* This app is based on the micro speech example created by the TensorFlow team.
==============================================================================*/

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_MODEL_SETTINGS_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_MODEL_SETTINGS_H_

constexpr int kMaxAudioSampleSize = 512;
constexpr int kAudioSampleFrequency = 16000;

// Variables depending on the time window processed
constexpr int kFeatureSliceSize = 40;
constexpr int kFeatureSliceCount = 499;
constexpr int kFeatureElementCount = (kFeatureSliceSize * kFeatureSliceCount);
constexpr int kFeatureSliceStrideMs = 20;
constexpr int kFeatureSliceDurationMs = 30;

// Variables for the model's output categories.
constexpr int kSilenceIndex = 0;
constexpr int kUnknownIndex = 1;

constexpr int kCategoryCount = 6;
extern const char* kCategoryLabels[kCategoryCount];

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_MODEL_SETTINGS_H_
