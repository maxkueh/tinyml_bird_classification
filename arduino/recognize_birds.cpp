/* This app is based on the micro speech example created by the TensorFlow team.
==============================================================================*/

#include "recognize_birds.h"

#include <limits>

RecognizeBirds::RecognizeBirds(tflite::ErrorReporter* error_reporter,
                                     int32_t average_window_duration_ms,
                                     uint8_t detection_threshold,
                                     int32_t suppression_ms,
                                     int32_t minimum_count)
    : error_reporter_(error_reporter),
      average_window_duration_ms_(average_window_duration_ms),
      detection_threshold_(detection_threshold),
      suppression_ms_(suppression_ms),
      minimum_count_(minimum_count),
      previous_results_(error_reporter) {
  previous_top_label_ = "silence";
  previous_top_label_time_ = std::numeric_limits<int32_t>::min();
}

TfLiteStatus RecognizeBirds::ProcessLatestResults(
    const TfLiteTensor* latest_results, const int32_t current_time_ms, int32_t* average_scores) {
  if ((latest_results->dims->size != 2) ||
      (latest_results->dims->data[0] != 1) ||
      (latest_results->dims->data[1] != kCategoryCount)) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "The results for recognition should contain %d elements, but there are "
        "%d in an %d-dimensional shape",
        kCategoryCount, latest_results->dims->data[1],
        latest_results->dims->size);
    return kTfLiteError;
  }

  if (latest_results->type != kTfLiteInt8) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "The results for recognition should be int8_t elements, but are %d",
        latest_results->type);
    return kTfLiteError;
  }

  if ((!previous_results_.empty()) &&
      (current_time_ms < previous_results_.front().time_)) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Results must be fed in increasing time order, but received a "
        "timestamp of %d that was earlier than the previous one of %d",
        current_time_ms, previous_results_.front().time_);
    return kTfLiteError;
  }

  // Add the latest results to the head of the queue.
  previous_results_.push_back({current_time_ms, latest_results->data.int8});

  // Prune any earlier results that are too old for the averaging window.
  const int64_t time_limit = current_time_ms - average_window_duration_ms_;
  while ((!previous_results_.empty()) &&
         previous_results_.front().time_ < time_limit) {
    previous_results_.pop_front();
  }

  // If there are too few results, assume the result will be unreliable and
  // bail.
  const int64_t how_many_results = previous_results_.size();
  const int64_t earliest_time = previous_results_.front().time_;
  const int64_t samples_duration = current_time_ms - earliest_time;
  if ((how_many_results < minimum_count_) ||
      (samples_duration < (average_window_duration_ms_ / 4))) {
    return kTfLiteOk;
  }

  // Calculate the average score across all the results in the window.
  //int32_t average_scores[kCategoryCount];
  for (int offset = 0; offset < previous_results_.size(); ++offset) {
    PreviousResultsQueue::Result previous_result =
        previous_results_.from_front(offset);
    const int8_t* scores = previous_result.scores;
    for (int i = 0; i < kCategoryCount; ++i) {
      if (offset == 0) {
        average_scores[i] = scores[i] + 128;
      } else {
        average_scores[i] += scores[i] + 128;
      }
    }
  }
  for (int i = 0; i < kCategoryCount; ++i) {
    average_scores[i] /= how_many_results;
  }

  return kTfLiteOk;
}
