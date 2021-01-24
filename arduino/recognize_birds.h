/* This app is based on the micro speech example created by the TensorFlow team.
==============================================================================*/

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_RECOGNIZE_COMMANDS_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_RECOGNIZE_COMMANDS_H_

#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "micro_features_micro_model_settings.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

// Partial implementation of std::dequeue, just providing the functionality
// that's needed to keep a record of previous neural network results over a
// short time period, so they can be averaged together to produce a more
// accurate overall prediction.
class PreviousResultsQueue {
 public:
  PreviousResultsQueue(tflite::ErrorReporter* error_reporter)
      : error_reporter_(error_reporter), front_index_(0), size_(0) {}

  // Data structure that holds an inference result, and the time when it
  // was recorded.
  struct Result {
    Result() : time_(0), scores() {}
    Result(int32_t time, int8_t* input_scores) : time_(time) {
      for (int i = 0; i < kCategoryCount; ++i) {
        scores[i] = input_scores[i];
      }
    }
    int32_t time_;
    int8_t scores[kCategoryCount];
  };

  int size() { return size_; }
  bool empty() { return size_ == 0; }
  Result& front() { return results_[front_index_]; }
  Result& back() {
    int back_index = front_index_ + (size_ - 1);
    if (back_index >= kMaxResults) {
      back_index -= kMaxResults;
    }
    return results_[back_index];
  }

  void push_back(const Result& entry) {
    if (size() >= kMaxResults) {
      TF_LITE_REPORT_ERROR(
          error_reporter_,
          "Couldn't push_back latest result, too many already!");
      return;
    }
    size_ += 1;
    back() = entry;
  }

  Result pop_front() {
    if (size() <= 0) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Couldn't pop_front result, none present!");
      return Result();
    }
    Result result = front();
    front_index_ += 1;
    if (front_index_ >= kMaxResults) {
      front_index_ = 0;
    }
    size_ -= 1;
    return result;
  }

  // Most of the functions are duplicates of dequeue containers, but this
  // is a helper that makes it easy to iterate through the contents of the
  // queue.
  Result& from_front(int offset) {
    if ((offset < 0) || (offset >= size_)) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Attempt to read beyond the end of the queue!");
      offset = size_ - 1;
    }
    int index = front_index_ + offset;
    if (index >= kMaxResults) {
      index -= kMaxResults;
    }
    return results_[index];
  }

 private:
  tflite::ErrorReporter* error_reporter_;
  static constexpr int kMaxResults = 50;
  Result results_[kMaxResults];

  int front_index_;
  int size_;
};

class RecognizeBirds {
 public:
  explicit RecognizeBirds(tflite::ErrorReporter* error_reporter,
                             int32_t average_window_duration_ms = 10000,
                             uint8_t detection_threshold = 150,
                             int32_t suppression_ms = 1500,
                             int32_t minimum_count = 3);

  // Call this with the results of running a model on sample data.
  TfLiteStatus ProcessLatestResults(const TfLiteTensor* latest_results,
                                    const int32_t current_time_ms,
                                    int32_t* average_scores);

 private:
  // Configuration
  tflite::ErrorReporter* error_reporter_;
  int32_t average_window_duration_ms_;
  uint8_t detection_threshold_;
  int32_t suppression_ms_;
  int32_t minimum_count_;

  // Working variables
  PreviousResultsQueue previous_results_;
  const char* previous_top_label_;
  int32_t previous_top_label_time_;
};

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_RECOGNIZE_COMMANDS_H_
