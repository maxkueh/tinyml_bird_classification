/* This app is based on the micro speech example created by the TensorFlow team.
==============================================================================*/


//Include Library for BLE functionality
#include <ArduinoBLE.h>

#include <TensorFlowLite.h>
#include "main_functions.h"
#include "audio_provider.h"
#include "feature_provider.h"
#include "micro_features_micro_model_settings.h"
#include "micro_features_model.h"
#include "recognize_birds.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

//BLE Service and corresponding characteristics
BLEService birdService("e50bf554-fdd9-4d9e-b350-86493ab13280");
BLEIntCharacteristic silenceChar("a58975f8-0fe3-40e5-af04-fedf80cba2a7", BLERead | BLENotify);
BLEIntCharacteristic unknownChar("4ae2b158-458d-48db-8c21-e3f9fc00958f", BLERead | BLENotify);
BLEIntCharacteristic parusmajorChar("408e957f-43d5-4486-9899-9de940b8de93", BLERead | BLENotify);
BLEIntCharacteristic turdusmerulaChar("74fdd9b2-aa8b-4418-b6ea-e8ba03e701df", BLERead | BLENotify);
BLEIntCharacteristic passerdomesticusChar("d7417ec3-dcf3-4baa-8ca8-c92670d555c1", BLERead | BLENotify);
BLEIntCharacteristic phylloscopuscollybitaChar("ce38bf27-669b-43d4-9a00-4d6545fac12a", BLERead | BLENotify);

namespace {
  
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;
RecognizeBirds* recognizer = nullptr;
int32_t previous_time = 0;

constexpr int kTensorArenaSize = 60 * 1024; //needed size for 10 Seconds of audio
uint8_t tensor_arena[kTensorArenaSize];
int8_t feature_buffer[kFeatureElementCount];
int8_t* model_input_buffer = nullptr;
}  // namespace

void setup() {
   
  if (!BLE.begin()) {
    Serial.println("starting BLE failed!");
    while (1);
  }
  
  //Set bird service as advertised service
  BLE.setLocalName("Bird Classifier");
  BLE.setAdvertisedService(birdService);

  //Setup BLE Characteristics
  birdService.addCharacteristic(silenceChar);
  birdService.addCharacteristic(unknownChar);
  birdService.addCharacteristic(parusmajorChar);
  birdService.addCharacteristic(turdusmerulaChar);
  birdService.addCharacteristic(passerdomesticusChar);
  birdService.addCharacteristic(phylloscopuscollybitaChar);
  
  BLE.addService(birdService);
  BLE.advertise();
  Serial.println("Bird Classifier active, waiting for connections...");

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  static tflite::MicroMutableOpResolver<4> micro_op_resolver(error_reporter);
  if (micro_op_resolver.AddDepthwiseConv2D() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddReshape() != kTfLiteOk) {
    return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] !=
       (kFeatureSliceCount * kFeatureSliceSize)) ||
      (model_input->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad input tensor parameters in model");
    return;
  }
  model_input_buffer = model_input->data.int8;

  // Prepare to access the audio spectrograms from the microphone
  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 feature_buffer);
  feature_provider = &static_feature_provider;

  static RecognizeBirds static_recognizer(error_reporter);
  recognizer = &static_recognizer;

  previous_time = 0;
}

void loop() {
  Serial.println("not connected");

  //Try to connect via BLE
  BLEDevice central = BLE.central();  

  //While connected, make bird classifications
  if (central) {
    Serial.print("Connected to central: ");
    Serial.println(central.address());
    
    while (central.connected()) {
  
      // Fetch the spectrogram for the current time.
      const int32_t current_time = LatestAudioTimestamp();
      int how_many_new_slices = 0;
      TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
          error_reporter, previous_time, current_time, &how_many_new_slices);
      if (feature_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "Feature generation failed");
        return;
      }
      previous_time = current_time;
      // If no new audio samples have been received since last time, don't bother
      // running the network model.
      if (how_many_new_slices == 0) {
        return;
      }
    
      // Copy feature buffer to input tensor
      for (int i = 0; i < kFeatureElementCount; i++) {
        model_input_buffer[i] = feature_buffer[i];
      }
    
      // Run the model on the spectrogram input and make sure it succeeds.
      TfLiteStatus invoke_status = interpreter->Invoke();
      if (invoke_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
        return;
      }
    
      // Obtain a pointer to the output tensor
      TfLiteTensor* output = interpreter->output(0);

      // Get average scores
      int32_t average_scores[kCategoryCount];
      TfLiteStatus process_status = recognizer->ProcessLatestResults(
          output, current_time, &*average_scores);      
      
      //print scores to serial monitor
      for (int i = 0; i < kCategoryCount; ++i) {
        TF_LITE_REPORT_ERROR(error_reporter,
                               "Average score in loop(): %d.",
                               average_scores[i]);
      }    
    
      if (process_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter,
                             "RecognizeBirds::ProcessLatestResults() failed");
        return;
      }

      //Update BLE Characteristics with each bird's average score over the last 10 seconds
      silenceChar.writeValue(average_scores[0]);
      unknownChar.writeValue(average_scores[1]);
      parusmajorChar.writeValue(average_scores[2]);
      turdusmerulaChar.writeValue(average_scores[3]);
      passerdomesticusChar.writeValue(average_scores[4]);
      phylloscopuscollybitaChar.writeValue(average_scores[5]);
    }
    
    Serial.print("Disconnected from central: ");
    Serial.println(central.address());
  }
}
