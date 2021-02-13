# TinyML app for bird classification via audio

A TinyML app that is able to classify birds based on their song. It was developed for the Arduino Nano 33 BLE Sense and is complemented by a [webapp](https://lucaslar.github.io/tinyml-bird-detection-bluetooth/).

To get the application running on your Arduino Nano 33 BLE Sense, follow these steps:

 1. Clone this repository
 2. Start the Arduino IDE and plug in your Arduino Nano
 3. Install "Arduino mbed-enabled Boards" via the Boards Manager
 4. Install the libraries "ArduinoBLE" (version 1.1.3) and "Arduino\_TensorFlowLite'" (version 2.4.0) via the Library Manager
 5. Select "Arduino Nano 33 BLE" as the used board
 6. Open "bird\_classifier\_application.ino" located in the cloned repository in the Arduino IDE
 7. Upload the application via the upload-button
