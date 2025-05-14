#include <WiFi.h>
#include <ArduinoJson.h>
#include <HTTPClient.h>

// Replace with your WiFi credentials
const char* ssid = "TECNO-TR109-CDBB";
const char* password = "30585069";

// Flask API URL (change the URL to match your Flask server's address)
const String apiURL = "https://solid-yodel-jj7rgq4qrrgphqjxq-5000.app.github.dev/latest_prediction";

void setup() {
  // Start serial communication
  Serial.begin(115200);

  // Connect to Wi-Fi
  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  
  // Wait until the Wi-Fi is connected
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");

  // Wait a bit for stability
  delay(1000);
  
  // Call the Flask API to get the latest prediction
  getLatestPrediction();
}

void loop() {
  // Nothing to do in the loop for now
}

// Function to make the HTTP GET request and display the result
void getLatestPrediction() {
  HTTPClient http;
  
  // Make the GET request
  http.begin(apiURL); // Specify the URL of your Flask API
  int httpCode = http.GET(); // Send the GET request

  // If the request is successful
  if (httpCode > 0) {
    String payload = http.getString();  // Get the response payload

    Serial.println("Response received:");
    Serial.println(payload);  // Print the JSON response

    // Parse the JSON response
    DynamicJsonDocument doc(1024);
    deserializeJson(doc, payload);

    // Extract and print the data from JSON
    const char* predicted_flour = doc["predicted_flour"];
    const char* timestamp = doc["timestamp"];

    Serial.print("Predicted Flour: ");
    Serial.println(predicted_flour);
    Serial.print("Timestamp: ");
    Serial.println(timestamp);
  } else {
    Serial.println("Error on HTTP request");
  }
  
  http.end();  // Close the HTTP connection
}
