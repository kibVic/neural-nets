#include <WiFi.h>
#include <ESP32WebServer.h>
#include <Camera.h>

const char* ssid = "your_wifi_ssid";
const char* password = "your_wifi_password";

ESP32WebServer server(80);

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  // Wait for WiFi connection
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("Connected to WiFi!");

  // Initialize camera
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL;
  config.ledc_timer = LEDC_TIMER;
  config.pin_d0 = 5;
  // Add other pins...
  // Initialize camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.println("Camera initialization failed");
    return;
  }

  // Route to capture image
  server.on("/capture", HTTP_GET, handleCaptureImage);

  server.begin();
}

void handleCaptureImage() {
  // Capture image
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    server.send(500, "text/plain", "Failed to capture image");
    return;
  }

  // Send the image to the server or save to local storage
  server.send_P(200, "text/plain", "Image captured successfully!");
  
  esp_camera_fb_return(fb);
}

void loop() {
  server.handleClient();
}
