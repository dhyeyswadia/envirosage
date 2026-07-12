#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>

// Replace with your Wi-Fi network credentials
const char* ssid = "Wifi-Name";
const char* password = "Wifi-Password";

// Replace with your server's endpoint URL
const char* serverUrl = "http://localhost:5000/api/waste-data";
// Define pins for the ultrasonic sensor
const int trigPin = D5; // GPIO 14
const int echoPin = D6; // GPIO 12F      

// Define the depth of the dustbin in cm
const int dustbinDepth = 18; // Adjusted for your dustbin's height of 20 cm

WiFiClient wifiClient; // Declare WiFiClient object

// Store the last known distance to detect changes
long lastDistance = -1; // Initialize to an impossible value
unsigned long lastSendTime = 0; // Track time since last forced send
const unsigned long delayInterval = 300000; // 5 minutes in milliseconds

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  int retryCount = 0;
  while (WiFi.status() != WL_CONNECTED && retryCount < 20) {
    delay(1000);
    Serial.print(".");
    retryCount++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.println("WiFi connected!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println();
    Serial.println("Failed to connect to Wi-Fi. Restarting...");
    ESP.restart();
  }

  // Set the sensor pins
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
}

void loop() {
  // Measure distance
  long duration, distance;
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  duration = pulseIn(echoPin, HIGH);
  distance = (duration * 0.034 / 2); // Calculate distance in cm

  // Map distance to a filled capacity between 0 and 10
  int filled_capacity = map(distance, 0, dustbinDepth, 10, 0);
  filled_capacity = constrain(filled_capacity, 0, 10); // Ensure it's between 0 and 10

  Serial.print("Distance: ");
  Serial.print(distance);
  Serial.println(" cm");
  Serial.print("Filled Capacity: ");
  Serial.print(filled_capacity);
  Serial.println("/10");

  // Get the current time in milliseconds
  unsigned long currentTime = millis();

  // Check if distance has changed significantly or if it's time for a periodic update
  if (distance != lastDistance || (currentTime - lastSendTime > delayInterval)) {
    // Send data to the server
    sendData(filled_capacity);

    // Update the last known distance and time
    lastDistance = distance;
    lastSendTime = currentTime;
  }

  delay(2000); // Delay 2 seconds before taking the next measurement to avoid rapid triggering
}

void sendData(int filled_capacity) {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(wifiClient, serverUrl); // Use WiFiClient with HTTPClient
    http.addHeader("Content-Type", "application/json");

    // Create JSON payload
    String jsonPayload = "{\"dustbin_id\": 7, \"location\": \"Arrey\", \"filled_capacity\": " + String(filled_capacity) + "}";
    int httpResponseCode = http.POST(jsonPayload);

    if (httpResponseCode > 0) {
      Serial.print("Data sent successfully, response code: ");
      Serial.println(httpResponseCode);
    } else {
      Serial.print("Error in sending data: ");
      Serial.println(httpResponseCode);
    }
    http.end();
  } else {
    Serial.println("WiFi disconnected, attempting to reconnect...");
    WiFi.begin(ssid, password);
  }
}
