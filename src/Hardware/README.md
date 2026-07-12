# Hardware Setup Documentation

## 📋 Overview
This section details the hardware implementation of the Smart Waste Management System, which uses ultrasonic sensors to measure waste levels in bins and transmit data wirelessly for real-time monitoring and predictive analysis.

## 🔧 Hardware Components

### 1. NodeMCU ESP8266
**Description**: A low-cost Wi-Fi microcontroller module based on the ESP8266 chip.

**Specifications**:
- Operating Voltage: 3.3V
- Digital I/O Pins: 17
- Analog Input Pins: 1
- Clock Speed: 80/160 MHz
- Flash Memory: 4MB
- Built-in Wi-Fi: 802.11 b/g/n

**Why NodeMCU?**
- **Cost-effective**: Significantly cheaper than Arduino + Wi-Fi shield combination
- **Integrated Wi-Fi**: Built-in wireless capabilities eliminate need for additional modules
- **Sufficient Processing Power**: Adequate for sensor data processing and network communication
- **Large Community Support**: Extensive documentation and libraries available
- **Low Power Consumption**: Suitable for battery-powered deployments

### 2. HC-SR04 Ultrasonic Sensor
**Description**: Non-contact distance measurement sensor using ultrasonic waves.

**Specifications**:
- Operating Voltage: 5V (works with 3.3V)
- Operating Current: 15mA
- Measuring Range: 2cm - 400cm
- Measuring Angle: 15°
- Accuracy: ±3mm
- Operating Frequency: 40kHz

**Why HC-SR04?**
- **Affordable**: Low cost per unit for large-scale deployment
- **Reliable**: Proven technology for distance measurement
- **Weather Resistant**: Performs well in various environmental conditions
- **Simple Interface**: Easy to integrate with microcontrollers
- **No Physical Contact**: Hygienic solution for waste measurement

## 🔌 Connection Diagram

```
HC-SR04          NodeMCU ESP8266
-------          ---------------
VCC      ------>    3V3
GND      ------>    GND
Trig     ------>    D5 (GPIO 14)
Echo     ------>    D6 (GPIO 12)
```

### Detailed Pin Connections

| HC-SR04 Pin | NodeMCU Pin | GPIO | Purpose |
|-------------|-------------|------|---------|
| VCC | 3V3 | - | Power supply (3.3V) |
| GND | GND | - | Common ground |
| Trig | D5 | GPIO 14 | Trigger ultrasonic pulse |
| Echo | D6 | GPIO 12 | Receive echo signal |

### Connection Explanation

1. **VCC → 3V3**: 
   - Provides power to the ultrasonic sensor
   - NodeMCU's 3.3V is sufficient for HC-SR04 operation
   - Ensures stable sensor readings

2. **GND → GND**: 
   - Establishes common ground reference
   - Essential for proper signal communication
   - Prevents floating ground issues

3. **Trig → D5 (GPIO 14)**:
   - Output pin from NodeMCU
   - Sends 10μs pulse to trigger ultrasonic burst
   - GPIO 14 chosen for its stability and availability

4. **Echo → D6 (GPIO 12)**:
   - Input pin to NodeMCU
   - Receives echo pulse duration
   - GPIO 12 selected for interrupt capability

## 📦 Physical Setup

### Dustbin Configuration
- **Standard Height**: 20 cm
- **Adjusted Software Height**: 18 cm (accounts for sensor mounting and error margin)
- **Sensor Mounting**: Top inside edge of dustbin
- **Orientation**: Sensor facing downward toward waste

### Power Supply Options
1. **USB Cable**: Direct power from USB port or power bank
2. **5V Power Adapter**: Wall adapter for permanent installations
3. **Battery Pack**: For portable deployments (with voltage regulator)

## 🔄 Alternative Hardware Options

### Microcontroller Alternatives

#### 1. **Arduino Uno + ESP8266 Module**
- **Pros**: More GPIO pins, familiar platform
- **Cons**: Higher cost, larger size, requires separate Wi-Fi module
- **Use Case**: When more sensors or actuators needed

#### 2. **Raspberry Pi Zero W**
- **Pros**: Full Linux OS, more processing power, camera support
- **Cons**: Higher cost, more power consumption, overkill for simple sensing
- **Use Case**: When image processing or complex edge computing required

#### 3. **ESP32**
- **Pros**: Dual-core, Bluetooth + Wi-Fi, more GPIO
- **Cons**: Slightly higher cost, more power consumption
- **Use Case**: When Bluetooth connectivity or more processing needed

### Sensor Alternatives

#### 1. **JSN-SR04T Waterproof Ultrasonic Sensor**
- **Pros**: Waterproof, more durable, better for outdoor use
- **Cons**: Higher cost, larger size
- **Use Case**: Outdoor bins or harsh environments

#### 2. **VL53L0X ToF (Time of Flight) Sensor**
- **Pros**: Higher accuracy, smaller size, laser-based
- **Cons**: Higher cost, shorter range (up to 2m)
- **Use Case**: Smaller bins or when high precision needed

#### 3. **Sharp GP2Y0A21YK IR Distance Sensor**
- **Pros**: Compact, no blind zone
- **Cons**: Affected by ambient light, non-linear output
- **Use Case**: Indoor bins with consistent lighting

#### 4. **Load Cell + HX711**
- **Pros**: Direct weight measurement, very accurate
- **Cons**: Requires bin modification, higher cost
- **Use Case**: When weight-based measurement preferred

## 🚀 Quick Start Guide

1. **Assemble Hardware**
   - Connect components as per the connection diagram
   - Ensure firm connections using jumper wires
   - Mount sensor securely on dustbin lid

2. **Power Setup**
   - Connect USB cable to NodeMCU
   - Verify power LED is ON

3. **Upload Code**
   - Install Arduino IDE
   - Add ESP8266 board support
   - Upload the provided Arduino sketch

4. **Configuration**
   - Update Wi-Fi credentials in code
   - Adjust dustbin height if needed
   - Set server endpoint URL

## ⚙️ Calibration

1. **Empty Bin Calibration**
   - Measure distance with empty bin
   - Note the maximum distance reading
   - Update `dustbinDepth` in code

2. **Full Bin Threshold**
   - Define minimum distance for "full" status
   - Typically 2-3 cm from sensor

3. **Test Readings**
   - Place objects at various levels
   - Verify accurate percentage calculations

## 🛠️ Troubleshooting

| Issue | Possible Cause | Solution |
|-------|----------------|----------|
| No readings | Power issue | Check connections and power supply |
| Erratic readings | Loose connections | Secure all jumper wires |
| Always shows full | Sensor too close | Increase mounting height |
| Wi-Fi connection fails | Wrong credentials | Verify SSID and password |
| Data not transmitting | Server issues | Check server URL and connectivity |

## 🔮 Future Hardware Enhancements

1. **Temperature/Humidity Sensor**: Monitor environmental conditions
2. **GPS Module**: Track bin locations for large deployments
3. **LED Indicators**: Visual feedback for bin status

## 📈 Scalability Considerations

- **Mesh Networking**: Use ESP-MESH for large area coverage
- **LoRaWAN Integration**: For long-range, low-power deployments
- **Edge Computing**: Process data locally to reduce bandwidth
- **Industrial Sensors**: Upgrade to industrial-grade components for harsh environments

---

*For software implementation details, see `waste_bin_monitor.ino`*
