# CV-controlled Robot Arm

A computer vision controlled robotic arm project using Raspberry Pi, PCA9685 servo driver, and 3 servos for precise movement control.

## Overview

This project implements a robotic arm that can be controlled through computer vision input. The system uses a Raspberry Pi as the main controller, a PCA9685 PWM servo driver for precise servo control, and three servos to create a functional robotic arm with multiple degrees of freedom.

## Hardware Requirements

- **Raspberry Pi** (any model with GPIO pins)
- **PCA9685 16-Channel 12-bit PWM/Servo Driver**
- **3 Servo Motors** (compatible with PWM control)
- **Jumper wires** for connections
- **Breadboard** (optional, for prototyping)
- **Power supply** (appropriate for your servos)

## Hardware Setup

### PCA9685 Wiring

Connect the PCA9685 to your Raspberry Pi as follows:

<img width="640" height="480" alt="PCA9685 Wiring Diagram" src="https://github.com/user-attachments/assets/74db2490-4256-4ee4-9b0c-5a7c3370aa6c" />

*Image Credit: Adafruit*

**Pin Connections:**
- **VCC** → 3.3V
- **GND** → Ground
- **SDA** → GPIO 2 (SDA)
- **SCL** → GPIO 3 (SCL)

### Servo Connections

Connect your three servos to the PCA9685:
- **Servo 1** → Channel 0
- **Servo 2** → Channel 1  
- **Servo 3** → Channel 2

Each servo requires:
- **Signal wire** → PCA9685 channel pin
- **Power wire** → External power supply positive
- **Ground wire** → Common ground with Raspberry Pi and power supply

⚠️ **Important:** Use an external power supply for servos to avoid overloading the Raspberry Pi.

## Software Installation

### Prerequisites

Ensure your Raspberry Pi is running a recent version of Raspberry Pi OS with Python 3 installed.

### Enable I2C

Enable I2C communication on your Raspberry Pi:

```bash
sudo raspi-config
```

Navigate to `Interfacing Options` → `I2C` → `Enable`

### Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

Execute the main script to start the CV-controlled robot arm:

```bash
python main.py
```

### Basic Operation

1. Ensure all hardware connections are secure
2. Power on your Raspberry Pi and external servo power supply
3. Run the main script
4. The computer vision system will begin processing input
5. The robot arm will respond to detected movements or objects

## Configuration

Modify servo positions, CV parameters, and other settings in the configuration section of `main.py` or dedicated config files.

## Troubleshooting

### Common Issues

**Servos not responding:**
- Check power supply connections
- Verify I2C is enabled
- Confirm PCA9685 address (default: 0x40)

**I2C communication errors:**
- Check SDA/SCL connections
- Verify I2C is enabled in raspi-config
- Test with `i2cdetect -y 1`

### Debugging

Enable debug mode by modifying the logging level in `main.py` or use:

```bash
python main.py --debug
```

## Additional Resources

- [Adafruit PCA9685 Documentation](https://learn.adafruit.com/adafruit-16-channel-servo-driver-with-raspberry-pi/overview)
- [Raspberry Pi GPIO Documentation](https://www.raspberrypi.org/documentation/usage/gpio/)
- [OpenCV Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Acknowledgments

- Adafruit for the PCA9685 library and documentation
- Raspberry Pi Foundation for the excellent hardware platform
- OpenCV community for computer vision tools
