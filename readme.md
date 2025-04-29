# Real-Time Flame Detection using MLX90640 and EFR32MG24

This project implements a real-time flame detection system using a SparkFun MLX90640 thermal infrared sensor array and a Silicon Labs EFR32MG24 microcontroller. It utilizes TensorFlow Lite for Microcontrollers (TFLM) to run a lightweight machine learning model directly on the embedded device for low-latency detection.

## Overview

The system works as follows:
1.  **Data Acquisition:** A dedicated firmware (`firmware/data_acquisition/`) running on the EFR32MG24 reads raw thermal frames (32x24 pixels) from the MLX90640 sensor via I2C and streams the data over the serial (USB VCOM) port.
2.  **Data Collection (Host PC):** A Python script (`training/collect_flame_data.py`) captures the streamed serial data and saves it into labeled CSV files (`data/flame.csv`, `data/noflame.csv`).
3.  **Model Training (Host PC):** Another Python script (`training/create_flame_tinyml.py`) loads the CSV data, trains a simple Keras neural network, converts the trained model to TensorFlow Lite (Float32 format), and exports it as a C array (`.cc` and `.h` files).
4.  **Inference (EFR32MG24):** The main inference firmware (`firmware/flame_detector/`) runs on the EFR32MG24. It includes the TFLM library and the exported C model (`model/flame_detector_float_model.cc` / `.h`). It continuously reads frames from the MLX90640, preprocesses the data (normalization), feeds it into the TFLM interpreter, and prints the classification result (Flame Detected / No Flame Detected) along with the confidence score to the serial console.

## Hardware Requirements

* **Microcontroller Board:** [Silicon Labs EFR32MG24 Dev Kit (BRD2601B Rev A01)](https://www.silabs.com/development-tools/wireless/efr32mg24-dev-kit) (or similar EFR32xG24 board)
* **Thermal Sensor:** [SparkFun MLX90640 Thermal IR Array (SEN-14843)](https://www.sparkfun.com/products/14843) (or equivalent MLX90640 breakout board)
* **Connection:** Jumper wires for I2C connection (`VCC`, `GND`, `SDA`, `SCL`)
* **Interface:** USB Cable (Type A to Micro-B or Type C, depending on the Dev Kit) for programming, power, and serial communication.

## Software Requirements

* **Host PC:** Windows, macOS, or Linux
* **IDE:** [Simplicity Studio v5](https://www.silabs.com/developers/simplicity-studio)
* **SDK:** Gecko SDK Suite v4.4.2 (or compatible, installable via Simplicity Studio) with EFR32MG24 support.
* **Python:** Python 3.x (tested with 3.8+)
* **Python Libraries:** Install using pip (preferably in a virtual environment):
    ```bash
    pip install numpy pandas tensorflow scikit-learn pyserial
    ```
    * To create a `requirements.txt` file after installation:
    ```bash
    pip freeze > requirements.txt
    ```
* **Serial Terminal:** A program to view serial output (e.g., Tera Term, PuTTY, `screen`, Simplicity Studio Serial Console).

## Repository Structure

├── data/                     # Contains collected CSV data (Example data might be included)
│   ├── flame.csv
│   └── noflame.csv
├── firmware/
│   ├── data_acquisition/     # Simplicity Studio project for streaming raw data (app.c)
│   └── flame_detector/       # Simplicity Studio project for TFLM inference (app.cpp)
│       └── model/            # Generated TFLite model C files go here
│           ├── flame_detector_float_model.cc
│           └── flame_detector_float_model.h
├── training/                 # Python scripts for data handling and model training
│   ├── collect_flame_data.py # Script to collect data from serial port
│   ├── create_flame_tinyml.py# Script to train model and generate C array
│   └── requirements.txt      # Python dependencies list (optional)
├── .gitignore                # Git ignore file for Simplicity Studio/Python projects
└── README.md                 




## Setup and Usage

### 1. Hardware Connection

Connect the MLX90640 sensor to the EFR32 Dev Kit using the I2C interface. Refer to the pinout diagrams for your specific board (`BRD2601B`) and sensor breakout.
* Sensor `VCC` -> Board `3.3V`
* Sensor `GND` -> Board `GND`
* Sensor `SDA` -> Board I2C `SDA` pin (e.g., `PC04` on BRD2601B for I2C0)
* Sensor `SCL` -> Board I2C `SCL` pin (e.g., `PC05` on BRD2601B for I2C0)
*(Verify these pins using the Simplicity Studio Hardware Configurator or board schematic if using different pins/I2C instance).*
* Connect the EFR32 Dev Kit to your PC via USB.

### 2. Data Collection (Optional - Use provided data or collect your own)

* Open the `firmware/data_acquisition/` project in Simplicity Studio.
* Build and flash the firmware to the EFR32 board (`Run` -> `Debug As` -> `Silicon Labs ARM Program`).
* Identify the serial (VCOM) port assigned to the EFR32 on your PC (e.g., `COM3`, `/dev/ttyACM0`).
* Run the `training/collect_flame_data.py` script:
    ```bash
    cd training
    python collect_flame_data.py
    ```
* Follow the script's prompts to select the serial port, baud rate (usually `115200`), and record frames for both 'flame' and 'no-flame' scenarios. Ensure the script saves the CSV files into the `data/` directory.

### 3. Model Training (Optional - Use provided model or train your own)

* Ensure `flame.csv` and `noflame.csv` are present in the `data/` directory.
* Run the `training/create_flame_tinyml.py` script:
    ```bash
    cd training
    python create_flame_tinyml.py
    ```
* This will train the model and generate `flame_detector_float.tflite`, `flame_detector_float_model.cc`, and `flame_detector_float_model.h` (likely in a subfolder like `training/output_model/`).
* **Crucially, copy the generated `.cc` and `.h` files (`flame_detector_float_model.cc`, `flame_detector_float_model.h`) into the `firmware/flame_detector/model/` directory**, overwriting any existing files if necessary.

### 4. Build and Flash Inference Firmware

* Open the `firmware/flame_detector/` project in Simplicity Studio.
* Verify the project configuration (open the `.slcp` file -> Software Components tab) includes all necessary components: `TensorFlow Lite Micro`, `I2CSPM` (for the correct instance, e.g., `sensor`), `Sleep Timer`, `IO Stream: USART` (instance `vcom`), `C++ Support`, `Micrium OS Kernel`, etc.
* Ensure the model files (`model/flame_detector_float_model.cc` and `.h`) are correctly added to the project structure within Simplicity Studio and are part of the build (they should compile without errors).
* Build the project (Hammer icon or `Project` -> `Build Project`).
* Flash the firmware to the EFR32 board (Debug icon or `Run` -> `Debug As` -> `Silicon Labs ARM Program`).

### 5. Run and Monitor

* If you started a debug session, press Resume (F8 or the Play icon). If you only flashed, reset the board or power cycle it.
* Open a serial terminal program (Tera Term, PuTTY, Simplicity Studio `Terminal` view).
* Connect to the EFR32's VCOM port with settings: **115200 baud, 8 data bits, no parity, 1 stop bit (8N1)**.
* You should see initialization messages followed by periodic output indicating whether a flame is detected:
    ```
    Initializing MLX90640...
    MLX Initialized.
    Refresh rate set to 8Hz.
    EEPROM Read & Params Extracted.
    Initializing TFLM Model...
    ... (TFLM Init messages) ...
    ML Model Initialized Successfully!
    Starting periodic timer (250 ms)...
    Initialization Complete. Running...
    ------------------------------------
    >>> No Flame Detected (Confidence: 0.041)
    ---
    >>> No Flame Detected (Confidence: 0.025)
    ---
    ```
* Introduce a flame (e.g., lighter, candle) into the sensor's field of view. The output should change:
    ```
    >>> Flame Detected! (Confidence: 0.987)
    ---
    ```

## Troubleshooting

* **No Serial Output:** Check COM port selection, baud rate (`115200`), cable connection. Ensure the `IO Stream: USART` component (instance `vcom`) is installed and configured correctly in the `.slcp` file. Check if firmware flashed successfully.
* **Sensor Errors (Init Fail, Read Error):** Double-check I2C wiring (`SDA`/`SCL` swap is common, ensure `VCC`/`GND` are correct). Verify the I2C pins configured in the `I2CSPM` component settings match the hardware connections (`PC04`/`PC05` for `I2CSPM` instance `sensor` if using defaults).
* **TFLM Errors (Allocation/Ops):** Increase `TENSOR_ARENA_SIZE` in `firmware/flame_detector/app.cpp` if you get allocation errors during `AllocateTensors()`. Ensure all required TensorFlow Lite ops (e.g., `AddFullyConnected`, `AddRelu`, `AddLogistic`) are registered in `init_ml_model` in `app.cpp` - match these to the layers used in your Keras model.
* **Build Errors:** Ensure all required SDK components listed in the `.slcp` file are installed via Simplicity Studio (`Install` button). Check that model C files (`.cc`, `.h`) are correctly located within the project view and part of the build. Ensure C++ support is enabled if using `.cpp` files.

## License

*(Optional: Add your chosen license here, e.g., MIT, Apache 2.0. You should also add a LICENSE file to the repository.)*

This project is licensed under the [NAME OF LICENSE] License - see the `LICENSE` file for details.

## Acknowledgments

*(Optional: Mention any libraries, code snippets, or tutorials that were helpful.)*
* Melexis for the MLX90640 sensor.
* Silicon Labs for the EFR32MG24 and Simplicity Studio.
* The TensorFlow Lite Micro team.
