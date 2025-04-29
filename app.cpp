// --- (Includes and Constants as before) ---
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits>

// Include app.h first (assuming it declares app_init/app_process_action)
#include "app.h"

#include "sl_sleeptimer.h"
#include "sl_i2cspm_instances.h"

#include "mlx90640/mlx90640.h"
#include "mlx90640/mlx90640_i2c.h"

// --- TFLM Core Headers (Needed for Manual Initialization) ---
// ****** THESE FILES MUST BE LOCATABLE BY THE COMPILER VIA INCLUDE PATHS ******
#include "tensorflow/lite/micro/micro_interpreter.h"    // REQUIRED
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h" // REQUIRED
#include "tensorflow/lite/schema/schema_generated.h"    // REQUIRED
#include "tensorflow/lite/micro/micro_log.h"            // For MicroPrintf

// --- Your Float32 Model Header ---
#include "flame_detector_float_model.h"

// --- Constants ---
#define MLX90640_EMISSIVITY 0.95f
#define MLX90640_TA_SHIFT   8
#define TIMER_INTERVAL_MS 250

#define ML_INPUT_WIDTH  32
#define ML_INPUT_HEIGHT 24
#define ML_INPUT_SIZE   (ML_INPUT_WIDTH * ML_INPUT_HEIGHT) // 768
#define ML_FLAME_THRESHOLD 0.8f

// --- TFLM Manual Initialization Globals ---
#define TENSOR_ARENA_SIZE (61440) // Example: 60KB - ADJUST AS NEEDED!
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

// Error Reporter and Op Resolver must persist
static tflite::MicroMutableOpResolver<10> micro_op_resolver; // Adjust <10> as needed

// --- Global Variables (MLX Sensor & TFLM Pointers) ---
sl_sleeptimer_timer_handle_t mlx90640_timer;
static paramsMLX90640 mlxParams;
uint16_t frameData[834];
uint16_t eeData[832];

static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* model_input = nullptr;
static TfLiteTensor* model_output = nullptr;

// --- TFLM Manual Initialization Function ---
// (Function definition is NOT inside extern "C" block anymore)
sl_status_t init_ml_model() {
    MicroPrintf("Initializing ML Model (Manual Setup)...\n");

    // 1. Add required Operators (CUSTOMIZE THIS SECTION)
    TfLiteStatus add_status;
    // Example operators:
    add_status = micro_op_resolver.AddConv2D();
    if(add_status != kTfLiteOk) { MicroPrintf("Error: Failed to add Conv2D op\n"); return SL_STATUS_FAIL;}
    add_status = micro_op_resolver.AddMaxPool2D();
     if(add_status != kTfLiteOk) { MicroPrintf("Error: Failed to add MaxPool2D op\n"); return SL_STATUS_FAIL;}
    add_status = micro_op_resolver.AddReshape();
     if(add_status != kTfLiteOk) { MicroPrintf("Error: Failed to add Reshape op\n"); return SL_STATUS_FAIL;}
    add_status = micro_op_resolver.AddFullyConnected();
    if(add_status != kTfLiteOk) { MicroPrintf("Error: Failed to add FullyConnected op\n"); return SL_STATUS_FAIL;}
     add_status = micro_op_resolver.AddLogistic(); // For Sigmoid output activation
     if(add_status != kTfLiteOk) { MicroPrintf("Error: Failed to add Logistic (Sigmoid) op\n"); return SL_STATUS_FAIL;}
    // ... Add other ops needed by your model ...
    MicroPrintf("  Operators added to resolver.\n");


    // 2. Load the Model
    const tflite::Model* model = tflite::GetModel(flame_detector_float_tflite);
    if (!model) { MicroPrintf("Error: Failed to get model from TFLite C array.\n"); return SL_STATUS_FAIL; }
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Error: Model schema version %lu mismatch! Expected %d\n", model->version(), TFLITE_SCHEMA_VERSION);
        return SL_STATUS_FAIL;
    }
    MicroPrintf("  Model loaded successfully. Schema version %lu.\n", model->version());


    // 3. Create the Interpreter instance
    //    ****** RESTORED the micro_error_reporter argument ******
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;
    MicroPrintf("  Interpreter created.\n");


    // 4. Allocate Tensors
    MicroPrintf("  Allocating tensors (Arena Size: %d bytes)...\n", TENSOR_ARENA_SIZE);
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        MicroPrintf("Error: AllocateTensors() failed! Status: %d. Increase TENSOR_ARENA_SIZE.\n", allocate_status);
        return SL_STATUS_FAIL;
    }
    MicroPrintf("  Tensors allocated successfully. Arena Used: %u bytes\n", (unsigned int)interpreter->arena_used_bytes());


    // 5. Get Input / Output Tensors
    model_input = interpreter->input(0);
    model_output = interpreter->output(0);
    if (!model_input || !model_output) { MicroPrintf("Error: Failed to get input/output tensor pointers.\n"); return SL_STATUS_FAIL; }


    // 6. Verify Tensor Properties
     MicroPrintf("Verifying TFLM Tensor properties...\n");
     MicroPrintf("  Input Tensor: Type=%s, Dims=[%d, %d], Size=%u bytes\n", TfLiteTypeGetName(model_input->type), model_input->dims->data[0], model_input->dims->data[1], model_input->bytes);
     MicroPrintf("  Output Tensor: Type=%s, Dims=[%d, %d], Size=%u bytes\n", TfLiteTypeGetName(model_output->type), model_output->dims->data[0], model_output->dims->data[1], model_output->bytes);

     if (model_input->type != kTfLiteFloat32 || model_input->bytes != (ML_INPUT_SIZE * sizeof(float))) {
         MicroPrintf("Error: Input tensor type/size mismatch! Expected FLOAT32[%d], Got %s[%u bytes]\n", ML_INPUT_SIZE, TfLiteTypeGetName(model_input->type), model_input->bytes);
         return SL_STATUS_FAIL;
    }
     if (model_output->type != kTfLiteFloat32 || model_output->bytes != (1 * sizeof(float))) {
         MicroPrintf("Error: Output tensor type/size mismatch! Expected FLOAT32[1], Got %s[%u bytes]\n", TfLiteTypeGetName(model_output->type), model_output->bytes);
         return SL_STATUS_FAIL;
    }

    MicroPrintf("ML Model Initialized Successfully (Manual Setup)!\n");
    return SL_STATUS_OK;
}

// --- Function to Read RAW IR Data ---
// (No changes needed here, definition NOT inside extern "C")
sl_status_t read_raw_ir_data() {
    sl_status_t status;
    int retries = 3;
    status = mlx90640_GetFrameData(frameData);
    while (status != SL_STATUS_OK && retries > 0) {
        MicroPrintf("Warning: mlx90640_GetFrameData failed (Status: 0x%lX), retrying...\n", status);
        sl_sleeptimer_delay_millisecond(20);
        status = mlx90640_GetFrameData(frameData);
        retries--;
    }
    if (status != SL_STATUS_OK) {
        MicroPrintf("Error: Failed to get MLX90640 frame data after retries (Status: 0x%lX)!\n", status);
    }
    return status;
}

// --- Function to Run Inference ---
// (No changes needed here, definition NOT inside extern "C")
void run_flame_inference() {
    if (!interpreter || !model_input || !model_output) {
        MicroPrintf("Error: TFLM Interpreter not initialized!\n");
        return;
    }
    // 1. Prepare Input Data (Normalize)
    for (int i = 0; i < ML_INPUT_SIZE; ++i) {
        model_input->data.f[i] = static_cast<float>(frameData[i]) / 65535.0f;
    }
    // 2. Run Inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        MicroPrintf("Error: TFLM Invoke() failed (Status: %d)\n", invoke_status);
        return;
    }
    // 3. Process Output
    float probability_flame = model_output->data.f[0];
    // 4. Make Prediction and Print Result
    if (probability_flame > ML_FLAME_THRESHOLD) {
        printf(">>> Flame Detected! (Confidence: %.2f)\n", probability_flame);
    } else {
        printf(">>> No Flame Detected (Confidence: %.2f)\n", probability_flame);
    }
    printf("---\n");
}

// --- Timer Callback Function ---
// (No changes needed here, definition NOT inside extern "C")
// Note: This is a C callback type, but the function itself is C++
void mlx90640_timer_callback(sl_sleeptimer_timer_handle_t *handle, void *data) {
    (void)handle; (void)data;
    if (read_raw_ir_data() == SL_STATUS_OK) {
        run_flame_inference();
    } else {
         MicroPrintf("Warning: Skipping inference due to sensor data read error.\n");
         printf("---\n");
    }
}


// --- Application Initialization & Process Action ---
// Definitions are now standard C++, linkage is handled by app.h (presumably)
// or potentially relies on the SL_WEAK attribute and linker behaviour if app.h is missing guards.
// REMOVED the extern "C" block from around these definitions.

SL_WEAK void app_init(void) {
    sl_status_t status;
    printf("\n=== MLX90640 Flame Detection Application (Float32 Model - Manual TFLM Init) ===\n");

    // Initialize MLX90640 Sensor
    printf("Initializing MLX90640 Sensor...\n");
    int init_retries = 5;
    do {
        status = mlx90640_init(sl_i2cspm_sensor);
        if (status != SL_STATUS_OK) { MicroPrintf("  MLX90640 Init Failed (Status: 0x%lX), retrying...\n", status); sl_sleeptimer_delay_millisecond(500); init_retries--; }
        else { MicroPrintf("  mlx90640_init Successful!\n"); }
    } while (status != SL_STATUS_OK && init_retries > 0);
     if (status != SL_STATUS_OK) { MicroPrintf("Error: Failed to initialize MLX90640. Halting.\n"); while(1); }
    printf("  MLX90640 Initialized.\n");

    uint8_t refresh_rate = 0x05; // 8Hz
    status = mlx90640_SetRefreshRate(refresh_rate);
    if (status == SL_STATUS_OK) { printf("  MLX90640 Refresh Rate set to: %d Hz\n", 1 << (refresh_rate - 1)); }
    else { MicroPrintf("Warning: Failed to set MLX90640 refresh rate (Status: 0x%lX).\n", status); }

    printf("  Reading MLX90640 EEPROM...\n");
    status = mlx90640_DumpEE(eeData);
    if (status != SL_STATUS_OK) { MicroPrintf("Error: Failed to dump MLX90640 EEPROM (Status: 0x%lX). Halting.\n", status); while(1); }

    printf("  Extracting MLX90640 Parameters...\n");
    status = mlx90640_ExtractParameters(eeData, &mlxParams);
     if (status != SL_STATUS_OK) { MicroPrintf("Error: Failed to extract MLX90640 parameters (Status: 0x%lX). Halting.\n", status); while(1); }
    printf("  MLX90640 EEPROM read and parameters extracted.\n");
    printf("MLX90640 Sensor Setup Complete.\n");


    // Initialize the Machine Learning Model MANUALLY
    // This function definition is now standard C++
    status = init_ml_model();
    if (status != SL_STATUS_OK) {
         MicroPrintf("Error: Failed to initialize ML model (Manual Init). Halting.\n");
         while(1);
    }

    // Start the Periodic Timer
    printf("Starting periodic timer (%lu ms interval)...\n", (uint32_t)TIMER_INTERVAL_MS);
    // Pass the C++ function pointer; compatible if signature matches C callback type
    status = sl_sleeptimer_start_periodic_timer_ms(&mlx90640_timer,
                                                   TIMER_INTERVAL_MS,
                                                   mlx90640_timer_callback,
                                                   NULL, 0, 0);
    if (status != SL_STATUS_OK) {
       MicroPrintf("Error: Failed to start sleeptimer (Status: 0x%lX). Halting.\n", status);
       while(1);
    }

    printf("Initialization Complete. Application Running...\n");
    printf("--------------------------------------------\n");
}

SL_WEAK void app_process_action(void) {
    // Intentionally left empty.
}

// Note: No closing '}' for an extern "C" block here anymore.
