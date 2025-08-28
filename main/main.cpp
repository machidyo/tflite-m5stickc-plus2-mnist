#include <stdio.h>
#include "M5Unified.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
uint8_t *trace_map;

constexpr int kTensorArenaSize = 14000;
uint8_t tensor_arena[kTensorArenaSize];
}

void setupM5();
void setupDisplay();

void setup() {
  setupM5();

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  //model = tflite::GetModel(g_model);
  model = tflite::GetModel(result_tflite);  
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  static tflite::MicroMutableOpResolver<10> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddShape();
  resolver.AddStridedSlice();
  resolver.AddPack(); 
  // if (resolver.AddFullyConnected() != kTfLiteOk) {
  //   return;
  // }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }
  else {
    MicroPrintf("AllocateTensors() successed");
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  trace_map = new uint8_t[28 * 28];
  for(uint16_t i=0; i < 28 * 28; ++i){
    if ((i - 12) % 28 == 0 || (i - 12) % 28 == 1)
      trace_map[i] = 255;
    else
      trace_map[i] = 0;
  }

  printf("[LOG] debug.\n");
  int index = 0;
  for(uint16_t i=0; i < 28; ++i){
    for(uint16_t j=0; j < 28; ++j){
      index = 28 * i + j;
      if (trace_map[index] == 255)
        printf("x");
      else
        printf("o");
    }
    printf("\n");
  }

  printf("[LOG] setup is finished.\n");
}

void loop() {
  for(uint16_t i=0; i < 28 * 28; ++i){
    input->data.uint8[i] = trace_map[i];
  }

  // printf("[LOG] start to invoke.\n");
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed\n");
    return;
  }

  uint8_t max_index = 0;
  uint8_t max_value = 0;
  for(uint8_t i=0; i < 10; ++i){
    float result = output->data.uint8[i] / 255;
    printf("[LOG] %d = %f\n", i, result);
    if(max_value < output->data.uint8[i]){
      max_value = output->data.uint8[i];
      max_index = i;
    }
  }

  //M5.Display.fillScreen(BLACK);  
  //M5.Display.setCursor(10, 20);
  // 推論結果をLED表示 確度 50%以下はバツ表示
  if(max_value >= 128){
    //M5.Display.printf("result is %d\n", max_index);
    printf("[LOG] result is %d\n", max_index);
  }
  else{
    //M5.Display.printf("result is X.\n");
    printf("[LOG] result is X.\n");
  }

  // 普通にloopを回すと推論より早すぎてエラーが出るのでdelayさせるExamplesでは500ms待っていた
  M5.delay(3000);
}

void setupM5() {
  auto cfg = M5.config();
  M5.begin(cfg);

  setupDisplay();
}

void setupDisplay() {
  M5.Display.setRotation(3);
  M5.Display.setTextColor(WHITE);
  M5.Display.setTextSize(2);

  M5.Display.fillScreen(BLACK);
  M5.Display.setCursor(10, 20);
  M5.Display.printf("Hello world\n");
  printf("Hello world ver.3 %d\n");
}
