// Auto-generated macro to instanciate and initialize opcode resolver based on TFLite flatbuffers in config directory
#ifndef SL_TFLITE_MICRO_OPCODE_RESOLVER_H
#define SL_TFLITE_MICRO_OPCODE_RESOLVER_H

#define SL_TFLITE_MICRO_OPCODE_RESOLVER(opcode_resolver) \
static tflite::MicroMutableOpResolver<2> opcode_resolver; \
opcode_resolver.AddFullyConnected(); \
opcode_resolver.AddLogistic(); \


#endif // SL_TFLITE_MICRO_OPCODE_RESOLVER_H
