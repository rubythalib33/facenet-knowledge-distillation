# facenet-knowledge-distillation

I train knowledge distillation from open source face recognition models called facenet-pytorch

https://github.com/timesler/facenet-pytorch

Trained Done:
- MobileNetV2 models

## Benchmark
```
Device: CPU
InceptionResnetV1: Parameters = 27910327, Inference Time = 0.05453 seconds
EfficientNetB1: Parameters = 7794184, Inference Time = 0.02678 seconds
MobileNetV3-Large: Parameters = 5483032, Inference Time = 0.01192 seconds
MobileNetV2: Parameters = 3504872, Inference Time = 0.01510 seconds

Device: CUDA
InceptionResnetV1: Parameters = 27910327, Inference Time = 0.01620 seconds
EfficientNetB1: Parameters = 7794184, Inference Time = 0.01093 seconds
MobileNetV3-Large: Parameters = 5483032, Inference Time = 0.00557 seconds
MobileNetV2: Parameters = 3504872, Inference Time = 0.00481 seconds
```