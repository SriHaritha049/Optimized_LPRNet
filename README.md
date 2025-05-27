How to Run:
I have used Google Colab to build this and uploaded the ipynb file here. Running it would give the results of my work.
LPRNet Model Optimization Project Report

1. Introduction

This project focuses on systematically optimizing the LPRNet deep neural network for License Plate Recognition, with the primary objectives of enhancing:
Accuracy
Speed
Space Efficiency
2. Original Model Characteristics

2.1 LPRNet Architecture Overview

LPRNet is a lightweight, efficient deep neural network designed for real-time license plate recognition, characterized by:
Input Image Size: 94×24 pixels
Specialized preprocessing and normalization
Convolutional backbone for feature extraction
Connectionist Temporal Classification (CTC) loss function

2.2 Key Architectural Components

Input Processing
Fixed image size (94×24 pixels)
Normalization for improved training convergence
Convolutional Backbone
Convolutional layers for spatial feature extraction
Batch Normalization for feature map normalization
ReLU activation functions
Sequence Modeling
Spatial-to-sequence mapping approach
Eliminates explicit character segmentation
Output Mechanism
Character sequence prediction
Greedy decoding algorithm


3. Optimization Strategies
3.1 Pruning Optimization

Technique: L1 Unstructured Pruning
Target Layers: Conv2d and Linear layers
Pruning Percentage: 35%
Methodology:
Remove weights with smallest L1 norm
Permanently eliminate insignificant connections
Results:
Model Size Reduction
Accuracy: 89.8% (minimal decrease)
Pruning Insights
Careful layer sensitivity analysis
Balanced trade-off between model compression and accuracy
Focused on computationally intensive layers

3.2 Quantization Optimization

Approach: Dynamic Quantization
Conversion: Weights to int8 precision
Key Techniques:
Dynamic Quantization
Runtime weight precision reduction
Preserved activation floating-point precision
Layer Fusion
Combined compatible layers
Reduced computational redundancy
Targeted Conv2d, BatchNorm, and ReLU layers
Quantization Outcomes
Significant model size reduction
Improved inference speed
Minimal accuracy degradation

3.3 Machine Learning Compiler (MLC) Optimizations

Model Compilation
TorchScript tracing
Conversion to TVM Relay format
Advanced Optimizations
Type inference and simplification
Optimization techniques:
1.FoldScaleAxis
2.FuseOps
3.AlterOpLayout
4.EliminateCommonSubexpr
5. XGBoost autotuning


4. Performance Metrics Comparison

![image](https://github.ncsu.edu/schalic/Optimized_LPRNet_CSC591/assets/30720/5ab26aaf-b58b-4561-a6bb-929f1d8190d9)


6. Conclusion
While the optimizations improved the speed of the model, perfect results weren’t achieved. The accuracy was slightly sacrificed, especially after pruning, but the model size remained consistent throughout the process. These results emphasize the need to strike the right balance between the three key metrics: accuracy, speed, and size. Moving forward, a more targeted approach in applying these optimizations may help achieve even better performance while minimizing trade-offs.



