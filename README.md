# LPRNet Model Optimization Project

This project demonstrates systematic optimization of the LPRNet deep neural network for License Plate Recognition (LPR), focusing on three key metrics:

- **Accuracy**
- **Inference Speed**
- **Model Size (Space Efficiency)**

---

## 1. How to Run

The project has been developed and tested on **Google Colab**.  
To reproduce the results:

- Upload and open the provided `.ipynb` file in Google Colab.
- Run all cells to execute the full optimization pipeline and view results.

---

## 2. Original Model Characteristics

### 2.1 LPRNet Architecture Overview

LPRNet is a lightweight, real-time DNN model for license plate recognition. Key properties include:

- **Input Size:** 94×24 pixels
- **Preprocessing:** Normalization for training stability
- **Backbone:** Convolutional layers
- **Loss Function:** Connectionist Temporal Classification (CTC)

### 2.2 Key Architectural Components

- **Input Processing:**  
  - Fixed-size normalization (94×24 pixels)
- **Feature Extraction:**  
  - Conv2D layers + BatchNorm + ReLU activations
- **Sequence Modeling:**  
  - Spatial-to-sequence mapping (no explicit character segmentation)
- **Output Layer:**  
  - Greedy decoding for character sequence prediction

---

## 3. Optimization Strategies

### 3.1 Pruning Optimization

- **Technique:** L1 Unstructured Pruning  
- **Target Layers:** Conv2D & Linear  
- **Pruning Ratio:** 35%

#### Methodology:
- Remove weights with smallest L1-norm
- Permanently eliminate less important connections

#### Results:
- **Accuracy:** 89.8% (minimal drop)
- **Model Size:** Reduced
- **Insights:**  
  - Layer sensitivity analysis helped balance compression and accuracy.
  - Focus was on compute-heavy layers.

---

### 3.2 Quantization Optimization

- **Type:** Dynamic Quantization (weights to int8)

#### Techniques:
- **Dynamic Quantization:** Reduce runtime weight precision to int8
- **Layer Fusion:** Merge Conv2D + BatchNorm + ReLU layers to optimize execution

#### Results:
- Improved inference speed
- Further model size reduction
- **Accuracy Impact:** Minor degradation

---

### 3.3 Machine Learning Compiler (MLC) Optimizations

#### Toolchain:
- **TorchScript → TVM Relay Conversion**

#### Applied Optimizations:
- `FoldScaleAxis`
- `FuseOps`
- `AlterOpLayout`
- `EliminateCommonSubexpr`
- **XGBoost Autotuning** for kernel performance

#### Benefits:
- Compiler-level fusion and layout tuning
- TVM autotuning delivered faster inference

---

## 4. Performance Metrics Comparison

| Optimization Stage | Accuracy | Inference Time | Model Size |
|--------------------|----------|----------------|-------------|
| Original           | ✅ High  | ❌ Slow         | 📦 Large     |
| Pruned             | ⚠️ ~89.8% | ✅ Faster       | ✅ Smaller   |
| Quantized          | ⚠️ Slight drop | ✅ Fastest | ✅ Smallest  |
| TVM-Compiled       | ⚠️ Slight drop | 🚀 Optimized | ✅ Small     |

> 📊 ![Performance Graph](https://github.com/user-attachments/assets/9e7d1d02-a116-4a61-9884-13fb6af4b95e)

---

## 5. Conclusion

The optimization process successfully enhanced inference speed and reduced model size. However, a slight trade-off in accuracy—especially after pruning—was observed.

📌 **Key takeaway:** Striking the right balance between accuracy, speed, and size is crucial.

### Future Work:
- Apply layer-wise adaptive pruning
- Explore quantization-aware training (QAT)
- Refine TVM autotuning with calibration data

---

## Acknowledgments

- PyTorch, ONNX, TVM, and CUDA frameworks
- LPRNet original authors
