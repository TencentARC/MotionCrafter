# Dataset Preparation

To train **MotionCrafter**, please download the following datasets from their official project pages.  
Note that the paths such as `data_normed_1/...` refer to **preprocessed** versions used in our implementation.  
You should first download the **raw datasets** listed below and then run our preprocessing pipeline to obtain the normalized format.

---

## 📦 Dataset Download Links

### **DynamicReplica**
Replica synthetic indoor dataset used in dynamic scene reconstruction research.  
**Project Page:** https://github.com/facebookresearch/Replica-Dataset

### **GTA-SfM**
Synthetic outdoor dataset rendered from GTA for SfM, depth, and motion estimation tasks.  
**Reference:** https://github.com/HKUST-Aerial-Robotics/Flow-Motion-Depth

### **Kubric**
Large-scale synthetic multi-task dataset generated using the Kubric engine from Google Research.  
**Project Page:** https://github.com/google-research/kubric

### **MatrixCity**
Ultra-large-scale city-level synthetic dataset for driving and urban scene perception.  
**Project Page:** https://github.com/city-super/MatrixCity  

### **MVS-Synth (GTAV_720)**
Multi-view stereo synthetic dataset rendered from GTA-V.  
**Dataset Page:** https://phuang17.github.io/DeepMVS/mvs-synth.html

### **Spring**
High-quality synthetic dataset for optical flow, scene flow, stereo, and multi-task perception.  
**Benchmark Website:** https://spring-benchmark.org/

### **Point Odyssey**
Large synthetic long-term point tracking dataset.  
**Project Page:** https://pointodyssey.com/

### **Synthia**
Synthetic urban driving dataset widely used for segmentation and depth estimation.  
**Official Page:** https://synthia-dataset.net/

### **TartanAir**
AirSim-based large-scale SLAM and multi-modal perception dataset.  
**Dataset Page:** https://tartanair.org/

### **Virtual KITTI 2**
Photorealistic synthetic clone of KITTI for driving perception tasks.  
**Dataset Page:** https://europe.naverlabs.com/Research/Computer-Vision/Proxy-Virtual-Worlds/

### **BlinkVision**
Video + event-based multimodal synthetic benchmark.  
**Website:** https://www.blinkvision.org/

### **OmniWorld**
Multi-domain, multi-modal synthetic world model dataset.  
**Project Page:** https://yangzhou24.github.io/OmniWorld/

### **ScanNet++**
High-resolution indoor real-world 3D reconstruction dataset.  
**Project Website:** https://scannetpp.mlsg.cit.tum.de/scannetpp/

---

## 📝 Notes
- The datasets above provide **raw data**. To reproduce the training pipeline, run the preprocessing script to convert them into the `data_normed_1/...` structure.  
- Several datasets are large (100–500K frames); please ensure sufficient storage.
- Open-source preprocessing usage and unified script configuration are documented in `datasets/preprocess/README.md`.

