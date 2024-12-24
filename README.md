# DDoS Attack Image Detection System

## Overview
This project implements a Deep Learning-based system for detecting various types of DOS and DDoS attacks using transfer learning. The system converts network traffic data into images and utilizes pre-trained convolutional neural networks (CNNs) to classify different types of attacks.

## Features
- Support for multiple pre-trained models:
  - ResNet18
  - VGG16
  - Inception V3
  - MobileNet
  - ResNet50
- Image-based attack detection for 12 different classes
- Achieved over 80% accuracy on test data
- GPU acceleration support
- Automatic learning rate adjustment
- Gradient clipping for stable training

## Prerequisites
- Python 3.7+
- PyTorch
- torchvision
- CUDA-capable GPU (recommended)
- matplotlib
- numpy

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/ddos-attack-detection.git
cd ddos-attack-detection

# Install required packages
pip install torch torchvision matplotlib numpy
```

## Dataset Structure
The dataset should be organized as follows:
```
attackImagesPaper/
├── Train/
│   ├── 10_Syn.csvImgs/
│   ├── 11_TFTP.csvImgs_12k/
│   ├── 12_UDPLag_t.csvImgs/
│   └── ...
└── Test/
    ├── 10_Syn.csvImgs/
    ├── 11_TFTP.csvImgs_12k/
    ├── 12_UDPLag_t.csvImgs/
    └── ...
```

## Attack Classes
1. DrDoS_DNS
2. DrDoS_LDAP
3. DrDoS_MSSQL
4. DrDoS_NetBIOS
5. DrDoS_NTP
6. DrDoS_SNMP
7. DrDoS_SSDP
8. DrDoS_UDP
9. Normal Traffic
10. Syn
11. TFTP
12. UDPLag

## Model Architecture
- Input images are resized to 128x128 pixels
- Pre-trained models are modified for 12-class classification
- Adam optimizer with learning rate of 0.001
- Cross Entropy Loss function
- Learning rate scheduling with ReduceLROnPlateau
- Gradient clipping with max norm of 1.0

## Usage
1. Prepare your dataset in the required format
2. Choose and run the desired model script:
```bash
python resnet18_model.py  # For ResNet18
python vgg16_model.py     # For VGG16
python inception_model.py  # For Inception V3
python mobilenet_model.py # For MobileNet
python resnet50_model.py  # For ResNet50
```


## Performance Monitoring
The training process provides real-time monitoring of:
- Batch-wise loss and accuracy
- Epoch training loss and accuracy
- Epoch test loss and accuracy
- Learning rate adjustments

## Results
Each model is evaluated based on:
- Training accuracy
- Test accuracy
- Training loss
- Test loss

## Future Improvements
- Implementation of ensemble methods
- Support for additional pre-trained models
- Real-time traffic analysis capabilities
- Model compression for edge deployment
- Extended dataset support

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

