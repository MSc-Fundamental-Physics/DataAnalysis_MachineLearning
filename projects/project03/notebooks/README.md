# Project 03 Notebooks and Utilities

## Notebooks

### 7_UNetConvNext_physics_v2.ipynb
Implements and trains the UNet-ConvNext hybrid architecture with physics-informed components. This notebook focuses on building the model architecture definition and training loop setup, and integration of physics-based constraints to guide the learning process. It produces a csv file calles 7_loss_UNetConvNext_physics.csv

### 8_1_testing_UNetConvNext_physics_v2_loss_function.ipynb
Evaluates the loss function behavior of the trained UNet-ConvNext physics model. This notebook reads the 7_loss_UNetConvNext_physics.csv of the previous notebook for reporting the loss function curves.

### 8_2_testing_UNetConvNext_physics_v2_qualitative.ipynb
Provides qualitative assessment of the UNet-ConvNext physics model predictions. This notebook generates visual comparisons between model outputs and ground truth data, examines spatial patterns in predictions, error distributions, and visualizes how well the model captures physical phenomena. Useful for understanding model behavior beyond numerical metrics.

### 9_model_evaluation.ipynb
Comprehensive (numerical) evaluation framework for the trained model. This notebook computes one of the standard metrics **Variance Scaled Mean Square Error** (**VMSE**). It consolidates results from previous testing notebooks into a final evaluation **table report**.

## Utilities

### normalisation.py
Utility module providing the functions for normalisation and denormalisation stages in the notebooks. It uses a preloaded tensor-like array for **mean** and **sigma** to preprocess input data for model training and post-process predictions back to original scale using the same preloaded tensors. Essential for ensuring numerical stability and proper model input/output handling.