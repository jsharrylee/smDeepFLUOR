# smDeepFLUOR: Single-Molecule Deep Learning Fluorescence Classification

This repository contains a simple 3D convolutional neural network designed to classify the fluorescence states of individual molecules from single-molecule TIRF (Total Internal Reflection Fluorescence) images.

The workflow consists of the following steps:

Particle tracking is performed using the ImageJ plugin ‘Mosaic’, which generates an Excel file containing the trajectories of particles detected in the full field-of-view image (not included in the code).

Based on this file, each single particle is cropped from the large TIRF image stack.

The dataset is then split into training and test sets.

After loading, the data undergoes standardization.

Finally, the model is trained and validated on the preprocessed single-particle image patches.
