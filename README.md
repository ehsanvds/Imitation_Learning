# Imitation Learning
This project aims to build a model for self driving cars based on immitation learning using TensorFlow 2. A new set of data was collected from CARLA driving simulator. The neural network and procedure is based on a paper about conditional immitation learning by Codevilla et al. (2018).

### Files
- general_functions: various functions for reading files and visualization
- image_augmentation: the functions for generating augmented images
- network: the functions for constructing the convolutional neural network
- main: the main file to execute the process
- model_evaluation: reading test data and evaluating the model

### Reference
Codevilla, F., Müller, M., López, A., Koltun, V. and Dosovitskiy, A. 2018 End-to-end Driving via Conditional Imitation Learning, IEEE International Conference on Robotics and Automation (ICRA).
