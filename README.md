# grad_desc-04.py

Neural Net made to visualize underlying math operations, includes a nifty visualization as the network learns an XOR logic gate.

# digit_recognition_01.py

Configurable Artifical Neural Network applied to reading handwriting samples. Run `digit_recogintion` in Python3, and you will be given options for network configuration and training.

## Default Configurations
 * Network: 784 inputs (28x28 pixel input images), 1 hidden layer of 300 nodes, 10 ouputs
 * Training: 10,000 iterations, 0.2 learning rate
 * Starting Weights: Random values, or input random number seed for repeatable results

## Execution
Loads 60,000 training images, trains for the specified iterations, and returns mean system error. Loads 10,000 test images, runs each through the trained network, and returns the rate of mis-identifaction along with average pass/fail probabilites and a simple chart of mis-identification by digit.

## Viewing Images
After execution, you can view any of the 10,000 test images, or press enter to view each mis-identifed digit one at a time.

# sigmoids.py

Demonstration of using derivative of standard logistic function, until limit of floating points.