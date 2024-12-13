
# Neural Network Engine

**A Command-Line Image Processing Tool**

This project provides a powerful neural network engine designed for image processing tasks on the Linux command line. It enables you to train and run neural networks for various image-related applications.

## Command Line Interface (CLI)

The engine is invoked using the following command structure:

```bash
./nnNet <mode> [arguments]
```

### Modes

* **learn:** Trains a new neural network model based on the provided configuration file. It opens three threads to speed up the process.
* **run:** Applies a trained model to a specified input image for prediction or processing.

### Arguments (learn mode)

* `<config file>`: Path to the configuration file specifying training parameters.
* `<output file>`: Name of the file to store the trained model.
* `<image width>`: Integer value representing the width of training images in pixels.
* `<image height>`: Integer value representing the height of training images in pixels.
* `<iterations>`: Integer value specifying the number of training iterations or the lowest error percent at which to stop training.

### Tests 
* Tested on an i5 12th gen with a small dataset of 6 128x128 RGB png images. 
  * 6 images of different anime girls with different features and different hair color.
  * Two images for each hair color and a total of three different hair colors
  * Learning took around an hour total with a learn rate of 0.01 per image
  * each back-propagation per image pair took half a second. This is faster than Pytorch on CPU.
  * it can process two 128x128 RGB images with backwards propagation in one second.
  * It was able to distinguish all the images succesfully after training.
  * high learn rate speeds (greater than 0.01) will lead to poor accuracy due to the "roughness" of each gradient descend.

* Tested on Intel Xenon
  * much slower on Xenon
  * avoid using this CPU if you can
* RAM usage = `((Width*Height)^2)*24*(10^(-9))` Giga Bytes

**Example (learn mode):**

```bash
./nnNet learn config.txt model.nn 32 32 1000
```

This command trains a neural network using the `<config file>` file, stores the trained model in `<model name>`, and uses images of size 32x32 pixels for 1000 iterations.

### Arguments (run mode)

* `<model_name>`: Path to the previously trained neural network model.
* `<input image>`: Path to the image file for processing.
* `<image width>`: Integer value representing the width of the input image (must match training data).
* `<image height>`: Integer value representing the height of the input image (must match training data).
* `(output file PNG, optional)`: Path to the optional output PNG file (if not provided, relevant raw data is sent to stdout).

**Example (run mode):**

```bash
./nnNet run model.nn input.png 32 32 output.png
```

This command applies the `model.nn` model to the `input.png` image (32x32 pixels) and saves the processed result to `output.png`. If no output file is specified, the processed data is printed to the console.

## Important Notes

* Ensure that all training images have the same width and height for consistent results.
* Pre-process input images to match the dimensions used during training to achieve accurate processing.

## Configuration File

The configuration file defines the format and content of the training data. Its format is as follows:

```
config file format:

  PNG file : PNG file or digit      : floating point

  (input)  : (train image or recognition location) : (learning rate)

NO EXTRA SPACES
```

**Explanation:**

* Each line in the configuration file specifies three values separated by colons.
* The first value identifies the type of data:
    * `PNG file`: Represents a path to a PNG image file for training.
* The second value defines the purpose of the data:
    * `train.png`: The data is used as a training image for the neural network.
    * `recognition location`: The data is associated with a specific location within the image for recognition purposes.
* The third value specifies the learning rate applied to that particular data point during training. This is a floating-point number controlling the model's learning speed.

**Example Configuration File (`config.txt`):**

```
cat.png:train.png:0.01
house.png:train.png:0.02
```
```
cat.png:0:0.01
house.png:1:0.01
```

This example configures the engine to:

* Train the model using `cat.png` and `house.png` images with a learning rate of 0.01 and 0.02, respectively.
* Tell the difference between the cat and the house.

## Troubleshooting: NaN Values

If you encounter "NaN" (Not a Number) values during the training process, it often indicates unstable learning. Here are potential solutions:

* **Decrease the learning rate:** Adjust the learning rates in your configuration file to lower values.
* **Reduce the number of iterations:** Train the model for fewer iterations initially.

Experiment with these solutions.
