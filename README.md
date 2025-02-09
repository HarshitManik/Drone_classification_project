# Drone_classification_project
Bringing novelty in earlier drone classification mechanism by introducing different neural networks for inreased  accuracy, using limitied radar data for classification, converting raw radar signals to spectogram data, applying fast fourier transforms and doing feature extraction and also working real-time classification on drones.
For the spectogram based clasiification(i.e first code-drone_classification_spectogram_cnn)

# Classification part-1
# Spectrogram-Based CNN Classification

## Overview
This project converts `.mat` files containing raw audio data into spectrogram images and trains a Convolutional Neural Network (CNN) model to classify different flying objects based on their acoustic signatures.

## Procedure
### 1. Convert `.mat` Files to Spectrogram Images
- Load `.mat` files and extract the `rawdata` key.
- Convert the extracted audio data into a floating-point format.
- Generate spectrogram images using `matplotlib`.
- Save valid spectrogram images in separate class-wise folders.
- Log any `.mat` files that were skipped due to missing or invalid data.

### 2. Train a CNN Model for Classification
- Load the spectrogram images and split them into training and validation sets.
- Apply data augmentation techniques for better generalization.
- Define a CNN architecture with convolutional, pooling, and fully connected layers.
- Compile and train the model using a suitable optimizer and loss function.
- Evaluate the model using accuracy, loss curves, and a confusion matrix.

### 3. Generate Confusion Matrix
- Compare predicted vs. actual class labels.
- Plot a heatmap using `seaborn` to visualize misclassifications.
- Print the classification report with precision, recall, and F1-score.

## Results
- Successfully processed **1140** spectrogram images from valid `.mat` files.
- Identified and handled **skipped** files due to missing or incompatible data.
- Achieved classification results with CNN.

## Resources
IEEE- dataset(instruction included for reference)
Abstract 
RRN-ATR-Net is a radar echo data set containing the micro-Doppler signatures (m-Ds) of different aerial targets, with a total sample size of 1200. It is available for researchers interested in this field but having difficulty collecting data. The data set is acquired using Texas Instruments' AWR1642BOOST radar sensor and the DCA1000EVM high-speed data acquisition card. The target types subject to acquisition include quadrotor (Phantom3s), fixed-wing (Cessna182), helicopter (T-REX450), and bionic bird (Gogo Bird1020). These data are collected in the anechoic chamber to eliminate unnecessary interference. The acquisition method is to place the target statically in a fixed position within a range of 1.5 to 6 meters away from the radar, allowing only the blade or wing to move. 300 signal samples are collected for each type.

Instructions: 
Use the following matlab code to read and parse the data, as an example of a short-time Fourier transform of the signal:
%%
clear all;close all;clc;
num = 1;
Rawdata = load(strcat('fixed_wing\',num2str(num),'.mat'));
rawdata = Rawdata.target_bin_part;
nwin = 64;
wind = hamming(nwin);
nlap = nwin-2;
nfft = 128;
STFT_imag=spectrogram(rawdata,wind,nlap,nfft,10000,'centered','yaxis');
x = [0 0.4096];
y = [-5000 5000];
figure
imagesc(x,y,db(abs(STFT_imag)))
axis xy;
colormap('jet')
%%

## Usage
1. Clone this repository:
   ```bash
   git clone <https://github.com/HarshitManik/Drone_classification_project/edit/main/README.md>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the conversion script:
   ```bash
   python convert_mat_to_spectrogram.py
   ```
4. Train the model:
   ```bash
   python train_cnn.py
   ```
5. Evaluate performance:
   ```bash
   python evaluate_model.py
   ```

## Acknowledgments
- Libraries used: Python, NumPy, SciPy, Matplotlib, Seaborn, TensorFlow/Keras.

Feel free to contribute or raise issues if needed! 🚀

