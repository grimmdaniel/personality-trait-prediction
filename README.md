# Big Five personality trait prediction on First Impressions V2 dataset

### Description
In this assignment you have to build a multimodal deep neural network for personality trait detection using tf.keras. You have to work with the First Impressions V2 dataset, which contains short (~15 seconds long) video clip recordings of speakers talking to the camera.  We will extract and combine RGB frames with MFCCs and utilize both video and audio information sources to achieve a better prediction.

You can **use only these** 3rd party **packages:** `cv2, keras, matplotlib, numpy, pandas, sklearn, skimage, tensorflow, librosa`.

## Prepare dataset

* Download the Chalearn: First Impressions V2 dataset. Here you can find more information about the dataset: http://chalearnlap.cvc.uab.es/dataset/24/description/
Big5.zip contains all of the mp4 clips, and the ground truth annotations. The samples are mostly 15 seconds long video clips, with one speaker talking to the camera. There are five personality traits: Extraversion, Agreeableness, Conscientiousness, Neuroticism and Openness. All target variables have continous values between [0, 1].
(regression task)

* Preprocess the data.
  * Audio representation: 
    * Extract the audio from the video. (Tips: use ffmpeg.)
    * Extract 24 Mel Frequency Cepstral Coefficients from the audio. (Tips: use librosa.)
    * Calculate the mean number of (spectral) frames in the dataset.
    * Standardize the MFCCs sample-wise. (Tips: zero mean and unit variance)
    * Use pre-padding (Note: with 0, which is also the mean after standardization) to unify the length of the samples.
    * Audio representation per sample is a tensor with shape (N,M,1) where N is the number of coefficients (e.g. 24) and M is the number of audio frames.
  * Visual representation:
    * Extract the frames from the video. (Tips: use ffmpeg.)
    * Resize the images to 140x248 to preserve the aspect ratio. (Tips: You can use lower/higher resolution as well.)
    * Subsample the frames to reduce complexity (6 frames/video is enough).
    * Use random crop 128x128 for training, and center crop for the validation and test sets.
    * Apply other standard data augmentation techniques, and scaling [0, 1].
    * Video representation per sample is a tensor with shape (F,H,W,3) where F is the number of frames (e.g. 6), H and W are the spatial dimensions (e.g. 128).
  * Ground truth labels:
    * There are 5 targets. Plot the distributions of the 5 personality traits.
    * You have to deal with an enhanced 'regression-to-the-mean problem'.

* Use the original dataset train-valid-test splits defined by the authors.
* Create a generator, which iterates over the audio and visual representations. (Note: the generator should produce a tuple ([x0, x1], y), where x0 is the audio, x1 is the video representation, y is the ground truth. (Don't forget: y is 5x1 vector for every sample.)
* Print the size of each set, plot 3 samples: frames, MFCCs and their corresponding personality trait annotations. (Tips: use librosa for plotting MFCCs)

Alternative considerations. They may require additional steps:
* You can use Mean MFCCs vectors to further reduce complexity. Input of the corresponding subnetwork should be modified to accept inputs with shape (N, 1).
* You can use log-melspectrograms as well. Note, that raw spectrograms are displaying power. Mel scale should be applied on the frequency axis, and log on the third dimension (decibels are expected). You can use librosa for that (librosa.feature.melspectrogram, librosa.power_to_db)

## Create Model

* Create the audio subnetwork
  * Choose one of these:
    * BLSTM (64 units, return sequences) + Dropout 0.5 + BLSTM (64 units) + Dense (128 units, ReLU)
    * Conv1D (32 filters, 3x3) + BatchNorm + ReLU, Conv1D (32 filters, 3x3) + BatchNorm + ReLU, Conv1D (64 filters, 3x3) + BatchNorm + ReLU, LSTM (64 units) + Dropout 0.5 + Dense (128 units, ReLU)
    * Conv2D (32 filters, 3x3) + BatchNorm + ReLU, MaxPool2D, Conv2D (32 filters, 3x3) + BatchNorm + ReLU, MaxPool2D, Flatten, Dense (128 units, ReLU)
  * You can try other configurations, better submodels. Have a reason for your choice!
* Create the visual subnetwork
  * Choose a visual backbone, which is applied frame-wise (Tips: use TimeDistributed Layer for this):
    * VGG-like architecture (Conv2D + MaxPooling blocks)
    * ResNet50 / Inception architecture (Residual blocks, Inception cells)
  * Apply Max pooling over the time dimension to reduce complexity (or use GRU or LSTM for better temporal modelling)
  * You can try other configurations, better submodels (like 3D convolution nets). Have a reason for your choice!
* Model fusion:
  * Concatenate the final hidden representations of the audio and visual subnetwork.
  * Apply fully connected layers on it (256 units, ReLU), then an another dense layer (5 units, linear). (Tips: you can try sigmoid to squeeze the output between [0,1]. However, you might struggle with the 'regression-to-the-mean' problem, the model prediction will be around 0.5 all the time, which is a local minimum.)
  * You can feed multiple inputs to the Model using a list: 
  model = tf.keras.models.Model(inputs=[input_audio, input_video], outputs=output)

* Performance metric:
  * Use the 1-Mean Absolute Error (1-MAE) as the evaluation metric.

## Extra task (Optional)
  * It is an unnormalized metric, and because of the target variables follow a Gaussian distribution, the modell will achieve around 0.88
  * Try to beat it, by modifing any part of the task, e.g. subnetworks, input data, configurations, preprocessing steps. (0.89-0.90 is great, between 0.90-0.92 is possible as well)
 
## Final steps, evaluation

* Plot the training / validation curve.
* Plot the '1-MAE' performance metric.
* Calculate the coefficient of determination (R^2) regression metric on the train, validation and test sets after training! (Note, that monitoring this metric during training is not advised with small batch size, because it is noisy and misleading.)
