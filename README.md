# Music Genre Detection using Python

•	Converted Audio files into Spectrogram for chart pattern recognition.

•	Utilized deep learning models such as CRNN and Parallel CNN & RNN to achieve accuracy of 68.9% and 70.6%.




# Problem Statement -

Audio files are important source in Machine Learning for information extraction. Music Information Retrieval - MIR is the interdisciplinary science of retrieving information from music. MIR is a small but growing field of research with many real-world applications. Those involved in MIR may have a background in musicology, psychoacoustics, psychology, academic music study, signal processing, informatics, machine learning, optical music recognition, computational intelligence or some combination of these.

 

> For this project - We aim to make the computer detect the genre of music (.mp3 file) with utmost accuracy

 
 
# Introduction - 

Automatic genre classification of music is an important topic in Music Information Retrieval with many interesting applications. A solution to genre classification would allow for machine tagging of songs, which could serve as metadata for building song recommenders. 

There has been an explosion of musical content available on the internet. Some sites, such as Spotify and Pandora, carefully curate and manually tag the songs on their sites. Other sources, such as YouTube, have a wider variety of music, but many songs lack the metadata needed to be searched and accessed by users. One of the most important features of a song is its genre.

Automatic genre classification would make hundreds of thousands of songs by local artists available to users and improve the quality of existing music recommenders on.

In this project, we investigate the following question: Given a song, can we automatically detect its genre? We look at spectrogram of the audio file to determine its genre.



# Structure for Genre Detection Project

Our project is divided over 6 files named and described below :

1) spectrogram_playmusic.ipynb -   Contains the spectrograms of distinct genres.
2) load_fma_dataset.ipynb -        The FMA dataset is loaded & feature engineering was performed.                                                                       
3) Convert_to_npz.ipynb -          Code to convert .mp3 music files to npz.
4) CRNN_model.ipynb -              Deep Learning Model CRNN
5) CNN_RNN_Parellel.ipynb -        Deep Learning Model Parallel CNN RNN
6) Heuristics.ipynb -              Output based on above two deep learning models.


# Data Sets -

In case of Music there are a few different datasets with data — GTZan and Million Songs data set (MSD) are 2 of the ones most commonly used. But both of these data sets have limitations. GTZan only has 100 songs per genre and MSD has well 1 million songs but with only their metadata, raw audio files are not available.

 

We have decided to use the Free Music Archive Small dataset. The Free Music Archive (FMA), an open and easily accessible dataset suitable for evaluating several tasks in MIR & concerned with browsing, searching, and organizing large music collections.The FMA small data set that we will be using has 8 genres and 1000 songs per genre evenly distributed (balanced dataset) with 30 second audio files & related meta-data. The eight genres are Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop and Rock. FMA Small is split into a training set of 6400 songs, validation set of 800 songs & test set of 800 songs.


Steps we have thought for a genre classification problem -

 

1. Build Baseline Classifiers -  We aim to try various sklearn models like Decision Tree, Random Forest Classifier, Support Vector Classifier, Linear Regression, XG BOOST  etc. Feature set being metadata of music files that includes 140 MFCC - Mel Frequency Cepstral Coefficient features & some other important features in the dataset  for each song - audio .mp3 file.

    

2. Find the most important features out of all the features using XG BOOST/ ADABOOST etc by extending the baseline classifiers.

 

3. Using those important features &  train the baseline model again and check the metric chosen. 

4. Build the Deep Learning Models -  We will be building the feature set for DL Models by converting the audio files into a spectogram. A spectrogram shows the frequencies that make up the sound, from low to high, and how they change over time, from left to right.

After that we aim to try Convolutional Recurrent Model & Parallel Convolutional and Recurrent Model to work on the problem of (basically) spectrogram image classification.

For image classification the local images are correlated thereby producing nearby pixels to have similar intensities & colours. In spectrogram analysis  there are often harmonic correlation which are spread along frequency axis while local correlation may be weaker.

We chose the above mentioned model because convolutional model are good with image recognition task & on the other hand RNN excels in understanding the time series data. Cause in music time t is dependent on time t -1.    

5. After we build the our baseline models with most important features containing mfcc etc and deep learning model with the spectrogram data.

We aim to apply heuristics and give weights to the deep learning and improved baseline model to produce the highest metric decided.

 


Reasons to convert the data into spectrogram - The spectrogram is a powerful tool to identify sound features either in real-time or in post-processing on recorded materials. Its utility however strongly depends on the type of audio material to be examined and on the accurate tuning of the analysis parameters. In some cases it is powerful enough to recognize the features of interest more accurately and objectively than human hearing.

 

# Reasons for choosing MFCC 
MFCCs alone can be is used as features in speech recognition task.

The mel frequency cepstral coefficients (MFCCs) of a signal are a set of features which concisely describe the overall shape of a spectral envelope ie curve in the frequency-amplitude plane.




