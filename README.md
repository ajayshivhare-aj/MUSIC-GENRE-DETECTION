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



# Structure for Genre Detection Project - 

Our project is divided over 6 files named and described below :

1) spectrogram_playmusic.ipynb -   Contains the spectrograms of distinct genres.
2) load_fma_dataset.ipynb -        The FMA dataset is loaded & feature engineering was performed.                                                                       
3) Convert_to_npz.ipynb -          Code to convert .mp3 music files to npz.
4) CRNN_model.ipynb -              Deep Learning Model CRNN
5) CNN_RNN_Parellel.ipynb -        Deep Learning Model Parallel CNN RNN
6) Heuristics.ipynb -              Output based on above two deep learning models.




# Data Set - 

In the case of Music, there are a few different datasets with data — GTZan and Million Songs dataset (MSD) are 2 of the ones most commonly used. But both of these data sets have limitations. GTZan only has 100 songs per genre and MSD has well 1 million songs but with only their metadata, raw audio files are not available. 

We have decided to use the Free Music Archive Small dataset:
(Link: https://github.com/mdeff/fma)
The Free Music Archive (FMA), an open and easily accessible dataset suitable for evaluating several tasks in MIR & concerned with browsing, searching, and organizing large music collections. The FMA small data set that we will be using has 8 genres and 1000 songs per genre evenly distributed (balanced dataset) with 30-second audio files & related meta-data. The eight genres are Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, and Rock. FMA Small is split into a training set of 6400 songs, a validation set of 800 songs & test set of 800 songs.


 
## Reasons to convert the data into spectrogram - The spectrogram is a powerful tool to identify sound features either in real-time or in post-processing on recorded materials. Its utility however strongly depends on the type of audio material to be examined and on the accurate tuning of the analysis parameters. In some cases it is powerful enough to recognize the features of interest more accurately and objectively than human hearing.

 

# Reasons for choosing MFCC 
MFCCs alone can be is used as features in speech recognition task.

The mel frequency cepstral coefficients (MFCCs) of a signal are a set of features which concisely describe the overall shape of a spectral envelope ie curve in the frequency-amplitude plane.




