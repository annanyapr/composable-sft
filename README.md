
This repo contains the code required to replicate the LT-SFT models. There are two main top level files ipython noteboooks. The ideal way to replicate the results is to run the below two notebooks in collab. You would also have to upload the below model weights to collab. This work was jointly replicated by Manan Sharma(msharma2@andrew.cmu.edu) and Annanya Chauhan(annanyac@andrew.cmu.edu).

### SFT_Training_and_Evaluation.ipynb

This ipython notebook contains the code that was used to train the models and to evaluate the models on 

1. Sentiment analysis for low resource indonesian languages.
2. NLI for low resource american languages.

The generated SFT model for the SA task can be found at https://drive.google.com/drive/folders/1ijEWUZw6e34wv7eZRggRV0AvJ0aUoj8e?usp=sharing.

The generated SFT model for the NLI task can be found at https://drive.google.com/drive/folders/1BPxlTlDNJHqeIsUrBw5P8eLMY5u68JDN?usp=sharing

### Layer_Analysis.ipynb
 This ipython notebook contains the code to generate the histograms which show the distribution of parameters in the different type of SFTs.

### Reference 
The original repo is https://github.com/cambridgeltl/composable-sft