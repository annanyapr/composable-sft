  

This repo contains the code required to 

1. Train the LT-SFT models: Use the script examples/text-classification/train_nli.sh.
2. Train the SC-SFT models: To see how to train a SC-SFT model, please check examples/text-classification/train_nli_layer_wise.sh. Modify examples/text-classification/per_layer_percentage.json to change the distribution of the SC-SFT. For sa use examples/text-classification/train_sa_layer_wise.sh
3. Train the Rand-SFT: To see how to train a Rand-SFT model, please check examples/text-classification/train_nli_random.sh. For sa you have to use examples/text-classification/train_sa_random.sh
4. To train Language SFT check the scripts examples/language-modeling/train_mlm_layer_wise.sh and examples/language-modeling/train_mlm.sh
5. To evaluate the model check examples/text-classification/eval_nli.sh and examples/text-classification/eval_sa.sh.

There are two ipython noteboooks. 

### SFT_Training_and_Evaluation.ipynb
 
This ipython notebook demonstrates the flow to train the model and evaluate it. Do check it once

1. Sentiment analysis for low resource indonesian languages.

2. NLI for low resource american languages.

### Layer_Analysis.ipynb

This ipython notebook contains the code to generate the histograms which show the distribution of parameters in the different type of SFTs.


The generated weights for the LT-SFT model for the SA task can be found at https://drive.google.com/drive/folders/1ijEWUZw6e34wv7eZRggRV0AvJ0aUoj8e?usp=sharing.

  
The generated weights for the LT-SFT model for the NLI task can be found at https://drive.google.com/drive/folders/1BPxlTlDNJHqeIsUrBw5P8eLMY5u68JDN?usp=sharing

The generated weights for SC-SFT model for the NLI task can be found at https://drive.google.com/drive/folders/1_Sye8gx5UZp7pMJed5vV87sf8jccbz0s?usp=sharing

The generated weights for SC-SFT model for the SA task can be found at https://drive.google.com/drive/folders/13OScrbV0Y5TViqpwsQZjAGXWz6mRMMI5?usp=sharing



The generated weights for Rand-SFT model for the SA task can be found at https://drive.google.com/drive/folders/1IfdCovJ_6bprftIBZ440DS-OeHPPhAVC?usp=sharing

The generated weights for Rand-SFT with low parameters for the SA task can be found at https://drive.google.com/drive/folders/10EIoByO3N60T6fIiw9vjqs4ipL709d8F?usp=sharing

The generated weights for Rand-SFT without embedding for the SA task can be found at https://drive.google.com/drive/folders/1HWiUSyz4vwG1Zb_ANjiuHpK4oP0CKvfp?usp=sharing

The generated weights for the Language SFT using SC-SFT and LT-SFT for 5 american languages can be found at https://drive.google.com/drive/folders/1xerDU1C71zmUySjEYQNb0tk2_x7SsVpA?usp=sharing
  

### Reference
The original repo is https://github.com/cambridgeltl/composable-sft


This work was jointly done by Manan Sharma(msharma2@andrew.cmu.edu) and Annanya Chauhan(annanyac@andrew.cmu.edu).