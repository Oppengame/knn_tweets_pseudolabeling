# unsupervised_classification_tweets
Unsupervised Filtering Exercise 
# Relevant tweet extraction: filter ~5000 tweets

## Preporcessing

  Since I used roberta which uses subword tokens trained on twitter, I didn't clean the data too much. I used clean tweets,
  deduplicated. I replace company cashtag and name with generic token to prevent duplicates in data. That left me with
  4,699,045 unlabeled samples. My EDA process can be found in jupyter notebooks, attached.

P.S. Note that your list data was saved as strings for some reason.


## Training

To run the pipeline, use **`full_process`** function from `classify.py`. 
Final results are in `class_labeled.csv.gzip` file for full dataframe with predicted probabilities (not attached due to size), 
and `filtered_twitter_data.csv` (attached) for selected 5K samples.

I use a two-stage pipeline:
1. Use k-NN on BERT-like network embeddings to extract similar and dissimilar samples. 
   I split the labeled data into train and validation based on date (20% validation data). I tried different 
   feature extractors (vanilla roberta, vanilla bert, twitter roberta), different values of k for k-means, 
   max/mean for embedding aggregation, different values of k for k-NN, and different distances for k-NN (cosine and L2).
   The final parameters are set as default. The code for this part can be found in `knn_pseudolabel.py`. The recall was 
   relatively small (~5-10%).
2. Use k-NN results to generate pseudolabels (10k of positive and negative samples for closest/farthest samples 
   with more than 10 likes). Train sequence classification based on these labels (also split 80/20 based on date). 
   Due to the nature of pseudolabels the training is very short to prevent overfitting to noise in all stages. 
   Similarly, to prevent overfitting I used different, though similar model at this stage.
   Use predictions as final probablity of tweet relevance. Extract top 5k 
   samples as representatives. Recall is still low (either due to noise in data or due to large amount of relevant data).
   Possibly, we should use both k-NN distance and probability to filter. 
   Using this predictions for manual relabeling can yield much better results potentially.
