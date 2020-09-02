## Sentiment Analysis for Movies review (Turkish) 

This project shows a comparison between three different approaches to train and test sentiment classification model to identify people’s opinions in Turkish language and label them as positive or negative, based on the emotions people’s express within them. Two Machine Learning techniques are namely Logistic Regression and Naive Bayes. Then Neural Networks using Embedding and 1D convolutional layers. Finally, the state-of-the-art pretrained BERT model is used to yield the best results. the following figures are yield from testing the models. 

Preprocessing|Naive Bayer|Logistic Regression|Deep Learning|BERT
---|----|----|----|-----
No |87%|88%|88%|95%
Yes|82%|82%|83%|88%

### How to use it: 
There is a shell script `run.sh` that makes it easy to run python scripts. It has commnds to train and test three applications. 
The first app is the machine learning model (logistic regression and naive bayes). In the shell script, uncomment the training command then 
in your terminal run either one of these commands depend which techniques you want to use to train your model `sh run.sh LR` or `sh run.sh NB`
One the modle is done training you can run the app and test your results. Once you run the `app.py`





### Technologies used:
TensorFlow, Keras, sklearn, Transformer, Torch, Flask   
