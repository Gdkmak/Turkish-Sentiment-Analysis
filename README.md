## Sentiment Analysis for Movies review (Turkish) 

This project shows a comparison between three different approaches to train and test sentiment classification model to identify people’s opinions in Turkish language and label them as positive or negative, based on the emotions people’s express within them. Two Machine Learning techniques are namely Logistic Regression and Naive Bayes. Then Neural Networks using Embedding and 1D convolutional layers. Finally, the state-of-the-art pretrained BERT model is used to yield the best results. the following figures are yield from testing the models. 

Preprocessing|Naive Bayer|Logistic Regression|Deep Learning|BERT
---|----|----|----|-----
No |87%|88%|88%|95%
Yes|82%|82%|83%|88%

### How to use it: 
There is a shell script `run.sh` that makes it easy to run python scripts. It has commnds to train and test three applications.

- Machine Learning:  

The first app is the machine learning model (logistic regression and naive bayes). In the shell script, uncomment the training command then 
in your terminal run either one of these commands (depends on which ML techniques you want to use to train your model) `sh run.sh LR` or `sh run.sh NB`
Once the modle is done training you can run the app and test your results `app.py`. Here **Flask** comes to the play. It will create an instance of web application that starts on (defulat) `localhost:5000`. Navigate to `predict` route and add two parameters `model` and `sentence` you need to test. The url will be like this `localhost:5000/predict?sentencce=YOUR SENTENCE&model=LR or NB`  

The result yielded is probabiliy of a given sentencce to be positive and negative 

![results](image/probability.png)

- Deep Learning: 

Similar to above but you don't have any parameters to assign the command. Once you run the `app.py` test your model by navigating `localhost:5000/predict?sentencce=YOUR SENTENCE`


- BERT:

Similar to the deep learning, 

***Note:*** We are using CUDA to run the code on GPU, you need to setup your computer to be GPU enabled with CUDA and CuDNN.


### Technologies used:
TensorFlow, Keras, sklearn, Transformer, Torch, Flask   
