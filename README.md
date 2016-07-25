# ErrorTextClassification

Use Word2vec and Neural Networks to classify the error feedback of the passenger seat screen on the plane. First use Word2vec get the vector for each word, then feed vector of each error info sentence into Nerual Nets.

## Word2vec Trained Result
* Word Vector Plot
![Word](https://github.com/coroner4817/ErrorTextClassification/raw/master/output/word_vec_plot_2016-07-24-21-41.png)
* Class Vector Plot
![Class](https://github.com/coroner4817/ErrorTextClassification/raw/master/output/class_vec_plot_2016-07-24-21-41.png)

## Error Text Training Methods

#### <code>nltk.NaiveBayesClassifier</code> Acc ~ 48%  
#### <code>Softmax Regression</code> Acc ~ 77%  
#### <code>Nerual Networks</code> Acc ~ 97%  

## Reference
* [Stanford cs224d](http://cs224d.stanford.edu/index.html)
