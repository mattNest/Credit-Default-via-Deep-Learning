# Credit Default Prediction via Deep Time Series Features Extraction

Traditionally, banks use **non-time dependency** models, such as linear models, to deal with credit scoring problems. However, the models would not be robust for more complicated real-world data and applications. Thus, we would like to leverage time-related deep learning models to provide better insights for this problem.

<br> We use **Default of Credit Cards Clients Dataset** collected by the UCI Machine Learning Repository. This dataset contains information on default payments,demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005. We would like to understand the features' insights before we put into the deep learning models. Thus, with features’ names for each columns, we believe this dataset would provide us more insights than most of other datasets that only have anonymous features.

<br> Then, we plot the heatmap for all the features to check the correlations
between each feature. We would examine the heatmap by two categories
that stated above ( static features and dynamic features). Apparently,
some of the dynamic features are highly correlated with time ( the
darker part in the heatmap ). Thus, we would try to leverage this part by
applying some of the sequential deep learning models such as RNN/LSTM
later in the model implementation part.

<br> ![img](https://i.imgur.com/pess7uV.png)

<br> The architecture of our hybrid model: 
![img](https://i.imgur.com/pei6Ef7.png)

We use nn.BCEWithLositsLoss from PyTorch. This loss combines a Sigmoid later and the BCELoss into one single
class. Besides, we also pass in tensor pos_weight = [3.54] into the loss
function to solve the imbalance data problem. This approach would simply adjust the weight, or you can say the gradient, of the model to
put more weight on the insufficient label.
![img](https://i.imgur.com/ftsEHIw.png)

<br> The architecture of using LSTM to extract latent from dynamic features
![img](https://i.imgur.com/ep2Jdpr.png)

### For more detailed descrpition, please refer to the report in this repo.
