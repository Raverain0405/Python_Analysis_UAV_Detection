# Python_Analysis_UAV_Detection

This is an analysis of the UAV intrusion dataset. The goal is to predict wether the drone is in intrusion from his wifi data.

Datasets and variables explained here:  http://mason.gmu.edu/~lzhao9/materials/data/UAV/


We found these variables to have a strong correlation with the target variable:
- all Size_mean variables (3 links)
- all Size_median variables (3 links)


In summary, we tested some machine learning algorithms with the selected variables above and found that these predict the target variable very well (>= 99% accuracy):
- Logistic Regression
- Support vector classifier
- Stochastic gradient descent classifier


We then generalized the process for the 3 datasets and combined them together. Surprisingly, we still got an extremely good accuracy (>= 99%)


