# ontram_pytorch

This repo contains the code to implement ordinal neural network transformation models (ONTRAMs) in Pytorch. ONTRAMs are interpretable deep learning based models which can input structured and unstructured data while they provide interpretable parameter estimates for the respective input. A detailed introduction into ONTRAMs can be found in the paper ["Deep and interpretable regression models for ordinal outcomes"](https://www.sciencedirect.com/science/article/pii/S003132032100443X).

# Project structure

The folder `./ontram_pytorch` contains all the functions/classes to build, train and evaluate an ontram:

- <a href="https://github.com/liherz/ontram_pytorch/ontram_pytorch/fit_ontram.py">`fit_ontram.py`</a>: Function to fit multiple ontrams with up to two imaging and one tabular dataset.
- <a href="https://github.com/liherz/ontram_pytorch/ontram_pytorch/predict_ontram.py">`predict_ontram.py`</a>: Function to predict a probability density function, classes etc.
- <a href="https://github.com/liherz/ontram_pytorch/ontram_pytorch/helper_predict.py">`helper_predict.py`</a>: Helper functions for the predict_ontram function.
- <a href="https://github.com/liherz/ontram_pytorch/ontram_pytorch/helper_predict.py">`helper_predict.py`</a>: Helper functions for the predict_ontram function.
- <a href="https://github.com/liherz/ontram_pytorch/ontram_pytorch/loss.py">`loss.py`</a>: Negative log likelihood for ordinal outcomes.
- <a href="https://github.com/liherz/ontram_pytorch/ontram_pytorch/metrics.py">`metrics.py`</a>: Some simple metrics for analysis such as accuracy, AUC, sensitivity and specificity.
- <a href="https://github.com/liherz/ontram_pytorch/ontram_pytorch/ontram_models.py">`ontram_models.py`</a>: Neural network building blocks like simple and complex intercepts and the class of ordinal neural network transformation models to combine the different building blocks.

The two notebooks `test_implementation_cifar10.ipynb` and `test_implementation_wine.ipynb` present to simple cases to test the implementation on imaging or tabular data. Other examples of implementations with keras can be found, for example in the github repos of our [xAI Paper](https://github.com/liherz/xAI_paper) or the application to [Functional outcome prediction of DL vs. neurologists](https://github.com/liherz/functional_outcome_prediction_dl_vs_neurologists).
