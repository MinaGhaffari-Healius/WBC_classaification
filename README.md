**White blood cell classification using convolutional neural network developed**  

<<<<<<< HEAD
To train the model run train.py. The train parameters can be set in configuration.py
The best models will get saved in Models directory.  
=======
To train the model run train.py. The train parameters can be set in configuration.py  
The best models will get saved in Models directory.     
>>>>>>> cc381c9ea24cd63a3fecb4bdee5a7bc8cf42de9c
to evaluate the model and plot the confusion matrix run evaluate.py and set the model in line load_model (line 11) to be the best saved model in Model directory.   




Dataset:
[Kaggle WBC dataset](https://www.kaggle.com/datasets/paultimothymooney/blood-cells?sortBy=relevance&group=everyone&search=includeamin&page=1&pageSize=20&datasetId=9232)


**Requirements:**
keras==2.8.0 
tensorflow==2.8.1  
opencv_python==4.5.5.64  
matplotlib==3.5.2  
numpy==1.22.4  
pandas==1.4.2  
scikit_learn==1.1.1  
seaborn==0.11.2  

