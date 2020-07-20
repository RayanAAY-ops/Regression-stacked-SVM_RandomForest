# Regression-stacked-SVM_RandomForest
This is my implementation of a stacked regressor using optimized SVM  and  random Forest using Optuna.The actual inputs of the combined regressor is  a latent representation  of 59 numerical inputs compressed into 3 ,extracted using an auto-encoder implemented under Keras   

My goal was to focus on the model instead of the data itself and all the visualisation/preprocessing behind .

I actually scored 0.15,(top 20%) with a very simple auto-encoder architecture and a lazy data preprocessing .

With a deeper one ,the result would probably be better. (DM me if you achieve better with a more complex architecture).

### STEPS OF EXECUTIONS ###

1.  pip install -r requirements.txt && mkdir sample

2.  python3 preprocessing.py    

3.  python3 AE_train.py

4.  python3 main.py

