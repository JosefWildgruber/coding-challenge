# Run code

To run the code please setup a python 3.9 environment (for example conda environment),
go to the Project folder and run the following command to install the dependencies

pip install -r requirements.txt

Once the dependencies are installed you can run train.py

python train.py

This will run the model training on the train file in the input folder.
Once training is done, the artifacts (pipeline.pkl, scores.csv and expected_columns.csv)
will be stored under trained_models/.

Once training is completed you can use the model to predict unknown data from the test_file in the
input folder. To do so just run inference.py

python inference.py

Predictions will be stored in output/predictions.csv


# EDA

Running eda.py will create all graphics used in my documentation and save them to graphics/

python eda.py


# Documentation

Please find the documentation of my approach to solve this challenge in doc/challenge_doc.pdf