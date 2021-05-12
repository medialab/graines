# graines
Classification of Twitter users using multimodal embeddings

## Download the code
```
git clone https://github.com/medialab/graines.git
cd graines
```

## Install requirements
* create a virtual environment with python 3.8
* activate it
* run `pip install -r requirements.txt`

## Create the ground truth
* locate the `non_graines_metadata.csv` and `graines_metadata.csv` files inside the `graines` repo
* run `python create_ground_truth.py`
* the ground truth is saved in a csv file : "`graines_et_non_graines.csv`". 
The seeds get the label 1 and the non-seeds the label 0.

## Create your own embeddings
Have a look at the [tfidf_on_descriptions.py](https://github.com/medialab/graines/blob/main/tfidf_on_descriptions.py) file: the matrix should be saved
as a `name_of_your_embedding_model.npy` matrix, and have exactly 411 rows. 
The vectors corresponding to each user should be in the same order as the users in `graines_et_non_graines.csv`.
You can run `python tfidf_on_descriptions.py` to get an example of the embedding matrix.

## Run the test
`python main.py --model name_of_your_embedding_model` (without .npy in the name of the model)
The results are run 5 times with a different train/test split. 

To save a complete report to [results_binary_classif.csv](https://github.com/medialab/graines/blob/main/results_binary_classif.csv),
run 

`python main.py --model name_of_your_embedding_model --report`

To try a different classifier, run 

`python main.py --model name_of_your_embedding_model --classifier SVM_RBF_kernel`



## Push your code and results
The [.gitignore](.gitignore) file should prevent you from loading the users personnal data or any Twitter data we collected.
```
git add .
git commit -m "name of your commit"
git push
```
