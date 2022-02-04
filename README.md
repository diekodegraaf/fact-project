# Assessing the Reliability of Word Embedding Gender Bias Measures


**Prepare Folders**
```shell
mkdir data/embed/
mkdir data/results/
```

**Requirements**

Our experiments are performed on ```Python 3.7```. 
```shell
conda create -n reliability_bias python=3.7
conda activate reliability_bias
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Download and Preprocess Data**

Follow the instructions in ```data/train_corpora/```.

**Train Word Embeddings** 

1. Skip-gram with Negative Sampling  
We use 48 threads as default. You can change it to fit your own machine.
   
```shell
python train_sgns.py --corpus wikitext --num_threads 48
python train_sgns.py --corpus reddit_ask_science --num_threads 48
python train_sgns.py --corpus reddit_ask_historians --num_threads 48
```

2. GloVe
First clone the repository from GitHub and compile
```shell
git clone https://github.com/stanfordnlp/glove
cd glove && make
```
Move ```gloveW.sh``` to the ```glove``` folder. 
Use a vector size of 300 and run it to train 32 different models.
Save the embeddings of WikiText-103, r/AskScience, and r/AskHistorians 
at ```EMBEDDING_FOLDER/glove```. 
For ```EMBEDDING_FOLDER``` see ```embed_folders``` in ```paths.py```. 

**Calculate Word Embedding Gender Bias Scores**

After training word embeddings, 
we calculate gender bias scores of words regarding each word embedding model.
```shell
%SGNS
python calc_bias_scores.py --embed_folder data/embed/wikitext-103/sgns --vocab_path data/embed/wikitext-103/vocab.txt --embed_type sgns --base_pair_type gender --bias_score_path data/results/bias_scores/wikitext-103/sgns.pkl 
python calc_bias_scores.py --embed_folder data/embed/wikitext-103/glove --vocab_path data/embed/wikitext-103/vocab.txt --embed_type glove --base_pair_type gender --bias_score_path data/results/bias_scores/wikitext-103/glove.pkl
python calc_bias_scores.py --embed_folder data/embed/reddit/askscience/sgns --vocab_path data/embed/reddit/askscience/vocab.txt --embed_type sgns --base_pair_type gender --bias_score_path data/results/bias_scores/reddit/askscience/sgns.pkl
python calc_bias_scores.py --embed_folder data/embed/reddit/askscience/glove --vocab_path data/embed/reddit/askscience/vocab.txt --embed_type glove --base_pair_type gender --bias_score_path data/results/bias_scores/reddit/askscience/glove.pkl
python calc_bias_scores.py --embed_folder data/embed/reddit/askhistorians/sgns --vocab_path data/embed/reddit/askhistorians/vocab.txt --embed_type sgns --base_pair_type gender --bias_score_path data/results/bias_scores/reddit/askhistorians/sgns.pkl
python calc_bias_scores.py --embed_folder data/embed/reddit/askhistorians/glove --vocab_path data/embed/reddit/askhistorians/vocab.txt --embed_type glove --base_pair_type gender --bias_score_path data/results/bias_scores/reddit/askhistorians/glove.pkl
```
Optionally calculate scores for sexual orientation scores of words regarding each word embedding model.

```shell
python calc_bias_scores.py --embed_folder data/embed/wikitext-103/sgns --vocab_path data/embed/wikitext-103/vocab.txt --embed_type sgns --base_pair_type sexual_orientation --bias_score_path data/results/bias_scores/wikitext-103/so_sgns.pkl 
python calc_bias_scores.py --embed_folder data/embed/wikitext-103/glove --vocab_path data/embed/wikitext-103/vocab.txt --embed_type glove --base_pair_type sexual_orientation --bias_score_path data/results/bias_scores/wikitext-103/so_glove.pkl
python calc_bias_scores.py --embed_folder data/embed/reddit/askscience/sgns --vocab_path data/embed/reddit/askscience/vocab.txt --embed_type sgns --base_pair_type sexual_orientation --bias_score_path data/results/bias_scores/reddit/askscience/so_sgns.pkl
python calc_bias_scores.py --embed_folder data/embed/reddit/askscience/glove --vocab_path data/embed/reddit/askscience/vocab.txt --embed_type glove --base_pair_type sexual_orientation --bias_score_path data/results/bias_scores/reddit/askscience/so_glove.pkl
python calc_bias_scores.py --embed_folder data/embed/reddit/askhistorians/sgns --vocab_path data/embed/reddit/askhistorians/vocab.txt --embed_type sgns --base_pair_type sexual_orientation --bias_score_path data/results/bias_scores/reddit/askhistorians/so_sgns.pkl
python calc_bias_scores.py --embed_folder data/embed/reddit/askhistorians/glove --vocab_path data/embed/reddit/askhistorians/vocab.txt --embed_type glove --base_pair_type sexual_orientation --bias_score_path data/results/bias_scores/reddit/askhistorians/so_glove.pkl
```

**Estimate Reliability and Run Experiments** 

Run ```reliability_analyses.ipynb``` 
after you have calculated word embedding gender bias scores. 

If you want to train your own word embeddings and run reliability estimation and analyses, 
please refer to ```reliability_metrics.py```. 
```ReliabilityEstimator``` can help you get the job done. 

**Regression Analyses**  

See ```mlr/```. 


**Citation**

If you find this repository useful, please consider citing our paper
```
@inproceedings{du-etal-2021-assessing,
    title = "Assessing the Reliability of Word Embedding Gender Bias Measures",
    author = "Du, Yupei  and
      Fang, Qixiang  and
      Nguyen, Dong",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.785",
    pages = "10012--10034",
    abstract = "Various measures have been proposed to quantify human-like social biases in word embeddings. However, bias scores based on these measures can suffer from measurement error. One indication of measurement quality is reliability, concerning the extent to which a measure produces consistent results. In this paper, we assess three types of reliability of word embedding gender bias measures, namely test-retest reliability, inter-rater consistency and internal consistency. Specifically, we investigate the consistency of bias scores across different choices of random seeds, scoring rules and words. Furthermore, we analyse the effects of various factors on these measures{'} reliability scores. Our findings inform better design of word embedding gender bias measures. Moreover, we urge researchers to be more critical about the application of such measures",
}
```

**Contact original authors** 

If you have questions/issues, 
either open an issue or contact Yupei Du (y.du@uu.nl) directly. 
