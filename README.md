# Python code for the 4th China Physiological Signal Challenge 2021

## What's in this repository?

We implement a classifier based on deep learning, which features RR interval of ECG lead signal. The code uses two main scripts (described below) to run and test the 2021 challenge's algorithm.

## How do I run these scripts?

You can run this baseline method by installing the requirements

    pip install requirements.txt

and running 

    python entry_2021.py <data_path> <result_save_path>

where '<data_path>' is the folder path of the test set, '<result_save_path>' is the folder path of your detection results. 

## How do I run my code and save my results?

After obtaining the test results, you can evaluate the scores of your method by running

    python score_2021.py <ans_path> <result_save_path>

where '<ans_path>' is the folder save the answers, which is the same path as '<data_path>' while the data and annotations are stored with 'wfdb' format. '<result_save_path>' is the folder path of your detection results.
