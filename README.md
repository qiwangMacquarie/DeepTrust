# DeepTrust

## Overview
This repository is Python implementation of the method proposed in "DeepTrust: A Deep User Model of Homophily Effect for Trust Prediction ", Qi Wang, Weiliang Zhao, Jian Yang, Jia Wu, Wenbin Hu, Qianli Xing, ICDM 2019. It is constituted of the follows sections:
input/ contains example data of Epinions;
src/ contains the implementation of the proposed DeepTrust method.


## Requirements
 - python == 3.7
 - numpy == 1.18.1 (We recommend you to use [Anaconda](https://anaconda.org/anaconda/numpy).)
 - tensorflow == 1.14.0 (https://github.com/tensorflow/tensorflow)
 - networkx == 2.4


## Usage
### Input and options

- Input
  * We store the input file in database and the structure of the input file is: (Trustor, Trustee, Trust Value).
  Some examples of input file are as follows:
Trustor     | Trustee       | Trust Value                         
      ----------    |------------|------------------------------------ 
      1     | 2        | 0           
      2     | 3        | 1
      3     | 4        | 0                     
      4     |5         | 1       
               
 
  The first two columns indicates the IDs of user pairs. The third column is the trust values between user pairs, 
  with value "1" indicating there are trust relations between users while value "0" indicating without trust relations.
  You can set an input graph in different forms as you want such as .txt format.

- Options
  * We present a table of brief explanations on main parameters. 

      Parameter     | Type       | Explanation                         | Default
      ----------    |------------|------------------------------------ |-------------
      --niter       | INT        | Number of epochs to train           |5
      --batch_size  | INT        | Size of batch to train in each epoch|5000
      --lr          | FLOAT      | Learning rate                       |0.1
      --decay       | FLOAT      | Deacy speed of learning rate        |1.1
      --negNum      | INT        | Number of negative instances        |5
      --save_dir    | PATH       | Path to save runing log             |log/

   
## Demo Examples
Train our DeepTrust model on the deafult Epinions dataset, output the performance on trust prediction task. 
For trust prediction task, we use widely used prediction accurcy as evaluation metric. 


We run python src/main.py with default epoch number (5) and test ratio (20%), and the result is as follows:

	begin to train the model at Fri Jun 26 10:49:13 2020
	Creating negative instaces finished: 47970.70s
	Loading total training trust pairs finished: 784.53s
	load data done at Sat Jun 27 00:21:48 2020
	Epoch #0  Finished 
	Epoch #0  | Trust Prediction Accuracy 05: 0.482533
	Epoch #0  | Trust Prediction Accuracy 06: 0.482533
	Epoch #0  | Trust Prediction Accuracy 07: 0.451233
	Epoch #1  Finished 
	Epoch #1  | Trust Prediction Accuracy 05: 0.490467
	Epoch #1  | Trust Prediction Accuracy 06: 0.490433
	Epoch #1  | Trust Prediction Accuracy 07: 0.444800
	Epoch #2  Finished 
	Epoch #2  | Trust Prediction Accuracy 05: 0.494967
	Epoch #2  | Trust Prediction Accuracy 06: 0.494667
	Epoch #2  | Trust Prediction Accuracy 07: 0.436800
	Epoch #3  Finished 
	Epoch #3  | Trust Prediction Accuracy 05: 0.496633
	Epoch #3  | Trust Prediction Accuracy 06: 0.495467
	Epoch #3  | Trust Prediction Accuracy 07: 0.432133
	Epoch #4  Finished 
	Epoch #4  | Trust Prediction Accuracy 05: 0.497400
	Epoch #4  | Trust Prediction Accuracy 06: 0.495567
	Epoch #4  | Trust Prediction Accuracy 07: 0.429367
	Epoch #5  Finished 
	Epoch #5  | Trust Prediction Accuracy 05: 0.497767
	Epoch #5  | Trust Prediction Accuracy 06: 0.495500
	Epoch #5  | Trust Prediction Accuracy 07: 0.426967
	Finish at Sat Jun 27 01:11:19 2020

## Baselines
In our paper, we used the following methods for comparison:
(1) TP: Propagation of trust and distrust.
(2) MF: Combining content and link for classification using matrix factorization.
(3) hTrust: Exploiting homophily effectfor trust prediction. Source: https://www.cse.msu.edu/~tangjili/trust.html.
(4) Power-law: Power-law distribu-tion aware trust prediction. Source: https://github.com/ZW-ZHANG/Powerlaw_TP.


## Cite
If you find this repository useful in your research, please cite our paper:   
@inproceedings{wang2019deeptrust,
  title={DeepTrust: A Deep User Model of Homophily Effect for Trust Prediction},
  author={Wang, Qi and Zhao, Weiliang and Yang, Jian and Wu, Jia and Hu, Wenbin and Xing, Qianli},
  booktitle={2019 IEEE International Conference on Data Mining (ICDM)},
  pages={618--627},
  year={2019},
  organization={IEEE}
}








