# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 11:59:32 2018

@author: 45016577
"""

import pickle
import pymysql
import numpy as np
import random
from   gensim.models.doc2vec import Doc2Vec
import tensorflow as tf
import argparse
import logging 
import time
import networkx as nx



connect = pymysql.Connect(
    host='localhost',
    port=3306,
    user='root',
    passwd='123456',
    db='epinions_experiment',
    charset='utf8'
)
def get_user_related_itemID():
    user_related_itemID = []
    cursor2 = connect.cursor()
    sql ="SELECT * FROM user_id order by user_id"
    cursor2.execute(sql)
    for row in cursor2.fetchall():
        if row[0] is not None:
            user_related_itemID.append([])
            continue
        cursor2.close()
    cursor3=connect.cursor()
    sql ="SELECT * FROM user_review_more_than_30"
    cursor3.execute(sql)
    for row in cursor3.fetchall():
        # row[0]: usre_id,  row[3]:item_id  
        item_id  = row[3]
        if row[0]<=len(user_related_itemID):       
            user_related_itemID[row[0]-1]=user_related_itemID[row[0]-1]+[item_id]
    cursor3.close()
    user_related_itemID = np.array(user_related_itemID)
    return user_related_itemID

# user PMF feature
fr = open('user_PMF_train_data.txt','rb') 
user_PMF_data = pickle.load(fr)
fr.close()

# user_related_item_feature
fr1 = open('item_PMF_train_data.txt','rb')  
item_PMF_data = pickle.load(fr1)
fr1.close()


item_model = Doc2Vec.load("item_doc2vec_item_size21661.model")

item_list = get_user_related_itemID() 
sorted_item_list = [] 
for item in item_list:
    new_item = list(set(item))
    new_item.sort(key=item.index)
    sorted_item_list.append(new_item) 
sorted_item_list = np.array(sorted_item_list)  
i = 0
user_item_feature = [] 
for user_id in range(7151):
    user_related_item_feature = float(0)
    for item_id in sorted_item_list[user_id]: # trustor_related_item feature
        user_related_item_feature += item_PMF_data[item_id-1]
        user_related_item_feature += item_model.docvecs[item_id-1]
        i = i+2    
    user_related_item_feature = user_related_item_feature/i  
    user_item_feature.append(user_related_item_feature)  
    
#load doc2vec model to get review feature

model = Doc2Vec.load("Doc2vec_Users_plus_Items_Epinions_category_USize7151.model")

# load dataset and add negative sample

user_size = 7151
negNum = 2 #5
trustor_set = []
total_trust_pair=[]
train_trust_pair = []
trust_pair = []
# load train trust pair and negative instances

def load_trust_pair(args):
    global total_trust_pair
    global train_trust_pair
    start = time.time()
    cursor1=connect.cursor()
    sql ="SELECT * FROM user_trust_relation_time order by time limit 90000"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    cursor1.execute(sql)
    for row in cursor1.fetchall():                                                 
        trustor = row[1]
        trustee = row[3]
        trustor_set.append(trustor)
        trust_pair.append([trustor,trustee])
        total_trust_pair.append([trustor,trustee,1])
        train_trust_pair.append([trustor,trustee,1])
    cursor1.close()
    G = nx.DiGraph()
    G.add_edges_from(trust_pair)
    for i in G.nodes():
        for j in G.nodes():
            if nx.has_path(G,i,j):
                #a = nx.shortest_path(G,i,j)
                b = nx.shortest_path_length(G,i,j)
                if b == 2 and G.degree(j) >80:
                    trustor_set.append(i)
                    train_trust_pair.append([i,j,1])
    global length
    length = len(train_trust_pair)
    end = time.time()
    
    for trustor in trustor_set:
        neglist=[]
        for t in range(negNum):
            j = np.random.randint(user_size+1)
            while [trustor,j] in total_trust_pair or j in neglist:
                j = np.random.randint(user_size+1)
            neglist.append(j)
            total_trust_pair.append([trustor,j,0])
            train_trust_pair.append([trustor,j,0])
    print("Creating negative instaces finished: %.2fs" % (time.time() - end))
    print("Loading total training trust pairs finished: %.2fs" % (end - start))
# load test pair and negative instances
test_trust_pair = []
test_trustor_set = []
negNum_test = 5
cursor4 = connect.cursor()
sql = "SELECT pair_id,trustor_id,trustee_id FROM user_trust_relation_time WHERE pair_id not in (select pair_id from user_trust_relation_time_80) limit 30000"
cursor4.execute(sql)
for row in cursor4.fetchall():
    trustor = row[1]
    trustee = row[2]
    test_trustor_set.append(trustor)
    test_trust_pair.append([trustor,trustee,1])
cursor4.close()

    
for test_trustor in test_trustor_set:
    test_neglist=[]
    for r in range(negNum_test):
        m = np.random.randint(user_size+1)
        while [test_trustor,m] in test_trust_pair or m in test_neglist:
            m = np.random.randint(user_size+1)
        test_neglist.append(m)
        test_trust_pair.append([test_trustor,m,0])

'''

def load_trust_pair(args): #load training set
    global train_trust_pair
    fr = open('trust_data_90_train.txt','rb') 
    train_trust_pair = pickle.load(fr)
    fr.close()
    global length
    length = len(train_trust_pair)
fr1 = open('trust_data_90_test.txt','rb')
test_trust_pair = pickle.load(fr1)
fr1.close()
'''
class Dataset:
    def __init__(self, args):
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.batch_size = args.batch_size
        self.total_trust_pair = total_trust_pair
        
        self.train_trust_pair = train_trust_pair
        self.test_trust_pair = test_trust_pair
        
        self.selected_pair = []
        self.total_trust_pair_size = len(self.total_trust_pair)
        self.flag = False
        
        self.test_pair = []
    def next_batch(self):
        # select a minibatch of trustor-trustee pairs 
        start = self.index_in_epoch  
        if self.epochs_completed == 0 and start == 0:
            random.shuffle(self.train_trust_pair)             
        if start >= length*3:
            self.epochs_completed +=1
            self.selected_pair = self.test_trust_pair
            self.index_in_epoch =0
            random.shuffle(self.selected_pair)
        else:
            self.index_in_epoch += self.batch_size 
            end = self.index_in_epoch
            self.selected_pair = self.train_trust_pair[start:end]
        trustor_PMF_in = [] 
        trustee_PMF_in = []
        trustor_PMF_item = []
        trustee_PMF_item = []
        trustor_review_in = []
        trustee_review_in = []
        
        trust_label_in = []
        
        positive_pair = []
    
        for trust_pair in self.selected_pair:
            trustor_id = trust_pair[0]
            trustee_id = trust_pair[1]
            trust_label = trust_pair[2]
            
            if trust_label == 1:
                positive_pair.append([trustor_id,trustee_id])
                
            
            trustor_feature = user_PMF_data[trustor_id - 1]
            trustee_feature = user_PMF_data[trustee_id - 1]
            
            trustoritem = user_item_feature[trustor_id - 1]
            trusteeitem = user_item_feature[trustee_id - 1]
            
            trustor_text_feature = model.docvecs[trustor_id - 1]
            trustee_text_feature = model.docvecs[trustee_id - 1]
            
            trustor_PMF_in.append(trustor_feature) 
            trustor_PMF_item.append(trustoritem)
            trustor_review_in.append(trustor_text_feature)
            
            trustee_PMF_in.append(trustee_feature) 
            trustee_PMF_item.append(trusteeitem)
            trustee_review_in.append(trustee_text_feature)
            
            trust_label_in.append(trust_label)
        return self.epochs_completed,self.index_in_epoch,trust_label_in,trustor_PMF_in,trustee_PMF_in,trustor_PMF_item,trustee_PMF_item,trustor_review_in,trustee_review_in,self.selected_pair,positive_pair
        
def main(args):
    # log
    logging.basicConfig(filename="log".format(time.strftime("%m-%d_%H_%M_%S", time.localtime())), 
    level=logging.INFO,format='%(asctime)s %(message)s\t',datefmt='%Y-%m-%d %H:%M:%S')    
    logging.info('begin to load data')
    print ('begin to train the model at ' + time.asctime())
    load_trust_pair(args) 
    dataset = Dataset(args)  
    
    print ('load data done at ' + time.asctime())
    logging.info('load data done')
    
    

    TRIGRAM_D = 35 
   
    L1_N = 30 
    L2_N = 25
    # input placeholder
    trustor_rating_batch = tf.placeholder(tf.float32) 
    trustor_item_batch = tf.placeholder(tf.float32)
    trustor_review_batch = tf.placeholder(tf.float32)
    
    trustee_rating_batch = tf.placeholder(tf.float32) 
    trustee_item_batch = tf.placeholder(tf.float32)
    trustee_review_batch = tf.placeholder(tf.float32)
    
    trust_relation_konwn = tf.placeholder(tf.float32)
    
    lr = tf.placeholder(tf.float32)  # learning rate
   
    # layer1
    l1_par_range = np.sqrt(6.0 / (TRIGRAM_D + L1_N)) 
    
    weight1 = tf.Variable(tf.random_uniform([TRIGRAM_D, L1_N], -l1_par_range, l1_par_range))
    bias1 = tf.Variable(tf.random_uniform([L1_N], -l1_par_range, l1_par_range))
 
    trustor_11 = tf.matmul(trustor_rating_batch, weight1) + bias1
    trustor_12 = tf.matmul(trustor_item_batch, weight1) + bias1
    trustor_13 = tf.matmul(trustor_review_batch, weight1) + bias1
    
    trustee_11 = tf.matmul(trustee_rating_batch, weight1) + bias1
    trustee_12 = tf.matmul(trustee_item_batch, weight1) + bias1
    trustee_13 = tf.matmul(trustee_review_batch, weight1) + bias1
    
    trustor_11_out = tf.nn.relu(trustor_11) 
    trustor_12_out = tf.nn.relu(trustor_12)
    trustor_13_out = tf.nn.relu(trustor_13)
    
    trustee_11_out = tf.nn.relu(trustee_11)
    trustee_12_out = tf.nn.relu(trustee_12)
    trustee_13_out = tf.nn.relu(trustee_13)
    
    #layer2   
    l2_par_range = np.sqrt(6.0 / (L1_N + L2_N))

    weight2 = tf.Variable(tf.random_uniform([L1_N, L2_N], -l2_par_range, l2_par_range))
    bias2 = tf.Variable(tf.random_uniform([L2_N], -l2_par_range, l2_par_range))

    trustor_21 = tf.matmul(trustor_11_out, weight2) + bias2
    trustor_22 = tf.matmul(trustor_12_out, weight2) + bias2
    trustor_23 = tf.matmul(trustor_13_out, weight2) + bias2
    
    trustee_21 = tf.matmul(trustee_11_out, weight2) + bias2
    trustee_22 = tf.matmul(trustee_12_out, weight2) + bias2
    trustee_23 = tf.matmul(trustee_13_out, weight2) + bias2
    
    trustor_21_out = tf.nn.relu(trustor_21) 
    trustor_22_out = tf.nn.relu(trustor_22) 
    trustor_23_out = tf.nn.relu(trustor_23) 
    
    trustee_21_out = tf.nn.relu(trustee_21)
    trustee_22_out = tf.nn.relu(trustee_22)
    trustee_23_out = tf.nn.relu(trustee_23)
    
   
    trustor_out = tf.concat([trustor_21_out,trustor_22_out,trustor_23_out],1)
    trustee_out = tf.concat([trustee_21_out,trustee_22_out,trustee_23_out],1)
    
    # 求trustor 与 trustee 的 拼接向量的 cosine similarity
    trustor_out_norm = tf.sqrt(tf.reduce_sum(tf.square(trustor_out),1)) 
    trustee_out_norm = tf.sqrt(tf.reduce_sum(tf.square(trustee_out),1))
    
    trustor_mul_trustee = tf.reduce_sum(tf.multiply(trustor_out,trustee_out),1) 
    prod = tf.multiply(trustor_out_norm,trustee_out_norm)
    
    cos_sim = tf.truediv(trustor_mul_trustee,prod) # cosine-similarity
        
   
    loss = tf.multiply(tf.reduce_sum(tf.square(cos_sim - trust_relation_konwn)),0.0001)
     
    # train step optimizer
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    
    config = tf.ConfigProto()  # log_device_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        epoch = 0
        tr_loss = 0
        step = 0

        while epoch <= args.niter :
            epochs_completed,index_in_epoch,trust_label_batch,\
            trustor_PMF_batch,trustee_PMF_batch,\
            trustor_PMF_item_batch,trustee_PMF_item_batch,\
            trustor_review_in_batch,trustee_review_in_batch,testing_pair,positive_pair_test= dataset.next_batch() 
            if epochs_completed == epoch + 1:               
                #training loss
                tr_loss /= args.trainingset_size
                print("Epoch #%-2d Finished | Train Loss: %-4.3f" % (epoch, tr_loss)) 
                logging.info('Epoch_%04d_finished\ttraing_loss = %.6f'%(epoch, tr_loss))
                tr_loss = 0               
                #test step
                test_loss,_cos_sim = sess.run([loss,cos_sim],feed_dict = {trustor_rating_batch:trustor_PMF_batch,trustor_item_batch:trustor_PMF_item_batch,
                                                 trustor_review_batch:trustor_review_in_batch,trustee_rating_batch:trustee_PMF_batch,
                                                 trustee_item_batch:trustee_PMF_item_batch,trustee_review_batch:trustee_review_in_batch,
                                                 trust_relation_konwn:trust_label_batch})
               
                finalvalue = 0
               
                for value in _cos_sim:
                    if value >=0.5:
                        finalvalue +=1
                test_size = len(test_trust_pair)
                test_loss/=test_size
                step = 0
                print("Epoch #%-2d | Test Loss: %-4.3f" %(epoch, test_loss))  
                i = 0   
            
                similarity_ordered_pair_05 = []
                similarity_ordered_pair_06 = []
                similarity_ordered_pair_07 = []
                test_pair_with_similarity = [] 
                for trust_pair in testing_pair:
                    trust_pair.append(_cos_sim[i])
                    i+=1
                    test_pair_with_similarity.append(trust_pair)
                test_pair_with_similarity = np.array(test_pair_with_similarity)
                test_pair_with_similarity_ordered = test_pair_with_similarity[np.lexsort(-test_pair_with_similarity.T)]
                test_pair_with_similarity_ordered = np.array(test_pair_with_similarity_ordered)
                k=0
                l=0
                m=0
                for index_05 in range(len(positive_pair_test)):
                      if test_pair_with_similarity_ordered[index_05][3] >=0.5:
                          k+=1
                          trustor_id = test_pair_with_similarity_ordered[index_05][0]
                          trustee_id = test_pair_with_similarity_ordered[index_05][1]
                          similarity_ordered_pair_05.append([trustor_id,trustee_id])
                for index_6 in range(len(positive_pair_test)):
                     if test_pair_with_similarity_ordered[index_6][3] >=0.6:
                          l+=1
                          trustor_id = test_pair_with_similarity_ordered[index_6][0]
                          trustee_id = test_pair_with_similarity_ordered[index_6][1]
                          similarity_ordered_pair_06.append([trustor_id,trustee_id])
                for index_7 in range(len(positive_pair_test)):
                     if test_pair_with_similarity_ordered[index_7][3] >=0.7:
                          m+=1
                          trustor_id = test_pair_with_similarity_ordered[index_7][0]
                          trustee_id = test_pair_with_similarity_ordered[index_7][1]
                          similarity_ordered_pair_07.append([trustor_id,trustee_id])
                intersection_05 = [v for v in similarity_ordered_pair_05 if np.all(v in positive_pair_test)]
                intersection_06 = [v for v in similarity_ordered_pair_06 if np.all(v in positive_pair_test)]
                intersection_07 = [v for v in similarity_ordered_pair_07 if np.all(v in positive_pair_test)]
               
                prediction_accuracy_05 = len(intersection_05)/len(positive_pair_test)
                prediction_accuracy_06= len(intersection_06)/len(positive_pair_test)
                prediction_accuracy_07 = len(intersection_07)/len(positive_pair_test)
                print("Epoch #%-2d | Trust Prediction Accuracy 05: %f" %(epoch, prediction_accuracy_05))
                print("Epoch #%-2d | Trust Prediction Accuracy 06: %f" %(epoch, prediction_accuracy_06))
                print("Epoch #%-2d | Trust Prediction Accuracy 07: %f" %(epoch, prediction_accuracy_07))
                epoch += 1
            else:
                #train step
                _, _tr_loss = sess.run([train_step,loss],feed_dict = {trustor_rating_batch:trustee_PMF_batch,trustor_item_batch:trustor_PMF_item_batch,
                                                 trustor_review_batch:trustor_review_in_batch,trustee_rating_batch:trustee_PMF_batch,
                                                 trustee_item_batch:trustee_PMF_item_batch,trustee_review_batch:trustee_review_in_batch,
                                                 trust_relation_konwn:trust_label_batch,
                                                lr:args.lr/pow(epochs_completed + 1,args.decay)})
                tr_loss+=_tr_loss
                step += 1
    print ('Finish at ' + time.asctime())            
    logging.info('done')
                
if __name__ == '__main__':
    # parse argv
    parser = argparse.ArgumentParser(description = 'trust_prediction_model')
    parser.add_argument('--data_root', action='store', dest='data_root', default='data/')
    parser.add_argument('--save_dir', action='store', dest='save_dir', default='log/')

    parser.add_argument('--niter', action='store', dest='niter', default=5, type=int)
    parser.add_argument('--trainingset_size', action='store', dest='trainingset_size', default=500000, type=int)
    parser.add_argument('--batch_size', action='store', dest='batch_size', default=5000, type=int)

    parser.add_argument('--user_size', action='store', dest='user_size', default=7151, type=int)
    
    parser.add_argument('--lr', action='store', dest='lr', default=0.1, type=float)#0.001
    parser.add_argument('--decay', action='store', dest='decay', default=1.1, type=float)
    args=parser.parse_args()
    main(args)
