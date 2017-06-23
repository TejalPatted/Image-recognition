# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 23:32:21 2017

@author: Tejal
"""
import numpy as np
import random as rd

# Read pgm file

def read_pgm(input_file):

    with open(input_file, 'rb') as c:
    #op=open(c)
        f=c.readline()
        f=c.readline()
        (width, height) = [int(j) for j in c.readline().split(' ')]
        depth=int(c.readline())
        endian='>'
        pgm=np.fromfile(c,dtype='u1'if depth < 256 else endian+'u2')
        pgm = [int(str(x)) / float(depth) for x in pgm.tolist()]
    return pgm

def read_list(f_name):
    images=[]
    label=[]

    with open (f_name) as op:
        for i in op.readlines():
            label.append(i)
            i=i.strip()
            images.append(read_pgm(i))
    img_data = np.array(images)
    labels = np.array(label)
    return (img_data,labels)
                 
   
# Create the network structure using the specifications given - of number of neurons needed in hidden and output layer
def create_nwk(n_hid, n_out):
    network = list()
    
    hid_lay = [{"weights":[rd.uniform(-1,1) for i in range(len(img_data[0]))] } for j in range(n_hid)]
    network.append(hid_lay)                       
    out_lay = [{"weights":[rd.uniform(-1,1) for i in range(n_hid)] } for j in range(n_out)]
    network.append(out_lay)
    return(network)                           

# Compute the predicted output    
def fwd_bias(rec,nwk,flag):
    inp_list = []
    hid_out = []
    inp_list.append(rec)
#Find output of each neuron in hidden layer
    for i in nwk[0]:
        wts = np.array(i['weights'])
#        X*wts
        sum_xw = np.dot(wts,rec.T)
#        Output after applying sigmoid function 
        sig_out = 1/(1+np.exp(-sum_xw))
        hid_out.append(sig_out)
# Hidden layer outputs        
    inp_list.append(np.array(hid_out))
        
# Output layer output       
    out_sum_xw = np.dot(nwk[1][0]['weights'],inp_list[1].T)
#    print(out_sum_xw)
    sig_out = 1/(1+np.exp(-out_sum_xw))

# Append output layer output to inp_list
    inp_list.append(sig_out)
    if flag == 0:
        return(inp_list)
    else:
        return(inp_list[2])
        
    

# Find the actual output using the label of the record
# If label has down -> output = 1 else 0    
def find_label(l):

    if 'down' in l:
        return 1
    else:
        return 0
        
def calc_der(pred_out):
    return (pred_out*(1-pred_out))
    
# Backpropogate the error to find new weights        
def back_prop(act_out,inp_list,nwk,l_rate):
#    w_new_out = []
    wt_new_list = []
        
#   Base case error -> error of output layer = 2(x-y)*derivative
    e1 = (inp_list[2]-act_out)*calc_der(inp_list[2])
    
#   Weighted error for hidden layer neuron 
    e2 = np.array(nwk[1][0]['weights'])*e1

## Adjust weights between input and hidden layer 
## j indicates the neuron in hidden layer
    for j in range(len(nwk[0])):
        wt_new_list = nwk[0][j]['weights']-l_rate*e2[j]*calc_der(inp_list[1][j])*np.array(inp_list[0])
        nwk[0][j]['weights']=wt_new_list
        wt_new_list = []

#    Adjust weights between hidden & output layer
    nwk[1][0]['weights'] = np.array(nwk[1][0]['weights']) - (e1 * np.array(inp_list[1])*l_rate)
    

    return nwk
        
# Compute weights of the network for n_epochs using SGD approach    
def compute_wts(n_epoch,l_rate):
    nwk = create_nwk(100,1)
    for i in range(n_epoch):
        for rec_num,img in enumerate(img_data):
     
            inp_list = fwd_bias(img,nwk,0)
   
    # actual label
            l = list_data[rec_num].lower().split('_')
            act_out = find_label(l)
            nwk = back_prop(act_out,inp_list,nwk,l_rate)
        if i%100 == 0:
            print "In iteration -" , i
    return nwk
    
# Predict the label
def predict(test_data,labels):
    output = [["Record number","Actual label","Predicted label"]]
    acc = 0
    error = 0.0
    for i,rec in enumerate(test_data):

        pred_out = fwd_bias(rec,nwk,1)
        l = labels[i].lower().split('_')
        act_out = find_label(l)
        error+= (act_out-pred_out)**2
        act_out = "Down" if act_out == 1 else "Not Down"
        if pred_out>0.5:
            pred_out = "Down"
            o1 = [i,act_out,pred_out] 
        else:
            pred_out = "Not Down"
            o1 = [i,act_out,pred_out]
        output.append(o1)
        if pred_out == act_out:
            acc+=1
    error = error**(1/2)/float(len(test_data))
    acc = acc/float(len(test_data))*100
    output = np.array(output)
    return(output,acc,error)
            
        
        
        
# Read training data     
img_data,list_data= read_list('downgesture_train.list')
print "Data read completed"
nwk = compute_wts(1000,0.1)
print "Weights computation completed"
test_img,test_labels = read_list('downgesture_test.list')
print "Test data read completed"
out,accuracy,error = predict(test_img,test_labels)
print "Prediction completed"
print "Accuracy = %.2f"%accuracy,"%"
print("Least square error = %f" %error)
print ("Labels for test data")
print out





                          
   
    

    
