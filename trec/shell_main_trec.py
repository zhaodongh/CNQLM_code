import sys
import os
import subprocess
import time
batch_size = [80,100,120]
num_filters=[65,80]
l2_reg_lambda=[0.00001,0.0001,0.0006]
learning_rate = [0.01,0.001,0.0001,0.00001]
#filter_sizes=['40','10','30']
count = 0
for batch in batch_size:
	for num in num_filters:
		for l2 in l2_reg_lambda:
			for rate in learning_rate:
				print ('The ', count, 'excue\n')
				count += 1
				subprocess.call('python train.py --batch_size %d --num_filters %d --l2_reg_lambda %f --learning_rate %f' % (batch,num,l2,rate), shell = True)
