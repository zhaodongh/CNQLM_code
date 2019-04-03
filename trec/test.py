import numpy as np 
import math
embedding = np.zeros((30, 50))
for i in range(30):
        embedding[i] = np.random.uniform(+((2*math.pi)/30)*i, +((2*math.pi)/30)*(i+1), 50)
print(embedding)