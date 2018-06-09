import numpy as np

x_train = np.array([i for i in range(10)])
y_train = np.array([i+10 for i in range(10)])

def gen():
    print('generator initiated')
    idx = 0
    
    while True:
        yield x_train[idx], y_train[idx]
        print('generator yielded a batch %d' % idx)
        idx += 1

test = gen()

#print(next(test))
#print(next(test))
#print(next(test))
#print(next(test))

from utils import *

gen = get_batches(x_train, y_train, batch_size=2)

#print(next(gen))
#print(next(gen))
#print(next(gen))
#print(next(gen))
#print(next(gen))
#x, y = get_batches(x_train, y_train, batch_size=4)
#print(x)
#print(y)


ids = fashon_parsing_data_ids()

print(ids[:23])

#gen = train_img_generator()
#next(gen)
#next(gen)
batch_size = 16
train_steps, train_batches = batch_iter(batch_size)

print("start ")
print("_"*50)
print(next(train_batches))
print(next(train_batches))
print(next(train_batches))
print(next(train_batches))
print(next(train_batches))
print(next(train_batches))
print(next(train_batches))
print(next(train_batches))
print(next(train_batches))
print(next(train_batches))
print(next(train_batches))
print(next(train_batches))
print(next(train_batches))
print(next(train_batches))
print(next(train_batches))
print(next(train_batches))
