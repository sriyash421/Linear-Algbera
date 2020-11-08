import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tqdm import trange

def convolve2d(A,B) :
    A_ = A.flatten()
    n = A.shape[1]
    toeplitz_row = np.zeros(A_.shape[0])
    for i in trange(B.shape[0]) :
        toeplitz_row[i*n:i*n+B.shape[1]] = B[i]
    #Creating the row for toeplitz matrix
    A_conv  = np.zeros(A_.shape[0])
    # Row-by-row Matrix multiplication, as toeplitz matrix is very large to load together
    for i in trange(A_.shape[0]) :
        temp_row = np.zeros(A_.shape[0])
        temp_row[i:] = toeplitz_row[0:A_.shape[0]-i]
        A_conv[i] = np.dot(temp_row, A_)
    A_conv = A_conv.reshape(A.shape[0],n)
    return A_conv
    
labels = ["Original","1a. Convoluted","1b. Convoluted","1c. Convoluted","2. Convoluted","3. Convoluted"]
D = cv2.cvtColor(imread("dog.jpg"), cv2.COLOR_BGR2GRAY)
C = cv2.cvtColor(imread("cat.jpg"), cv2.COLOR_BGR2GRAY)
#required kernels
B_list = [np.ones((i,i))/(i*i) for i in [5,10,20]]
temp = np.array([[1,-1]])
B_list.extend([temp, temp.T])

#results
cat_results = [C]
dog_results = [D]

for (B,label) in zip(B_list,labels):
    print(B)
    #2D convolution
    DB = convolve2d(D,B)
    CB = convolve2d(C,B)
    cat_results.append(CB)
    dog_results.append(DB)

#plotting the results
fig, axs = plt.subplots(2,6,figsize=(32,32))
plt.gray()
for i,ax in enumerate(axs.T.flat) :
    if i<6:
        ax.imshow(dog_results[i], cmap='gray')
        ax.axis('off')
        ax.set_title(f'{labels[i]} Dog', fontsize=10)
    else:
        ax.imshow(cat_results[i-6], cmap='gray')
        ax.axis('off')
        ax.set_title(f'{labels[i-6]} Cat', fontsize=10)
        
plt.tight_layout(pad=2)
plt.show()
