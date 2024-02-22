# Model architecture
## 1. Networks
### ．Two layers network

| Layers            | Details   |
|-------------------|-----------|
| Affine 1          | 100 nodes |
| ReLU 1            |           |
| Affine 2          | 50 nodes  |
| ReLU 2            |           |
| Affine            |           |
| Softmax           |           |
| Cross entropy error |           |


#### Testing accuracy : 0.896 
1. Epoch: 100  
2. Batch sizw: 200  
3. Learning rate: 0.01   
4. Optimizer: SGD

### ．Three layers network

| Layers            | Details   |
|-------------------|-----------|
| Affine 1          | 50 nodes  |
| ReLU 1            |           |
| Affine 2          | 50 nodes  |
| ReLU 2            |           |
| Affine 3          | 50 nodes  |
| ReLU 3            |           |
| Affine            |           |
| Softmax           |           |
| Cross entropy error |           |

#### Testing accuracy : 0.908
1. Epoch: 100  
2. Batch sizw: 200  
3. Learning rate: 0.005  
4. Optimizer: SGD  

### ．Network learning
Mini-batch -> Gragient -> Update parameters  
1. Randomly select a mini-batch of training data and labels.
2. Calculate the gradients.
3. Use the optimizer to update the parameters. 
 
Keep repeating these steps to adjust the weights until the epochs is reached.

## 2. Layers
### ．Affine
```js
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
```

### ．ReLU
```js
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        #x<=0:true, x>0:false
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout
```

### ．Softmax-with-Loss
```js
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True) #avoid overflow 
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
```
```js
def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)

    batch_size = y.shape[0]
    return -np.sum(t*np.log(y + 1e-7))/batch_size
```
```js
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None #loss
        self.y = None #solfmax output
        self.t = None #training data(one-hot)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size 
        return dx
```


# PCA Method
PCA is a commonly used technique for reducing the dimensionality of a dataset by transforming it into a lower-dimensional space while retaining the most important features or principal components.  
We use 1470 grayscale fruit images as training data, including three types of fruits: carambula, lychee, and pear, with 490 images of each category. Each image has 1024 pixels, and we employed PCA to reduce the 1024 features to 2 dimensions before training the neural network.
```js
def PCA_func(train_data, test_data):
    total_data = np.concatenate(((train_data, test_data)),axis=0) #1-dim
    pca = PCA(n_components=2)
    total_data = pca.fit(total_data) #train model

    train_PCA_data = pca.transform(train_data)
    test_PCA_data = pca.transform(test_data)
    return train_PCA_data, test_PCA_data
```
```js
def print_pca(data,label):
    pca = PCA(n_components=2)
    markers = ['.', '.', '.']
    colors = ('red', 'blue', 'orange')
    classes = ['Carambula', 'Lychee', 'Pear']
    labels = [0., 1., 2.]  
    x_2D = pca.fit(data).transform(data)

    for c, i, target_name, m in zip(colors, labels, classes, markers):
        plt.scatter(x_2D[label == i, 0], x_2D[label == i, 1], c=c, label=target_name, marker=m)

    plt.xlabel('PCA-feature-1')
    plt.ylabel('PCA-feature-2')
    plt.legend(classes, loc='upper right')
```
<img width="301" alt="image" src="https://user-images.githubusercontent.com/128220508/236402649-90d72222-18bd-4575-8f86-1feac44349e3.png"><img width="301" alt="image" src="https://user-images.githubusercontent.com/128220508/236402724-772f3f16-a479-4d3a-9706-81dc21a50d6a.png">. 

# Decision Regions
```js
def plot_decision_regions(X, Y, Y_pred, classifier):
    # X:feature data, Y:label data, Y_pred: 
    X = np.delete(X, 2, axis=1) #delete bias
    Y = np.argmax(Y, axis=1)
    Y_pred = np.argmax(Y_pred, axis=1)

    resolution = 0.01

    markers = ('.', '.', '.')
    colors = ('red', 'blue', 'orange')
    cmap = ListedColormap(colors[:len(np.unique(Y_pred))])

    # decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # coordinate matrix
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))  
    bias = np.ones((xx1.size, 1))
    
    input_data = np.c_[xx1.ravel(), xx2.ravel(), bias]
    Z = classifier.predict(input_data)
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx1.shape) 


    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap) #contour plot
    plt.xlim(xx1.min(), xx1.max()) #limit x-axis
    plt.ylim(xx2.min(), xx2.max()) #limit y-axis

    # plot
    for idx, cl in enumerate(np.unique(Y_pred)):
        plt.scatter(x=X[Y == cl, 0], y=X[Y == cl, 1], alpha=0.6, c=[cmap(idx)], marker=markers[idx], label=cl)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='lower left')

```

Plot decision regions  
```js
#train
plot_decision_regions(train_data, train_label, train_predict, network)
plt.title('Decision Region - Train Data')
plt.show()

#test
plot_decision_regions(test_data, test_label, test_predict, network)
plt.title('Decision Region - Test Data')
plt.show()
```
### Two layers network
![image](https://user-images.githubusercontent.com/128220508/236423961-da470504-951b-47b6-b3eb-592dfdb11b39.png)![image](https://user-images.githubusercontent.com/128220508/236423973-fccb5736-f918-43e1-8a76-abea1990995c.png)

### Three layers network
![image](https://user-images.githubusercontent.com/128220508/236424038-65ee6e76-209b-43df-b625-9203236fa26e.png)![image](https://user-images.githubusercontent.com/128220508/236424058-77c8e9e7-6e0b-4219-94e2-02a4a7541686.png).   
The images indicate that the model has been trained effectively as it has classified most of the features.





