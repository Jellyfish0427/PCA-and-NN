## Model architecture
### Two layers network
<img width="442" alt="截圖 2023-05-04 上午1 00 55" src="https://user-images.githubusercontent.com/128220508/235987538-39ec131a-2398-4d16-85f7-ba7a281bd9a6.png">  
#### testing accuracy : 0.896 
1. Epoch: 100  
2. Batch sizw: 200  
3. Learning rate: 0.01   
4. Optimizer: SGD

### Three layers network
<img width="435" alt="截圖 2023-05-04 上午1 04 48" src="https://user-images.githubusercontent.com/128220508/235988449-07149991-dee9-4a56-a084-47d7c1d8a8f0.png">  
#### testing accuracy : 0.908
1. Epoch: 100  
2. Batch sizw: 200  
3. Learning rate: 0.005  
4. Optimizer: SGD  

### Network learning
Mini-batch -> Gragient -> Update parameters  
1. Randomly select a mini-batch of training data and labels.
2. Calculate the gradients.
3. Use the optimizer to update the parameters.
Keep repeating these steps to adjust the weights until the epochs is reached.

## PCA Method
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
### Training data PCA
<img width="301" alt="image" src="https://user-images.githubusercontent.com/128220508/236402649-90d72222-18bd-4575-8f86-1feac44349e3.png">  
### Testing data PCA
<img width="301" alt="image" src="https://user-images.githubusercontent.com/128220508/236402724-772f3f16-a479-4d3a-9706-81dc21a50d6a.png">. 



