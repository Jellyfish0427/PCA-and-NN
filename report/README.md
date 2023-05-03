## Model architecture
### Two layers network
<img width="442" alt="截圖 2023-05-04 上午1 00 55" src="https://user-images.githubusercontent.com/128220508/235987538-39ec131a-2398-4d16-85f7-ba7a281bd9a6.png">  

1. Epoch: 100  
2. Batch sizw: 200  
3. Learning rate: 0.01   
4. Optimizer: SGD   

### Three layers network
<img width="435" alt="截圖 2023-05-04 上午1 04 48" src="https://user-images.githubusercontent.com/128220508/235988449-07149991-dee9-4a56-a084-47d7c1d8a8f0.png">  

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

