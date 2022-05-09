import numpy as np

class Knn_classifier:
    def __init__(self, train_samples, train_labels):
        self.train_samples = train_samples
        self.train_labels  = train_labels
    
    def classify_image(self, test_image, num_neighbors = 3, metric = 'l2'): 
     
        distances = None
        if(metric == 'l2'):   
            distances = np.sqrt(np.sum((self.train_samples - test_image) ** 2, axis = 1))
        elif(metric == 'l1'):
            distances = np.sum(abs(self.train_samples - test_image), axis = 1)
        else:
            print('Error! Metric {} is not defined!'.format(metric))
            exit(1)
            
        sort_index = np.argsort(distances)
        sort_index = sort_index[:num_neighbors]
        nearest_labels = self.train_labels[sort_index]
        histc = np.bincount(nearest_labels)
        
        return np.argmax(histc)
    
              
    def predict(self, test_samples, num_neighbors = 3, metric = 'l2'):
        num_test_samples = test_samples.shape[0] 
        predicted_labels = np.zeros((num_test_samples))
        
        for i in range(num_test_samples): 
            predicted_labels[i] = self.classify_image(test_samples[i, :], num_neighbors = num_neighbors, metric = metric)
        
        return predicted_labels
