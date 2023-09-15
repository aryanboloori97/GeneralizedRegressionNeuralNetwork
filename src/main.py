
class GRNN(object):
    
    
    def __init__(self, x_sample, y_sample, std=1.5):
        self.std = std
        self.x_sample = x_sample
        self.y_sample = y_sample
        
    def _euclidean_distance(self, x_input):
        
        return np.sqrt(np.sum((x_input - self.x_sample)**2, axis=-1))
    
    
    def _activation(self, distance):
        
        return np.exp(-(distance**2)/(2 * (self.std**2)))
                      
                      
    def _numerator(self, distance):
        
        activations = self._activation(distance)
        
        return np.sum(self.y_sample  *  activations, axis=-1)

    def _denominator(self, distance):
        
        activations = self._activation(distance)
        
        return np.sum(activations, axis=-1)    
                      
                      
    def predict(self, x_input):
        
        distance = self._euclidean_distance(x_input[:,np.newaxis,:])
        predicted = self._numerator(distance) / self._denominator(distance)
        self.predicted = predicted
        return predicted
    
    def mean_squared_error(self, y_real):
        return np.sum((self.predicted - y_real)**2)
    
    
    def root_mean_squared_error(self, y_real):
        return np.sqrt(self.mean_squared_error(y_real))
    
