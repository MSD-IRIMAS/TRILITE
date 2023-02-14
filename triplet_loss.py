from tensorflow.keras import backend as k
import tensorflow as tf

def triplet_loss_function(alpha):
    
    def temp(ytrue, ypred):
        
        ref = ypred[:, :, 0]
        pos = ypred[:, :, 1]
        neg = ypred[:, :, 2]
        
        ref = k.cast(ref,dtype=ref.dtype)
        pos = k.cast(pos, dtype=ref.dtype)
        neg = k.cast(neg, dtype=ref.dtype)
        
        loss = k.maximum(k.sum(tf.math.subtract(tf.math.add(k.square(ref-pos),alpha),k.square(ref-neg)),axis=1),0)
        
        return loss

    return temp