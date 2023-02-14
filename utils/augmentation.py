import numpy as np
from .utils import znormalisation

def triplet_generation(x):
    
    w1 = np.random.choice(np.linspace(start=0.6,stop=1,num=1000),size=1)
    w2 = (1 - w1) / 2

    ref = np.random.permutation(x[:])

    n = int(ref.shape[0])
    l = int(ref.shape[1])

    pos = np.zeros(shape=ref.shape)
    neg = np.zeros(shape=ref.shape)

    all_indices = np.arange(start=0, stop=n)

    for i in range(n):

        temp_indices = np.delete(arr=all_indices, obj=i)
        indice_neg = int(np.random.choice(temp_indices, size=1))

        temp_indices = np.delete(arr=all_indices, obj=[i,indice_neg])
        indice_b = int(np.random.choice(temp_indices, size=1))

        indice_c = int(np.random.choice(temp_indices, size=1))

        indice_b2 = int(np.random.choice(temp_indices, size=1))

        indice_c2 = int(np.random.choice(temp_indices, size=1))

        a = ref[i].copy()

        nota = ref[indice_neg].copy()

        b = ref[indice_b].copy()
        c = ref[indice_c].copy()

        b2 = ref[indice_b2].copy()
        c2 = ref[indice_c2].copy()

        # MixingUp

        pos[i] = w1 * a + w2 * b + w2 * c
        neg[i] = w1 * nota + w2 * b2 + w2 * c2

        # Masking

        start_pos_neg = int(np.random.randint(low=0, high=l-1, size=1))
        stop_pos_neg = int(np.random.randint(low=start_pos_neg + (l - 1 - start_pos_neg) //
                       10, high=start_pos_neg + (l - 1 - start_pos_neg) * 4//10 + 1, size=1))

        noise_pos_left = np.random.random(size=start_pos_neg)
        noise_pos_left /= 5
        noise_pos_left -= 0.1
        noise_pos_right = np.random.random(size=l-stop_pos_neg-1)
        noise_pos_right /= 5
        noise_pos_right -= 0.1

        pos[i, 0:start_pos_neg] = noise_pos_left
        pos[i, stop_pos_neg+1:l] = noise_pos_right
                       
        noise_neg_left = np.random.random(size=start_pos_neg)
        noise_neg_left /= 5
        noise_neg_left -= 0.1
        noise_neg_right = np.random.random(size=l-stop_pos_neg-1)
        noise_neg_right /= 5
        noise_neg_right -= 0.1

        neg[i, 0:start_pos_neg] = noise_neg_left
        neg[i, stop_pos_neg+1:l] = noise_neg_right

    pos = znormalisation(pos)
    neg = znormalisation(neg)

    return ref, pos, neg