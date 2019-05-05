import numpy as np



def generator(features, labels, batch_size, width_pixel):
    while True:
        for i in range(int(np.floor(len(features) / float(batch_size)))): # -1 just for the one case
            feature_new = np.zeros((batch_size, features[batch_size*i].shape[0], width_pixel,
                                    features[batch_size*i].shape[2]))
            label_new = np.zeros((batch_size, labels[batch_size*i].shape[0], width_pixel,
                                    labels[batch_size*i].shape[2]))
            for c in range(batch_size):
                #if batch_size*i+c >= len(features):
                    #break
                feature_new[c] = features[batch_size*i+c]
                label_new[c] = labels[batch_size*i+c]
            yield feature_new, label_new