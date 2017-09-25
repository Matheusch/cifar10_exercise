from sklearn.manifold import TSNE
import numpy as np
import os
from subprocess import call
import tarfile
import pickle
import sklearn
import sklearn.linear_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np



def load_data(train_batches):
    data = []
    labels = []
    for data_batch_i in train_batches:
        featuresFile = np.load(data_batch_i)
        data.append(featuresFile['representations'])
        labels.append(featuresFile['labels'])        
    
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    length = len(labels)
    print ('data.shape',data.shape)
    print ('labels.shape',labels.shape)
    return data, labels


trainFeatures, trainLabels = load_data(["data_batch_{}_tensorflow.npz".format(i) for i in range(1, 6)])


samplesCount = 25000
tsneFeaturesFilename = str(samplesCount) + '_tsne_cnn_features.npz'
if os.path.exists(tsneFeaturesFilename):
    print('tSNE features found')
    tsne_features = np.load(tsneFeaturesFilename)['tsne_features']
else:
    print('tSNE features not found (test)')
    tsne = TSNE(n_components=2, verbose=1, perplexity=35, n_iter=250)
    tsneInput = trainFeatures[:samplesCount]
    reshaped_features = np.reshape(tsneInput, [tsneInput.shape[0], np.prod(tsneInput.shape[1:])]) 
    print ('reshaped_features.shape', reshaped_features.shape)
    tsne_features = tsne.fit_transform(reshaped_features)
    np.savez(tsneFeaturesFilename, tsne_features=tsne_features)
print('tsne features obtained')

plt.figure(figsize=(20,20))
plt.scatter(tsne_features[:,0], tsne_features[:,1], c=plt.cm.jet(trainLabels/10), s=10, edgecolors='none')
plt.show()




    


