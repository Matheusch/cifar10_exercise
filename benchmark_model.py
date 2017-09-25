import os
from subprocess import call
import tarfile
import pickle
import sklearn
import sklearn.linear_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from skimage.feature import hog
from skimage import color, exposure
from skimage import data as skimage_data


cifar_python_directory = os.path.abspath("cifar-10-batches-py")


def unpickle(file):    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data(train_batches):
    data = []
    labels = []
    for data_batch_i in train_batches:
        d = unpickle(
            os.path.join(cifar_python_directory, data_batch_i)
        )
        data.append(d[b'data'])
        labels.append(np.array(d[b'labels']))
    
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    length = len(labels)    
    return data.reshape(length, 3, 32, 32), labels

def showRandomEachClassImages(data, labels):
    fig = plt.figure()
    actualIdx = 1
    for i in range(10):    
        for j in range(10):
            randomIdx = np.random.randint(5000)
            foundIdx = [idx for idx, element in enumerate(labels) if element == i][randomIdx]
            reverted = data[foundIdx,:,0:32,0:32]
            img = np.swapaxes(np.swapaxes(reverted,0,2),0,1)
            ax = fig.add_subplot(10, 10, actualIdx)
            ax.axis('off')
            plt.imshow(img)
            actualIdx = actualIdx + 1
    plt.show()

def getHogFeatures(data, featuresFilename):
    samplesCount = data.shape[0]
    features = np.array([])
    if os.path.exists(featuresFilename):
        features = np.load(featuresFilename)

    if (features.size == 0):
        axesSwapped = data[0,:,0:32,0:32]
        image = np.swapaxes(np.swapaxes(axesSwapped,0,2),0,1)
        gray = color.rgb2gray(image)
        imgDescriptor = hog(gray, orientations=8, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), block_norm='L2')

        features = np.empty((samplesCount, imgDescriptor.shape[0]))
        
        for i in range(samplesCount):
            axesSwapped = data[i,:,0:32,0:32]
            image = np.swapaxes(np.swapaxes(axesSwapped,0,2),0,1)
            gray = color.rgb2gray(image)
            imgDescriptor = hog(gray, orientations=8, pixels_per_cell=(8, 8),
                            cells_per_block=(1, 1), block_norm='L2')
            features[i] = imgDescriptor

        np.save(featuresFilename, features, allow_pickle=False)
    return features
    

filename = "cifar-10-python.tar.gz"
print("Start downloading")
if not os.path.exists(filename):
    call(
        "wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        shell=True
    )
    print("Downloading has ended.")
else:
    print("You've already downloaded the data.")

cifar_python_directory = os.path.abspath("cifar-10-batches-py")
if os.path.exists(cifar_python_directory):
    print("Already extracted")
else:    
    tar = tarfile.open(filename, "r:gz")
    tar.extractall()
    tar.close()


trainData, trainLabels = load_data(["data_batch_{}".format(i) for i in range(1, 6)])
showRandomEachClassImages(trainData, trainLabels)


trainFeaturesFilename = 'features_train.npy'
trainFeatures = getHogFeatures(trainData, trainFeaturesFilename)



##from sklearn.decomposition import PCA
##pca = PCA(n_components=2)
##X_r = pca.fit(trainFeatures).transform(trainFeatures)
##print ('PCA singular_values:', pca.singular_values_ )
##print ('PCA explained_variance_ratio:', pca.explained_variance_ratio_ )
##
##plt.figure(figsize=(20,20))
##plt.scatter(X_r[:,0], X_r[:,1], c=plt.cm.jet(trainLabels/10), s=10, edgecolors='none')
##plt.show()
##plt.title('PCA of HOG descriptors')


##from sklearn.manifold import TSNE
##samplesCount = 25000
##tsneFeaturesFilename = str(samplesCount) + '_tsne_benchmark_features.npz'
##if os.path.exists(tsneFeaturesFilename):
##    print('tSNE features found')
##    tsne_features = np.load(tsneFeaturesFilename)['tsne_features']
##else:
##    print('tSNE features not found (test)')
##    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=250)
##    tsneInput = trainFeatures[:samplesCount]
##    reshaped_features = np.reshape(tsneInput, [tsneInput.shape[0], np.prod(tsneInput.shape[1:])]) 
##    print ('reshaped_features.shape', reshaped_features.shape)
##    tsne_features = tsne.fit_transform(reshaped_features)
##    np.savez(tsneFeaturesFilename, tsne_features=tsne_features)
##print('tsne features obtained')
##
##plt.figure(figsize=(20,20))
##plt.scatter(tsne_features[:,0], tsne_features[:,1], c=plt.cm.jet(trainLabels/10), s=10, edgecolors='none')
##plt.show()



trainedModelFilename = 'trained_benchmark_model.pkl'
clf = []

from sklearn.externals import joblib

if os.path.exists(trainedModelFilename):    
    clf = joblib.load(trainedModelFilename)     
else:
    from sklearn import svm
    clf = svm.SVC(kernel='rbf',decision_function_shape='ovo',verbose=True)    
    clf.fit(trainFeatures, trainLabels)
    joblib.dump(clf, trainedModelFilename)




print ('Score calculation...')
testData, testLabels = load_data(["test_batch"])
testFeaturesFilename = 'features_test.npy'
testFeatures = getHogFeatures(testData, testFeaturesFilename)

score = clf.score(testFeatures, testLabels)
print ('final benchmark score: ', score)













