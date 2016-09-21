from theano import function, config, shared, sandbox
import theano.tensor as T
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.pool import MaxPool, AvgPool
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from multiprocessing import Pipe
import multiprocessing as mp
from six.moves import cPickle
import logging
import math
import numpy as np
import scipy.linalg
from sklearn import metrics
import time
from multiprocessing import Process, Queue
import numpy as np
import SharedArray as sa
from sklearn.metrics import accuracy_score
import os

#WARNING FOR AVERAGE POOLING THIS RELIES ON THIS FORK OF PYLEARN2:
# https://github.com/Vaishaal/pylearn2

#logging.getLogger('theano.gof.cmodule').setLevel(logging.DEBUG)

def unpickle(infile):
    import cPickle
    fo = open(infile, 'rb')
    outdict = cPickle.load(fo)
    fo.close()
    return outdict

def load_cifar_processed():
    npzfile = np.load("./cifar_processed")
    return (npzfile['XTrain'], npzfile['yTrain']), (npzfile['XTest'], npzfile['yTest'])

def load_cifar(center=False):
    train_batches = []
    train_labels = []
    for i in range(1,6):
        cifar_out = unpickle("../cifar/data_batch_{0}".format(i))
        train_batches.append(cifar_out["data"])
        train_labels.extend(cifar_out["labels"])

    # Stupid bull shit to get pixels in correct order
    X_train= np.vstack(tuple(train_batches)).reshape(-1, 32*32, 3)
    X_train = X_train.reshape(-1,3,32,32)
    mean_image = np.mean(X_train, axis=0)[np.newaxis, :, :]
    y_train = np.array(train_labels)
    cifar_out = unpickle("../cifar/test_batch")
    X_test = cifar_out["data"].reshape(-1, 32*32, 3)
    X_test = X_test.reshape(-1,3,32,32)
    X_train = X_train
    X_test = X_test
    y_test = cifar_out["labels"]
    return (X_train, np.array(y_train)), (X_test, np.array(y_test))

def conv(data, filter_gen, feature_batch_size, num_feature_batches, data_batch_size, cuda_convnet=True, symmetric_relu=True, start_feature_batch=0, pool_type='avg', pool_size=14, pool_stride=14, pad=0, bias=1.0, ps=6):


    strideStart = pool_size/2.0
    outX = int(math.ceil(((data.shape[2] - ps + 1) - pool_size)/float(pool_stride))) + 1
    outY = int(math.ceil(((data.shape[3] - ps + 1) - pool_size)/float(pool_stride))) + 1

    print("Pool Size ", pool_size)
    print("Pool Stride", pool_stride)
    print ("out size: ", outX, "x",  outY)
    outFilters = feature_batch_size*num_feature_batches
    if (symmetric_relu):
        outFilters = 2*outFilters

    print "Out Shape ", outX, "x", outY, "x", outFilters
    XFinal = np.zeros((data.shape[0], outFilters, outX, outY), 'float32')

    XBlock = None
    FTheano = None
    filters = []
    numImages = data.shape[0]
    # Convert to cuda-convnet order
    if (cuda_convnet):
        data = data.transpose(1,2,3,0)

    # POOL OP CREATION
    if (cuda_convnet):
        if (pool_type == 'avg'):
            pool_op = AvgPool(ds=pool_size, stride=pool_stride)
        elif (pool_type == 'max'):
            pool_op = MaxPool(ds=pool_size, stride=pool_stride)
        else:
            raise Exception('Unsupported pool type')

    else:
        pool_op = lambda X: T.signal.pool.pool_2d(X, (pool_size, pool_size), ignore_border=False, mode='max')

    if (cuda_convnet):
        conv_op = FilterActs(pad=pad)
    else:
        conv_op = lambda X, F: T.nnet.conv2d(X, F)

    CHANNEL_AXIS = 1
    for j in range(num_feature_batches):
        F = filter_gen(feature_batch_size)
        if (cuda_convnet):
            F = F.transpose(1,2,3,0)
            CHANNEL_AXIS = 0

        filters.append(F)
        if (FTheano == None):
            FTheano = shared(F.astype('float32'))
        else:
            FTheano.set_value(F.astype('float32'))

        start_filters = j*feature_batch_size
        end_filters = (j+1)*feature_batch_size

        if symmetric_relu:
            start_filters *= 2
            end_filters *= 2

        for i in range(int(np.ceil(numImages/float(data_batch_size)))):
                start = i*data_batch_size
                end = min((i+1)*data_batch_size, numImages)
                print "FEATURE BATCH #", (j + start_feature_batch), "DATA BATCH #", i,  " SIZE IS ", end - start
                if (cuda_convnet):
                    XBlock_cpu = data[:, :, :, start:end]
                else:
                    XBlock_cpu = data[start:end, :, :, :]

                if (XBlock == None):
                    XBlock = shared(XBlock_cpu)
                else:
                    XBlock.set_value(XBlock_cpu)

                # CONV
                XBlock_conv_out = conv_op(XBlock, FTheano)

                # RELU
                XBlock0 = T.nnet.relu(XBlock_conv_out - bias, 0)
                if (symmetric_relu):
                    XBlock1 = T.nnet.relu(-1.0 * XBlock_conv_out - bias, 0)

                XBlock0 = pool_op(XBlock0)
                if (symmetric_relu):
                    XBlock1 = pool_op(XBlock1)
                    XBlockOut = np.concatenate((XBlock0.eval(), XBlock1.eval()), axis=CHANNEL_AXIS)
                else:
                    XBlockOut = np.array(XBlock0.eval())

                if (cuda_convnet):
                    XBlockOut = XBlockOut.transpose(3,0,1,2)
                    F = F.transpose(3,0,1,2)
                XFinal[start:end,start_filters:end_filters,:,:] = XBlockOut

    filters = np.concatenate(filters,axis=0)
    return (XFinal, filters)

def preprocess(train, test, min_divisor=1e-8, zca_bias=0.1):
    origTrainShape = train.shape
    origTestShape = test.shape

    train = np.ascontiguousarray(train, dtype=np.float32).reshape(train.shape[0], -1)
    test = np.ascontiguousarray(test, dtype=np.float32).reshape(test.shape[0], -1)


    print "PRE PROCESSING"
    nTrain = train.shape[0]

    # Zero mean every feature
    train = train - np.mean(train, axis=1)[:,np.newaxis]
    test = test - np.mean(test, axis=1)[:,np.newaxis]

    # Normalize
    train_norms = np.linalg.norm(train, axis=1)/55.0
    test_norms = np.linalg.norm(test, axis=1)/55.0

    # Get rid of really small norms
    train_norms[np.where(train_norms < min_divisor)] = 1
    test_norms[np.where(test_norms < min_divisor)] = 1

    # Make features unit norm
    train = train/train_norms[:,np.newaxis]
    test = test/test_norms[:,np.newaxis]


    whitening_means = np.mean(train, axis=0)
    data_means = np.mean(train, axis=1)


    zeroCenterTrain = (train - whitening_means[np.newaxis, :])

    trainCovMat = 1.0/nTrain * zeroCenterTrain.T.dot(zeroCenterTrain)

    (E,V) = np.linalg.eig(trainCovMat)

    E += zca_bias
    sqrt_zca_eigs = np.sqrt(E)
    inv_sqrt_zca_eigs = np.diag(np.power(sqrt_zca_eigs, -1))
    global_ZCA = V.dot(inv_sqrt_zca_eigs).dot(V.T)
    print global_ZCA[:4,:4]

    train = (train - whitening_means).dot(global_ZCA)
    test = (test - whitening_means).dot(global_ZCA)

    return (train.reshape(origTrainShape), test.reshape(origTestShape))

def learnPrimal(trainData, labels, W=None, reg=0.1):
    '''Learn a model from trainData -> labels '''

    trainData = trainData.reshape(trainData.shape[0],-1)
    n = trainData.shape[0]
    X = np.ascontiguousarray(trainData, dtype=np.float32).reshape(trainData.shape[0], -1)
    if (W == None):
        W = np.ones(n)[:, np.newaxis]

    print "X SHAPE ", trainData.shape
    print "Computing XTX"
    sqrtW = np.sqrt(W)
    X *= sqrtW
    XTWX = X.T.dot(X)
    print "Done Computing XTX"
    idxes = np.diag_indices(XTWX.shape[0])
    XTWX[idxes] += reg
    y = np.eye(max(labels) + 1)[labels]
    XTWy = X.T.dot(W * y)
    model = scipy.linalg.solve(XTWX, XTWy)
    return model

def learnDual(gramMatrix, labels, reg=0.1, W=None, TOT_FEAT=1, NUM_TRAIN=1):
    ''' Learn a model from K matrix -> labels '''
    print ("Learning Dual Model better")

    n = gramMatrix.shape[0]
    if (W == None):
        W = np.ones(n)

    W_inv = W.copy()
    W_inv = 1.0/W_inv

    y = np.eye(max(labels) + 1)[labels]
    print "REG IS ", reg
    diag_indices= np.diag_indices(gramMatrix.shape[0])
    gramMatrix[diag_indices] += reg*W_inv
    model = scipy.linalg.solve(gramMatrix, y, sym_pos=True)
    gramMatrix[diag_indices] -= reg*W_inv

    return model

def evaluatePrimalModel(data, model):
    data = data.reshape(data.shape[0],-1)
    raw = data.dot(model)
    yHat = np.argmax(raw, axis=1)
    return yHat, raw


def evaluateDualModel(kMatrix, model, TOT_FEAT=1):
    kMatrix /= TOT_FEAT
    y = kMatrix.dot(model)
    kMatrix *= TOT_FEAT
    yHat = np.argmax(y, axis=1)
    return yHat

def trainAndEvaluateDualModel(KTrain, KTest, labelsTrain, labelsTest, reg=0.1, TOT_FEAT=1, W=None):
    model = learnDual(KTrain,labelsTrain, reg=reg, TOT_FEAT=TOT_FEAT, W=W)
    predTrainLabels = evaluateDualModel(KTrain, model, TOT_FEAT=TOT_FEAT)
    predTestLabels = evaluateDualModel(KTest, model, TOT_FEAT=TOT_FEAT)
    train_acc = metrics.accuracy_score(labelsTrain, predTrainLabels)
    test_acc = metrics.accuracy_score(labelsTest, predTestLabels)
    return train_acc, test_acc

def trainAndEvaluatePrimalModel(XTrain, XTest, labelsTrain, labelsTest, reg=0.1, W=None):
    model = learnPrimal(XTrain, labelsTrain, reg=reg, W=W)
    predTrainLabels, _ = evaluatePrimalModel(XTrain, model)
    predTestLabels, _ = evaluatePrimalModel(XTest, model)
    train_acc = metrics.accuracy_score(labelsTrain, predTrainLabels)
    test_acc = metrics.accuracy_score(labelsTest, predTestLabels)
    print "CONFUSION MATRIX"
    print metrics.confusion_matrix(labelsTest, predTestLabels)
    return train_acc, test_acc

def convolveAndAccumulateGramAsync(XTrain, XTest, labelsTrain, labelsTest, filter_gen,  num_feature_batches=1, regs=[0.1], pool_size=14, FEATURE_BATCH_SIZE=1024, CUDA_CONVNET=True, DATA_BATCH_SIZE=1280, pool_stride=14, ps=6, symmetric_relu=True):
    print("RELOADING MOTHER FUCKER 3")
    X = np.vstack((XTrain, XTest))
    parent, child = Pipe()

    outX = int(math.ceil(((X.shape[2] - ps + 1) - pool_size)/float(pool_stride))) + 1
    outY = int(math.ceil(((X.shape[3] - ps + 1) - pool_size)/float(pool_stride))) + 1

    relu = 1
    if (symmetric_relu):
        relu = 2
    XBatchShape = (X.shape[0],FEATURE_BATCH_SIZE*relu*outX*outY)
    # Initialize shared memory arrays
    XBatchShared = np.memmap("/run/shm/xbatch", dtype="float32", mode="w+", shape=XBatchShape)
    trainKernelShared = np.memmap("/run/shm/trainKernel", mode="w+", shape=(XTrain.shape[0], XTrain.shape[0]), dtype='float32')
    testKernelShared = np.memmap("/run/shm/testKernel", mode="w+", shape=(XTest.shape[0], XTrain.shape[0]), dtype='float32')
    squaredNormTrainShared = np.memmap("/run/shm/xtestNorms", mode="w+", shape=(XTrain.shape[0]), dtype='float32')
    squaredNormTestShared = np.memmap("/run/shm/xtrainNorms", mode="w+", shape=(XTest.shape[0]), dtype='float32')

    p = Process(target=accumulateGramAndSolveAsync, args=(child, XTrain.shape[0], XTest.shape[0], regs, labelsTrain, labelsTest, XBatchShape))
    p.start()
    for i in range(1, (num_feature_batches + 1)):
        print("Convolving features")
        time1 = time.time()
        (XBatch, filters) = conv(X, filter_gen, FEATURE_BATCH_SIZE, 1, DATA_BATCH_SIZE, CUDA_CONVNET, symmetric_relu=symmetric_relu, start_feature_batch=i-1, pool_size=pool_size, pool_stride=pool_stride, ps=ps)
        print "XBATCH SHAPE ", XBatch.shape
        time2 = time.time()
        print 'Convolving features took {0} seconds'.format((time2-time1))
        print("Sending features")
        time1 = time.time()
        np.copyto(XBatchShared, XBatch.reshape(XBatch.shape[0], -1))
        XBatchShared.flush()
        parent.send(i)
        time2 = time.time()
        print 'Sending features took {0} seconds'.format((time2-time1))
    parent.send(-1)
    parent.recv()
    print("Receiving kernel from child")
    trainKernelLocal = np.zeros((XTrain.shape[0], XTrain.shape[0]))
    testKernelLocal = np.zeros((XTest.shape[0], XTrain.shape[0]))
    squaredNormTrainLocal = np.zeros(XTrain.shape[0],dtype='float32')
    squaredNormTestLocal  = np.zeros(XTest.shape[0],dtype='float32')


    np.copyto(trainKernelLocal, trainKernelShared)
    np.copyto(testKernelLocal, testKernelShared)

    np.copyto(squaredNormTrainLocal, squaredNormTrainShared)
    np.copyto(squaredNormTestLocal, squaredNormTestShared)

    parent.close()
    child.close()
    os.remove("/run/shm/trainKernel")
    os.remove("/run/shm/testKernel")
    os.remove("/run/shm/xtestNorms")
    os.remove("/run/shm/xtrainNorms")
    os.remove("/run/shm/xbatch")
    return trainKernelLocal, testKernelLocal, squaredNormTrainLocal, squaredNormTestLocal

def accumulateGramAndSolveAsync(pipe,  numTrain, numTest, regs, labelsTrain, labelsTest, XBatchShape):
    trainKernel = np.zeros((numTrain, numTrain), dtype='float32')
    testKernel= np.zeros((numTest, numTrain), dtype='float32')

    XBatchShared = np.memmap("/run/shm/xbatch", mode="r+", shape=XBatchShape, dtype='float32')
    trainKernelShared = np.memmap("/run/shm/trainKernel", mode="r+", shape=(numTrain, numTrain), dtype='float32')
    testKernelShared = np.memmap("/run/shm/testKernel", mode="r+", shape=(numTest, numTrain), dtype='float32')
    squaredNormTrainShared = np.memmap("/run/shm/xtestNorms", mode="r+", shape=(numTrain,), dtype='float32')
    squaredNormTestShared = np.memmap("/run/shm/xtrainNorms", mode="r+", shape=(numTest,), dtype='float32')

    # Local copy
    XBatchLocal = np.zeros(XBatchShared.shape, dtype='float32')
    print("CHILD Process Spun yupyup")
    TOT_FEAT = 0
    while(True):
        m = pipe.recv()
        if m == -1:
            break
        time1 = time.time()
        print("Receiving (ASYNC) Batch {0}".format(m))
        np.copyto(XBatchLocal, XBatchShared)
        time2 = time.time()
        print 'Receiving (ASYNC) took {0} seconds'.format((time2-time1))
        XBatchTrain = XBatchLocal[:50000,:]
        XBatchTest = XBatchLocal[50000:,:]

        squaredNormTrainShared += np.sum(XBatchTrain * XBatchTrain, axis=1)
        squaredNormTestShared += np.sum(XBatchTest * XBatchTest, axis=1)

        print("XBATCH DTYPE " + str(XBatchTest.dtype))
        print("Accumulating (ASYNC) Gram")
        time1 = time.time()
        TOT_FEAT += XBatchTrain.shape[1]
        trainKernel += XBatchTrain.dot(XBatchTrain.T)
        testKernel += XBatchTest.dot(XBatchTrain.T)
        time2 = time.time()
        print 'Accumulating (ASYNC) Batch {1} gram took {0} seconds'.format((time2-time1), m)


    np.copyto(trainKernelShared, trainKernel)
    np.copyto(testKernelShared, testKernel)
    trainKernelShared.flush()
    testKernelShared.flush()
    squaredNormTrainShared.flush()
    squaredNormTestShared.flush()
    pipe.send(1)

def patchify_all_imgs(X, patch_shape, pad=True, pad_mode='constant', cval=0):
    out = []
    print X.shape
    X = X.transpose(0,2,3,1)
    i = 0
    for x in X:
        dim = x.shape[0]
        patches = patchify(x, patch_shape, pad, pad_mode, cval)
        out_shape = patches.shape
        out.append(patches.reshape(out_shape[0]*out_shape[1], patch_shape[0], patch_shape[1], -1))
    return np.array(out)

def patchify(img, patch_shape, pad=True, pad_mode='constant', cval=0):
    ''' Function borrowed from:
    http://stackoverflow.com/questions/16774148/fast-way-to-slice-image-into-overlapping-patches-and-merge-patches-to-image
    '''
    #FIXME: Make first two coordinates of output dimension shape as img.shape always

    if pad:
        pad_size= (patch_shape[0]/2, patch_shape[0]/2)
        img = np.pad(img, (pad_size, pad_size, (0,0)),  mode=pad_mode, constant_values=cval)

    img = np.ascontiguousarray(img)  # won't make a copy if not needed

    X, Y, Z = img.shape
    x, y= patch_shape
    shape = ((X-x+1), (Y-y+1), x, y, Z) # number of patches, patch_shape
    # The right strides can be thought by:
    # 1) Thinking of `img` as a chunk of memory in C order
    # 2) Asking how many items through that chunk of memory are needed when indices
#    i,j,k,l are incremented by one
    strides = img.itemsize*np.array([Y*Z, Z, Y*Z, Z, 1])
    patches = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    return patches

def make_empirical_filter_gen(patches, labels, MIN_VAR_TOL=0, seed=0):
    np.random.seed(seed)
    patches = patches.reshape(patches.shape[0]*patches.shape[1],*patches.shape[2:])
    all_idxs = np.random.choice(patches.shape[0], patches.shape[0], replace=False)
    curr_idx = [0]
    def empirical_filter_gen(num_filters):
        idxs = all_idxs[curr_idx[0]:curr_idx[0]+num_filters]
        curr_idx[0] += num_filters
        unfiltered = patches[idxs].astype('float32').transpose(0,3,1,2)
        old_shape = unfiltered.shape
        unfiltered = unfiltered.reshape(unfiltered.shape[0], -1)
        unfiltered_vars = np.var(unfiltered, axis=1)
        filtered = unfiltered[np.where(unfiltered_vars > MIN_VAR_TOL)]
        out = filtered[:num_filters].reshape(num_filters, *old_shape[1:])
        return out
    return empirical_filter_gen

def make_balanced_empirical_filter_gen(patches, labels):
    ''' NUM_FILTERS MUST BE DIVISIBLE BY NUM_CLASSES '''
    def empirical_filter_gen(num_filters):
        filters = []
        for c in range(NUM_CLASSES):
            patch_ss = patches[np.where(labels == c)]
            patch_ss = patch_ss.reshape(patch_ss.shape[0]*patch_ss.shape[1], *patch_ss.shape[2:])
            idxs = np.random.choice(patch_ss.shape[0], num_filters/NUM_CLASSES, replace=False)
            unfiltered = patch_ss[idxs].astype('float32').transpose(0,3,1,2)
            old_shape = unfiltered.shape
            unfiltered = unfiltered.reshape(unfiltered.shape[0], -1)
            unfiltered_vars = np.var(unfiltered, axis=1)
            filtered = unfiltered[np.where(unfiltered_vars > MIN_VAR_TOL)]
            out = filtered[:num_filters].reshape(num_filters/NUM_CLASSES, *old_shape[1:])
            filters.append(out)
        return np.concatenate(filters, axis=0)
    return empirical_filter_gen

def estimate_bandwidth(patches):
    patch_norms = np.linalg.norm(patches.reshape(patches.shape[0], -1), axis=1)
    return np.median(patch_norms)

def make_gaussian_filter_gen(bandwidth, patch_size=6, channels=3):
    ps = patch_size
    def gaussian_filter_gen(num_filters):
        out = np.random.randn(num_filters, channels, ps, ps).astype('float32') * bandwidth
        print out.shape
        return out
    return gaussian_filter_gen


def make_gaussian_cov_filter_gen(patches, sub_sample=100000):
    patches = patches.reshape(patches.shape[0]*patches.shape[1],*patches.shape[2:])
    idxs = np.random.choice(patches.shape[0], sub_sample, replace=False)
    patches = patches[idxs, :, :, :]
    patches = patches.reshape(patches.shape[0], -1)
    means = patches.mean(axis=0)[:,np.newaxis]
    covMatrix = 1.0/(patches.shape[0]) * patches.T.dot(patches) - means.dot(means.T)
    covMatrixRoot = np.linalg.cholesky(covMatrix).astype('float32')
    print(covMatrixRoot.shape)
    def gaussian_filter_gen(num_filters):
        out = np.random.randn(num_filters, 3*6*6).astype('float32').dot(covMatrixRoot)
        return out.reshape(out.shape[0], 3, 6, 6)
    return gaussian_filter_gen

def make_gaussian_cc_cov_filter_gen(patches, labels, patch_size=6, channels=3, sub_samples=10000):
    ''' NUM_FILTERS MUST BE DIVISBLE BY NUM_CLASSES '''
    covMatrixRoots = []
    bws = []
    for c in range(NUM_CLASSES):
        patch_ss = patches[np.where(labels == c)]
        patch_ss = patch_ss.reshape(patch_ss.shape[0]*patch_ss.shape[1], *patch_ss.shape[2:])
        bw = estimate_bandwidth(patch_ss)
        bws.append(bw)
        idxs = np.random.choice(patch_ss.shape[0], sub_samples, replace=False)
        patch_ss = patch_ss.reshape(patch_ss.shape[0], -1)
        means = patch_ss.mean(axis=0)[:,np.newaxis]
        covMatrix = 1.0/(patch_ss.shape[0]) * (patch_ss.T.dot(patch_ss) - means.dot(means.T))
        #covMatrix =  1.0  * np.eye(patch_ss.shape[1]) * 10.0/bw
        covMatrixRoot = np.linalg.cholesky(covMatrix).astype('float32')
        covMatrixRoots.append(covMatrixRoot)

    def gaussian_filter_gen(num_filters):
        ps = patch_size
        filters = []
        for c in range(NUM_CLASSES):
            out = np.random.randn(num_filters/NUM_CLASSES, channels*ps*ps).astype('float32').dot(covMatrixRoots[c])
            filters.append(out.reshape(out.shape[0], channels, ps, ps))
        return np.concatenate(filters, axis=0)
    return gaussian_filter_gen







if __name__ == "__main__":
    # Load CIFAR

    NUM_FEATURE_BATCHES=512
    DATA_BATCH_SIZE=(1280)
    FEATURE_BATCH_SIZE=(1024)
    NUM_TRAIN = 50000
    NUM_TEST = 10000
    NUM_CLASSES = 10
    POOL_TYPE ='avg'
    FILTER_GEN ='empirical'
    BANDWIDTH = 1.0
    LAMBDAS = [1e-1/FEATURE_BATCH_SIZE, 1e-2/FEATURE_BATCH_SIZE, 1e-3/FEATURE_BATCH_SIZE, 1e-4/FEATURE_BATCH_SIZE, 1e-5/FEATURE_BATCH_SIZE]
    CUDA_CONVNET = True
    SCALE = 55.0
    BIAS = 1.25
    MIN_VAR_TOL = 1e-4
    TOT_FEAT = FEATURE_BATCH_SIZE*NUM_FEATURE_BATCHES

    np.random.seed(10)
    (XTrain, labelsTrain), (XTest, labelsTest) = load_cifar_processed()
    patches = patchify_all_imgs(XTrain, (6,6), pad=False)
    if FILTER_GEN == 'gaussian':
       filter_gen = make_gaussian_filter_gen(1.0)
    elif FILTER_GEN == 'empirical':
        filter_gen = make_empirical_filter_gen(patches, labelsTrain)
    elif FILTER_GEN == 'empirical_balanced':
        filter_gen = make_balanced_empirical_filter_gen(patches, labelsTrain)
    elif FILTER_GEN == 'gaussian_cov':
        filter_gen = make_gaussian_cov_filter_gen(patches)
    elif FILTER_GEN == 'gaussian_cc_cov':
        filter_gen = make_gaussian_cc_cov_filter_gen(patches, labelsTrain)
    else:
        raise Exception('Unknown FILTER_GEN value')


    '''
    X = np.vstack((XTrain, XTest))
    time1 = time.time()
    (Xlevel1, filters) = conv(X, filter_gen, FEATURE_BATCH_SIZE, 1, DATA_BATCH_SIZE, CUDA_CONVNET, pool_size=2, symmetric_relu=False)
    time2 = time.time()
    print 'Convolutions with {0} filters took {1} seconds'.format(NUM_FEATURE_BATCHES*FEATURE_BATCH_SIZE, (time2-time1))
    Xlevel1Train = Xlevel1[:50000,:,:,:]
    Xlevel1Test = Xlevel1[50000:,:,:,:]
    patches = patchify_all_imgs(Xlevel1Train, (3,3), pad=False)
    patches = patches.reshape(patches.shape[0]*patches.shape[1],*patches.shape[2:])
    bw = estimate_bandwidth(patches)
    filter_gen = make_gaussian_filter_gen(10.0/bw, patch_size=3, channels=128)
    print ('level 1 shape ' + str(Xlevel1.shape))
    (XFinal, filters) = conv(Xlevel1, filter_gen, FEATURE_BATCH_SIZE*10, NUM_FEATURE_BATCHES, DATA_BATCH_SIZE, CUDA_CONVNET, pool_size=6, bias=1.0)
    print('level 2 shape ' + str(XFinal.shape))

    XFinalTrain = XFinal[:50000,:,:,:].reshape(NUM_TRAIN,-1)
    XFinalTest = XFinal[50000:,:,:,:].reshape(NUM_TEST,-1)
    print "Output train data shape ", XFinalTrain.shape
    print "Output test data shape ", XFinalTest.shape
    print "Output filters shape ", filters.shape
    convTrainAcc, convTestAcc = trainAndEvaluatePrimalModel(XFinalTrain, XFinalTest, labelsTrain, labelsTest, reg=LAMBDAS[0])
    print "(conv) train: ", convTrainAcc, "(conv) test: ", convTestAcc
    print("STARTING LEVEL 2")
    '''
    featurizeTrainAndEvaluateDualModelAsync(XTrain, XTest, labelsTrain, labelsTest, filter_gen, num_feature_batches=NUM_FEATURE_BATCHES, solve_every_iter=NUM_FEATURE_BATCHES/4, regs=LAMBDAS)



