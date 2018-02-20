# pip install mumpy and matplotlib, this runs on WINdows with py3.4
import numpy as np
import matplotlib.pyplot as plt


# the following activation functions and partial derivatives use standard
# equations rather than numpy functions for illustrative purposes.

# sigmoid activation function, with optional derivation
def sigmoid(x, deriv=False):
    if deriv == True:
        return x*(1-x)
    return 1/(1+np.exp(-x))

# hyperbolic tangent activation function, with optional derivation
def tanh(x, deriv=False):
    if deriv == True:
        return 1.0 - (2/(1+np.exp(-2*x))-1)**2
    return 2/(1+np.exp(-2*x))-1

# rectified linear unit, leaky
def relu(x, deriv=False):
    if deriv == True:
        return 1 if x.any() > 0 else 0.01
    return np.maximum(x, 0)

# generate heatmap Z axis
def intensity(map):
    l1 = sigmoid(np.dot(map, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    return sigmoid(np.dot(l2, syn2))

# inc argument is pixel size relative to x/y scales, which are both 1.0 at max.
# so inc = 0.1 means the heat map is 10x10, and 0.05 makes it 20x20. Each pixel
# is calculated at each call, so obviously higher res has a higher load
def plotit(inc, block=False):
    i1 = 0
    mapx = []
    while i1 < 1/inc+1:
        mapx.append(round(inc*i1,2))
        i1 += 1
    mapy = mapx
    mapz = [[intensity([x, y])[0] for y in mapy]for x in mapx]
    plot = plt.pcolormesh(mapx, mapy, mapz)
    plot
    if i == 0:
        plot.set_clim(vmin=0, vmax=1)
        plt.colorbar()
    plt.title('iteration ' + str(i) + ', error: ' + str(error*100)[:4] + '%')
    plt.show(block=block)
    plt.pause(0.001)

    
# training data, 4 examples of 2 inputs, shape (4, 2)
X = np.array(
    [
        [0,0],
        [0,1],
        [1,0],
        [1,1],
    ]
)
# training data, target output for 4 examples (4, 1), this is an XOR gate which
# is true when either - but not both - input is true. The inverse is an XNOR.
y = np.array(
    [
        [0],
        [1],
        [1],
        [0],
    ]
)
# assign random weights to synapses to get started. Uncomment random.seed to
# get repeatable output. seeds 3 and 8 are pretty distinct.
# np.random.seed(8)

inputs = 2
hidden1 = 4
hidden2 = 4
outputs = 1

# synapse layer one, HIDDEN LAYER 1, two inputs to four neurons
syn0 = 2*np.random.random((inputs, hidden1)) - 1
# synapse layer two, HIDDEN LAYER 2, four neurons to four neurons
syn1 = 2*np.random.random((hidden1, hidden2)) - 1
# synapse layer three, OUTPUT LAYER , four neurons to one output
syn2 = 2*np.random.random((hidden2 , outputs)) - 1
# neurons can be added to each layer, but the second value for each layer must
# equal the first value of the following layer.

#define activation functions for each layer of neurons
l1func = sigmoid
l2func = sigmoid
l3func = sigmoid

i = 0
plt.ion()
plt.show()
# calculate starting error
l0 = X
l1 = sigmoid(np.dot(l0, syn0))
l2 = sigmoid(np.dot(l1, syn1))
l3 = sigmoid(np.dot(l2, syn2))
l3_error = y - l3
error = np.mean(np.abs(l3_error))
# training step, loop with training data until the output
# matches the target output to total system error defined here
n = 100
while error > 0.02:
    if i % n == 0:
        # generate heat map data each nth iteration.
        # lower inc and/or n values obviously increase execution time.
        plotit(inc=0.05)
    # each iteration calculates the effect each weight (synapse) has on total
    # error, and then a delta value used to update the weights in each layer.
    i += 1
    l0 = X
    l1 = l1func(np.dot(l0, syn0))
    l2 = l2func(np.dot(l1, syn1))
    l3 = l3func(np.dot(l2, syn2))
    l3_error = y - l3
    error = np.mean(np.abs(l3_error))
    l3_delta = l3_error * l3func(l3,deriv = True)
    l2_error = l3_delta.dot(syn2.T)
    l2_delta = l2_error * l2func(l2,deriv = True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * l1func(l1,deriv = True)
    # update synapse weights based on calculated error
    syn2 += l2.T.dot(l3_delta)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    
print('trained in ' + str(i) + ' iterations, ' + str(error*100)[:4] +
      '% error')
# plot final output at defined error, higher res because we're done iterating
plotit(0.01, block=True)