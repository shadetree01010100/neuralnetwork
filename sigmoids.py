import math 


def sigmoid(x, deriv=False, L=1, k=1, x0=0):
    fx = L / (1 + math.e ** -(k * (x- x0)))
    if not deriv:
        return fx
    return fx * (1 - fx)

def inv_sigmoid(y, L=1, k=1, x0=0):
    return -math.log((L - y) / (k * y)) / math.log(math.e) + x0

foo = 0.999
while True:
    # move along X axis of sigmoid, using Y values as input, calculating the
    # derivative and then incrementing Y by that slope each step.
    step = sigmoid(inv_sigmoid(foo), deriv=True)
    if step > foo:
        # inadequate floating point precision would cause negative 'foo' value
        print('!!! ', step)
        break
    foo -= step
    # clamp range
    # foo = min(max(foo, 0.001), 0.999)
    print(foo)
