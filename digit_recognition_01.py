import struct
from neurons2 import *

def image_input():
    with open('handwriting/train-images.idx3-ubyte', 'rb') as train_images:
        train_images.seek(4, 0)
        count = struct.unpack('>I', train_images.read(4))[0]
        print('loading {} training images ...'.format(count))
        train_images.seek(16, 0)
        return np.array([[r / 255 for r in \
            [b for b in train_images.read(784)]] for c in range(count)
        ])

def image_labels():
    print('loading training image labels ...')
    with open('handwriting/train-labels.idx1-ubyte', 'rb') as train_labels:
        train_labels.seek(4, 0)
        count = struct.unpack('>I', train_labels.read(4))[0]
        # todo: refactor to 10 outputs representing 0-9
        labels = []
        for c in range(count):
            output = []
            label = [b for b in train_labels.read(1)][0]
            output.extend([1.0 if r == label else 0.0 for r in range(10)])
            labels.append(output)
        return np.array(labels)

inputs = 784
hidden = 300
outputs = 10
activation = 'logistic'
X = image_input()
y = image_labels()
print('building network ... {}->{}->{}'.format(inputs, hidden, outputs))
nn = NeuralNetwork([inputs,hidden,outputs], activation)
print('training ...')
nn.fit(X,y)

# run test data and determine error rate
with open('handwriting/t10k-images.idx3-ubyte', 'rb') as test_images:
    with open('handwriting/t10k-labels.idx1-ubyte', 'rb') as test_labels:
        test_images.seek(4, 0)
        count = struct.unpack('>I', test_images.read(4))[0]
        failures = 0
        passes = 0
        fail_conf = 0
        pass_conf = 0
        print('\ntesting {} images ...'.format(count))
        errors = {}
        for c in range(count):
            image_offset = c * 784 + 16
            label_offset = c + 8
            test_images.seek(image_offset, 0)
            test_labels.seek(label_offset, 0)
            label = [n for n in test_labels.read(1)][0]
            X = np.array(
                [r / 255 for r in [b for b in test_images.read(784)]]
            )
            output = [n for n in nn.predict(X)]
            prediction = [(a, b) for a,b in enumerate(output) \
                if b == max(output)
            ][0]
            if prediction[0] != label:
                if prediction[0] not in errors.keys():
                    errors[prediction[0]] = 1
                errors[prediction[0]] += 1
                fail_conf += prediction[1]
                failures += 1
                avg_fail_conf = round(fail_conf / failures * 100, 1)
            else:
                pass_conf += prediction[1]
                passes += 1
                avg_pass_conf = round(pass_conf / passes * 100, 1)
        print('\n{}% mis-identification rate'.format(
            round(failures / count * 100, 1)
        ))
        print('{}% average pass confidence'.format(avg_pass_conf))
        print('{}% average fail confidence'.format(avg_fail_conf))
        print('\nmisidentifaction percentage by digit:')
        for k in sorted(errors.keys()):
            percent = round(errors[k] / failures * 10)
            print('{} |{}{}|'.format(
                k, '#' * percent, ' ' * (10 - percent)
            ))
        # view and test individual images from test images
        while True:
            print('\ntest image number (Q to Quit): ', end = '')
            user = input()
            if user.upper() == 'Q':
                break
            try:
                image_offset = int(user) * 784 + 16
                label_offset = int(user) + 8                
                test_images.seek(image_offset, 0)
                test_labels.seek(label_offset, 0)
                label = str([b for b in test_labels.read(1)][0])
                X = np.array([r / 255 for r in \
                    [b for b in test_images.read(784)]
                ])
                output = [n for n in nn.predict(X)]
                prediction = [(a, round(b * 100, 2)) for a, b in \
                    enumerate(output) if b == max(output)][0]
                losers = [b for b in output if b != max(output)]
                runner_up = [(a, round(b * 100, 2)) for a, b in \
                    enumerate(losers) if b == max(losers)][0]
                test_images.seek(image_offset, 0)
                for r in range(28):
                    row = [b / 255 for b in test_images.read(28)]
                    print(
                        ''.join(
                            ['#' if b > .5 else '+' if b > 0 else ' ' \
                                for b in row
                            ]
                        )
                    )
                if int(label) == prediction[0]:
                    print('{} SUCCESS!'.format(user))
                else:
                    print('{} FAILURE!'.format(user))
                print('label:\t\t{}'.format(label))
                print('predicted:\t{}\t{}%'.format(
                    prediction[0], prediction[1]
                ))
                print(
                    'runner up: \t{}\t{}%'.format(runner_up[0], runner_up[1])
                )
            except Exception as e:
                print(e)