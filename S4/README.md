EVA S4 (Exploring classifier networks using MNIST)

How to run:
'Run All' cells

How to change network:
Replace the network class name in the last cell and run.

Best performing networks:

Net13:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 28, 28]             160
       BatchNorm2d-2           [-1, 16, 28, 28]              32
         MaxPool2d-3           [-1, 16, 14, 14]               0
           Dropout-4           [-1, 16, 14, 14]               0
            Conv2d-5           [-1, 32, 14, 14]           4,640
       BatchNorm2d-6           [-1, 32, 14, 14]              64
         MaxPool2d-7             [-1, 32, 7, 7]               0
           Dropout-8             [-1, 32, 7, 7]               0
            Conv2d-9             [-1, 16, 5, 5]           4,624
      BatchNorm2d-10             [-1, 16, 5, 5]              32
          Dropout-11             [-1, 16, 5, 5]               0
           Linear-12                   [-1, 10]           4,010
================================================================
Total params: 13,562

Accuracy on test set: 99.06 in 20 epochs

This uses conv layers, Max Pool layers, batch norm, dropout, and FC layer.

Net14:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 28, 28]             160
       BatchNorm2d-2           [-1, 16, 28, 28]              32
         MaxPool2d-3           [-1, 16, 14, 14]               0
           Dropout-4           [-1, 16, 14, 14]               0
            Conv2d-5           [-1, 32, 14, 14]           4,640
       BatchNorm2d-6           [-1, 32, 14, 14]              64
         MaxPool2d-7             [-1, 32, 7, 7]               0
           Dropout-8             [-1, 32, 7, 7]               0
            Conv2d-9             [-1, 16, 5, 5]           4,624
      BatchNorm2d-10             [-1, 16, 5, 5]              32
          Dropout-11             [-1, 16, 5, 5]               0
           Conv2d-12             [-1, 10, 3, 3]           1,450
        AvgPool2d-13             [-1, 10, 1, 1]               0
================================================================
Total params: 11,002

Accuracy: 98.85 for 20 epochs.

This uses conv layers, Max Pool layers, batch norm, dropout, and GAP layer.

