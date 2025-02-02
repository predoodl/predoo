import json


if __name__ == '__main__':
    op_info = [{"OpName": "conv2d",
                  "InputShape": [[1,3,224,224],[1,1,1,1],[1,3,112,112],
                                 [1,3,56,56],[1,3,28,28],[1,3,14,14],[1,3,7,7]],
                  "MeanErr_threshold": 5e-4,
                  "MaxErr_threshold": 0.0025,
                  "data_format": ["NCHW","NHWC"],
                  "filters": 8,
                  "kernel_size": 3,
                  "strides": (1,1),
                  "padding": "valid"
                  },
                  {"OpName": "BatchNormalization",
                  "InputShape": [[1,3,224,224],[1,1,1,1],[1,3,112,112],
                                 [1,3,56,56],[1,3,28,28],[1,3,14,14],[1,3,7,7]],
                  "MeanErr_threshold": 5e-5,
                  "MaxErr_threshold": 0.001,
                  "momentum": 0.99,
                  "epsilon": 0.001,
                  },
                  {"OpName": "MaxPooling2D",
                   "InputShape": [[1, 3, 224, 224], [1, 1, 1, 1], [1, 3, 112, 112],
                                  [1, 3, 56, 56], [1, 3, 28, 28], [1, 3, 14, 14], [1, 3, 7, 7]],
                   "MeanErr_threshold": 1e-4,
                   "MaxErr_threshold": 0.0005,
                   "pool_size": 2,
                   "strides": 1,
                   "padding": "valid"
                  },
                  {"OpName": "Relu",
                   "InputShape": [ [1, 1, 1, 1], [1, 3, 56, 56], [1, 3, 28, 28],
                                   [1, 3, 14, 14], [1, 3, 7, 7],[2,2],[1,1]],
                   "MeanErr_threshold": 1e-4,
                   "MaxErr_threshold": 0.0003,
                  },
                  {"OpName": "Sigmoid",
                   "InputShape": [[1, 1, 1, 1], [1, 3, 56, 56], [1, 3, 28, 28],
                                  [1, 3, 14, 14], [1, 3, 7, 7], [2, 2], [1, 1]],
                   "MeanErr_threshold": 1e-4,
                   "MaxErr_threshold": 0.0003,
                   },
                  {"OpName": "Tanh",
                   "InputShape": [[1, 1, 1, 1], [1, 3, 56, 56], [1, 3, 28, 28],
                                  [1, 3, 14, 14], [1, 3, 7, 7], [2, 2], [1, 1]],
                   "MeanErr_threshold": 1e-4,
                   "MaxErr_threshold": 0.0002,
                   },
                  {"OpName": "Softmax",
                   "InputShape": [[1, 1, 1, 1], [1, 3, 56, 56], [1, 3, 28, 28],
                                  [1, 3, 14, 14], [1, 3, 7, 7], [2, 2], [1, 1]],
                   "MeanErr_threshold": 1e-5,
                   "MaxErr_threshold": 0.0003,
                   },

                  ]
    json_str = json.dumps(op_info, indent=4)

    with open('OpInfo.json', 'w') as f:  # 创建一个params.json文件
        f.write(json_str)  # 将json_str写到文件中


