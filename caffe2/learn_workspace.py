from caffe2.python import workspace, model_helper
import numpy as np
x = np.random.rand(4,3,2)
print(x)
print(x.shape)
workspace.FeedBlob("my_x", x)
x2 = workspace.FetchBlob("my_x")
print(x2)
