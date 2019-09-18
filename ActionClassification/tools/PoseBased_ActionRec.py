# import required packages and global variables
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import os
import io
import cPickle as pickle
import os.path as osp
import numpy as np
import math
import cv2 as cv
import glob
import datetime
# import extra liraries required for designing the network
import lmdb
import shutil
import time
import caffe2.python.predictor.predictor_exporter as pe
from caffe2.proto import caffe2_pb2
from caffe2.python.predictor import mobile_exporter
from caffe2.python import (
    brew,
    core,
    model_helper,
    net_drawer,
    optimizer,
    visualize,
    workspace,
    memonger
)
# If you would like to see some really detailed initializations,
# you can change --caffe2_log_level=0 to --caffe2_log_level=-1
core.GlobalInit(['caffe2', '--caffe2_log_level=-1'])
print("Necessities for action recognition network is imported!")
### Defining global variables
DTYPE = np.float32
height = 1080  # frame height in pixel
width = 1920  # frame width in pixel
fps = 30.0
col_ch = 4
sigma = 2
resize_scale = 0.125
sub_sample =4000
crop = False
classes=[
    'Sitting',
    'Sit-to-Stand',
    'Standing',
    'Walking',
    'Stand-to-Sit'
]
keypoints = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle']
# Define dataset-specific parameters, and declare model training parameters
# Paths to the init & predict net output locations
init_net_out = 'TuftsAction_init_net.pb'
predict_net_out = 'TuftsAction_predict_net.pb'

# Paths to LMDBs
training_lmdb_path = osp.join('/data','BehnazData','Results', 'PoseBased_ActionRec', 'training_normalized_valid_lmdb')
validation_lmdb_path = osp.join('/data', 'BehnazData', 'Results', 'PoseBased_ActionRec', 'validation_normalized_valid_lmdb')
testing_lmdb_path = osp.join('/data', 'BehnazData', 'Results', 'PoseBased_ActionRec', 'testing_normalized_valid_lmdb')

# Dataset specific params
data_db_type = "lmdb"
image_width = int(width * resize_scale)                # input image width
image_height = int(height * resize_scale)               # input image height
image_channels = 14 * col_ch                           # input image channels
num_classes = 5                                        # number of action classes

# Training params
num_epoch = 30
batch_size =  50        # total batch size
validation_interval = 50                               # validate every <validation_interval> training iterations
checkpoint_iters = 500                                 # output checkpoint db every <checkpoint_iters> iterations
base_learning_rate = 0.005          # initial learning rate (scale with total batch size)
step_size = 1                                     # influence the learning rate after 10 epochs
weight_decay = 1e-3                                     # weight decay (L2 regularization)

root_folder = os.path.join('..','classification_net','SingleGPU')
# Create root_folder if not already there
if not os.path.isdir(root_folder):
    os.makedirs(root_folder)

# Resetting workspace with root_folder argument sets root_folder as working directory
workspace.ResetWorkspace(root_folder)

### Defining helper functions
def AddInput(model, db, db_type, batch_size, mirror=0):
    # load the data
    data_f32, label = brew.db_input(
        model,
        blobs_out=["data_f32", "label"],
        batch_size=batch_size,
        db=db,
        db_type=db_type,
    )
    data_org = model.Cast(data_f32, "data_org", to=core.DataType.FLOAT)
    # mirroring tensor data randomly
    if mirror and np.random.uniform() > 0.49:
        data_org = workspace.FetchBlob("data_org")
        data = np.flip(data_org, axis=2)
        workspace.FeedBlob("data", data)
    else:
        data = model.Copy(data_org, "data")
        # prevent back-propagation: optional performance improvement; may not be observable at small scale
    data = model.StopGradient(data, data)


# Helper function for maintaining the correct height and width dimensions after
# convolutional and pooling layers downsample the input data
def update_dims(height, width, kernel, stride, pad):
    new_height = ((height - kernel + 2 * pad) // stride) + 1
    new_width = ((width - kernel + 2 * pad) // stride) + 1
    return new_height, new_width


# Defining the action classification network model
def Add_Action_Tufts_Model(model, num_classes, image_height, image_width, image_channels, is_test=0):
    ################################## Block 1 ############################
    # Convolutional layer 1
    #    conv1_1 = brew.conv(model, 'data', 'conv1_1', dim_in=image_channels, dim_out=64, kernel=3, stride=2, pad=0)
    #    h,w = update_dims(height=image_height, width=image_width, kernel=3, stride=2, pad=0)
    # ReLU layer 1
    #    relu1_1 = brew.relu(model, conv1_1, 'relu1_1')
    # Batch normalization layer 1
    #    bn1_1 = brew.spatial_bn(model, relu1_1, 'bn1_1', dim_in=64, epsilon=1e-3, momentum=0.1, is_test=is_test)
    # Drop out with p=0.25
    #    dropout1_1 = brew.dropout(model, bn1_1, 'dropout1_1', ratio=0.25, is_test=is_test)

    # Convolutional layer 2
    #    conv1_2 = brew.conv(model, dropout1_1, 'conv1_2', dim_in=64, dim_out=64, kernel=3, stride=1, pad=0)
    #    h,w = update_dims(height=h, width=w, kernel=3, stride=1, pad=0)
    # ReLU layer 1
    #    relu1_2 = brew.relu(model, conv1_2, 'relu1_2')
    # Batch normalization layer 1
    #    bn1_2 = brew.spatial_bn(model, relu1_2, 'bn1_2', dim_in=64, epsilon=1e-3, momentum=0.1, is_test=is_test)
    # Drop out with p=0.25
    #    dropout1_2 = brew.dropout(model, bn1_2, 'dropout1_2', ratio=0.25, is_test=is_test)
    ##################################### Block 2 ##########################
    # Convolutional layer 3
    conv2_1 = brew.conv(model, 'data', 'conv2_1', dim_in=image_channels, dim_out=128, kernel=3, stride=2, pad=0)
    h, w = update_dims(height=image_height, width=image_width, kernel=3, stride=2, pad=0)
    # ReLU layer 1
    relu2_1 = brew.relu(model, conv2_1, 'relu2_1')
    # Batch normalization layer 1
    bn2_1 = brew.spatial_bn(model, relu2_1, 'bn2_1', dim_in=128, epsilon=1e-3, momentum=0.1, is_test=is_test)
    # Drop out with p=0.25
    dropout2_1 = brew.dropout(model, bn2_1, 'dropout2_1', ratio=0.25, is_test=is_test)

    # Convolutional layer 4
    conv2_2 = brew.conv(model, dropout2_1, 'conv2_2', dim_in=128, dim_out=128, kernel=3, stride=1, pad=0)
    h, w = update_dims(height=h, width=w, kernel=3, stride=1, pad=0)
    # ReLU layer 1
    relu2_2 = brew.relu(model, conv2_2, 'relu2_2')
    # Batch normalization layer 1
    bn2_2 = brew.spatial_bn(model, relu2_2, 'bn2_2', dim_in=128, epsilon=1e-3, momentum=0.1, is_test=is_test)
    # Drop out with p=0.25
    dropout2_2 = brew.dropout(model, bn2_2, 'dropout2_2', ratio=0.25, is_test=is_test)
    ##################################### Block 3 ############################
    # Convolutional layer 5
    conv3_1 = brew.conv(model, dropout2_2, 'conv3_1', dim_in=128, dim_out=256, kernel=3, stride=2, pad=0)
    h, w = update_dims(height=h, width=w, kernel=3, stride=2, pad=0)
    # ReLU layer 1
    relu3_1 = brew.relu(model, conv3_1, 'relu3_1')
    # Batch normalization layer 1
    bn3_1 = brew.spatial_bn(model, relu3_1, 'bn3_1', dim_in=256, epsilon=1e-3, momentum=0.1, is_test=is_test)
    # Drop out with p=0.25
    dropout3_1 = brew.dropout(model, bn3_1, 'dropout3_1', ratio=0.25, is_test=is_test)

    # Convolutional layer 4
    conv3_2 = brew.conv(model, dropout3_1, 'conv3_2', dim_in=256, dim_out=256, kernel=3, stride=1, pad=0)
    h, w = update_dims(height=h, width=w, kernel=3, stride=1, pad=0)
    # ReLU layer 1
    relu3_2 = brew.relu(model, conv3_2, 'relu3_2')
    # Batch normalization layer 1
    bn3_2 = brew.spatial_bn(model, relu3_2, 'bn3_2', dim_in=256, epsilon=1e-3, momentum=0.1, is_test=is_test)
    # Drop out with p=0.25
    dropout3_2 = brew.dropout(model, bn3_2, 'dropout3_2', ratio=0.25, is_test=is_test)

    # Global average pooling
    pool1 = brew.average_pool(model, dropout3_2, 'pool1', global_pooling=True)
    # Fully connected layers
    pred = brew.fc(model, pool1, 'fc1', dim_in=256, dim_out=num_classes)
    # Softmax layer
    softmax, loss = model.SoftmaxWithLoss([pred, 'label'], ['softmax', 'loss'])
    brew.accuracy(model, [softmax, 'label'], 'accuracy')
    model.net.MultiClassAccuracy([softmax, 'label'], ['accuracy_per_class', 'amount_per_class'])
    return [loss]


def AddOptimizerOps_fixsgd(model):
    optimizer.build_sgd(
        model,
        base_learning_rate=0.01,
        policy="fixed",
        momentum=0.9,
        weight_decay=0.004
    )


def AddOptimizerOps_adam(model):
    # Use adam as optimization function
    optimizer.build_adam(
        model,
        base_learning_rate=base_learning_rate
        #        policy="step",
        #        momentum=0.9,
        #        weight_decay=0.004
    )


def AddOptimizerOps_sgd(model):
    """Add optimizer ops."""
    optimizer.build_sgd(model, base_learning_rate=0.01,
                        policy='step', stepsize=1, gamma=0.999,
                        momentum=0.9, nesterov=False)


def AddOptimizerOps_nestsgd(model):
    brew.add_weight_decay(model, weight_decay)
    iter = brew.iter(model, "iter")
    lr = model.net.LearningRate(
        [iter],
        "lr",
        base_lr=base_learning_rate,
        policy="step",
        stepsize=step_size,
        gamma=0.1,
    )
    for param in model.GetParams():
        param_grad = model.param_to_grad[param]
        param_momentum = model.param_init_net.ConstantFill(
            [param], param + '_momentum', value=0.0
        )

        # Update param_grad and param_momentum in place
        model.net.MomentumSGDUpdate(
            [param_grad, param_momentum, lr, param],
            [param_grad, param_momentum, param],
            # almost 100% but with room to grow
            momentum=0.9,
            # netsterov is a defenseman for the Montreal Canadiens, but
            # Nesterov Momentum works slightly better than standard momentum
            nesterov=1,
        )


def AddAccuracy(model):
    accuracy = brew.accuracy(model, ["softmax", "label"], "accuracy")
    return accuracy


def OptimizeGradientMemory(model, loss):
    model.net._net = memonger.share_grad_blobs(
        model.net,
        loss,
        set(model.param_to_grad.values()),
        namescope="memaction",
        share_activations=False,
    )
### Adding check-points
# Create uniquely named directory under root_folder to output checkpoints to
unique_timestamp = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
checkpoint_dir = os.path.join(root_folder, unique_timestamp)
os.makedirs(checkpoint_dir)
print("Checkpoint output location: ", checkpoint_dir)

# Add checkpoints to a given model
def AddCheckpoints(model, checkpoint_iters, db_type):
    ITER = brew.iter(model, "iter")
    model.Checkpoint([ITER] + model.params, [],
                           db=os.path.join(unique_timestamp, "action_tufts_checkpoint_%05d.lmdb"),
                           db_type="lmdb", every=checkpoint_iters)

## Defining Training net and Test net creating functions
arg_scope = {"order": "NCHW"}
# TRAINING MODEL
def createTrainModel(training_lmdb_path, batch_size):
    """Create and return a training model, complete with training ops."""
    train_model = model_helper.ModelHelper(name='train_net', arg_scope=arg_scope)
    AddInput(train_model, db=training_lmdb_path, db_type=data_db_type, batch_size=batch_size, mirror=0)
    losses = Add_Action_Tufts_Model(train_model,num_classes, image_height, image_width, image_channels, is_test=0)
    train_model.AddGradientOperators(losses)
    AddOptimizerOps_adam(train_model)
    AddCheckpoints(train_model, checkpoint_iters, db_type="lmdb")
    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net, overwrite=True)
    return train_model

# VALIDATION MODEL
def createValidationModel(validation_lmdb_path, batch_size):
    """Create and return a test model. Does not include training ops."""
    val_model = model_helper.ModelHelper(name='val_net', arg_scope=arg_scope, init_params=False)
    AddInput(val_model, db=validation_lmdb_path, db_type=data_db_type, batch_size=batch_size)
    losses = Add_Action_Tufts_Model(val_model,num_classes, image_height, image_width, image_channels, is_test=1)
    workspace.RunNetOnce(val_model.param_init_net)
    workspace.CreateNet(val_model.net, overwrite=True)
    return val_model

def CreateDeployModel():
    deploy_model = model_helper.ModelHelper(name="deploy_net", arg_scope=arg_scope, init_params=False)
    Add_Action_Tufts_Model(deploy_model,num_classes, image_height, image_width, image_channels, is_test=1)
    workspace.RunNetOnce(val_model.param_init_net)
    workspace.CreateNet(val_model.net, overwrite=True)
    return deploy_model

### train and validate
# initialize the logging variables
val_loss = np.zeros(num_epoch)
val_total_accuracy = np.zeros(num_epoch)
train_loss = np.zeros(num_epoch)
train_accuracy = np.zeros(num_epoch)
val_class_accuracy = np.zeros((num_epoch, num_classes))
val_class_count = np.zeros(num_classes, dtype=int)
val_count = 0
tot_itr_count = 0
total_time = 0
# defining GPU device and training/ validation networks
device = 0
with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, device)):
    train_model = createTrainModel(training_lmdb_path, batch_size)
    val_model = createValidationModel(validation_lmdb_path, batch_size=batch_size)

# iteraring the forward/ backword pass for optimizing variables
train_iter_per_epoch = train_data_count // batch_size
val_iter_per_epoch = validation_count // batch_size
# Now, we run the network (forward & backward pass)
for epoch in range(1, num_epoch+1):
    t1 = time.time()
    accuracies = []
    losses = []
    for itr in range(1, train_iter_per_epoch+1):
        # Stopwatch start!
        tot_itr_count += 1
        workspace.RunNet(train_model.net)
        accuracies.append(workspace.FetchBlob('accuracy'))
        losses.append(workspace.FetchBlob('loss'))
        #if not tot_itr_count % disp_interval:
    train_loss[val_count] = np.array(losses).mean()
    train_accuracy[val_count] = np.array(accuracies).mean()
    t2 = time.time()
    dt = t2 - t1
    total_time += dt
    # Validate every epoch
    print("...epoch:{}/{}   el_time:{}".format(epoch, num_epoch, dt))
    print("training loss:{}, train_accuracy:{}".format(train_loss[val_count], train_accuracy[val_count]))
    losses = []
    accuracies = []
    accuracies_per_class = []
    class_count = []
    for _ in range(val_iter_per_epoch):
        workspace.RunNet(val_model.net)
        losses.append(workspace.FetchBlob('loss'))
        accuracies.append(workspace.FetchBlob('accuracy'))
        accuracies_per_class.append(workspace.FetchBlob('accuracy_per_class'))
        if epoch == num_epoch:
            class_count.append(workspace.FetchBlob('amount_per_class'))
    val_loss[val_count] = np.array(losses).mean()
    val_total_accuracy[val_count] = np.array(accuracies).mean()
    val_class_accuracy[val_count, :] = np.array(accuracies_per_class).mean(axis=0)
    if epoch == num_epoch:
        val_class_count = np.array(class_count).sum(axis=0)
    print("Validation Loss:{}, Validation total accuracy:{}, Per class validation accuracy:{}"
          .format(val_loss[val_count],val_total_accuracy[val_count], val_class_accuracy[val_count, :] ))
    val_count += 1

print("Per class data count: Sitting={}, Sit-to-Stand={}, Standing={}, Walking={}, Stand-to-Sit={}"
      .format(val_class_count[0], val_class_count[1],
            val_class_count[2], val_class_count[3],
            val_class_count[4]))
print('total elapsed time is {}'.format(total_time))

### Save trained model
# Run init net and create main net
#workspace.RunNetOnce(deploy_model.param_init_net)
#workspace.CreateNet(deploy_model.net, overwrite=True)

# Use mobile_exporter's Export function to acquire init_net and predict_net
init_net, predict_net = mobile_exporter.Export(workspace, val_model.net, val_model.params)

# Locations of output files
full_init_net_out = os.path.join(checkpoint_dir, init_net_out)
full_predict_net_out = os.path.join(checkpoint_dir, predict_net_out)

# Simply write the two nets to file
with open(full_init_net_out, 'wb') as f:
    f.write(init_net.SerializeToString())
with open(full_predict_net_out, 'wb') as f:
    f.write(predict_net.SerializeToString())
print("Model saved as " + full_init_net_out + " and " + full_predict_net_out)
