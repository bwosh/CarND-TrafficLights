#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1))
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


from keras import backend as K
from keras.models import Sequential, model_from_json

sess = K.get_session()
K.set_learning_phase(0)
with open("model.json", 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
weights_path = "model.h5"
model.load_weights(weights_path)
model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['acc'])
K.set_learning_phase(0)

# get dataset
from train_classifier import RedNoRedDataset
red_folders=['./../../data/crops/coco_red/']
nored_folders=['./../../data/crops/coco_no_red/']
dataset = RedNoRedDataset( 32, red_folders, nored_folders)
dataset_train, dataset_test = dataset.split()


# verify
from torch.utils.data import DataLoader
testloader = DataLoader(dataset_test, batch_size=1, num_workers=0, shuffle=False)

def generator(loader, epochs):
    for e in range(epochs):
        for batch_idx,batch in enumerate(loader): 
            input, clsid = batch

            input = input.numpy().transpose(0,2,3,1)
            clsid = clsid.numpy()
            yield input, clsid

x = model.evaluate_generator(generator(testloader, 1), 
        steps=len(testloader), 
        callbacks=None, 
        max_queue_size=1, 
        workers=1, use_multiprocessing=False, verbose=1)

print("DONE.")
print(x)
