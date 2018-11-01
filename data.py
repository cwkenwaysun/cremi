from cremi.io import CremiFile
from cremi import Annotations
from cremi.evaluation import NeuronIds, Clefts, SynapticPartners

import cv2
import numpy as np
import h5py
import scipy.misc
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt



file = CremiFile("data/sample_A+_20160601.hdf", "r")
print type(file)
print file

#print file.read_neuron_ids()

print file.has_neuron_ids_confidence()
print file.read_raw().data

data = file.read_raw().data
data = np.array(data[:,:,:])
for i in range(125):
    plt.imshow(data[i,:,:])
plt.imsave(arr=data[50,:,:], fname="image")

'''
neuron_ids_evaluation = NeuronIds(truth.read_neuron_ids())
(voi_split, voi_merge) = neuron_ids_evaluation.voi(test.read_neuron_ids())
adapted_rand = neuron_ids_evaluation.adapted_rand(test.read_neuron_ids())

clefts_evaluation = Clefts(test.read_clefts(), truth.read_clefts())
fp_count = clefts_evaluation.count_false_positives()
fn_count = clefts_evaluation.count_false_negatives()
fp_stats = clefts_evaluation.acc_false_positives()
fn_stats = clefts_evaluation.acc_false_negatives()

synaptic_partners_evaluation = SynapticPartners()
fscore = synaptic_partners_evaluation.fscore(
    test.read_annotations(),
    truth.read_annotations(),
    truth.read_neuron_ids())
'''