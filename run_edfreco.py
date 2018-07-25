import os


os.system('export PYTHONPATH=/Users/laurent/Desktop/facenet-master/src')

print()
print()
print()


os.system('python3 src/align/align_dataset_mtcnn.py datasets/EDF datasets/EDF_aligned')



os.system('python3 src/classifier.py TRAIN datasets/EDF_aligned pretrained_models/20180408-102900/20180408-102900.pb classifier_models/edf_classifier.pkl')

os.system('python3 src/real_time_edf_recognition.py')
