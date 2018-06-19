# face_reco_app

## Download models

https://drive.google.com/file/d/1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz/view

To save in a pretrained_models directory



## HOWTO face recognition:

### Pre-settings:
Place at root of facet-master
export PYTHONPATH=/Users/laurent/Desktop/facenet-master/src


### Take pictures:
python3 src/take_pictures.py --person Laurent --n_pictures 10 datasets/Perso

### Align data:
python3 src/align/align_dataset_mtcnn.py datasets/Perso datasets/Perso_aligned

### Train classifier:
python3 src/classifier.py TRAIN datasets/Perso_aligned pretrained_models/20180408-102900/20180408-102900.pb classifier_models/model_perso2.pkl

### Live face recognition:
BE CAREFUL: change pretrained model/ classifier model in face.py file

python3 src/real_time_face_recognition.py

### Live face verification:
BE CAREFUL: change pretrained model/ classifier model in face2.py file

python3 src/real_time_face_verification.py
