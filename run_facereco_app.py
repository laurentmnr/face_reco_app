import os


os.system('export PYTHONPATH=/Users/laurent/Desktop/facenet-master/src')

print()
print()
print()

while True:
    addperson= input("Add a person? If yes, Press ENTER, otherwise press N \n")
    if len(addperson)>1 and addperson[-1]=='N':
        break
    print()
    name = input("Enter your name: \n")
    os.system('python3 src/take_pictures.py --person '+name+' --n_pictures 3 datasets/Perso')


    print()
    print()
    print()




os.system('python3 src/align/align_dataset_mtcnn.py datasets/Perso datasets/Perso_aligned')



os.system('python3 src/classifier.py TRAIN datasets/Perso_aligned pretrained_models/20180408-102900/20180408-102900.pb classifier_models/model_perso2.pkl')

os.system('python3 src/real_time_face_recognition.py')
