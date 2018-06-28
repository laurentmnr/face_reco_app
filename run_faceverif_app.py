import os


os.system('export PYTHONPATH=/Users/laurent/Desktop/facenet-master/src')

print()
print()
print()


name = input("Enter your name: \n")
os.system('python3 src/take_pictures.py --person '+name+' --n_pictures 3 datasets/Perso')
os.system('python3 src/align/align_dataset_mtcnn.py datasets/Perso datasets/Perso_aligned')
os.system('python3 src/real_time_face_verification.py')
