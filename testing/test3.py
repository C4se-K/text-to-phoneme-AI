import os

DIR_NAME = os.path.join(os.path.dirname(__file__), 'American-English')
SUB_DIR = ['Consonants', 'Conventions', 'Vowels']
file_list = []

for dir in SUB_DIR:
    directory_path = os.path.join(DIR_NAME, dir)
    new_list = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    file_list += (new_list)
     

#print list of files
for file in file_list:
    print(file)

print(file_list.__len__())