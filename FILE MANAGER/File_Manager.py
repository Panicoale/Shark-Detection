import os
import shutil

#INITIAL SETTING

INPUT_path='/Users/panicoale/PycharmProjects/OpenLabeling/OpenLabeling/main/input'
INPUT_txt_path='/Users/panicoale/PycharmProjects/OpenLabeling/OpenLabeling/main/output/YOLO_darknet'
OUTPUT_path='/home/alessandro/Shark/darknet-master/build/darknet/x64/data/obj'

txt_files = []
text='/home/alessandro/Shark/darknet-master/build/darknet/x64/data/obj/'
# r=root, d=directories, f = files
directories=-1
for r, d, f in os.walk(INPUT_path):
    for dir in d:
        directories+=1
        new_dir=[]
        new_dir.append(r+'/'+dir)
        new_dir = os.path.join(r, dir)
        print(new_dir)

        # COPIA FILE JPG IN OUTPUT FOLDER
        files = os.listdir(new_dir)
        for file_name in files:
            full_file_name = os.path.join(new_dir, file_name)


            if '.jpg' in file_name:
                #shutil.copy(full_file_name, OUTPUT_path)
                print(full_file_name)
                #print('\n')



                #SCRIVI IL PATH TEMPORANEO DEL FILE TXT
                #new_dir_txt=[]
                new_dir_txt = os.path.join(INPUT_txt_path, dir)
                full_file_name2 = os.path.join(new_dir_txt, file_name[0:-3]+'txt')
                #shutil.copy(full_file_name2, OUTPUT_path)
                print(full_file_name2)

                #destination_path
                #TXT_full_file_name
                #txt_files.append(full_file_name2)

                #COPIA FILE TXT IN OUTPUT FOLDER



        # src_files = os.listdir(src)
        # for file_name in src_files:
        #     full_file_name = os.path.join(src, file_name)
        #     if os.path.isfile(full_file_name):
        #         shutil.copy(full_file_name, dest)

print(txt_files)
# with open('train2.txt', 'w') as f:
#     for item in files:
#         f.write("%s\n" % item)
#        for f in os.walk(path):
#         if '.jpg' in file:
#
#             files.append(text+file)
#
# for f in files:
#     print(f)
#
# with open('train2.txt', 'w') as f:
#     for item in files:
#         f.write("%s\n" % item)
