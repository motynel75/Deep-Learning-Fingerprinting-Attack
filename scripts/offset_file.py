import os
import glob

#This script rename file label
data_loc = "/Users/Amine/Desktop/Imperial College - CS/Spring term/Msc Project/packet_logs_formated_credential"

jc_cred_list =[]
tsp_cred_list =[]

for filename in glob.iglob(data_loc+'/*'):
    file_name = filename.split('/')[-1]
    file_name_plit = file_name.split('_')

    if file_name_plit[0] == 'jc' :
        jc_cred_list.append(file_name)

    if file_name_plit[0] == 'tsp':
        tsp_cred_list.append(file_name)

print(jc_cred_list)
print(tsp_cred_list)


for i in range(len(jc_cred_list)) :
    os.rename(jc_cred_list[i], 'jc_'+str(i)+'.txt')

for i in range(len(tsp_cred_list)) :
    os.rename(tsp_cred_list[i], 'tsp_'+str(i)+'.txt')
