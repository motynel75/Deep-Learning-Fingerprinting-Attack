import os
import glob


# extracting file name
for filename in glob.iglob('/Users/Amine/Desktop/Imperial College - CS/Spring term/Msc Project/packet_logs/*'):

    path = filename
    filename = filename.split('/')[-1]
    filename2 = filename.split('_')


    if filename2[0] == 'jc':
        number = int(filename2[-1].strip('.txt')) + 25
        filename2[-1] = str(number)+'.txt'

    filename2 = '_'.join(filename2)
    print(filename2)

    #create processed file
    filepath = os.path.join('/Users/Amine/Desktop/Imperial College - CS/Spring term/Msc Project/packet_logs_formated_credential_100/', filename2)
    processed_file = open(filepath, "w+", newline='')

    with open(path, 'r+', newline='') as fp:
       line = fp.readline()
       first_timestamp = line.split()[2][1:]
       first_timestamp = float(first_timestamp)
       

       while line:
           filtered_line = []
           cur_timestamp = str(float(line.split()[2][1:])-first_timestamp)

           #Hardcoded source/destination
           dir = line.split()[5]+line.split()[6]+line.split()[7]
           if dir == '10.0.0.1==>10.0.0.2' :
               filtered_line.append(cur_timestamp)
               filtered_line.append('1')

           elif dir == '10.0.0.2==>10.0.0.1' :
               filtered_line.append(cur_timestamp)
               filtered_line.append('-1')

           #writing new line in file
           for item in filtered_line :
               if filtered_line and float(filtered_line[0]) >= 0 and float(filtered_line[0]) <= 5.0:
                   processed_file.write("%s " % item)

           processed_file.write("\n")

           if filtered_line:
               pass


           line = fp.readline()

    processed_file.close()
