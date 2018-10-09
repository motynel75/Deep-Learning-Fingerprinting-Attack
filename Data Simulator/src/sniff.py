from __future__ import print_function

import logging
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

from scapy.all import *
import os
import argparse

## Parsing argument
parser = argparse.ArgumentParser()
parser.add_argument("label")
args = parser.parse_args()

label = str(args.label)
## Create a Packet Counter
counter = 0

# Saving captured traffic into .txt file
file = open('/home/amine/packet_logs/'+label+'.txt',"w") # Modify target path

## Define Custom Action function
def custom_action(packet):

    if hasattr(packet[0][1],"src") :

        global counter
        counter += 1
        res = 'Packet Timestamps :'+ str(packet.time)+' Packet #{}: {} ==> {}'.format(counter, packet[0][1].src, packet[0][1].dst)
        file.write(res + os.linesep)
        return res

## Setup sniff, filtering for IP traffic
packets = sniff(prn=custom_action)
