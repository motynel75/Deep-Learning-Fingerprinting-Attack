import os
import subprocess
import signal
from subprocess import call

for iter in range(0,10):
    print("batch number: ",iter)
    proc = subprocess.call('sudo python3 transactions-simulator.py '+str(iter),shell=True)
