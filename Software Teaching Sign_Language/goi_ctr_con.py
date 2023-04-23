from Chuyen_co_dau_thanh_khong_dau import ma_hoa_telex
import time
import serial
from serial import Serial
serialcom = serial.Serial('COM4',9600)
serialcom.timeout = 1
s = " Việt nam"
print (s)
print(len(s))
output = []
time.sleep(1)
for char in s:
    output= ma_hoa_telex(char)
    print (output[0])
    serialcom.write(output[0].encode())
    time.sleep(4)
    if (len(output[1]) != 0 ):
        print (output[1])
        serialcom.write(output[1].encode())
        time.sleep(4)
    
