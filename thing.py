cnt=0
f_cnt =0
import os
for file in os.listdir('/vol/bitbucket/sna21/dataset/UBI_FIGHTS/multi/pose_outputs/normal'):
    cnt += 1
print("Length of Normal pose folder:", cnt)

for file in os.listdir('/vol/bitbucket/sna21/dataset/UBI_FIGHTS/multi/pose_outputs/fight'):
    f_cnt += 1
print("Length of Fight pose folder:", f_cnt)