import cv2
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--label_file", required=True, help="label_file path")
parser.add_argument("--gen_dir", required=True, help="test dir")
parser.add_argument("--out_dir", required=True, help="output dir")
parser.add_argument("--batch_size", default=4, help="batch size when doing testing")
parser.add_argument("--num", default=500, help="number of output images(bs*bs*number)")
parser.add_argument("--checkpoint_iter", default=10000, help="loaded checkpoint iteration")
opt = parser.parse_args()

out_dir = opt.out_dir
num = opt.num
label_file = opt.label_file
gen_dir = opt.gen_dir
checkpoint_iter = opt.checkpoint_iter
batch_size = opt.batch_size

#label_file = '/home/sjqian/WFLW/train_98pt.txt'
#gen_dir = '/home/sjqian/stylealign/log/test_0003/'
#out_dir = '/home/sjqian/stylealign/log/test_0003/'
#num = 500
#batch_size = 4
#checkpoint_iter = 10000

label = open(label_file)
label = label.readlines()


os.makedirs(os.path.join(out_dir,'output'),exist_ok = True)
os.makedirs(os.path.join(out_dir,'output/Image/'), exist_ok = True)
new_label = open(os.path.join(out_dir,'output')+'label.txt,'w')
"transfer_{:07}.png".format(global_step)
for i in tqdm(range(num)):
   cur = cv2.imread(gen_dir + 'transfer_{:07}.png'.format(checkpoint_iter+i+1))
   for j in range(batch_size):
       cur_landmark = label[i* batch_size+ j + batch_size][:(label[i* batch_size + j + batch_size].rfind(' '))]
       for n in range(batch_size):
           cur_name = str(i*batch_size*batch_size + j * batch_size + n) + '.jpg'
           now = cur[256*(n+1):256*(n+2),(j+1)*256:(j+2)*256,:]
           now = cv2.resize(now,(384,384),interpolation=cv2.INTER_CUBIC)
           cv2.imwrite(out_dir+'output/Image/'+cur_name,now)
           new_label.write(cur_landmark+' '+ cur_name+'\n')
new_label.close()
