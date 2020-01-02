import numpy as np
import cv2
import math
from numpy import linalg as LA
import time


def distance(p1, p2):
  return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))


def curve_interp(src, samples, index):
  assert(src.shape[0] > 2)
  assert(samples >= 2)
  # length = np.zeros((src.shape[0]-1))

  # time_0 = time.time()

  src_1 = src[0:src.shape[0]-1, :]
  src_2 = src[1:src.shape[0], :]
  src_delta = src_1 - src_2
  length = np.sqrt(src_delta[:,0]**2 + src_delta[:,1]**2)
  assert(length.shape[0] == src.shape[0]-1)

  # for i in range(length.shape[0]):
  #   length[i] = distance(src[i, :], src[i+1, :])

  # time_1 = time.time()
  # if index == 0:
  #   print('   time_0_1_{}:{}'.format(index, time_1-time_0))


  accu_length = np.zeros((src.shape[0]))
  for i in range(1,accu_length.shape[0]):
    accu_length[i]=accu_length[i-1]+length[i-1]
  dst = np.zeros((samples, 2))
  pre_raw = 0

  # time_2 = time.time()
  # if index == 0:
  #   print('   time_1_2_{}:{}'.format(index, time_2-time_1))


  step_interp = accu_length[accu_length.shape[0]-1] / float(samples-1);
  dst[0, :] = src[0, :]
  dst[dst.shape[0]-1, :] = src[src.shape[0]-1, :]
  for i in range(1,samples-1):
    covered_interp = step_interp*i
    while (covered_interp > accu_length[pre_raw+1]):
      pre_raw += 1
      assert(pre_raw<accu_length.shape[0]-1)
    dx = (covered_interp - accu_length[pre_raw]) / length[pre_raw]
    dst[i, :] = src[pre_raw, :] * (1.0 - dx) + src[pre_raw+1, :] * dx
    # dst[i, 0]=src[pre_raw, 0] * (1.0 - dx) + src[pre_raw+1, 0] * dx
    # dst[i, 1]=src[pre_raw, 1] * (1.0 - dx) + src[pre_raw+1, 1] * dx

  # time_3 = time.time()
  # if index == 0:
  #   print('   time_2_3_{}:{}'.format(index, time_3-time_2))


  return dst


def curve_fitting(points, samples, index):
  num_points = points.shape[0]
  assert(num_points > 1)
  valid_points = [points[0]]
  for i in range(1,num_points):
    if (distance(points[i, :], points[i-1, :]) > 0.001):
      valid_points.append(points[i, :])
  assert(len(valid_points) > 1)
  valid_points = np.asarray(valid_points)
  functions = np.zeros((valid_points.shape[0]-1, 9))

  if valid_points.shape[0] == 2:
    functions[0, 0] = LA.norm(valid_points[0, :] - valid_points[1, :])
    functions[0, 1] = valid_points[0, 0]
    functions[0, 2] = (valid_points[1, 0] - valid_points[0, 0]) / functions[0, 0]
    functions[0, 3] = 0
    functions[0, 4] = 0
    functions[0, 5] = valid_points[0, 1]
    functions[0, 6] = (valid_points[1, 1] - valid_points[0, 1]) / functions[0, 0]
    functions[0, 7] = 0
    functions[0, 8] = 0
  else:
    Mx = np.zeros((valid_points.shape[0]))
    My = np.zeros((valid_points.shape[0]))
    A = np.zeros((valid_points.shape[0]-2))
    B = np.zeros((valid_points.shape[0]-2))
    C = np.zeros((valid_points.shape[0]-2))
    Dx = np.zeros((valid_points.shape[0]-2))
    Dy = np.zeros((valid_points.shape[0]-2))
    for i in range(functions.shape[0]):
      functions[i, 0] = LA.norm(valid_points[i, :] - valid_points[i+1, :])
    for i in range(A.shape[0]):
      A[i] = functions[i, 0]
      B[i] = 2.0 * (functions[i, 0] + functions[i+1, 0])
      C[i] = functions[i+1, 0]
      Dx[i] = 6.0 * ((valid_points[i+2, 0]-valid_points[i+1, 0]) / functions[i+1, 0] - (valid_points[i+1, 0]-valid_points[i, 0]) / functions[i, 0])
      Dy[i] = 6.0 * ((valid_points[i+2, 1]-valid_points[i+1, 1]) / functions[i+1, 0] - (valid_points[i+1, 1]-valid_points[i, 1]) / functions[i, 0])

    C[0] = C[0] / B[0]
    Dx[0]=Dx[0] / B[0]
    Dy[0]=Dy[0] / B[0]
    for i in range(1,A.shape[0]):
      tmp=B[i] - A[i] * C[i-1]
      C[i]=C[i]/tmp
      Dx[i]=(Dx[i] - A[i]*Dx[i-1]) / tmp
      Dy[i]=(Dy[i] - A[i]*Dy[i-1]) / tmp
    Mx[valid_points.shape[0]-2]=Dx[valid_points.shape[0]-3]
    My[valid_points.shape[0]-2]=Dy[valid_points.shape[0]-3]
    for i in range(valid_points.shape[0]-4,-1,-1):
      Mx[i+1]=Dx[i]-C[i]*Mx[i+2]
      My[i+1]=Dy[i]-C[i]*My[i+2]
    Mx[0]= 0
    Mx[valid_points.shape[0]-1]=0
    My[0]=0
    My[valid_points.shape[0]-1]=0

    for i in range(functions.shape[0]):
      functions[i, 1] = valid_points[i, 0]
      functions[i, 2] = (valid_points[i+1, 0]-valid_points[i, 0]) / functions[i, 0] - (2.0*functions[i, 0]*Mx[i]+functions[i, 0]*Mx[i+1])/6.0
      functions[i, 3] = Mx[i] / 2.0
      functions[i, 4] = (Mx[i+1]-Mx[i]) / (6.0*functions[i, 0])
      functions[i, 5] = valid_points[i, 1]
      functions[i, 6] = (valid_points[i+1, 1]-valid_points[i, 1])/functions[i, 0] - (2.0*functions[i, 0]*My[i]+functions[i, 0]*My[i+1])/6.0
      functions[i, 7] = My[i] / 2.0
      functions[i, 8] = (My[i+1]-My[i])/(6.0*functions[i, 0])

  # samples_per_segment = 20
  # time_0 = time.time()

  samples_per_segment = samples*1/functions.shape[0]+1
  # samples_per_segment = samples*100/functions.shape[0]+1

  rawcurve=np.zeros((functions.shape[0]*int(samples_per_segment), 2))
  for i in range(functions.shape[0]):
    step = functions[i, 0]/samples_per_segment
    for j in range(int(samples_per_segment)):
      t = step*j
      rawcurve[i*int(samples_per_segment)+j,:] = np.asarray([functions[i, 1]+functions[i, 2]*t+functions[i, 3]*t*t+functions[i, 4]*t*t*t, functions[i, 5] + functions[i, 6]*t + functions[i, 7]*t*t+functions[i, 8]*t*t*t])

  # samples_per_segment = samples*100/functions.shape[0]+1
  # rawcurve=[]
  # for i in range(functions.shape[0]):
  #   step = functions[i, 0]/samples_per_segment
  #   for j in range(samples_per_segment):
  #     t = step*j
  #     rawcurve.append([functions[i, 1]+functions[i, 2]*t+functions[i, 3]*t*t+functions[i, 4]*t*t*t, functions[i, 5] + functions[i, 6]*t + functions[i, 7]*t*t+functions[i, 8]*t*t*t])
  # rawcurve = np.asarray(rawcurve)

  # print('samples_per_segment:{}'.format(samples_per_segment))
  # print('functions.shape[0]:{}'.format(functions.shape[0]))

  # time_1 = time.time()
  # if index == 0 or index == 1:
    # print(' time_0_1_{}:{}'.format(index, time_1-time_0))

  curve_tmp = curve_interp(rawcurve, samples, index)

  # time_2 = time.time()
  # if index == 0 or index == 1:
  #   print(' time_1_2_{}:{}'.format(index, time_2-time_1))

  return curve_tmp


def points_to_heatmap_68points(points, heatmap_num, heatmap_size, label_size, sigma):
    for i in range(points.shape[0]):
        points[i] *= (float(heatmap_size) / float(label_size))

    align_on_curve = [0]*heatmap_num
    curves = [0]*heatmap_num

    align_on_curve[0] = np.zeros((17, 2)) # contour
    align_on_curve[1] = np.zeros((5, 2)) # left eyebrow
    align_on_curve[2] = np.zeros((5, 2)) # right eyebrow
    align_on_curve[3] = np.zeros((4, 2)) # nose bridge
    align_on_curve[4] = np.zeros((5, 2)) # nose tip
    align_on_curve[5] = np.zeros((4, 2)) # left top eye
    align_on_curve[6] = np.zeros((4, 2)) # left bottom eye
    align_on_curve[7] = np.zeros((4, 2)) # right top eye
    align_on_curve[8] = np.zeros((4, 2)) # right bottom eye
    align_on_curve[9] = np.zeros((7, 2)) # up up lip
    align_on_curve[10] = np.zeros((5, 2)) # up bottom lip
    align_on_curve[11] = np.zeros((5, 2)) # bottom up lip
    align_on_curve[12] = np.zeros((7, 2)) # bottom bottom lip

    for i in range(17):
        align_on_curve[0][i] = points[i]

    for i in range(5):
        align_on_curve[1][i] = points[i+17]

    for i in range(5):
        align_on_curve[2][i] = points[i+22]

    for i in range(4):
        align_on_curve[3][i] = points[i+27]

    for i in range(5):
        align_on_curve[4][i] = points[i+31]

    for i in range(4):
        align_on_curve[5][i] = points[i+36]

    align_on_curve[6][0] = points[36]
    align_on_curve[6][1] = points[41]
    align_on_curve[6][2] = points[40]
    align_on_curve[6][3] = points[39]

    align_on_curve[7][0] = points[42]
    align_on_curve[7][1] = points[43]
    align_on_curve[7][2] = points[44]
    align_on_curve[7][3] = points[45]

    align_on_curve[8][0] = points[42];
    align_on_curve[8][1] = points[47];
    align_on_curve[8][2] = points[46];
    align_on_curve[8][3] = points[45];

    for i in range(7):
        align_on_curve[9][i] = points[i+48]

    for i in range(5):
        align_on_curve[10][i] = points[i+60]

    align_on_curve[11][0] = points[60]
    align_on_curve[11][1] = points[67]
    align_on_curve[11][2] = points[66]
    align_on_curve[11][3] = points[65]
    align_on_curve[11][4] = points[64]

    align_on_curve[12][0] = points[48]
    align_on_curve[12][1] = points[59]
    align_on_curve[12][2] = points[58]
    align_on_curve[12][3] = points[57]
    align_on_curve[12][4] = points[56]
    align_on_curve[12][5] = points[55]
    align_on_curve[12][6] = points[54]

    heatmap = np.zeros((heatmap_size, heatmap_size, heatmap_num))
    for i in range(heatmap_num):
        curve_map = np.full((heatmap_size, heatmap_size), 255, dtype=np.uint8)

        valid_points = [align_on_curve[i][0, :]]
        for j in range(1, align_on_curve[i].shape[0]):
            if (distance(align_on_curve[i][j, :], align_on_curve[i][j-1, :]) > 0.001):
                valid_points.append(align_on_curve[i][j, :])

        if len(valid_points) > 1:
      # time_1_0 = time.time()
            curves[i] = curve_fitting(align_on_curve[i], align_on_curve[i].shape[0] * 10, i)
            for j in range(curves[i].shape[0]):
                if (int(curves[i][j, 0] + 0.5) >= 0 and int(curves[i][j, 0] + 0.5) < heatmap_size and
                    int(curves[i][j, 1] + 0.5) >= 0 and int(curves[i][j, 1] + 0.5) < heatmap_size):
                    curve_map[int(curves[i][j, 1] + 0.5), int(curves[i][j, 0] + 0.5)] = 0

      # time_1_1 = time.time()
      # print('time_1_0_1:{}'.format(time_1_1-time_1_0))

    # time_1_1 = time.time()
    # distance transform
        image_dis = cv2.distanceTransform(curve_map, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # time_1_2 = time.time()
    # print('time_1_1_2:{}'.format(time_1_2-time_1_1))

    # gaussian map generation
        image_dis = image_dis.astype(np.float64)
        image_gaussian = (1.0 / (2.0*np.pi*(sigma**2))) * np.exp(-1.0 * image_dis**2 / (2.0*sigma**2))
        image_gaussian = np.where(image_dis<(3.0*sigma), image_gaussian, 0)

    # for k in range(curve_map.shape[0]):
    #   for t in range(curve_map.shape[1]):
    #     gaussian_val = (1.0 / (2.0*np.pi*np.power(sigma,2))) * np.exp(-1.0 * np.power(image_dis[k, t],2) / (2.0*np.power(sigma,2)))
    #     if image_dis[k, t] < (3.0 * sigma):
    #       image_dis[k, t] = gaussian_val
    #     else:
    #       image_dis[k, t] = 0

    # time_1_3 = time.time()
    # print('time_1_2_3:{}'.format(time_1_3-time_1_2))

    # normalised to [0,1]
        maxVal = image_gaussian.max()
        minVal = image_gaussian.min()

        if maxVal == minVal:
            image_gaussian = 0
        else:
            image_gaussian = (image_gaussian-minVal) / (maxVal-minVal)

    # for k in range(curve_map.shape[0]):
    #   for t in range(curve_map.shape[1]):
    #     if maxVal == minVal:
    #       image_gaussian[k, t] = 0
    #     else:
    #       image_gaussian[k, t] = (image_gaussian[k, t] - minVal) / (maxVal-minVal)

    # time_1_4 = time.time()
    # print('time_1_3_4:{}'.format(time_1_4-time_1_3))

        heatmap[:, :, i] = image_gaussian

    return heatmap



def points_to_heatmap(points, heatmap_num, heatmap_size, label_size, sigma):
  # resize points label
  for i in range(points.shape[0]):
    points[i] *= (float(heatmap_size) / float(label_size))

  # heatmap generation
  align_on_curve = [0]*heatmap_num
  curves = [0]*heatmap_num
  align_on_curve[0] = np.zeros((33, 2)) # contour
  align_on_curve[1] = np.zeros((5, 2)) # left top eyebrow
  align_on_curve[2] = np.zeros((5, 2)) # right top eyebrow
  align_on_curve[3] = np.zeros((4, 2)) # nose bridge
  align_on_curve[4] = np.zeros((9, 2)) # nose tip
  align_on_curve[5] = np.zeros((5, 2)) # left top eye
  align_on_curve[6] = np.zeros((5, 2)) # left bottom eye
  align_on_curve[7] = np.zeros((5, 2)) # right top eye
  align_on_curve[8] = np.zeros((5, 2)) # right bottom eye
  align_on_curve[9] = np.zeros((7, 2)) # up up lip
  align_on_curve[10] = np.zeros((5, 2)) # up bottom lip
  align_on_curve[11] = np.zeros((5, 2)) # bottom up lip
  align_on_curve[12] = np.zeros((7, 2)) # bottom bottom lip
  align_on_curve[13] = np.zeros((5, 2)) # left bottom eyebrow
  align_on_curve[14] = np.zeros((5, 2)) # left bottom eyebrow

  for i in range(33):
    align_on_curve[0][i] = points[i]

  for i in range(5):
    align_on_curve[1][i] = points[i+33]

  for i in range(5):
    align_on_curve[2][i] = points[i+38]

  for i in range(4):
    align_on_curve[3][i] = points[i+43]

  align_on_curve[4][0] = points[80]
  align_on_curve[4][1] = points[82]
  for i in range(5):
    align_on_curve[4][i+2] = points[i+47]
  align_on_curve[4][7] = points[83]
  align_on_curve[4][8] = points[81]
  
  align_on_curve[5][0] = points[52]
  align_on_curve[5][1] = points[53]
  align_on_curve[5][2] = points[72]
  align_on_curve[5][3] = points[54]
  align_on_curve[5][4] = points[55]

  align_on_curve[6][0] = points[55]
  align_on_curve[6][1] = points[56]
  align_on_curve[6][2] = points[73]
  align_on_curve[6][3] = points[57]
  align_on_curve[6][4] = points[52]

  align_on_curve[7][0] = points[58]
  align_on_curve[7][1] = points[59]
  align_on_curve[7][2] = points[75]
  align_on_curve[7][3] = points[60]
  align_on_curve[7][4] = points[61]

  align_on_curve[8][0] = points[61]
  align_on_curve[8][1] = points[62]
  align_on_curve[8][2] = points[76]
  align_on_curve[8][3] = points[63]
  align_on_curve[8][4] = points[58]

  for i in range(7):
    align_on_curve[9][i] = points[i+84]

  for i in range(5):
    align_on_curve[10][i] = points[i+96]

  align_on_curve[11][0] = points[96]
  align_on_curve[11][1] = points[103]
  align_on_curve[11][2] = points[102]
  align_on_curve[11][3] = points[101]
  align_on_curve[11][4] = points[100]

  align_on_curve[12][0] = points[84]
  align_on_curve[12][1] = points[95]
  align_on_curve[12][2] = points[94]
  align_on_curve[12][3] = points[93]
  align_on_curve[12][4] = points[92]
  align_on_curve[12][5] = points[91]
  align_on_curve[12][6] = points[90]

  align_on_curve[13][0] = points[33]
  align_on_curve[13][1] = points[64]
  align_on_curve[13][2] = points[65]
  align_on_curve[13][3] = points[66]
  align_on_curve[13][4] = points[67]

  align_on_curve[14][0] = points[68]
  align_on_curve[14][1] = points[69]
  align_on_curve[14][2] = points[70]
  align_on_curve[14][3] = points[71]
  align_on_curve[14][4] = points[42]

  heatmap = np.zeros((heatmap_size, heatmap_size, heatmap_num))
  for i in range(heatmap_num):
    curve_map = np.full((heatmap_size, heatmap_size), 255, dtype=np.uint8)

    valid_points = [align_on_curve[i][0, :]]
    for j in range(1, align_on_curve[i].shape[0]):
      if (distance(align_on_curve[i][j, :], align_on_curve[i][j-1, :]) > 0.001):
        valid_points.append(align_on_curve[i][j, :])

    if len(valid_points) > 1:
      # time_1_0 = time.time()
      curves[i] = curve_fitting(align_on_curve[i], align_on_curve[i].shape[0] * 10, i)
      for j in range(curves[i].shape[0]):
        if (int(curves[i][j, 0] + 0.5) >= 0 and int(curves[i][j, 0] + 0.5) < heatmap_size and
          int(curves[i][j, 1] + 0.5) >= 0 and int(curves[i][j, 1] + 0.5) < heatmap_size):
          curve_map[int(curves[i][j, 1] + 0.5), int(curves[i][j, 0] + 0.5)] = 0

      # time_1_1 = time.time()
      # print('time_1_0_1:{}'.format(time_1_1-time_1_0))

    # time_1_1 = time.time()
    # distance transform
    image_dis = cv2.distanceTransform(curve_map, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # time_1_2 = time.time()
    # print('time_1_1_2:{}'.format(time_1_2-time_1_1))

    # gaussian map generation
    image_dis = image_dis.astype(np.float64)
    image_gaussian = (1.0 / (2.0*np.pi*(sigma**2))) * np.exp(-1.0 * image_dis**2 / (2.0*sigma**2))
    image_gaussian = np.where(image_dis<(3.0*sigma), image_gaussian, 0)

    # for k in range(curve_map.shape[0]):
    #   for t in range(curve_map.shape[1]):
    #     gaussian_val = (1.0 / (2.0*np.pi*np.power(sigma,2))) * np.exp(-1.0 * np.power(image_dis[k, t],2) / (2.0*np.power(sigma,2)))
    #     if image_dis[k, t] < (3.0 * sigma):
    #       image_dis[k, t] = gaussian_val
    #     else:
    #       image_dis[k, t] = 0

    # time_1_3 = time.time()
    # print('time_1_2_3:{}'.format(time_1_3-time_1_2))

    # normalised to [0,1]
    maxVal = image_gaussian.max()
    minVal = image_gaussian.min()

    if maxVal == minVal:
      image_gaussian = 0
    else:
      image_gaussian = (image_gaussian-minVal) / (maxVal-minVal)
  
    # for k in range(curve_map.shape[0]):
    #   for t in range(curve_map.shape[1]):
    #     if maxVal == minVal:
    #       image_gaussian[k, t] = 0
    #     else:
    #       image_gaussian[k, t] = (image_gaussian[k, t] - minVal) / (maxVal-minVal)

    # time_1_4 = time.time()
    # print('time_1_3_4:{}'.format(time_1_4-time_1_3))

    heatmap[:, :, i] = image_gaussian

  return heatmap



def points_to_heatmap_98pt(points, heatmap_num, heatmap_size, label_size, sigma):
  # resize points label
  for i in range(points.shape[0]):
    points[i] *= (float(heatmap_size) / float(label_size))

  # heatmap generation
  align_on_curve = [0]*heatmap_num
  curves = [0]*heatmap_num
  align_on_curve[0] = np.zeros((33, 2)) # contour
  align_on_curve[1] = np.zeros((5, 2)) # left top eyebrow
  align_on_curve[2] = np.zeros((5, 2)) # right top eyebrow
  align_on_curve[3] = np.zeros((4, 2)) # nose bridge
  align_on_curve[4] = np.zeros((5, 2)) # nose tip
  align_on_curve[5] = np.zeros((5, 2)) # left top eye
  align_on_curve[6] = np.zeros((5, 2)) # left bottom eye
  align_on_curve[7] = np.zeros((5, 2)) # right top eye
  align_on_curve[8] = np.zeros((5, 2)) # right bottom eye
  align_on_curve[9] = np.zeros((7, 2)) # up up lip
  align_on_curve[10] = np.zeros((5, 2)) # up bottom lip
  align_on_curve[11] = np.zeros((5, 2)) # bottom up lip
  align_on_curve[12] = np.zeros((7, 2)) # bottom bottom lip
  align_on_curve[13] = np.zeros((5, 2)) # left bottom eyebrow
  align_on_curve[14] = np.zeros((5, 2)) # right bottom eyebrow


  for i in range(33):
    align_on_curve[0][i] = points[i]

  for i in range(5):
    align_on_curve[1][i] = points[i+33]

  for i in range(5):
    align_on_curve[2][i] = points[i+42]

  for i in range(4):
    align_on_curve[3][i] = points[i+51]

  for i in range(5):
    align_on_curve[4][i] = points[i+55]
  
  for i in range(5):
    align_on_curve[5][i] = points[i+60]

  align_on_curve[6][0] = points[60]
  align_on_curve[6][1] = points[67]
  align_on_curve[6][2] = points[66]
  align_on_curve[6][3] = points[65]
  align_on_curve[6][4] = points[64]

  for i in range(5):
    align_on_curve[7][i] = points[i+68]

  align_on_curve[8][0] = points[68]
  align_on_curve[8][1] = points[75]
  align_on_curve[8][2] = points[74]
  align_on_curve[8][3] = points[73]
  align_on_curve[8][4] = points[72]

  for i in range(7):
    align_on_curve[9][i] = points[i+76]

  for i in range(5):
    align_on_curve[10][i] = points[i+88]

  align_on_curve[11][0] = points[88]
  align_on_curve[11][1] = points[95]
  align_on_curve[11][2] = points[94]
  align_on_curve[11][3] = points[93]
  align_on_curve[11][4] = points[92]

  align_on_curve[12][0] = points[76]
  align_on_curve[12][1] = points[87]
  align_on_curve[12][2] = points[86]
  align_on_curve[12][3] = points[85]
  align_on_curve[12][4] = points[84]
  align_on_curve[12][5] = points[83]
  align_on_curve[12][6] = points[82]

  align_on_curve[13][0] = points[33]
  align_on_curve[13][1] = points[41]
  align_on_curve[13][2] = points[40]
  align_on_curve[13][3] = points[39]
  align_on_curve[13][4] = points[38]

  align_on_curve[14][0] = points[50]
  align_on_curve[14][1] = points[49]
  align_on_curve[14][2] = points[48]
  align_on_curve[14][3] = points[47]
  align_on_curve[14][4] = points[46]

  heatmap = np.zeros((heatmap_size, heatmap_size, heatmap_num))
  for i in range(heatmap_num):
    curve_map = np.full((heatmap_size, heatmap_size), 255, dtype=np.uint8)

    valid_points = [align_on_curve[i][0, :]]
    for j in range(1, align_on_curve[i].shape[0]):
      if (distance(align_on_curve[i][j, :], align_on_curve[i][j-1, :]) > 0.001):
        valid_points.append(align_on_curve[i][j, :])

    if len(valid_points) > 1:
      curves[i] = curve_fitting(align_on_curve[i], align_on_curve[i].shape[0] * 10, i)
      for j in range(curves[i].shape[0]):
        if (int(curves[i][j, 0] + 0.5) >= 0 and int(curves[i][j, 0] + 0.5) < heatmap_size and
          int(curves[i][j, 1] + 0.5) >= 0 and int(curves[i][j, 1] + 0.5) < heatmap_size):
          curve_map[int(curves[i][j, 1] + 0.5), int(curves[i][j, 0] + 0.5)] = 0

    # distance transform
    image_dis = cv2.distanceTransform(curve_map, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # gaussian map generation
    image_dis = image_dis.astype(np.float64)
    image_gaussian = (1.0 / (2.0*np.pi*(sigma**2))) * np.exp(-1.0 * image_dis**2 / (2.0*sigma**2))
    image_gaussian = np.where(image_dis<(3.0*sigma), image_gaussian, 0)

    # normalised to [0,1]
    maxVal = image_gaussian.max()
    minVal = image_gaussian.min()

    if maxVal == minVal:
      image_gaussian = 0
    else:
      image_gaussian = (image_gaussian-minVal) / (maxVal-minVal)

    heatmap[:, :, i] = image_gaussian

  return heatmap


def points_to_landmark_map(points, heatmap_num, heatmap_size, label_size, sigma):
  # set heatmap_num to 1 to save memeory
  # assert(heatmap_num == 1)

  # resize points label
  for i in range(points.shape[0]):
    points[i] *= (float(heatmap_size) / float(label_size))

  # heatmap generation
  heatmap = np.zeros((heatmap_size,heatmap_size,heatmap_num))

  for i in range(len(points)):
    pt = points[i]
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= heatmap.shape[1] or ul[1] >= heatmap.shape[0] or
            br[0] < 0 or br[1] < 0):
      # If not, just continue
      continue

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], heatmap.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], heatmap.shape[0]) - ul[1]
    # Image range
    heatmap_x = max(0, ul[0]), min(br[0], heatmap.shape[1])
    heatmap_y = max(0, ul[1]), min(br[1], heatmap.shape[0])

    if heatmap_num == 1:
      for j in range(heatmap_x[1]-heatmap_x[0]):
        for k in range(heatmap_y[1]-heatmap_y[0]):
          heatmap[heatmap_y[0]+k, heatmap_x[0]+j] = max(g[g_y[0]+k, g_x[0]+j], heatmap[heatmap_y[0]+k, heatmap_x[0]+j])
    else:
      heatmap[heatmap_y[0]:heatmap_y[1], heatmap_x[0]:heatmap_x[1], i] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    # heatmap[heatmap_y[0]:heatmap_y[1], heatmap_x[0]:heatmap_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]


  return heatmap
