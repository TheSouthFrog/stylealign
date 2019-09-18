import numpy as np
import math
from numpy import linalg as LA


def distance(p1, p2):
  return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))


def curve_interp(src, samples, index):
  assert(src.shape[0] > 2)
  assert(samples >= 2)

  src_1 = src[0:src.shape[0]-1, :]
  src_2 = src[1:src.shape[0], :]
  src_delta = src_1 - src_2
  length = np.sqrt(src_delta[:,0]**2 + src_delta[:,1]**2)
  assert(length.shape[0] == src.shape[0]-1)

  accu_length = np.zeros((src.shape[0]))
  for i in range(1,accu_length.shape[0]):
    accu_length[i]=accu_length[i-1]+length[i-1]
  dst = np.zeros((samples, 2))
  pre_raw = 0

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

  curve_tmp = curve_interp(rawcurve, samples, index)

  return curve_tmp

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
          heatmap[heatmap_y[0]+k, heatmap_x[0]+j] = max(g[g_y[0]+k, g_x[0]+j], heatmap[heatmap_y[0]+k,heatmap_x[0]+j])
    else:
      heatmap[heatmap_y[0]:heatmap_y[1], heatmap_x[0]:heatmap_x[1], i] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

  return heatmap
