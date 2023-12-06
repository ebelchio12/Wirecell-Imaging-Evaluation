# https://scikit-spatial.readthedocs.io/en/stable/index.html
from skspatial.objects import Line, Points
from skspatial.plotting import plot_3d
import matplotlib.pyplot as plt
import numpy as np
import ROOT
from ROOT import TCanvas
from ROOT import gStyle
from scipy.linalg import norm
import math

gStyle.SetOptFit(True)

# create a list of 3D points using x,y,z,q lists as input
# x = [5,4,8,..,7,3]
# same for y, z and q.
# 
# lists must have the same size
def makeXYZQList(x,y,z,q):
  xyzqlist = []
  for i in range(len(x)):
    xyzq = [float(x[i]), float(y[i]), float(z[i]), float(q[i])]
    xyzqlist.append(xyzq)
  return xyzqlist


def getXYZList(xyzqlist):
  xyzlist = []
  for i in range(len(xyzqlist)):
    xyzq = xyzqlist[i]
    xyz = [float(xyzq[0]), float(xyzq[1]), float(xyzq[2])]
    xyzlist.append(xyz)
  return xyzlist


def getX(xyzqlist):
  v = []
  for i in range(len(xyzqlist)):
    xyzq = xyzqlist[i]
    v.append(xyzq[0])
  return v


def getY(xyzqlist):
  v = []
  for i in range(len(xyzqlist)):
    xyzq = xyzqlist[i]
    v.append(xyzq[1])
  return v


def getZ(xyzqlist):
  v = []
  for i in range(len(xyzqlist)):
    xyzq = xyzqlist[i]
    v.append(xyzq[2])
  return v


def getQ(xyzqlist):
  v = []
  for i in range(len(xyzqlist)):
    xyzq = xyzqlist[i]
    v.append(xyzq[3])
  return v


# select point in the specified range
def selectPoints(points, xmin, xmax, ymin, ymax, zmin, zmax):
  points_selected = []
  for i in range(len(points)):
    point = points[i]
    if point[0] > float(xmin) and  point[0] < float(xmax) and point[1] > float(ymin) and  point[1] < float(ymax) and point[2] > float(zmin) and  point[2] < float(zmax):
      #print('\n xyz={}'.format(xyz[i]))
      points_selected.append(point)

  return points_selected


# https://scikit-spatial.readthedocs.io/en/stable/api_reference/Line/methods/skspatial.objects.Line.best_fit.html#skspatial.objects.Line.best_fit
# Fit a line to multiple 3D points.
# xyzlist = [[2,4,1], [6,2,9], ..., [7,0,2]]
# return fitted line 
def FitBestLine3D(xyzlist, 
                  xmin_plt,
                  xmax_plt,
                  ymin_plt,
                  ymax_plt,
                  bool_make_plot
                  ):
  points = Points(xyzlist)
  line_fit = Line.best_fit(points)

  # print direction vector and centroid
  #print('\n line.direction.round(3)={}'.format(line_fit.direction.round(3)))
  #print('\n line.point={}'.format(line_fit.point))
  #
  # make a plot (optional)
  '''
  if bool_make_plot==True:
    plot_3d(
      #line_fit.plotter(t_1=-7, t_2=7, c='k'),
      line_fit.plotter()#,
      #points.plotter(c='b', depthshade=False),
      #points.plotter(),
    )

    plt.xlim([xmin_plt, xmax_plt])
    plt.ylim([ymin_plt, ymax_plt])
    #plt.zlim([0, 400])
    plt.xlabel("x  [cm]")
    plt.ylabel("y [cm]")
    plt.show()
  '''
  return line_fit

'''
# calculate perpedincular 3D distance between point and line
# see: https://www.nagwa.com/en/explainers/939127418581/
def calculateDistance(line, point):
  P = point
  A = line.point #any point in the line here (using centroid point)
  APvec = np.array([P[0]-A[0],P[1]-A[1],P[2]-A[2]])
  dvec = np.array(line.direction) # unit direction vector
  crossvec = np.cross(APvec, dvec)
  cross_mag = np.sqrt(crossvec.dot(crossvec)) 
  distance = cross_mag
  return distance


# check which points are inside sphere
def getInsideSphere(center, radius, points):
  inside = []
  cx = center[0]
  cy = center[1]
  cz = center[2]
  cq = center[3]
  r = radius
  for i in range(len(points)):
    point = points[i]
    x = point[0]
    y = point[1]
    z = point[2]
    q = point[3]
    
    #if (x - cx)**2 + (y - cy)**2 + (z - cz)**2 < r**2 and abs(q-cq) < 50:
    if (x - cx)**2 + (y - cy)**2 + (z - cz)**2 < r**2:
      inside.append(point)
 
  # get the one with closest q
  closest_q = []
  n_inside = len(inside)
  if n_inside > 0 :
    mindiff = float('+inf')
    idx=-999
    for i in range(n_inside):
      point = inside[i]
      q = point[3]
      diff = abs(cq-q)
      if(diff < mindiff):
        mindiff = diff
        idx = i     

    closest_q.append(inside[idx])

  return inside#closest_q 
'''

# plot objects into canvas
# object named 'skipobject_name' is not treated as histogram
# histograms must have the same binwidth
def make_canvas(histograms, cname, skipobject_name='aname'):
  # normalization (minimum entries)
  norm=10e6
  for i in range(len(histograms)):
    if histograms[i].GetName() == skipobject_name:
      continue
    if histograms[i].GetEntries() < norm:
      norm=histograms[i].GetEntries()   

  # draw normalized and get maxy
  c = TCanvas('temp')
  c.cd()
  maxy=-10e6
  imaxy = -1
  isDraw = False
  for i in range(len(histograms)):
    if histograms[i].GetName() == skipobject_name:
      continue
    if isDraw == False:
      isDraw = True
      hnorm = histograms[i].DrawNormalized("hist", norm)
      for ibin in range(hnorm.GetNbinsX()):
        ybin = hnorm.GetBinContent(ibin)
        if ybin > maxy:
          maxy=ybin
          imaxy=i
    else:
      hnorm = histograms[i].DrawNormalized("hist same", norm)
      for ibin in range(hnorm.GetNbinsX()):
        ybin = hnorm.GetBinContent(ibin)
        if ybin > maxy:
          maxy=ybin
          imaxy=i

  # final draw
  c2 = TCanvas(cname)
  c2.cd()
  histograms[imaxy].DrawNormalized("hist", norm)
  for i in range(len(histograms)):
    if i == imaxy:
      continue
    else:
      if histograms[i].GetName() == skipobject_name:
        histograms[i].Draw("same")
      else:
        histograms[i].DrawNormalized("hist same", norm)

  c2.BuildLegend()
  return c2


# make histograms using input list
def make_histogram(inlist, nbins=100, xmin=-.1, xmax=5, xlabel='x', ylabel='y', title='title', hname='h', color=4):
  h = ROOT.TH1F(hname, title, nbins, xmin, xmax)
  h.SetLineColor(color)
  h.GetXaxis().SetTitle(xlabel)
  h.GetYaxis().SetTitle(ylabel)
  for i in range(len(inlist)):
    h.Fill(inlist[i])
  return h


# https://stackoverflow.com/questions/32317247/how-to-draw-a-cylinder-using-matplotlib-along-length-of-point-x1-y1-and-x2-y2
#
# Starting from a xyz coordinate system (CS), create an orthonormal CS' (n1,n2,n3) 
# with origin at R0 w.r.t. CS and with unitary n1 vector in the same direction as 
# line's direction vector V.
# 
# More info about CS' at: https://drive.google.com/file/d/1XB9fLdnAQNxoJHQ1O-jgE1z_44NFAoh0/view?usp=sharing
#
# theta0XZ: track angle (in degrees!) from simulation
# theta0YZ MUST be zero in track simulation!!
# 
def getNewBasis(line, theta0XZ_in_degrees):
  # convert input angle from degrees to radians
  theta0XZ = math.radians(theta0XZ_in_degrees)
  # direction vector
  direction_vector = np.array(line.direction) # unit direction vector
  V = direction_vector
  #print('\n V(x,y,z) = ({},{},{})'.format(V[0], V[1], V[2]))

  # make sure V point to the positive XZ octant (Vx>0, Vz>0)
  if V[0]<0 or V[2]<0:
    print('\n Inverting V orientation..')
    V = -V
  #print('\n V(x,y,z) = ({},{},{})'.format(V[0], V[1], V[2]))
  # unitary n1 vector in the same direction as V
  n1 = V / norm(V)
  # make some vector not in the same direction as n1
  not_n1 = np.array([1, 0, 0])
  if (n1 == not_n1).all():
    not_n1 = np.array([0, 1, 0])
  # make n2 vector perpendicular to n1
  n2 = np.cross(n1, not_n1)
  # normalize n2
  n2 /= norm(n2)
  # make unit vector perpendicular to n1 and n2
  n3 = np.cross(n1, n2)
  # make sure n1, n2 and n3 correspond to xhat', yhat' and zhat'
  # in a 2D rotation about y-axis (see CS' info above)
  theta = -((math.pi/2)-theta0XZ) # 2D rotation angle about y-axis
  yhat_prime = np.array([0, 1, 0]) # in a 2D rotation about y-axis
  zhat_prime = np.array([math.sin(theta), 0, math.cos(theta)]) # in a 2D rotation about y-axis
  xhat_prime = np.array([math.cos(theta), 0, -math.sin(theta)]) # in a 2D rotation about y-axis
  #print('\n 1 - np.dot(n1, xhat_prime)= {}'.format(1 - np.dot(n1, xhat_prime)))
  #print('\n 1 - np.dot(n2, yhat_prime)= {}'.format(1 - np.dot(n2, yhat_prime)))
  #print('\n 1 - np.dot(n3, zhat_prime)= {}'.format(1 - np.dot(n3, zhat_prime)))
  max_angle = math.radians(10) # use 10 degrees as maximum angle between CS' unit vectors and CS 2D-rotated unit vectors
  dot1 = np.dot(n1, xhat_prime)
  dot2 = np.dot(n2, yhat_prime)
  dot3 = np.dot(n3, zhat_prime)
  if dot1 > math.cos(max_angle) and dot2 > math.cos(max_angle) and dot3 > math.cos(max_angle):
    n1_correct = True
  else:
    print("\n Something wrong with unit vectors orientation! Fix it")

  return np.array([n1,n2,n3]) 


# Find (N1,N2,N3) coordinates of input point P = (x,y,z) 
# in a new coordinate system (n1,n2,n3) (see definition above). 
def getNewCoordinates(line, mypoint, theta0XZ_in_degrees):
  x = mypoint[0]
  y = mypoint[1]
  z = mypoint[2]
  q = mypoint[3]
  P = [x, y, z]

  # get basis
  basis = getNewBasis(line, theta0XZ_in_degrees)
  n1 = basis[0]
  n2 = basis[1]
  n3 = basis[2]

  # R0: origin of CS' (in CS coordinates)
  # P_prime: vector going from R0 to P (in CS coordinates)
  # N1,N2,N3: coordinates of P_prime (in CS' coordinates)
  R0 = line.point
  #print('\n R0={}'.format(R0))
  P_prime = P - R0
  N1 = np.dot(P_prime, n1)   
  N2 = np.dot(P_prime, n2)   
  N3 = np.dot(P_prime, n3)   
  #print('\n R0={}'.format(R0))
  #print('\n P={}'.format(P))
  #print('\n P"=P-R0 ={}'.format(P_prime))
  #print('\n x"=P".n1 ={}'.format(N1))
  #print('\n y"=P".n2 ={}'.format(N2))
  #print('\n z"=P".n3 ={}'.format(N3))

  return np.array([N1,N2,N3,q])


def getOldCoordinates(line, mypoint_prime, theta0XZ_in_degrees):
  # old basis
  ihat = np.array([1,0,0])
  jhat = np.array([0,1,0])
  khat = np.array([0,0,1])
  # R0: origin of CS' (in CS coordinates)
  R0 = line.point
  # get new basis
  basis = getNewBasis(line, theta0XZ_in_degrees)
  n1 = basis[0]
  n2 = basis[1]
  n3 = basis[2]
  # N1,N2,N3
  N1 = mypoint_prime[0]
  N2 = mypoint_prime[1]
  N3 = mypoint_prime[2]
  # P_prime: vector going from R0 to P (in CS' coordinates)
  P_prime = N1*n1 + N2*n2 + N3*n3
  # old coordinates
  x = np.dot(P_prime + R0, ihat)
  y = np.dot(P_prime + R0, jhat)
  z = np.dot(P_prime + R0, khat)

  return np.array([x,y,z])


# project list of P=(x,y,z) points on new coordinate system having z-axis along line
def project(xyzqlist, line, theta0XZ_in_degrees):
  xyzqlist_prime = []
  for i in range(len(xyzqlist)):
    xyzq = xyzqlist[i]
    if len(xyzq) != 4:
      print('\n project(): points in input list must be [x,y,z,q]. Fix it.\n')
      exit()
    xyzq_prime = getNewCoordinates(line, xyzq, theta0XZ_in_degrees)
    xyzqlist_prime.append(xyzq_prime)
  return xyzqlist_prime


def gauss(fname, constant, mean, sigma, xmin, xmax):
  gaus = ROOT.TF1(fname,"gaus",xmin,xmax);
  gaus.SetParameters(constant, mean, sigma)
  return gaus


def fitGaus(histogram, fname):
  hmax = histogram.GetMaximum()
  #hmean = histogram.GetMean()
  hmean = histogram.GetXaxis().GetBinCenter(histogram.GetMaximumBin());
  hstd = histogram.GetRMS()
  xmin = hmean - 2.5*hstd
  xmax = hmean + 2.5*hstd
  #f1 = ROOT.TF1(fname, "gaus", xmin, xmax)
  f1 = gauss(fname, hmax, hmean, hstd, xmin, xmax)
  fitStatus = histogram.Fit(fname, "R")
  if fitStatus != 0:
    print('\n Error occurred in fit! Check histogram and fit function.\n')
  return f1


# https://root-forum.cern.ch/t/smearing-an-energy-according-to-a-crystal-ball-function/35395/2
# [random] https://root-forum.cern.ch/t/how-does-tf1-getrandom-work/36031/2
# [define gauss] http://hep.bu.edu/~jlraaf/2011REU/root_lecture02.pdf
# smear values according to input histogram
# values_to_smear: array
#def smear(values_to_smear, fname, gauss_constant, gauss_mean, gauss_sigma, gauss_xmin, gauss_xmax):
def smear(values_to_smear, sigma, n):
  smeared = []
  constant = 1/(sigma*math.sqrt(2*math.pi))
  xmin = -5*sigma
  xmax = 5*sigma
  rdn = ROOT.TRandom()
  #gaus = gauss(fname, gauss_constant, gauss_mean, gauss_sigma, gauss_xmin, gauss_xmax)
  for i in range(len(values_to_smear)):
    #x = values_to_smear[i] + gaus.GetRandom()
    mean = values_to_smear[i]
    #fname = 'gaus_' + str(i)
    #gaus = gauss(fname, constant, mean, sigma, xmin, xmax)
    for k in range(n):
      #x = mean + gaus.GetRandom()
      x = rdn.Gaus(mean, sigma)
      smeared.append(x)
  return smeared
