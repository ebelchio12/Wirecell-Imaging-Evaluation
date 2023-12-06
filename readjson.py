# Script to read in json file from imaging clusters and
# save x,y,z and q
#
# author: Ewerton B
# usage: python readjson.py
#

import json
from BestFit3D import *
from getMuonTrackZRange import *
import numpy as np
from pathlib import Path
import math

# ROOT colors
kBlue=4
kRed=2
kGreen=8

#datapath = '/sbnd/data/users/ebatista/imaging-handscan/XYMuon/singlemuon_1GeV_X0_-150_Y0_0_Z0_50_Theta0XZ_10_Theta0YZ_0/data'
datapath = '/sbnd/data/users/ebatista/imaging-handscan/XYMuon/bee_singlemuon_nthreshold_0d0/data'
eventsToProcess = 1
nskip=0#eventsToProcess-1
track_length_for_fit = 30
zmin_improved= 50
zmax_improved= 63
use_improved = False

# subdirs labels
lab1 = 'truthDepo'
lab2 = 'imaging'

# Get muon track ranges for zcut
muontrack_zrangefile='/sbnd/data/users/ebatista/imaging-handscan/XYMuon/muontracks_imaging-fit_zrange-formatted_60events.txt'
zrange_evt_xmin_xmax = getMuonTrackZminZmax(muontrack_zrangefile)
#print("zrange_evt_xmin_xmax:", zrange_evt_xmin_xmax)

vsigx=[]
vsigy=[]


# loop over events
for ievt in range(eventsToProcess):

  if ievt < nskip:
    continue

  print('\n ******************')
  print('\n Event {}'.format(ievt))
  print('\n ******************')

  subdir = str(ievt)
  fname1 = str(ievt) + '-' + lab1 + '.json'
  fname2 = str(ievt) + '-' + lab2 + '.json'
  inputfile1 = datapath + '/' + subdir + '/' + fname1
  inputfile2 = datapath + '/' + subdir + '/' + fname2
  anglefile = datapath + '/' + subdir + '/angle.txt'
  angle=-99
  with open(anglefile, 'r') as myfile:
    angle = float(myfile.read().rstrip())
  print('\n angle[evt {}]={}'.format(str(ievt),angle))

  # debug info
  if ievt == 0:
    print('\n Processing {} events (Nevents must be <= number of folders in {})..'.format(eventsToProcess,datapath))
    print('\n Reading input file {}..'.format(inputfile1))
    print('\n Reading input file {}..'.format(inputfile2))

  # load data
  fin1 = open(inputfile1)
  fin2 = open(inputfile2)
  data1 = json.load(fin1)
  data2 = json.load(fin2)

  x1 = data1['x']
  y1 = data1['y']
  z1 = data1['z']
  q1 = data1['q']

  x2 = data2['x']
  y2 = data2['y']
  z2 = data2['z']
  q2 = data2['q']
  
  # remove duplicates
  #x = list(dict.fromkeys(x))
  #y = list(dict.fromkeys(y))
  #z = list(dict.fromkeys(z))
  #q = list(dict.fromkeys(q))

  # check vector length
  lenx1 = len(x1)
  leny1 = len(y1)
  lenz1 = len(z1)
  lenq1 = len(q1)

  #print('\n lenx1={}'.format(str(lenx1)))
  #print('\n lenq1={}'.format(str(lenq1)))

  if lenx1 != leny1 or lenx1 != lenz1 or lenx1 != lenq1:
    print("WRONGa")
  elif leny1 != lenz1 or leny1 != lenq1:  
    print("WRONGb")
  elif lenz1 != lenq1:  
    print("WRONGc")

  # create list of points for fit (pre-select them first)
  xyzq1 = makeXYZQList(x1,y1,z1,q1)
  xyzq2 = makeXYZQList(x2,y2,z2,q2)
  print('\n npoints(truth)={}'.format(len(xyzq1)))
  print('\n npoints(imaging)={}'.format(len(xyzq2)))
  xmin = float('-inf')
  xmax = float('+inf')
  ymin = float('-inf')
  ymax = float('+inf')
  '''
  zmin = zmin_improved#float('-inf')
  zmax = zmin + track_length_for_fit * math.cos(math.radians(angle)) # (old=zmax=65)
  if use_improved == True:
    zmax = zmax_improved
  '''
  zmin = float(zrange_evt_xmin_xmax[ievt][1])
  zmax = float(zrange_evt_xmin_xmax[ievt][2])
  if zmin == 9999 and zmax == 9999:
    continue
  xyzq1_selected = selectPoints(xyzq1, xmin, xmax, ymin, ymax, zmin, zmax)
  xyzq2_selected = selectPoints(xyzq2, xmin, xmax, ymin, ymax, zmin, zmax)
  print('\n nselected(truth)={}'.format(len(xyzq1_selected)))
  print('\n nselected(imaging)={}'.format(len(xyzq2_selected)))
  print('\n Fit points: zmin={}, zmaz={} \n\n'.format(zmin, zmax))
  q1_selected = getQ(xyzq1_selected)
  xyz1_selected = getXYZList(xyzq1_selected)
  #xyz2_selected = getXYZList(xyzq2_selected)

  # fit best 3D line from truthDepos
  line_fit = FitBestLine3D(xyz1_selected, -201.9, 0, -200, 200, True)

  # plot x,y,z
  x1_selected = getX(xyzq1_selected)
  y1_selected = getY(xyzq1_selected)
  z1_selected = getZ(xyzq1_selected)
  x2_selected = getX(xyzq2_selected)
  y2_selected = getY(xyzq2_selected)
  z2_selected = getZ(xyzq2_selected) 
  hz1 = make_histogram(z1_selected, 100, zmin-5, zmax+5, 'z [cm]', 'Entries', 'truthDepos', 'hz1', kGreen)
  hz2 = make_histogram(z2_selected, 100, 0, 500, 'z [cm]', 'Entries', 'imaging', 'hz2', kBlue)
  #cz1 = make_canvas([hz1], 'cz1')
  
  # project list of P=(x,y,z) points on new coordinate system P'=(x',y',z') (x'-axis along fitted line)
  theta0XZ_in_degrees = angle # track angle in simulation (theta0YZ must be zero!!!)
  xyzq1_selected_prime = project(xyzq1_selected, line_fit, theta0XZ_in_degrees)
  x1_selected_prime = getX(xyzq1_selected_prime)
  y1_selected_prime = getY(xyzq1_selected_prime)
  z1_selected_prime = getZ(xyzq1_selected_prime)
  xyzq2_selected_prime = project(xyzq2_selected, line_fit, theta0XZ_in_degrees)
  x2_selected_prime = getX(xyzq2_selected_prime)
  y2_selected_prime = getY(xyzq2_selected_prime)
  z2_selected_prime = getZ(xyzq2_selected_prime)
  
  # radial distance between (y',z') points and fitted line
  r2 = []
  for i in range(len(y2_selected_prime)):
    y2_ = y2_selected_prime[i]
    z2_ = z2_selected_prime[i]
    r = np.sqrt(y2_**2 + z2_**2)
    r2.append(r)
  
  hr2 = make_histogram(r2, 60, 0, 6, 'radial distance to fitted line [cm]', 'Entries', 'imaging', 'hr2', kBlue)
  #lr = [hr2]
  #cr = make_canvas(lr, 'cr')
  
  # plot z'y' 
  nbinsy1=400
  ymin1=-0.4
  ymax1=0.4
  nbinsz1=100
  zmin1=-1
  zmax1=1
  #
  nbinsy2=30
  ymin2=-7
  ymax2=7
  nbinsz2=60
  zmin2=-5
  zmax2=5
  #hx1_prime = make_histogram(x1_selected_prime, nbins2, xmin, xmax, "x' [cm]", 'Entries', 'truthDepos', 'hx1_prime', kGreen)
  hy1_prime = make_histogram(y1_selected_prime, nbinsy1, ymin1, ymax1, "y' [cm]", 'Entries', 'truthDepos', 'hy1_prime', kGreen)
  hz1_prime = make_histogram(z1_selected_prime, nbinsz1, zmin1, zmax1, "z' [cm]", 'Entries', 'truthDepos', 'hz1_prime', kGreen)

  #hx2_prime = make_histogram(x2_selected_prime, nbins2, xmin, xmax, "x' [cm]", 'Entries', 'imaging', 'hx2_prime', kBlue)
  hy2_prime = make_histogram(y2_selected_prime, nbinsy2, ymin2, ymax2, "y' [cm]", 'Entries', 'imaging', 'hy2_prime', kBlue)
  hz2_prime = make_histogram(z2_selected_prime, nbinsz2, zmin2, zmax2, "z' [cm]", 'Entries', 'imaging', 'hz2_prime', kBlue)

  # fit gaussian to histograms and calculate sigx, sigy
  fname = 'mygaus'
  fy1_prime =  fitGaus(hy1_prime, fname)
  fy2_prime =  fitGaus(hy2_prime, fname)
  fz1_prime =  fitGaus(hz1_prime, fname)
  fz2_prime =  fitGaus(hz2_prime, fname)
  sigy_prime_tru = fy1_prime.GetParameters()[2]
  sigy_prime_img = fy2_prime.GetParameters()[2]
  sigz_prime_tru = fz1_prime.GetParameters()[2]
  sigz_prime_img = fz2_prime.GetParameters()[2]

  # draw with fit function
  cz1_prime = make_canvas([hz1_prime, fz1_prime], 'cz1_prime', fname)
  cy1_prime = make_canvas([hy1_prime, fy1_prime], 'cy1_prime', fname)
  cz2_prime = make_canvas([hz2_prime, fz2_prime], 'cz2_prime', fname)
  cy2_prime = make_canvas([hy2_prime, fy2_prime], 'cy2_prime', fname)

  # calculate sigma
  theta0XZ = math.radians(angle)
  theta=-((math.pi/2)-theta0XZ)
  sigz_prime = math.sqrt(sigz_prime_img**2 - sigz_prime_tru**2)
  sigy_prime = math.sqrt(sigy_prime_img**2 - sigy_prime_tru**2)
  sigy = sigy_prime
  sigz = sigy
  sigx = math.sqrt( (sigz_prime**2 - sigy**2 * math.cos(theta)**2)/math.sin(theta)**2 )
  print('\n (sigx, sigy, sigz) = ({}, {}, {}) mm\n'.format(sigx*10, sigy*10, sigz*10))
  vsigx.append(sigx)
  vsigy.append(sigy)

  # smear x, y coordinates
  nmulti=1
  x1_smeared = smear(x1_selected, sigx, nmulti)
  y1_smeared = smear(y1_selected, sigy, nmulti)
  z1_smeared = smear(z1_selected, sigz, nmulti)
  
  # plot x1, y1, z1 after smearing
  hx1_smeared = make_histogram(x1_smeared, 100, -154, -144, "x [cm]", 'Entries', 'truthDepos (smeared)', 'hx1_smeared', kRed)
  hy1_smeared = make_histogram(y1_smeared, 100, -5, 5, "y [cm]", 'Entries', 'truthDepos (smeared)', 'hy1_smeared', kRed)
  hz1_smeared = make_histogram(z1_smeared, 100, 40, 80, "z [cm]", 'Entries', 'truthDepos (smeared)', 'hz1_smeared', kRed)
  #lx_prime = [hx2_prime, hx1_prime_smeared]
  #ly_prime = [hy2_prime, hy1_prime_smeared]
  lx = [hx1_smeared, hx2]
  ly = [hy1_smeared, hy2]
  lz = [hz1_smeared, hz2]
  '''
  cx = make_canvas(lx, 'cx')
  cy = make_canvas(ly, 'cy')
  cz = make_canvas(lz, 'cz')
  '''
  
  # radial distance between smeared (x',y') points and fitted line
  xyzq1_smeared = makeXYZQList(x1_smeared, y1_smeared, z1_smeared, q1)
  xyzq1_smeared_prime = project(xyzq1_smeared, line_fit, theta0XZ_in_degrees)
  y1_prime_smeared = getY(xyzq1_smeared_prime)
  z1_prime_smeared = getZ(xyzq1_smeared_prime)
  r1_smeared = []
  for i in range(len(y1_prime_smeared)):
    y1_ = y1_prime_smeared[i]
    z1_ = z1_prime_smeared[i]
    r = np.sqrt(y1_**2 + z1_**2)
    r1_smeared.append(r)

  hr1_smeared = make_histogram(r1_smeared, 60, 0, 6, 'radial distance to fitted line [cm]', 'Entries', 'truthDepos (smeared)', 'hr1_smeared', kRed)
  lr = [hr2, hr1_smeared]
  #cr = make_canvas(lr, 'cr')
  cz1 = make_canvas([hz1], 'cz1')

  '''
  # get points inside sphere centered at truth points
  # and radius equal to mean distance of imaging 
  # points to fitted line
  mean = np.mean(distances2)
  std = np.std(distances2)
  radius = mean + 0*std #float('+inf')
  nmatched = []
  qdiff = []
  xdiff = []
  ydiff = []
  zdiff = []
  #qdiff =  10 #float('+inf') #0.5
  # loop over truth points
  for ip in range(len(xyzq1_selected)):
    center = xyzq1_selected[ip]
    points_inside = getInsideSphere(center, radius, xyzq2_selected)
    n = len(points_inside)
    nmatched.append(n)
    if n > 0:
      cx = center[0]
      cy = center[1]
      cz = center[2]
      cq = center[3]
      matched = points_inside[0]
      x = matched[0]
      y = matched[1]
      z = matched[2]
      q = matched[3]
      xdiff.append(cx-x)
      ydiff.append(cy-y)
      zdiff.append(cz-z)
      qdiff.append(abs(cq-q))
    '''
  '''
    nq = 0
    if n == 1: #more than 1 img matched
      for i2 in range(n): #plot difference between qtruth and qimg
        q1 = center[3]
        q2 = points_inside[i2][3]
        if(abs(q1-q2)<qdiff):
          nq = nq + 1
        #qmatched.append(abs(q1-q2))
      #print('\n nq={}'.format(nq))  
    '''
  '''
  c1 = make_histogram(nmatched, 21, -.5, 20.5, '# img points within radius centered at truth point', 'Entries', 'Imaging','c1')
  c_q = make_histogram(qdiff, 100, 0, 250, 'matched img point (q_img - q_truth) [electrons]', 'Entries', 'Imaging','c2')
  c_x = make_histogram(xdiff, 100, -50, 50, 'matched img point (x_img - x_truth) [cm]', 'Entries', 'xdiff','c_x')
  c_y = make_histogram(ydiff, 100, -50, 50, 'matched img point (y_img - y_truth) [cm]', 'Entries', 'ydiff','c_y')
  c_z = make_histogram(zdiff, 100, -50, 50, 'matched img point (z_img - z_truth) [cm]', 'Entries', 'zdiff','c_z')
  '''

  
  # write smeared truth points to JSON
  foutname1 = str(ievt) + '-' + lab1 + '_smeared.json'
  outputfile1 = datapath + '/' + subdir + '/' + foutname1
  print('\n Saving output file {}..'.format(outputfile1))
  Path(outputfile1).write_text(json.dumps({"runNo": 0, "subRunNo": 0, "eventNo": 0, "geom": "sbnd", "type": "cluster", "x":x1_smeared, "y":y1_smeared, "z": z1_smeared, "q": q1_selected}))

  # close files 
  fin1.close()
  fin2.close()

# make sigma histograms after loop
hsigx = make_histogram(vsigx, 100, 0, 0.5, 'sigma_x [cm]', 'Entries', 'title', 'hsigx', kBlue)
hsigy = make_histogram(vsigy, 100, 0, 1.5, 'sigma_y [cm]', 'Entries', 'title', 'hsigy', kBlue)
csigx = make_canvas([hsigx], 'csigx')
csigy = make_canvas([hsigy], 'csigy')
