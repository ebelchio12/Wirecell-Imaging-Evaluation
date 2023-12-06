# Script to read in json file from imaging clusters and
# save x,y,z and q
#
# author: Ewerton B
# usage: python readjson.py
#

import json
from pathlib import Path
from BestFit3D import *

# ROOT colors
kBlue=4
kRed=2
kGreen=8

#datapath = '/sbnd/data/users/ebatista/imaging-handscan/XYMuon/bee_singlemuon_nthreshold_0d0/data'
#datapath = '/sbnd/data/users/ebatista/imaging-handscan/XYMuon/LynnSamples/data'
datapath = '/sbnd/data/users/ebatista/imaging-handscan/BNBSamples/bnb/bee/data'
eventsToProcess = 499
nskip=380
lab1 = 'truthDepo'

sigx=0.1565
sigy=0.6784
sigz=sigy


# loop over events
for ievt in range(eventsToProcess):

  if ievt <= nskip:
    continue

  #print('\n ******************')
  print('\n Event {}'.format(ievt))
  #print('\n ******************')

  subdir = str(ievt)
  fname1 = str(ievt) + '-' + lab1 + '.json'
  inputfile1 = datapath + '/' + subdir + '/' + fname1
  
  # debug info
  if ievt == 0:
    print('\n Processing {} events (Nevents must be <= number of folders in {})..'.format(eventsToProcess,datapath))
    print('\n Reading input file {}..'.format(inputfile1))

  # load data
  fin1 = open(inputfile1)
  data1 = json.load(fin1)

  x1 = data1['x']
  y1 = data1['y']
  z1 = data1['z']
  q1 = data1['q']

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
  #xyzq1 = makeXYZQList(x1,y1,z1,q1)
  print('\n [ievt {}] npoints(truth)={}'.format(ievt, lenx1))
  #xmin = float('-inf')
  #xmax = float('+inf')
  #ymin = float('-inf')
  #ymax = float('+inf')
  #zmin = float('-inf')
  #zmax = float('+inf')
  #xyzq1_selected = selectPoints(xyzq1, xmin, xmax, ymin, ymax, zmin, zmax)
  #print('\n nselected(truth)={}'.format(len(xyzq1_selected)))
  #q1_selected = getQ(xyzq1_selected)
  #xyz1_selected = getXYZList(xyzq1_selected)
  #xyz1_selected = getXYZList(xyzq1)

  # plot x,y,z
  #x1_selected = getX(xyzq1_selected)
  #y1_selected = getY(xyzq1_selected)
  #z1_selected = getZ(xyzq1_selected)
  #q1_selected = getQ(xyzq1_selected)
  '''
  hx1 = make_histogram(x1, 100, -200, 200, 'x [cm]', 'Entries', 'truthDepos', 'hx1', kGreen)
  hy1 = make_histogram(y1, 100, -200, 200, 'y [cm]', 'Entries', 'truthDepos', 'hy1', kGreen)
  hz1 = make_histogram(z1, 100, 0, 500, 'z [cm]', 'Entries', 'truthDepos', 'hz1', kGreen)
  '''
  #cx1 = make_canvas([hx1], 'cx1')
  #cy1 = make_canvas([hy1], 'cy1')
  #cz1 = make_canvas([hz1], 'cz1')
  
  # smear x, y coordinates
  nmulti=1
  x1_smeared = smear(x1, sigx, nmulti)
  y1_smeared = smear(y1, sigy, nmulti)
  z1_smeared = smear(z1, sigz, nmulti)
  
  # plot x1, y1, z1 after smearing
  '''
  hx1_smeared = make_histogram(x1_smeared, 100, -200, 200, "x [cm]", 'Entries', 'truthDepos (smeared)', 'hx1_smeared', kRed)
  hy1_smeared = make_histogram(y1_smeared, 100, -200, 200, "y [cm]", 'Entries', 'truthDepos (smeared)', 'hy1_smeared', kRed)
  hz1_smeared = make_histogram(z1_smeared, 100, 0, 500, "z [cm]", 'Entries', 'truthDepos (smeared)', 'hz1_smeared', kRed)
  lx = [hx1_smeared, hx1]
  ly = [hy1_smeared, hy1]
  lz = [hz1_smeared, hz1]
  cx = make_canvas(lx, 'cx')
  cy = make_canvas(ly, 'cy')
  cz = make_canvas(lz, 'cz')
  ''' 
  # write smeared truth points to JSON
  foutname1 = str(ievt) + '-' + lab1 + '_smeared.json'
  outputfile1 = datapath + '/' + subdir + '/' + foutname1
  print('\n Saving output file {}..'.format(outputfile1))
  Path(outputfile1).write_text(json.dumps({"runNo": 0, "subRunNo": 0, "eventNo": str(ievt), "geom": "sbnd", "type": "cluster", "x":x1_smeared, "y":y1_smeared, "z": z1_smeared, "q": q1}))

  # close files 
  fin1.close()

