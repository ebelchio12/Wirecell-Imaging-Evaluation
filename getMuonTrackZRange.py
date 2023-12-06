

def getMuonTrackZminZmax(rangefile):
  # Initialize empty list for zcut
  evtnumber_zmin_zmax = []

  # Open the file with ranges for z-cut
  with open(rangefile, 'r') as file:
    # Read lines from the file
    lines = file.readlines()

    # Split each line based on the comma and store in respective lists
    for line in lines:
      values = line.strip().split(' ')
      evtnumber_zmin_zmax.append([values[0], values[1], values[2]])

  return evtnumber_zmin_zmax

# Print the results
#rangefile='/sbnd/data/users/ebatista/imaging-handscan/XYMuon/muontracks_imaging-fit_zrange-formatted.txt'
#mylist = getMuonTrackZminZmax(rangefile)
#print("evtnumber_zmin_zmax:", mylist)
