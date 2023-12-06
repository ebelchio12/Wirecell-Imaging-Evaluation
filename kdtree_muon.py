#!/usr/bin/env python
# coding: utf-8

# In[66]:


import sys
import os
data_path = '/sbnd/data/users/ebatista/imaging-handscan/BNBSamples/bnb/bee/data' #os.environ.get("DATA")
app_path = os.environ.get("APP")
sys.path.append(data_path+'/wirecell/helper')
import wc_img as wc_img
from tqdm import tqdm
from scipy.spatial import KDTree, cKDTree


import importlib
____ = importlib.reload(wc_img)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# In[67]:


# path = "/sbnd/data/users/ebatista/imaging-handscan/XYMuon/smeared-samples/data"
# path = "/exp/sbnd/data/users/lynnt/wirecell/img/samples/muon/data"
#path =  "/exp/sbnd/data/users/lynnt/wirecell/img/samples/bnb/data" #commented Ewerton
path = '/sbnd/data/users/ebatista/imaging-handscan/BNBSamples/bnb/bee/data'
#img_list, tru_list = wc_img.load_data(path,nevents=10) #commented Ewerton
img_list, tru_list = wc_img.load_data(path, nevents=500, img_str="imaging", tru_str="truthDepo_smeared")

# In[68]:


tru_match_list = []
tru_miss_list  = []
img_match_list = []
img_miss_list  = []
for i in tqdm(range(len(img_list))):
    img_match, img_miss, tru_match, tru_miss,  = wc_img.find_nearest(img_list[i],tru_list[i],max_dist=2)
    tru_match_list.append(tru_match)
    tru_miss_list.append(tru_miss)
    img_match_list.append(img_match)
    img_miss_list.append(img_miss)


# In[69]:


fig, axes = plt.subplots(1,3,figsize=(15,5))
for event in range(len(tru_match_list)): 
    tru_mat_x = tru_match_list[event][:,0]; tru_mat_y = tru_match_list[event][:,1]; tru_mat_z = tru_match_list[event][:,2]
    img_mat_x = img_match_list[event][:,0]; img_mat_y = img_match_list[event][:,1]; img_mat_z = img_match_list[event][:,2]
    
    if event < 5:
      axes[0].hist(tru_mat_x - img_mat_x,bins=np.linspace( -5,5,51),histtype="step",lw=2,density=True,label=f"evt {event}")
      axes[1].hist(tru_mat_y - img_mat_y,bins=np.linspace( -5,5,51),histtype="step",lw=2,density=True,label=f"evt {event}")
      axes[2].hist(tru_mat_z  - img_mat_z,bins=np.linspace(-5,5,51),histtype="step",lw=2,density=True,label=f"evt {event}")
    else:
      axes[0].hist(tru_mat_x - img_mat_x,bins=np.linspace( -5,5,51),histtype="step",lw=2,density=True)
      axes[1].hist(tru_mat_y - img_mat_y,bins=np.linspace( -5,5,51),histtype="step",lw=2,density=True)
      axes[2].hist(tru_mat_z  - img_mat_z,bins=np.linspace(-5,5,51),histtype="step",lw=2,density=True)

axes[0].legend(); axes[1].legend(); axes[2].legend()
axes[0].set_ylabel("Area Normalized"); axes[1].set_ylabel("Area Normalized"); axes[2].set_ylabel("Area Normalized")
axes[0].set_xlabel(r"$x_{\mathrm{imaging}} - x_{\mathrm{truth}}$",fontsize=15)
axes[1].set_xlabel(r"$y_{\mathrm{imaging}} - y_{\mathrm{truth}}$",fontsize=15)
axes[2].set_xlabel(r"$z_{\mathrm{imaging}} - z_{\mathrm{truth}}$",fontsize=15)


plt.suptitle("xyz Difference between Imaging Point and Smeared Truth-Matched Point",fontsize=15)
plt.show()


# In[70]:


event = 8
img = img_list[event]
tru = tru_list[event]


# In[71]:


fig, axes = plt.subplots(1,3,figsize=(25,5))
axes[0].scatter(tru[:,0],tru[:,1],marker='.',alpha=0.2,s=2,color='red' ,label='smeared truth')
axes[0].scatter(img[:,0],img[:,1],marker='.',alpha=1.0,s=2,color='blue',label='imaging')
ax0_xmin, ax0_xmax = (axes[0].get_xlim())
ax0_ymin, ax0_ymax = (axes[0].get_ylim())
axes[0].set_xlabel('x [cm]', fontsize=12); axes[0].set_ylabel('y [cm]', fontsize=12)
axes[0].set_title("Front View", fontsize=15)

axes[1].scatter(tru[:,2],tru[:,0],marker='.',alpha=0.2,s=2,color='red' ,label='smeared truth')
axes[1].scatter(img[:,2],img[:,0],marker='.',alpha=1.0,s=2,color='blue',label='imaging')
ax1_xmin, ax1_xmax = (axes[1].get_xlim())
ax1_ymin, ax1_ymax = (axes[1].get_ylim())
axes[1].set_xlabel('z [cm]', fontsize=12); axes[1].set_ylabel('x [cm]', fontsize=12)
axes[1].set_title("Top View", fontsize=15)

axes[2].scatter(tru[:,2],tru[:,1],marker='.',alpha=0.2,s=2,color='red' ,label='smeared truth')
axes[2].scatter(img[:,2],img[:,1],marker='.',alpha=1.0,s=2,color='blue',label='imaging')
ax2_xmin, ax2_xmax = (axes[2].get_xlim())
ax2_ymin, ax2_ymax = (axes[2].get_ylim())
axes[2].set_xlabel('z [cm]', fontsize=12); axes[2].set_ylabel('y [cm]', fontsize=12)
axes[2].set_title("Side View", fontsize=15)

axes[0].legend(); axes[1].legend(); axes[2].legend()

plt.suptitle("Image matched to smeared truth (2 cm max)", fontsize=18, fontweight='bold', y=1.01)
# legend_elements = [Line2D([0], [0], marker='.', color='w', markerfacecolor='orange',  markersize=10, label='truth miss'),
#                    Line2D([0], [0], marker='.', color='w', markerfacecolor='green', markersize=10, label='truth match')]
# for ax in axes:
#     ax.legend(handles=legend_elements)
plt.show() 


# In[10]:


xmin = -110; xmax = -90
ymin = -10; ymax = 10
zmin = 100; zmax = 150


# In[11]:


fig, axes = plt.subplots(1,3,figsize=(25,5))
axes[0].scatter(tru[:,0],tru[:,1],marker='.',alpha=0.2,s=2,color='red' ,label='smeared truth')
axes[0].scatter(img[:,0],img[:,1],marker='.',alpha=1.0,s=2,color='blue',label='imaging')
axes[0].set_xlim(xmin,xmax)
axes[0].set_ylim(ymin,ymax)
axes[0].set_xlabel('x [cm]', fontsize=12); axes[0].set_ylabel('y [cm]', fontsize=12)
axes[0].set_title("Front View", fontsize=15)

axes[1].scatter(tru[:,2],tru[:,0],marker='.',alpha=0.2,s=2,color='red' ,label='smeared truth')
axes[1].scatter(img[:,2],img[:,0],marker='.',alpha=1.0,s=2,color='blue',label='imaging')
axes[1].set_xlim(zmin,zmax)
axes[1].set_ylim(xmin,xmax)
axes[1].set_xlabel('z [cm]', fontsize=12); axes[1].set_ylabel('x [cm]', fontsize=12)
axes[1].set_title("Top View", fontsize=15)

axes[2].scatter(tru[:,2],tru[:,1],marker='.',alpha=0.2,s=2,color='red' ,label='smeared truth')
axes[2].scatter(img[:,2],img[:,1],marker='.',alpha=1.0,s=2,color='blue',label='imaging')
axes[2].set_xlim(zmin,zmax)
axes[2].set_ylim(ymin,ymax)
axes[2].set_xlabel('z [cm]', fontsize=12); axes[2].set_ylabel('y [cm]', fontsize=12)
axes[2].set_title("Side View", fontsize=15)

axes[0].legend(); axes[1].legend(); axes[2].legend()

plt.suptitle("Image matched to smeared truth (2 cm max)", fontsize=18, fontweight='bold', y=1.01)
# legend_elements = [Line2D([0], [0], marker='.', color='w', markerfacecolor='orange',  markersize=10, label='truth miss'),
#                    Line2D([0], [0], marker='.', color='w', markerfacecolor='green', markersize=10, label='truth match')]
# for ax in axes:
#     ax.legend(handles=legend_elements)
plt.show() 


# In[8]:


tru_match_arr, tru_miss_arr, img_match_arr, img_miss_arr = wc_img.find_nearest(tru,img,max_dist=2)
print("[Ewerton] % of tru points without a position match: ",    np.round(len(tru_miss_arr)/len(tru) *100,2))
print("% of total charge of points without a match: ", np.round(np.sum(tru_miss_arr[:,3])/np.sum(tru[:,3]) *100,2))

fig, axes = plt.subplots(1,3,figsize=(25,5))
axes[0].scatter(tru_match_arr[:,0],tru_match_arr[:,1],marker='.',alpha=0.2,s=2,color='green',label='tru match')
axes[0].scatter(tru_miss_arr[:,0],tru_miss_arr[:,1],  marker='.',alpha=1.0,s=2,color='orange',label='tru miss')
ax0_xmin, ax0_xmax = (axes[0].get_xlim())
ax0_ymin, ax0_ymax = (axes[0].get_ylim())
axes[0].set_xlabel('x [cm]', fontsize=12); axes[0].set_ylabel('y [cm]', fontsize=12)
axes[0].set_title("Front View", fontsize=15)

axes[1].scatter(tru_match_arr[:,2],tru_match_arr[:,0],marker='.',alpha=0.2,s=2,color='green',label='tru match')
axes[1].scatter(tru_miss_arr[:,2],tru_miss_arr[:,0]  ,marker='.',alpha=1.0,s=2,color='orange',label='tru miss')
ax1_xmin, ax1_xmax = (axes[1].get_xlim())
ax1_ymin, ax1_ymax = (axes[1].get_ylim())
axes[1].set_xlabel('z [cm]', fontsize=12); axes[1].set_ylabel('x [cm]', fontsize=12)
axes[1].set_title("Top View", fontsize=15)

axes[2].scatter(tru_match_arr[:,2],tru_match_arr[:,1],marker='.',alpha=0.2,s=2,color='green',label='tru match')
axes[2].scatter(tru_miss_arr[:,2],tru_miss_arr[:,1],  marker='.',alpha=1.0,s=2,color='orange',label='tru miss')
ax2_xmin, ax2_xmax = (axes[2].get_xlim())
ax2_ymin, ax2_ymax = (axes[2].get_ylim())
axes[2].set_xlabel('z [cm]', fontsize=12); axes[2].set_ylabel('y [cm]', fontsize=12)
axes[2].set_title("Side View", fontsize=15)

plt.suptitle("Imaging ", fontsize=18, fontweight='bold', y=1.01)
legend_elements = [Line2D([0], [0], marker='.', color='w', markerfacecolor='orange',  markersize=10, label='truth miss'),
                   Line2D([0], [0], marker='.', color='w', markerfacecolor='green', markersize=10, label='truth match')]
for ax in axes:
    ax.legend(handles=legend_elements)
plt.show() 


# In[9]:


plt.hist(tru_miss_arr[:,3], bins=np.linspace(0,7e3,50),histtype="step",density=True)
plt.hist(tru_match_arr[:,3],bins=np.linspace(0,7e3,50),histtype="step",density=True)
plt.show()


# In[10]:


img_match_arr, img_miss_arr, tru_match_arr, tru_miss_arr = wc_img.find_nearest(img,tru,max_dist=2)
print("number of img points without a match: ", len(img_miss_arr))

fig, axes = plt.subplots(1,3,figsize=(25,5))
axes[0].scatter(img_miss_arr[:,0], img_miss_arr[:,1], marker='.',alpha=1,  s=5,color='red',label='img miss')
axes[0].scatter(img_match_arr[:,0],img_match_arr[:,1],marker='.',alpha=0.2,s=5,color='blue',label='img match')
axes[0].set_xlim(ax0_xmin,ax0_xmax)
axes[0].set_ylim(ax0_ymin,ax0_ymax)
axes[0].set_xlabel('x [cm]', fontsize=12); axes[0].set_ylabel('y [cm]', fontsize=12)
axes[0].set_title("Front View", fontsize=15)

axes[1].scatter(img_miss_arr[:,2],img_miss_arr[:,0],  marker='.',alpha=1,  s=5,color='red',label='img miss')
axes[1].scatter(img_match_arr[:,2],img_match_arr[:,0],marker='.',alpha=0.2,s=5,color='blue',label='img match')
axes[1].set_xlim(ax1_xmin,ax1_xmax)
axes[1].set_ylim(ax1_ymin,ax1_ymax)
axes[1].set_xlabel('z [cm]', fontsize=12); axes[1].set_ylabel('x [cm]', fontsize=12)
axes[1].set_title("Top View", fontsize=15)

axes[2].scatter(img_miss_arr[:,2],img_miss_arr[:,1],  marker='.',alpha=1,  s=5,color='red',label='img miss')
axes[2].scatter(img_match_arr[:,2],img_match_arr[:,1],marker='.',alpha=0.2,s=5,color='blue',label='img match')
axes[2].set_xlim(ax2_xmin,ax2_xmax)
axes[2].set_ylim(ax2_ymin,ax2_ymax)
axes[2].set_xlabel('z [cm]', fontsize=12); axes[2].set_ylabel('y [cm]', fontsize=12)
axes[2].set_title("Side View", fontsize=15)

plt.suptitle("Image Matched to Truth (2 cm max)", fontsize=18, fontweight='bold', y=1.01)

legend_elements = [Line2D([0], [0], marker='.', color='w', markerfacecolor='red',  markersize=10, label='image miss'),
                   Line2D([0], [0], marker='.', color='w', markerfacecolor='blue', markersize=10, label='image match')]
for ax in axes:
    ax.legend(handles=legend_elements)

plt.show()


# In[13]:


print( "sum of matched img charge / total truth charge", np.round(np.sum(img_match_arr[:,3])/np.sum(tru[:,3:])*100,2))

