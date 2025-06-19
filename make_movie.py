import numpy as np
import pickle
import glob
import os
import utils

beta = 0.7
S0 = 0.04
tmyosin = 5.0
tviscous = 40.0
btype = 1
flow_align = 0.0
#Choose plot type: 
# 0 = single tenor component, 
# 1 = quiver plot of velocity field, 
# 2 = all M and pi components
plot_option = 0

L = 50.0
Ngrid = int(4*L + 1)
vskip = 11 #default 11, for quiver plot
x = np.linspace(-0.5*L,0.5*L,Ngrid)
y = np.linspace(-0.5*L,0.5*L,Ngrid)

movie_location = f'.//movies_L{L}/flow_align_{flow_align}/beta_{beta}/tmyosin_{tmyosin}/tviscous_{tviscous}/S0_{S0}/btype_{btype}/'
os.makedirs(movie_location, exist_ok=True)

sim_data_location = f'.//data_L{L}/flow_align_0.0/beta_{beta}/tmyosin_{tmyosin}/tviscous_{tviscous}/S0_{S0}/btype_{btype}/'
files = sorted(glob.glob(sim_data_location + '/data*.pickle'))
num_saved_steps = len(files)

#First plot each frame
for f_idx in range(num_saved_steps):
    
    f = sim_data_location+f'/data_{f_idx}.pickle'
    print(f't_idx = {f_idx}')
    sim_data = pickle.load(open(f,'rb'),encoding='latin1')
    figname = movie_location+'/Panel{}.png'.format(int(f_idx))
    utils.plot_field(plot_option, btype, figname, sim_data, x, y, vskip, Ngrid, beta, S0, tmyosin, tviscous, selected_field = 'gammadotyy', col_lower=-0.02, col_upper=0.02)

    #Optional arguments: 
    #If using plot option 0 - choose "selected_filed" and colour bar limits "col_lower" and "col_upper"
    #If using plot option 2 - choose Mxx and Myy colour bar limits with "Mxx_lower"/"Mxx_upper"
    #                       - choose Mxy colour bar limits with "Mxy_lower"/"Mxy_upper"
    #                       - choose pressure colour bar limits (same for all components) with "pi_lower"/"pi_upper"


utils.pngs_to_movie(movie_location,'gammadotyy.gif')


