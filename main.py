import os
from active_sheet import ActiveSheet
import time
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--beta", type = float, default = 0.5, help = "activity parameter")
parser.add_argument("-s", "--S0", type = float, default = 0.0, help = "boundary stress")
parser.add_argument("-m", "--tmyosin", type = float, default = 10., help = "myosin time")
parser.add_argument("-v", "--tviscous", type = float, default = 10., help = "visco-elastic time")
parser.add_argument("-f", "--flow_align", type = float, default = 0., help = "flow alignment parameter")
parser.add_argument("-l", "--sys_size", type = float, default = 50., help = "system size")
parser.add_argument("-t", "--tmax", type = float, default = 1000., help = "simulation time") #1000
parser.add_argument("-p", "--tplot", type = float, default = 10., help = "how often to save") #10 check if this line is consistent with the desktop version. 
parser.add_argument("-n", "--noise_amp", type = float, default = 0., help = "noise amplitude for initial condition") #Update on desktop version
parser.add_argument("-a", "--boundary", type = int, default = 1, help = "boundary type, 0 is uniax, 1 is biax, 2 is pull, 3 is push, 4 is pull only bottom")
parser.add_argument("-d", "--top_dir", type = str, default = './', help = "top directory where data is saved")

args = parser.parse_args()

#Assign params
tmyosin = args.tmyosin
tviscous = args.tviscous
beta = args.beta 

#Generally keep these fixed
shear_mod = 0.5 
bulk_mod = 1.0 
k0 = 8.
friction = 1.0  
m0 = 0.5  
D = 1.

btype = args.boundary
S0xx, S0yy, S0xy = utils.select_bcs(args.S0,btype)
    
#flow-align = 0: Jaumann, flow_align = 0.5: Upper Convected, flow_align = -0.5: Lower Convected.
flow_align = args.flow_align  

#simulation_params
sys_size = args.sys_size
Ngrid = int(4*sys_size + 1)
h = sys_size / (Ngrid - 1)
dt = 0.25 * (h ** 2) #default 0.25*(h**2)
tplot = args.tplot
tmax = args.tmax
noise_amp = args.noise_amp
gradlim = 0.1
cutgrad = True
myo_pert = False

#plotting_params
maxtsteps = round(tmax/dt)
nskip = round(tplot/dt)

vskip = 11 #For the quiver plot in simluator.plot()

dirname_data = args.top_dir+f'/data_L{args.sys_size}/flow_align_{flow_align}/beta_{beta}/tmyosin_{tmyosin}/tviscous_{tviscous}/S0_{args.S0}/btype_{args.boundary}/'

#exist_ok=True means that if a directory already exists, it won't produce and error. 
#Instead it will make a directory at the first point in the nest where there isn't a directory.
#e.g if I do os.makedirs('A/B/C', exist_o k=True), and A/B exists, then C will be made inside B. If none of them exist, A/B/C will be made, and so on.
os.makedirs(dirname_data, exist_ok=True)
   
simulator = ActiveSheet(dt, tmax, tplot, Ngrid, sys_size, D, tmyosin, tviscous, bulk_mod, shear_mod, k0, friction, m0, beta,
             gradlim, S0xx, S0yy, S0xy, cutgrad, myo_pert, flow_align, noise_amp, btype)

simulator.set_initial_conditions()
simulator.save_params(dirname_data,nskip)

t1 = time.time()

for k in range(maxtsteps):    
    simulator.integrate_one_step()
    
    if k%nskip == 0:  
        print(f't = {k*dt:.2f}')         
        simulator.save_state(k,dirname_data,nskip)
            
    simulator.update_current_values()

t2 = time.time()
run_time = t2-t1
print(f'run_time = {run_time}')

        
            
            
            
    
    



