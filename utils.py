import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
import glob

def pngs_to_movie(folder, output_file, fps=10):
    # Get all PNG files sorted by index
    image_files = sorted(
        [f for f in os.listdir(folder) if f.startswith("Panel") and f.endswith(".png")],
        key=lambda f: int(f.replace("Panel", "").replace(".png", ""))
    )

    # Read the first image to get dimensions
    img = mpimg.imread(os.path.join(folder, image_files[0]))
    fig, ax = plt.subplots()
    im = ax.imshow(img)
    ax.axis('off')  # Hide axes

    def update(frame):
        img = mpimg.imread(os.path.join(folder, image_files[frame]))
        im.set_array(img)
        return [im]

    ani = FuncAnimation(fig, update, frames=len(image_files), blit=True)
    writer = FFMpegWriter(fps=fps)
    ani.save(output_file, writer=writer)
    print(f"Saved movie to {output_file}")

def plot_field(plot_option, sim_data, x, y, vskip, Ngrid, beta, S0, tmyosin, tviscous, btype, figname, **kwargs):

    #Data is stored like so
    # data = {'t': step_number * self.dt, 'Mxx': Mxx,
    #                         'Myy': Myy, 'Mxy': Mxy, 'Pxx': Pxx, 'Pyy': Pyy, 'Pxy': Pxy, 'vx': vx, 'vy': vy,
    #                         'gammadotxx': gammadotxx, 'gammadotyy': self.gammadotyy, 'gammadotxy': self.gammadotxy}
    
    Mxx = sim_data['Mxx']
    Myy = sim_data['Myy']
    Mxy = sim_data['Mxy']
    Pxx = sim_data['Pxx']
    Pyy = sim_data['Pyy']
    Pxy = sim_data['Pxy']
    gammadotxx = sim_data['gammadotxx']
    gammadotyy = sim_data['gammadotyy']
    vx = sim_data['vx']
    vy = sim_data['vy']
    t = sim_data['t']
    S0xx, S0yy, S0xy = select_bcs(S0,btype)

    #Single tensor component 
    if plot_option == 0:
        selected_fieldname = kwargs.get('selected_field', None)
        selected_field = sim_data[selected_fieldname]
        fig = plt.figure()
        p = plt.pcolormesh(y,x, selected_field, cmap=plt.cm.jet, rasterized=True)
        #The rasrerized = True option avoids white lines when the figure is saved as pdf
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(selected_fieldname+f', beta = {beta}, ext. stress (xx) = {S0xx}, ext. stress (yy) = {S0yy}, tm = {tmyosin}, tv = {tviscous}')
        plt.colorbar(p)
        p.set_clim(kwargs.get('col_lower', 0),kwargs.get('col_upper', 1))
        ax = plt.gca()
        ax.set_aspect(1 / ax.get_data_ratio())
    
    #Velocity quiver plot
    if plot_option == 1:
        fig = plt.figure()
        plt.quiver(y[::vskip], x[::vskip], vx[1:(Ngrid-1),1:(Ngrid-1)][::vskip,::vskip], vy[1:(Ngrid-1),1:(Ngrid-1)][::vskip,::vskip], color='b', width = 0.0075)
        plt.title(f"velocity at t = {t:.3f}, beta = {beta}")
        ax = plt.gca()
        ax = plt.gca()
        ax.set_aspect(1 / ax.get_data_ratio())

    #Full grid
    if plot_option == 2:
        fig, axs = plt.subplots(2, 3, figsize = (11,12))
        # fig.xticks(fontsize = 14)
        # plt.cla() #clear current figure
        fig.suptitle(f'beta = {beta}, ext. stress (xx) = {S0xx}, ext. stress (yy) = {S0yy}, tm = {tmyosin}, tv = {tviscous} \n t = {t}', size = 18)

        axs[0,0].set_title('M_xx',fontsize=16)
        p00 = axs[0,0].pcolormesh(x,x, Mxx, cmap=plt.cm.jet,rasterized=True)
        cbar = fig.colorbar(p00, ax=axs[0,0])
        p00.set_clim(kwargs.get('Mxx_lower', 0),kwargs.get('Mxx_upper', 1))
        axs[0,0].set_aspect(1 / axs[0,0].get_data_ratio())
        axs[0,0].tick_params(axis='both', labelsize=14)
        cbar.ax.tick_params(labelsize=14) 

        axs[0,1].set_title("M_yy",fontsize=16)
        p01 = axs[0,1].pcolormesh(x,x, Myy, cmap=plt.cm.jet,rasterized=True)
        cbar = fig.colorbar(p01, ax=axs[0,1])
        p01.set_clim(kwargs.get('Mxx_lower', 0),kwargs.get('Mxx_upper', 1))
        axs[0,1].set_aspect(1 / axs[0,0].get_data_ratio())
        axs[0,1].tick_params(axis='both', labelsize=14)
        cbar.ax.tick_params(labelsize=14) 

        # #change the colourbar limits for Mxx and Myy for the initial condition if using a myosin perturbation,
        # #so that the initial condition can actually be seen.
        # if self.myo_pert and t == 0:
        #     delta_col_m = 0.0001
        #     p00.set_clim(self.m0 - delta_col_m, self.m0 + delta_col_m)
        #     p01.set_clim(self.m0 - delta_col_m, self.m0 + delta_col_m)

        axs[0,2].set_title("M_xy",fontsize=16)
        p02 = axs[0,2].pcolormesh(x,x, Mxy, cmap=plt.cm.jet,rasterized=True)
        cbar = fig.colorbar(p02, ax=axs[0,2])
        p02.set_clim(-kwargs.get('Mxy_lower', -0.2),kwargs.get('Mxy_upper', 0.2))
        axs[0,2].set_aspect(1 / axs[0,0].get_data_ratio())
        axs[0,2].tick_params(axis='both', labelsize=14)
        cbar.ax.tick_params(labelsize=14) 

        axs[1,0].set_title("pi_xx",fontsize=16)
        p10 = axs[1,0].pcolormesh(x,x, Pxx, cmap=plt.cm.jet,rasterized=True)
        cbar = fig.colorbar(p10, ax=axs[1,0])
        p10.set_clim(kwargs.get('pi_lower', -0.05),kwargs.get('pi_upper', 0.05))
        axs[1,0].set_aspect(1 / axs[0,0].get_data_ratio())
        axs[1,0].tick_params(axis='both', labelsize=14)
        cbar.ax.tick_params(labelsize=14) 

        axs[1,1].set_title("pi_yy",fontsize=16)
        p11 = axs[1,1].pcolormesh(x,x, Pyy, cmap=plt.cm.jet,rasterized=True)
        cbar = fig.colorbar(p11, ax=axs[1,1])
        p11.set_clim(kwargs.get('pi_lower', -0.05),kwargs.get('pi_upper', 0.05))
        axs[1,1].set_aspect(1 / axs[0,0].get_data_ratio())
        axs[1,1].tick_params(axis='both', labelsize=14)
        cbar.ax.tick_params(labelsize=14) 

        axs[1,2].set_title("pi_xy",fontsize=16)
        p12 = axs[1,2].pcolormesh(x,x, Pxy, cmap=plt.cm.jet,rasterized=True)
        cbar = fig.colorbar(p12, ax=axs[1,2])
        p12.set_clim(kwargs.get('pi_lower', -0.05),kwargs.get('pi_upper', 0.05))
        axs[1,2].set_aspect(1 / axs[0,0].get_data_ratio())
        axs[1,2].tick_params(axis='both', labelsize=14)
        cbar.ax.tick_params(labelsize=14) 

    plt.savefig(figname,dpi=500)
    plt.close(fig) 

def select_bcs(S0, btype):

    if btype == 0:
        S0xx = S0
        S0yy = 0.
        S0xy = 0.
    elif btype == 1:
        S0xx = S0
        S0yy = -S0
        S0xy = 0.
    elif btype == 2:
        S0xx = S0
        S0yy = S0
        S0xy = 0.
    elif btype == 3:
        S0xx = -S0
        S0yy = -S0
        S0xy = 0.
    elif btype == 4: #pull on bottom only 
        S0xx = 0
        S0yy = S0
        S0xy = 0

    return S0xx, S0yy, S0xy

