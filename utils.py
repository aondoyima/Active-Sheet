import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

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

def plot_field(plot_option, btype, figname, sim_data, x, y, vskip, Ngrid, beta, S0, tmyosin, tviscous, **kwargs):
    # Unpack fields
    Mxx, Myy, Mxy = sim_data['Mxx'], sim_data['Myy'], sim_data['Mxy']
    Pxx, Pyy, Pxy = sim_data['Pxx'], sim_data['Pyy'], sim_data['Pxy']
    gammadotxx, gammadotyy = sim_data['gammadotxx'], sim_data['gammadotyy']
    vx, vy, t = sim_data['vx'], sim_data['vy'], sim_data['t']
    S0xx, S0yy, S0xy = select_bcs(S0, btype)

    font_title = 18
    font_label = 16

    if plot_option == 0:
        selected_fieldname = kwargs.get('selected_field')
        if selected_fieldname not in sim_data:
            raise ValueError(f"Field '{selected_fieldname}' not in sim_data keys.")
        selected_field = sim_data[selected_fieldname]
        plt_label = plot_label(selected_fieldname)

        fig, ax = plt.subplots(figsize=(7, 6))
        p = ax.pcolormesh(y, x, selected_field, cmap='viridis', rasterized=True)
        p.set_clim(kwargs.get('col_lower', 0), kwargs.get('col_upper', 1))

        ax.tick_params(labelsize=font_label)
        ax.set_xlabel('$x$', fontsize=font_label)
        ax.set_ylabel('$y$', fontsize=font_label)
        ax.set_title(f"{plt_label} at $t$ = {sim_data['t']}, \n $\\beta$ = {beta}, $S_{{xx}}$ = {S0xx}, $S_{{yy}}$ = {S0yy}, $t_m$ = {tmyosin}, $t_v$ = {tviscous} \n", fontsize=font_title)
        ax.set_aspect('equal')
        cbar = fig.colorbar(p, ax=ax)
        cbar.ax.tick_params(labelsize=font_label)

    elif plot_option == 1:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.quiver(
            y[::vskip], x[::vskip],
            vx[1:Ngrid-1, 1:Ngrid-1][::vskip, ::vskip],
            vy[1:Ngrid-1, 1:Ngrid-1][::vskip, ::vskip],
            color='navy', width=0.005
        )
        ax.set_title(f"$\\vec{{v}}$ at $t$ = {t:.2f},  \n $\\beta$ = {beta}, $S_{{xx}}$ = {S0xx}, $S_{{yy}}$ = {S0yy}, $t_m$ = {tmyosin}, $t_v$ = {tviscous} \n", fontsize=font_title)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=font_label)
        ax.set_xlabel('$x$', fontsize=font_label)
        ax.set_ylabel('$y$', fontsize=font_label)

    elif plot_option == 2:
        fig, axs = plt.subplots(2, 3, figsize=(16, 12))
        fig.suptitle(
            f"$\\beta$ = {beta}, $S_{{xx}}$ = {S0xx}, $S_{{yy}}$ = {S0yy}, $t_m$ = {tmyosin}, $t_v$ = {tviscous}, $t$ = {t:.2f}",
            fontsize=font_title, y=0.85
        )

        fields = [
            ("$M_{xx}$", Mxx, 'viridis', kwargs.get('Mxx_lower', 0), kwargs.get('Mxx_upper', 1)),
            ("$M_{yy}$", Myy, 'viridis', kwargs.get('Mxx_lower', 0), kwargs.get('Mxx_upper', 1)),
            ("$M_{xy}$", Mxy, 'seismic', kwargs.get('Mxy_lower', -0.2), kwargs.get('Mxy_upper', 0.2)),
            ("$\\pi_{xx}$", Pxx, 'seismic', kwargs.get('pi_lower', -0.05), kwargs.get('pi_upper', 0.05)),
            ("$\\pi_{yy}$", Pyy, 'seismic', kwargs.get('pi_lower', -0.05), kwargs.get('pi_upper', 0.05)),
            ("$\\pi_{xy}$", Pxy, 'seismic', kwargs.get('pi_lower', -0.05), kwargs.get('pi_upper', 0.05)),
        ]

        for i, (ax, (title, field, cmap, vmin, vmax)) in enumerate(zip(axs.flat, fields)):
            row, col = divmod(i, 3)
            im = ax.pcolormesh(y, x, field, cmap=cmap, rasterized=True)
            im.set_clim(vmin, vmax)
            ax.set_title(title, fontsize=16)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=font_label)
            # Set x and y axis labels only where appropriate
            if row == 1:  # bottom row
                ax.set_xlabel('$x$', fontsize=font_label)
            if col == 0:  # first column
                ax.set_ylabel('$y$', fontsize=font_label)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2.5%", pad=0.05)
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=font_label)

        fig.subplots_adjust(wspace=0.5,hspace=-0.3)

    else:
        raise ValueError(f"Invalid plot_option {plot_option}. Must be 0, 1, or 2.")

    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave space for suptitle
    plt.savefig(figname, dpi=500, bbox_inches='tight')
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

def plot_label(field_name):
    if field_name=='Mxx':
        plot_label = '$M_{xx}$'
    elif field_name=='Myy':
        plot_label = '$M_{yy}$'
    elif field_name=='Mxy':
        plot_label = '$M_{xy}$'
    elif field_name=='Pxx':
        plot_label = '$\\pi_{xx}$'
    elif field_name=='Pyy':
        plot_label = '$\\pi_{yy}$'
    elif field_name=='Pxy':
        plot_label = '$\\pi_{xy}$'
    elif field_name=='gammadotxx':
        plot_label = '$\\dot{\\gamma}_{xx}$'
    elif field_name=='gammadotyy':
        plot_label = '$\\dot{\\gamma}_{yy}$'
    elif field_name=='vx':
        plot_label = '$v_x$'
    elif field_name=='vy':
        plot_label = '$v_y$'
    else:
        raise ValueError(f"Field '{field_name}' not in sim_data keys.")
    
    return plot_label

