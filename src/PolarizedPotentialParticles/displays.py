
from logging import config
from matplotlib.animation import FuncAnimation
import panel as pn
import matplotlib.pyplot as plt
import torch
from polarizedpotentialparticles.trainer import Trainer
from polarizedpotentialparticles.losses import gaussian_splat_from_image, gaussian_splat, gaussian_splat_data


class Displayer:
    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def _state_for_display(self, state):
        # state may be [N, C] or [B, N, C]; display first batch if present
        if hasattr(state, "shape") and len(state.shape) == 3:
            return state[0]
        return state

    def plot_loss(self):
        losses = [h["loss"] for h in self.trainer.history]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(losses, '.', label="Loss")
        ax.set_yscale("log")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.set_title("Training Loss Over Time")

        return fig

    def display_loss(self):
        fig = self.plot_loss()
        plt.close(fig)
        return pn.panel(fig, width=600, height=400)

    def display_rollout(self, rollout : list):
        # Create an animation of the particle positions over time
        fig, ax = plt.subplots(figsize=(6, 6))
        scat = ax.scatter([], [], s=100)  # Initialize an empty scatter plot

        x_lim = (-3, 3)
        y_lim = (-3, 3)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        def update(frame):
            pos = self._state_for_display(rollout[frame])[:, :2]  # Get the positions for the current frame


            scat.set_offsets(pos)  # Update the scatter plot with new positions
            
            return scat, 
    
        anim = FuncAnimation(fig, update, frames=len(rollout), interval=50, blit=True)

        # show as gif in notebook
        anim.save("animation.gif", writer="pillow")
        plt.close(fig)


        return pn.panel("animation.gif", width=600, height=600)



    def display_rollout_image(self, rollout : list):
        # Create an animation of the particle positions over time
        fig, ax = plt.subplots(figsize=(6, 6))
        scat = ax.scatter([], [], s=100)  # Initialize an empty scatter plot

        x_lim = (-1.1, 1.1)
        y_lim = (-1.1, 1.1)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        
        emoji_path = "C:/Users/jakob/Documents/work/PolarizedPotentialParticles/src/polarizedpotentialparticles/morphologies/" + self.trainer.config.loss_config.target + ".png"

        img_grid = gaussian_splat_from_image(emoji_path)

        # display the image in real life coordinates
        ax.imshow(img_grid, extent=(-1., 1., -1., 1.), origin='lower', cmap='gray', alpha=0.5)

        def update(frame):
            pos = self._state_for_display(rollout[frame])[:, :2]  # Get the positions for the current frame

            scat.set_offsets(pos)  # Update the scatter plot with new positions
            
            return scat, 
    
        anim = FuncAnimation(fig, update, frames=len(rollout), interval=200, blit=True)

        # show as gif in notebook
        anim.save("animation.gif", writer="pillow")
        plt.close(fig)


        return pn.panel("animation.gif", width=600, height=600)


    def display_rollout_image_gauss(self, rollout : list):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

        pos0 = self._state_for_display(rollout[0])[:, :2]
        pos0 = torch.tensor(pos0)
        img0 = gaussian_splat_data(pos0)
        im = ax.imshow(img0, extent=(-1., 1., -1., 1.), origin='lower', cmap='gray', alpha=1.)

        def update(frame):
            ro = rollout[frame][:,:2]
            ro = self._state_for_display(ro)
            ro = torch.tensor(ro)
            particle_grid = gaussian_splat_data(ro)
            im.set_data(particle_grid)
            return im,

        anim = FuncAnimation(fig, update, frames=len(rollout), interval=200, blit=True)
        anim.save("animation_gauss.gif", writer="pillow")
        plt.close(fig)
        return pn.panel("animation_gauss.gif", width=600, height=600)
    
    def display_rollout_image_gauss_difference(self, rollout : list):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

        emoji_path = "C:/Users/jakob/Documents/work/PolarizedPotentialParticles/src/polarizedpotentialparticles/morphologies/" + self.trainer.config.loss_config.target + ".png"
        img_grid = gaussian_splat_from_image(emoji_path)
        im = ax.imshow(img_grid, extent=(-1., 1., -1., 1.), origin='lower', cmap='gray', alpha=1.)

        ax.set_title("Diff: ")

        def update(frame):
            ro = rollout[frame][:,:2]
            ro = self._state_for_display(ro)
            ro = torch.tensor(ro)
            particle_grid = gaussian_splat_data(ro)

            difference_grid = (img_grid - particle_grid)**2
            im.set_data(difference_grid)

            ax.set_title(f"Diff: {difference_grid.sum().item():.4f}")

            return im,

        anim = FuncAnimation(fig, update, frames=len(rollout), interval=200, blit=True)
        anim.save("animation_gauss_difference.gif", writer="pillow")
        plt.close(fig)
        return pn.panel("animation_gauss_difference.gif", width=600, height=600)


    def display_rollout_as_static(self, rollout : list):
        # Plot every ten frames of the particles
        fig, ax = plt.subplots(figsize=(6, 6))
        num_frames = len(range(0, len(rollout), 10))

        rang = list(range(0, len(rollout), 10))
        for i, frame_idx in enumerate(rang):
            frame = self._state_for_display(rollout[frame_idx])
            pos = frame[:, :2]  # Get the positions for the current frame

            polarity = frame[:, self.trainer.config.N_spatial_dim:self.trainer.config.N_spatial_dim + 2 * self.trainer.config.N_polarizations]  # Get the polarizations for the current frame
            
            color = plt.cm.Blues(0.3 + 0.7 * i / max(1, num_frames - 1))  # Blue gradient from light to dark
            ax.scatter(pos[:, 0], pos[:, 1], s=100, alpha=0.5, c=[color])  # Plot the positions

            # Plot the polarization vectors as arrows if first or last frame
            if not (i == 0 or i == len(rang) - 1):
                continue
            for j in range(pos.shape[0]):
                ax.arrow(pos[j, 0], pos[j, 1], polarity[j, 0]*0.05, polarity[j, 1]*0.05, head_width=0.02, head_length=0.02, fc='r', ec='r')


  
        # close the plot and show as static image in notebook
        plt.close(fig)



        return pn.panel(fig, width=600, height=600)
    
    def display_rollout_3d(self, rollout : list):

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': '3d'})

        def update(frame_idx):
            ax.clear()
            frame = self._state_for_display(rollout[frame_idx])
            pos = frame[:, :3]  # Get the positions for the current frame
            time = frame_idx / len(rollout)  # Normalize time to [0, 1]

            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=100, alpha=0.5, c="Blue")  # Plot the positions in 3D

            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_zlim(-1.1, 1.1)
    
        anim = FuncAnimation(fig, update, frames=len(rollout), interval=200, blit=False)
        anim.save("animation_3d.gif", writer="pillow")
        plt.close(fig)
        return pn.panel("animation_3d.gif", width=600, height=600)




    def display_multiple(self, panels: list):
        return pn.Row(*panels, width=600, height=600)