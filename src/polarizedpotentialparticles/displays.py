
from logging import config
from matplotlib.animation import FuncAnimation
import panel as pn
import matplotlib.pyplot as plt
import torch
import numpy as np
from polarizedpotentialparticles.trainer import Trainer
from polarizedpotentialparticles.losses import gaussian_splat_from_image, gaussian_splat, gaussian_splat_data
from pathlib import Path

class Displayer:
    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self.px_size = 500

    def _state_for_display(self, state):
        # state may be [N, C] or [B, N, C]; display first batch if present
        if hasattr(state, "shape") and len(state.shape) == 3:
            return state[-1]
        return state

    def loss(self):
        losses = [h["loss"] for h in self.trainer.history]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(losses, '.', label="Loss")
        ax.set_yscale("log")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.set_title("Training Loss Over Time")
        plt.close(fig)
        return pn.panel(fig, width=self.px_size, height=int(self.px_size * 0.66))

    def loss_types(self, normalize: bool):
        img_losses = [h.get("img_loss", 0) for h in self.trainer.history]
        step_losses = [h.get("step_loss", 0) for h in self.trainer.history]

        if normalize:
            max_img_loss = max(img_losses)
            max_step_loss = max(step_losses) 
            img_losses = [l / max_img_loss for l in img_losses]
            step_losses = [l / max_step_loss for l in step_losses]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(img_losses, '.', label="Image Loss")
        ax.plot(step_losses, '.', label="Step Loss")
        ax.set_yscale("log")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.set_title("Training Loss Components Over Time")
        plt.close(fig)
        return pn.panel(fig, width=self.px_size, height=int(self.px_size * 0.66))
    
    def accuracy(self):
        accuracies = [h.get("accuracy", 0) for h in self.trainer.history]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(accuracies, '.', label="Accuracy")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.set_title("Training Accuracy Over Time")
        plt.close(fig)
        return pn.panel(fig, width=self.px_size, height=int(self.px_size * 0.66))

    def rollout(self, rollout : list):
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


        return pn.panel("animation.gif", width=self.px_size, height=self.px_size)



    def rollout_image(self, rollout : list):
        # Create an animation of the particle positions over time
        fig, ax = plt.subplots(figsize=(6, 6))
        scat = ax.scatter([], [], s=100)  # Initialize an empty scatter plot

        x_lim = (-1.1, 1.1)
        y_lim = (-1.1, 1.1)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        
        emoji_path = (Path(__file__).resolve().parent / "morphologies" / f"{self.trainer.config.loss_config.target}.png")

        img_grid = gaussian_splat_from_image(emoji_path)

        # display the image in real life coordinates
        ax.imshow(img_grid, extent=(-1., 1., -1., 1.), origin='lower', cmap='gray', alpha=0.5)

        def update(frame):
            pos = self._state_for_display(rollout[frame])[:, :2]  # Get the positions for the current frame

            scat.set_offsets(pos)  # Update the scatter plot with new positions
            ax.set_title(f"Frame {frame+1}/{len(rollout)}")
            return scat, 
    
        anim = FuncAnimation(fig, update, frames=len(rollout), interval=200, blit=True)

        # show as gif in notebook
        anim.save("animation.gif", writer="pillow")
        plt.close(fig)


        return pn.panel("animation.gif", width=self.px_size, height=self.px_size)
    
    def rollout_image_hidden(self, rollout : list):
        # Create an animation of the particle positions over time
        fig, ax = plt.subplots(figsize=(6, 6))
        scat = ax.scatter([], [], s=100)  # Initialize an empty scatter plot

        x_lim = (-1.1, 1.1)
        y_lim = (-1.1, 1.1)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        
        emoji_path = (Path(__file__).resolve().parent / "morphologies" / f"{self.trainer.config.loss_config.target}.png")

        img_grid = gaussian_splat_from_image(emoji_path)

        # display the image in real life coordinates
        ax.imshow(img_grid, extent=(-1., 1., -1., 1.), origin='lower', cmap='gray', alpha=0.5)

        def update(frame):
            pos = self._state_for_display(rollout[frame])[:, :2]  # Get the positions for the current frame

            colors = self._state_for_display(rollout[frame])[:, 2]  # Get the hidden state for the current frame

            scat.set_offsets(pos)  # Update the scatter plot with new positions
            scat.set_array(colors)  # Update the scatter plot with new colors based on hidden state
            # use a colormap to convert hidden state values to colors
            scat.set_cmap("viridis")
            # set color limits to be the same across frames for consistency
            _max, _min = np.max(colors), np.min(colors)
            scat.set_clim(_min, _max)

            ax.set_title(f"Frame {frame+1}/{len(rollout)}")
            return scat, 
    
        anim = FuncAnimation(fig, update, frames=len(rollout), interval=200, blit=True)

        # show as gif in notebook
        anim.save("animation_hidden.gif", writer="pillow")
        plt.close(fig)


        return pn.panel("animation_hidden.gif", width=self.px_size, height=self.px_size)



    def rollout_image_gauss(self, rollout : list):
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
            ax.set_title(f"Frame {frame+1}/{len(rollout)}")
            return im,

        anim = FuncAnimation(fig, update, frames=len(rollout), interval=200, blit=True)
        anim.save("animation_gauss.gif", writer="pillow")
        plt.close(fig)
        return pn.panel("animation_gauss.gif", width=self.px_size, height=self.px_size)
    
    def rollout_image_gauss_difference(self, rollout : list):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

        emoji_path = (Path(__file__).resolve().parent / "morphologies" / f"{self.trainer.config.loss_config.target}.png")

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

            ax.set_title(f"Diff: {difference_grid.mean().item():.4f} | frame {frame+1}/{len(rollout)}")

            return im,

        anim = FuncAnimation(fig, update, frames=len(rollout), interval=200, blit=True)
        anim.save("animation_gauss_difference.gif", writer="pillow")
        plt.close(fig)
        return pn.panel("animation_gauss_difference.gif", width=self.px_size, height=self.px_size)


    def rollout_as_static(self, rollout : list):
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



        return pn.panel(fig, width=self.px_size, height=self.px_size)
    
    def rollout_3d(self, rollout : list):

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


    def final_state(self, rollout : list, losses : list):
        first_state = self._state_for_display(rollout[0])
        final_state = self._state_for_display(rollout[-1])
        first_pos = first_state[:, :2]
        pos = final_state[:, :2]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(first_pos[:, 0], first_pos[:, 1], s=100, alpha=0.5, c="Red", label="Initial State")  # Plot the positions for the first frame
        ax.scatter(pos[:, 0], pos[:, 1], s=100, alpha=0.5, c="Blue", label="Final State")  # Plot the positions for the final frame
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        # remove axes
        ax.axis('off')
        ax.set_title("Final State | Loss: {:.4f}".format(losses[-1]))
        ax.legend()


        plt.close(fig)
        return pn.panel(fig, width=self.px_size, height=self.px_size)
    

    def dashboard(self, rollout, losses):
        to_display = []

        to_display.append(self.loss())
        to_display.append(self.loss_types(normalize = False))
        to_display.append(self.final_state(rollout, losses))
        # to_display.append(self.rollout_as_static(rollout))
        has_hidden_dim = self.trainer.config.particle_config.hidden_dim > 0
        if has_hidden_dim:
            to_display.append(self.rollout_image_hidden(rollout))
        else:
            to_display.append(self.rollout_image(rollout))
        to_display.append(self.rollout_image_gauss(rollout))
        to_display.append(self.rollout_image_gauss_difference(rollout))

        panel = self.display_multiple(to_display)

        return panel
    def display_multiple(self, panels: list):
        # Normalize all inputs to Panel objects so mixed pane types work.
        panel_objects = [pn.panel(p) for p in panels]

        if len(panel_objects) == 0:
            return pn.Spacer(width=1, height=1)

        # For small sets keep a single compact row.
        if len(panel_objects) <= 3:
            return pn.FlexBox(
                *panel_objects,
                flex_wrap="nowrap",
                justify_content="flex-start",
                align_items="flex-start",
                gap="0px",
            )

        # For larger sets wrap tightly instead of stretching columns,
        # which avoids large empty gutters between panes.
        return pn.FlexBox(
            *panel_objects,
            flex_wrap="wrap",
            justify_content="flex-start",
            align_items="flex-start",
            align_content="flex-start",
            gap="0px",
        )
