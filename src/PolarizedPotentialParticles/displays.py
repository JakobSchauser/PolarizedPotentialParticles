
from matplotlib.animation import FuncAnimation
import panel as pn
import matplotlib.pyplot as plt
from polarizedpotentialparticles.trainer import Trainer


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
        plt.figure(figsize=(10, 5))
        plt.plot(losses, '.', label="Loss")
        plt.yscale("log")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Loss Over Time")
        return plt.gcf()


    def display_loss(self):
        return pn.panel(self.plot_loss(), width=600, height=400)

    def display_rollout(self, rollout : list):
        # Create an animation of the particle positions over time
        fig, ax = plt.subplots(figsize=(6, 6))
        scat = ax.scatter([], [], s=100)  # Initialize an empty scatter plot

        extends = [-3, 3, -3, 3]

        ax.set_xlim(extends[0], extends[1])
        ax.set_ylim(extends[2], extends[3])

        def update(frame):
            pos = self._state_for_display(rollout[frame])[:, :2]  # Get the positions for the current frame
            scat.set_offsets(pos)  # Update the scatter plot with new positions
            return scat,

        anim = FuncAnimation(fig, update, frames=len(rollout), interval=50, blit=True)


        # show as gif in notebook
        anim.save("animation.gif", writer="pillow")
        plt.close(fig)


        return pn.panel("animation.gif", width=600, height=600)
    

    def display_rollout_as_static(self, rollout : list):
        # Plot every ten frames of the particles
        plt.figure(figsize=(6, 6))
        num_frames = len(range(0, len(rollout), 10))
        for i, frame_idx in enumerate(range(0, len(rollout), 10)):
            frame = self._state_for_display(rollout[frame_idx])
            pos = frame[:, :2]  # Get the positions for the current frame

            polarity = frame[:, self.trainer.config.N_spatial_dim:self.trainer.config.N_spatial_dim + 2 * self.trainer.config.N_polarizations]  # Get the polarizations for the current frame
            
            color = plt.cm.Blues(0.3 + 0.7 * i / max(1, num_frames - 1))  # Blue gradient from light to dark
            plt.scatter(pos[:, 0], pos[:, 1], s=100, alpha=0.5, c=[color])  # Plot the positions

            # Plot the polarization vectors as arrows if first or last frame
            if not (i == 0 or i == num_frames - 1):
                continue
            for j in range(pos.shape[0]):
                plt.arrow(pos[j, 0], pos[j, 1], polarity[j, 0]*0.05, polarity[j, 1]*0.05, head_width=0.02, head_length=0.02, fc='r', ec='r')


        # close the plot and show as static image in notebook
        plt.close()

        return pn.panel(plt.gcf(), width=600, height=600)
