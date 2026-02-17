
from matplotlib.animation import FuncAnimation
import panel as pn
import matplotlib.pyplot as plt
from polarizedpotentialparticles.trainer import Trainer


class Displayer:
    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def plot_loss(self):
        losses = [h["loss"] for h in self.trainer.history]
        plt.figure(figsize=(10, 5))
        plt.plot(losses, '.', label="Loss")
        plt.yscale("log")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()



    def display_loss(self):
        return pn.panel(self.plot_loss(), width=600, height=400)

    def display_rollout(self, steps = 50):
        states = self.trainer.rollout(steps=steps)  # Get the rollout states

        # Create an animation of the particle positions over time
        fig, ax = plt.subplots(figsize=(6, 6))
        scat = ax.scatter([], [], s=100)  # Initialize an empty scatter plot
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        def update(frame):
            pos = states[frame][:, :2]  # Get the positions for the current frame
            scat.set_offsets(pos)  # Update the scatter plot with new positions
            return scat,

        anim = FuncAnimation(fig, update, frames=len(states), interval=200, blit=True)
        plt.close(fig)  # Prevent the static plot from displaying
        return pn.panel(anim.to_jshtml(), width=600, height=600)
    
    def display_rollout_as_static(self, steps = 50):
        states = self.trainer.rollout(steps=steps)  # Get the rollout states



        # Plot every ten frames of the particles
        plt.figure(figsize=(6, 6))
        for i in range(0, len(states), 10):
            pos = states[i][:, :2]  # Get the positions for the current frame

            polarity = states[i][:, self.trainer.config.N_spatial_dim:self.trainer.config.N_spatial_dim + 2 * self.trainer.config.N_polarizations]  # Get the polarizations for the current frame
            
            plt.scatter(pos[:, 0], pos[:, 1], s=100, alpha=0.5)  # Plot the positions

            # Plot the polarization vectors as arrows
            for j in range(pos.shape[0]):
                plt.arrow(pos[j, 0], pos[j, 1], polarity[j, 0]*0.1, polarity[j, 1]*0.1, head_width=0.02, head_length=0.02, fc='r', ec='r')


        return pn.panel(plt.gcf(), width=600, height=600)