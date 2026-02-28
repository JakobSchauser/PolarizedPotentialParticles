
import torch
torch.cuda.empty_cache()
torch.cuda.synchronize()


# In[3]:


import torch
import numpy as np
import panel as pn
# from IPython.display import display, clear_output
# pn.extension()
import matplotlib.pyplot as plt


# In[4]:


from polarizedpotentialparticles.configs import Config, ParticleConfig, SimulationConfig, LossConfig
from polarizedpotentialparticles.trainer import Trainer
from polarizedpotentialparticles.displays import Displayer


# In[ ]:





# In[5]:


p_cfg = ParticleConfig()
t_cfg = SimulationConfig()
l_config = LossConfig()

l_config.target = "circle"

cfg = Config(particle_config=p_cfg, simulation_config=t_cfg, loss_config=l_config)

trainer = Trainer(cfg)
displayer = Displayer(trainer)


# In[6]:


# check if it is running on gpu
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {cfg.device}")


# # Todo
# Symmetrize
# 
# ~~N_nbs~~
# 
# Normalize different inputs

# In[7]:


# pretrain

steps = 10

for ep in range(100):
    print(ep,"/", "100", end="\r")
    trainer.train(steps, accumulate_loss=True, step_loss=True)


# # clear_output(wait=True)
# rollout = trainer.rollout(steps = 3*steps)

# to_display = []
# to_display.append(displayer.display_loss())
# # to_display.append(displayer.display_rollout_as_static(rollout))
# to_display.append(displayer.display_rollout_image(rollout))
# to_display.append(displayer.display_rollout_image_gauss(rollout))
# to_display.append(displayer.display_rollout_image_gauss_difference(rollout))
# # to_display.append(displayer.display_rollout_3d(rollout))

# display(displayer.display_multiple(to_display))


# In[ ]:


steps = 225

every = int(2000/steps)


d = 25

for ep in range(20000):
    print(ep,"/", "20000", end="\r")
    rnd = np.random.randint(-d, d) if d > 0 else 0
    trainer.train(steps + rnd, accumulate_loss=True, step_loss=True)

    # if (ep+1) % 50 == 0:
    #     clear_output(wait=True)
    #     rollout = trainer.rollout(steps = steps + d)

    #     to_display = []
    #     to_display.append(displayer.display_loss())
    #     # to_display.append(displayer.display_rollout_as_static(rollout))
    #     to_display.append(displayer.display_rollout_image(rollout))
    #     to_display.append(displayer.display_rollout_image_gauss(rollout))
    #     to_display.append(displayer.display_rollout_image_gauss_difference(rollout))
    #     # to_display.append(displayer.display_rollout_3d(rollout))

    #     display(displayer.display_multiple(to_display))

    # save the model
    if (ep+1) % 500 == 0:
        trainer.save_model("model2.pt")


