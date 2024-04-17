import gym

mountaincar = gym.make('MountainCarContinuous-v0', render_mode="human") 

observation = mountaincar.reset()
observation
print(observation)
mountaincar.render()

# for step in range(5):
#     t = mountaincar.step(mountaincar.action_space.sample())
#     mountaincar.render()
#     print(t)    