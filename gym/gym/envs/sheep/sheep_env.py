import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.sheep import rendering
from gym.envs.sheep import SheepGroup
from gym.envs.sheep import DogGroup
from pyglet.window import key
import pdb
import numpy as np
import time

'''
LEGEND
grey dot  = sheep
red dot   = dog
green dot = target
blue dot  = sheep center
'''

class SheepEnv(gym.Env):
  metadata = {'render.modes': ['human','rgb_array']}
  #Env Set-up variable
  SCREEN_WIDTH = 300
  SCREEN_HEIGHT = 300
  TARGET_X = 60
  TARGET_Y = 60
  #finishing radius can be changed later
  FINISH_RADIUS = 50
  SHEEP_RADIUS = 40
  Default_SheepCount = 30
  Default_DogCount = 1
  DISCRETE_Action_Count = 4 #Number of action when discrete number of action spaces is used
  FEATURE_Count = 9
  #FEATURE_Count = 1
  def __init__(self,obs_type='ram'):
    assert obs_type in ('ram', 'image')
    self._obs_type = obs_type
    np.random.seed(int(time.time()))
    print('initilize with seed '+ str(int(time.time())))
    self._seed()
    self.action_space = spaces.Discrete(self.DISCRETE_Action_Count)
    self.viewer = None

    self.sheepGroup = SheepGroup.SheepGroup( self.Default_SheepCount, self.Default_DogCount,self.SCREEN_WIDTH,self.SCREEN_HEIGHT);
    #Now, the list of dogs technically belongs to sheepGroup
    self.dogGroup = self.sheepGroup
    #self._reset()

    # Need to figure out the high for our case
    high = np.array([np.inf] * 6)
    if self._obs_type == 'ram':
        self.observation_space = spaces.Box(-high, high)
    elif self._obs_type == 'image':
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.SCREEN_HEIGHT,self.SCREEN_WIDTH,3))
    else:
        raise error.Error('Unrecognized obervation type: ()'.format(self._obs_type))

    return

  def if_done(self):
      #when it done we need to make sure the average radius of the herd is smaller than a fixed radius
      if(self.get_dist_sqr_to_target() <= self.FINISH_RADIUS*self.FINISH_RADIUS) and (self.get_cluster_dist_from_centroid()<= self.SHEEP_RADIUS):
          return True
      else:
          return False

  def get_reward(self):
      #start with sparse award for experiements
      reward = 0
      if(self.get_dist_sqr_to_target() <= 6.25*self.FINISH_RADIUS*self.FINISH_RADIUS) and (self.get_cluster_dist_from_centroid()<= 4*self.SHEEP_RADIUS):
          reward = reward + self.FINISH_RADIUS*2.5 - np.sqrt(self.get_dist_sqr_to_target())
      if(self.if_done()):
          reward = reward + 1000
          return reward
      else:
          reward = reward - 10
          return reward
  def _step(self, action=None):
    #TO-Do: Implementi Action for Shepherd
    assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
    self.sheepGroup.executeDogAction(action)

    self.sheepGroup.cleanPreviousState()
    self.sheepGroup.updateLocations()

    # #Handle Different Action Space Choice
    # #observation consists on distance of centroid to target and average distance of sheeps to centroid
    # #assume one dog situation
    # allDogLocations=self.sheepGroup.get_DogsLocation()
    # dog_to_sheep_centroid = self.get_firstDogToSheepCentroidDist()
    # #assume the dog can see the centroid location of the sheep
    # SheepCentroid = self.sheepGroup.get_sheep_centroid()
    ob = self._get_obs()
    return ob,self.get_reward(),self.if_done(),{}

  def _get_obs(self):
      if self._obs_type == 'ram':
          # observation consists on distance of centroid to target and average distance of sheeps to centroid
          # assume one dog situation
          allDogLocations = self.sheepGroup.get_DogsLocation()
          dog_to_sheep_centroid = self.get_firstDogToSheepCentroidDist()
          # assume the dog can see the centroid location of the sheep
          SheepCentroid = self.sheepGroup.get_sheep_centroid()
          #img = self.viewer.render(return_rgb_array=True)
          #print('the size of the image '+str(np.shape(img)))
          return [allDogLocations[0][0],  # simple discretization
                  allDogLocations[0][1],
                  SheepCentroid[0],
                  SheepCentroid[1],
                  dog_to_sheep_centroid,
                  self.get_dist_sqr_to_target(),
                  self.get_cluster_dist_from_centroid(),
                  self.TARGET_X, self.TARGET_Y]
      elif self._obs_type == 'image':
          img = self.viewer.render(return_rgb_array=True)
          return img

  def _seed(self, seed=None):
      self.np_random, seed = seeding.np_random(seed)
      seed2 = seeding.hash_seed(seed +1 )% 2**31
      return [seed2]

  def _reset(self):
      # Need to be changed later
      if (self.sheepGroup.SheepList is not None):
          for sheep in self.sheepGroup.SheepList:
              sheep.X = np.random.randint(0, self.SCREEN_WIDTH)
              sheep.Y = np.random.randint(0, self.SCREEN_HEIGHT)
              sheep.velocityX = np.random.random_integers(0, 1000) / 500
              sheep.velocityY = np.random.random_integers(0, 1000) / 500
      if (self.sheepGroup.DogList is not None):
          for dog in self.sheepGroup.DogList:
              dog.X = np.random.randint(0, self.SCREEN_WIDTH)
              #dog.Y = sheep.Y = np.random.randint(0, self.SCREEN_HEIGHT)
              dog.Y = np.random.randint(0, self.SCREEN_HEIGHT)
              ################################################################################3
              dog.velocityX = np.random.random_integers(0, 1000) / 400
              dog.velocityY = np.random.random_integers(0, 1000) / 400
      allDogLocations = self.sheepGroup.get_DogsLocation()
      dog_to_sheep_centroid = self.get_firstDogToSheepCentroidDist()
      SheepCentroid = self.sheepGroup.get_sheep_centroid()
      if self._obs_type == 'ram':
          return [allDogLocations[0][0], allDogLocations[0][1], SheepCentroid[0], SheepCentroid[1], dog_to_sheep_centroid, self.get_dist_sqr_to_target(),
             self.get_cluster_dist_from_centroid(), self.TARGET_X, self.TARGET_Y]
      elif self._obs_type == 'image':
          return self._render(mode='rgb_array')
      ###############################################################


  def get_firstDogToSheepCentroidDist(self):
      centroid = self.sheepGroup.get_sheep_centroid()
      return np.sqrt((self.sheepGroup.DogList[0].X-self.sheepGroup.centroid[0])**2+(self.sheepGroup.DogList[0].Y-self.sheepGroup.centroid[1])**2)

  # sqrt(sum of squares of distances) / (number of sheep)
  def get_cluster_dist_from_centroid(self):
    centroid = self.sheepGroup.get_sheep_centroid()
    sum_of_dist_sqr = 0
    for sheep in self.sheepGroup.SheepList:
      sum_of_dist_sqr += np.sqrt((sheep.X - centroid[0])**2 + (sheep.Y - centroid[1])**2)
    return sum_of_dist_sqr/len(self.sheepGroup.SheepList)

  def get_dist_sqr_to_target(self):
    centroid = self.sheepGroup.get_sheep_centroid()
    return (self.TARGET_X - centroid[0])**2 + (self.TARGET_Y - centroid[1])**2

  def key_press(self, symbol, modifier):
      key_moveSize = 30
      if symbol==key.LEFT: 
        self.dogGroup.DogList[0].X -= key_moveSize
        self.dogGroup.DogList[0].velocityX = -key_moveSize
      if symbol==key.RIGHT: 
        self.dogGroup.DogList[0].X += key_moveSize
        self.dogGroup.DogList[0].velocityX = key_moveSize
      if symbol==key.UP: 
        self.dogGroup.DogList[0].Y += key_moveSize
        self.dogGroup.DogList[0].velocityY = key_moveSize
      if symbol==key.DOWN: 
        self.dogGroup.DogList[0].Y -= key_moveSize
        self.dogGroup.DogList[0].velocityY = -key_moveSize
      if symbol==key.A: 
        self.dogGroup.DogList[1].X -= key_moveSize
        self.dogGroup.DogList[0].velocityX = -key_moveSize
      if symbol==key.D: 
        self.dogGroup.DogList[1].X += key_moveSize
        self.dogGroup.DogList[0].velocityX = key_moveSize
      if symbol==key.W: 
        self.dogGroup.DogList[1].Y += key_moveSize
        self.dogGroup.DogList[0].velocityY = key_moveSize
      if symbol==key.S: 
        self.dogGroup.DogList[1].Y -= key_moveSize
        self.dogGroup.DogList[0].velocityY = -key_moveSize

  def _render(self, mode='human', close=False):

      #if mode == 'rgb_array':
      #    return np.array(...)  # return RGB frame suitable for video
      #elif mode is 'human':
      #    ...  # pop up a window and render
    if close:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return

    curSheepList = self.sheepGroup.SheepList
    curDogList = self.dogGroup.DogList
    if self.viewer is None:
        self.viewer = rendering.Viewer(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        self.viewer.window.on_key_press = self.key_press

        target_translation = rendering.Transform()
        target_circ = rendering.make_circle(10)
        target_circ.set_color(0, 0.7, 0)
        target_circ.add_attr(target_translation)
        self.viewer.add_geom(target_circ)
        target_translation.set_translation(self.TARGET_X, self.TARGET_Y)

        self.centroid_translation = rendering.Transform()
        centroid_circ = rendering.make_circle(4)
        centroid_circ.set_color(0, 0, 0.9)
        centroid_circ.add_attr(self.centroid_translation)
        self.viewer.add_geom(centroid_circ)

        self.sheepTranlations = []
        for sheep in curSheepList:
            translation = rendering.Transform()
            circ = rendering.make_circle(7)
            circ.set_color(0.5, 0.5, 0.5)
            circ.add_attr(translation)
            self.viewer.add_geom(circ)
            self.sheepTranlations.append(translation)
        self.dogTranlations = []
        for dog in curDogList:
            translation = rendering.Transform()
            circ = rendering.make_circle(9)
            circ.set_color(1, 0, 0)
            circ.add_attr(translation)
            self.viewer.add_geom(circ)
            self.dogTranlations.append(translation)

    for ind, translation in enumerate(self.sheepTranlations):
        translation.set_translation(curSheepList[ind].X,curSheepList[ind].Y)
    for ind, translation in enumerate(self.dogTranlations):
        translation.set_translation(curDogList[ind].X, curDogList[ind].Y)
    
    centroid_pos = self.sheepGroup.get_sheep_centroid()
    self.centroid_translation.set_translation(centroid_pos[0], centroid_pos[1]) 

    #print(self.viewer.render(return_rgb_array = mode=='rgb_array'))
    return self.viewer.render(return_rgb_array = mode=='rgb_array')
    