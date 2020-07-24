import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class DroneAutomaticDrivingEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.81
        self.massdrone = 2.3536
        self.i_x = 0.1676
        self.i_y = 0.1676
        self.i_z = 0.2794
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        
        self.Lidar1_joint_angle=math.pi/6
        self.Lidar2_joint_angle=math.pi/3
        self.Lidar3_joint_angle=-math.pi/6
        self.Lidar4_joint_angle=-math.pi/3

        high_observation = np.array([
            np.finfo(np.float32).max,np.finfo(np.float32).max,
            np.finfo(np.float32).max,np.finfo(np.float32).max,
            np.finfo(np.float32).max,np.finfo(np.float32).max,
            np.finfo(np.float32).max,np.finfo(np.float32).max,
            np.finfo(np.float32).max,np.finfo(np.float32).max,
            np.finfo(np.float32).max,np.finfo(np.float32).max,
            np.finfo(np.float32).max,np.finfo(np.float32).max,
            np.finfo(np.float32).max,np.finfo(np.float32).max,
            np.finfo(np.float32).max,np.finfo(np.float32).max,
            np.finfo(np.float32).max,np.finfo(np.float32).max,
            np.finfo(np.float32).max,np.finfo(np.float32).max
             ])
        
        self.action_space = spaces.Box(np.array([0,-1]),np.array([1,1]),dtype=np.float32)
        self.observation_space = spaces.Discrete(5)
        #[0]: x_dot_input=5 m/s    psi_input= + pi/12 rad
        #[1]: x_dot_input=5 m/s    psi_input= 0 rad
        #[2]: x_dot_input=5 m/s    psi_input= - pi/12 rad
        
        #[3]: x_dot_input=0 m/s    psi_input= + pi/12 rad
        #[4]: x_dot_input=0 m/s    psi_input= 0 rad
        #[5]: x_dot_input=0 m/s    psi_input= - pi/12 rad/

        #[6]: x_dot_input=2.5 m/s    psi_input= + pi/12 rad
        #[7]: x_dot_input=2.5 m/s    psi_input= 0 rad
        #[8]: x_dot_input=2.5 m/s    psi_input= - pi/12 rad


        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        
        self.wall_up=self.wall(-10.0,10.0,10.0,10.0)
        self.wall_down=self.wall(-10.0,-10.0,10.0,-10.0)
        self.wall_right=self.wall(10.0,-10.0,10.0,10.0)
        self.wall_left=self.wall(-10.0,-10.0,-10.0,10.0)
        self.Wall=np.array([self.wall_up,self.wall_down,self.wall_right,self.wall_left])
        self.x_goal_pre=0
        self.y_goal_pre=0
       
    def wall(self,x1,y1,x2,y2):
        return x1,y1,x2,y2
    
    def Lidar(self,x,y,angle):
        
        A=np.array([(0.0,0.0),(0.0,0.0),(0.0,0.0),(0.0,0.0)])
        Cos=math.cos(angle)
        Sin=math.sin(angle)
        for i in range(len(self.Wall)):
            
            Distance=0.0
            R=10.0
            x1=self.Wall[i,0]
            y1=self.Wall[i,1]
            x2=self.Wall[i,2]
            y2=self.Wall[i,3]
            
            if abs(angle-math.atan2(x2-x1,y2-y1)) > 0.0001 :
                    Det=(y1-y2)*R*Cos+(x2-x1)*R*Sin
                    X=(R*Cos*(x2*y1-x1*y2)+(x1-x2)*(R*Cos*y-R*Sin*x))/Det
                    Y=(R*Sin*(x2*y1-x1*y2)+(y1-y2)*(R*Cos*y-R*Sin*x))/Det
                    Distance=((X-x)**2+(Y-y)**2)**0.5
                    Angle=math.acos(((x2-x1)*Cos+(y2-y1)*Sin)/(((x2-x1)**2+(y2-y1)**2)**0.5))      
                    A[i,0]=Distance
                    A[i,1]=Angle
                    A[i,0]=10 if Distance > 10 else A[i,0]
                    A[i,1]=0 if Distance > 10 else A[i,1]
                    A[i,0]=10 if R*Cos*(X-x)+R*Sin*(Y-y) < 0 else A[i,0]
                    A[i,1]=0 if R*Cos*(X-x)+R*Sin*(Y-y) < 0 else A[i,1]
                    
            else:
                    A[i,0]=10
                    A[i,1]=0
            
        self.distance=A[A[:,0].argmin(),0]
        self.angle=A[A[:,0].argmin(),1]
                      
        return self.distance, self.angle

    def step(self, action):
        state = self.state
        
        x, x_dot,y,y_dot,phi,phi_dot,theta,theta_dot,psi,psi_dot,x_goal,y_goal = state[8:20]
        x_dot_input=state[20]
        psi_input=state[21]
        
        Z=0.9 #zeta
        A=5   #frequency
        K_d=2*A*Z
        K_p=A**2
        
        if action==0:
            x_dot_input=x_dot_input+0.05
            psi_input=psi_input
        elif action==1:
            x_dot_input=x_dot_input-0.05
            if x_dot_input < 0:
                x_dot_input=0       
            psi_input=psi_input
        elif action==2:
            x_dot_input=x_dot_input
            psi_input=psi_input+math.pi/72
        elif action==3:
            x_dot_input=x_dot_input
            psi_input=psi_input-math.pi/72
        elif action==4:
            x_dot_input=x_dot_input+0.1
            psi_input=psi_input
        #elif action==4:
        #    x_dot_input=0
        #    psi_input=psi-math.pi/12
        #elif action==5:
        #    x_dot_input=0
        #    psi_input=psi+math.pi/12
        #elif action==6:
        #    x_dot_input=2.5
        #    psi_input=psi+math.pi/12
        #elif action==7:
        #    x_dot_input=2.5
        #    psi_input=psi
        #elif action==8:
        #    x_dot_input=2.5
        #    psi_input=psi-math.pi/12
            
        x1_dot=x_dot*math.cos(psi)+y_dot*math.sin(psi)
        y1_dot=-x_dot*math.sin(psi)+y_dot*math.cos(psi)
        u_x=-K_d*(x1_dot-x_dot_input)
        u_y=-K_d*y1_dot
        u_z=0
        u_x_prime=u_x*math.cos(psi)-u_y*math.sin(psi)
        u_y_prime=u_x*math.sin(psi)+u_y*math.cos(psi)
        phi_input=u_x_prime/self.gravity
        theta_input=-u_y_prime/self.gravity
        
        phi_acc=-K_d*phi_dot-K_p*(phi-phi_input)
        theta_acc=-K_d*theta_dot-K_p*(theta-theta_input)
        psi_acc=-K_d*psi_dot-K_p*(psi-psi_input)
        x_acc=u_x_prime
        y_acc=u_y_prime
        z_acc=u_z
        
        x_pre=x
        y_pre=y
     
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc 
        y = y + self.tau * y_dot
        y_dot = y_dot + self.tau * y_acc
        phi = phi + self.tau * phi_dot
        phi_dot = phi_dot + self.tau * phi_acc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc
        psi = psi + self.tau * psi_dot
        psi_dot = psi_dot + self.tau * psi_acc
  
        Lidar1_distance,Lidar1_angle=self.Lidar(x,y,-self.Lidar1_joint_angle+psi)
        Lidar2_distance,Lidar2_angle=self.Lidar(x,y,-self.Lidar2_joint_angle+psi)
        Lidar3_distance,Lidar3_angle=self.Lidar(x,y,-self.Lidar3_joint_angle+psi)
        Lidar4_distance,Lidar4_angle=self.Lidar(x,y,-self.Lidar4_joint_angle+psi)
             
        DistanceToGoal=((x-x_goal)**2+(y-y_goal)**2)**0.5
        DistanceToGoal_pre=((x_pre-x_goal)**2+(y-y_goal)**2)**0.5
        
        done =  Lidar1_distance < 0.5 or Lidar2_distance < 0.5 or Lidar3_distance < 0.5                 or Lidar4_distance< 0.5 or abs(phi) > math.pi/3 or abs(theta) > math.pi/3 
        false =  Lidar1_distance < 0.5 or Lidar2_distance < 0.5 or Lidar3_distance < 0.5                 or Lidar4_distance< 0.5 or abs(phi) > math.pi/3 or abs(theta) > math.pi/3      
        done = bool(done)
        
        if not done:
            #if DistanceToGoal-DistanceToGoal_pre<0 and abs(x-x_goal)<2.5 and abs(y-y_goal)<2.5 and (x_dot**2+y_dot**2)**0.5>1.5:
            #              reward=1
            #elif DistanceToGoal-DistanceToGoal_pre<0 and (x_dot**2+y_dot**2)**0.5>1.5:
            #              reward=0
            if abs(x-x_goal)<1.5 and abs(y-y_goal)<1.5:
                          goal=self.np_random.uniform(low=-8, high=8, size=(2))
                          reward=10
                          x_goal, y_goal = goal 
                          #x_goal,y_goal=self.np_random.uniform(low=-8, high=8, size=(2))
            else:
                reward= -1
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            #if DistanceToGoal-DistanceToGoal_pre<0 and abs(x-x_goal)<2.5 and abs(y-y_goal)<2.5 and (x_dot**2+y_dot**2)**0.5>1.5:
            #              reward=1
            #elif DistanceToGoal-DistanceToGoal_pre<0 and (x_dot**2+y_dot**2)**0.5>1.5:
            #              reward=0
            if abs(x-x_goal)<1.5 and abs(y-y_goal)<1.5:
                          reward=5
                          done=True
                          #x_goal,y_goal=self.np_random.uniform(low=-8, high=8, size=(2))
            else:
                          reward=-20
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = -10
        
        
        x_delta=x_goal-x
        y_delta=y_goal-y
        position1=(x_delta**2+y_delta**2)**0
        position2=math.atan2(y_delta,x_delta)
        self.state = ( 
        Lidar1_distance,Lidar1_angle,
        Lidar2_distance,Lidar2_angle,
        Lidar3_distance,Lidar3_angle,
        Lidar4_distance,Lidar4_angle,
        x, x_dot,
        y,y_dot,
        phi,phi_dot,
        theta,theta_dot,
        psi,psi_dot,
        x_goal,y_goal,
        x_dot_input,psi_input,
        position1,position2
           )
       
        return np.array(self.state), reward, done, {}
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
                          
    def reset(self):
       
        Lidar1_distance,Lidar1_angle=self.Lidar(0.0,0.0,-math.pi/6.0)
        Lidar2_distance,Lidar2_angle=self.Lidar(0.0,0.0,-math.pi/3.0)
        Lidar3_distance,Lidar3_angle=self.Lidar(0.0,0.0,math.pi/6.0)
        Lidar4_distance,Lidar4_angle=self.Lidar(0.0,0.0,math.pi/3.0)
        goal=self.np_random.uniform(low=-8, high=8, size=(2))
        initial=self.np_random.uniform(low=-5, high=5, size=(2))
        angle=self.np_random.uniform(low=-math.pi, high=math.pi, size=(1))
        psi_input=angle[0]
        x_dot_input=0
        x_delta= goal[0]-initial[0]
        y_delta= goal[1]-initial[1]
        position1=(x_delta**2+y_delta**2)**0.5
        
        position2=math.atan2(y_delta,x_delta)
        self.state = ( 
        Lidar1_distance,Lidar1_angle,
        Lidar2_distance,Lidar2_angle,
        Lidar3_distance,Lidar3_angle,
        Lidar4_distance,Lidar4_angle,
        initial[0],0.0,
        initial[1],0.0,
        0.0,0.0,
        0.0,0.0,
        angle[0],0.0,
        goal[0],goal[1],
        x_dot_input,psi_input,
        position1,position2
           )
        self.x_goal_pre=0
        self.y_goal_pre=0
        
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600
        world_width = 30
        scale = screen_width/world_width
        drone_width=30
        drone_height=30
        x = self.state
        x_goal=x[18]
        y_goal=x[19]
        Lidar1_length=x[0]
        Lidar2_length=x[2]
        Lidar3_length=x[4]
        Lidar4_length=x[6]
       
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -drone_width/2, drone_width/2, drone_height/2, -drone_height/2
            
            drone = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.dronetrans = rendering.Transform()
            drone.add_attr(self.dronetrans)
            self.viewer.add_geom(drone)
            drone.set_color(.8,.6,.4)
            
            self.goaltrans =rendering.Transform()
            #translation=(screen_width/2,screen_width/2)
            self.goal = rendering.make_circle(drone_width/2)
            self.goal.add_attr(self.goaltrans)
            self.goal.set_color(1,0,0)
            self.viewer.add_geom(self.goal)
            self.wall_1 = rendering.Line((5*scale,5*scale), (25*scale,5*scale))
            self.wall_2 = rendering.Line((5*scale,5*scale),(5*scale,25*scale))
            self.wall_3 = rendering.Line((5*scale,25*scale),(25*scale,25*scale))
            self.wall_4 = rendering.Line((25*scale,5*scale),(25*scale,25*scale))
            self.wall_1.set_color(0,0,0)
            self.wall_2.set_color(0,0,0)
            self.wall_3.set_color(0,0,0)
            self.wall_4.set_color(0,0,0)
            self.viewer.add_geom(self.wall_1)
            self.viewer.add_geom(self.wall_2)
            self.viewer.add_geom(self.wall_3)
            self.viewer.add_geom(self.wall_4)
            self.Lidar__1trans=rendering.Transform()
            self.Lidar__2trans=rendering.Transform()
            self.Lidar__3trans=rendering.Transform()
            self.Lidar__4trans=rendering.Transform()
            self.Lidar__1 = rendering.Line((0,0), (0,10*scale))
            self.Lidar__2 = rendering.Line((0,0), (0,10*scale))
            self.Lidar__3 = rendering.Line((0,0), (0,10*scale))
            self.Lidar__4 = rendering.Line((0,0), (0,10*scale))
            self.Lidar__1.add_attr(self.Lidar__1trans)
            self.Lidar__2.add_attr(self.Lidar__2trans)
            self.Lidar__3.add_attr(self.Lidar__3trans)
            self.Lidar__4.add_attr(self.Lidar__4trans)
            self.Lidar__1.set_color(0,0,0)
            self.Lidar__2.set_color(0,0,0)
            self.Lidar__3.set_color(0,0,0)
            self.Lidar__4.set_color(0,0,0)
            self.viewer.add_geom(self.Lidar__1)
            self.viewer.add_geom(self.Lidar__2)
            self.viewer.add_geom(self.Lidar__3)
            self.viewer.add_geom(self.Lidar__4)
            self._drone_geom = drone

        if self.state is None: return None
        
        dronex = x[8]*scale+screen_height/2
        droney = x[10]*scale+screen_width/2
        if not ((x_goal==self.x_goal_pre) and (y_goal==self.y_goal_pre)) :
            
            self.goaltrans.set_translation(screen_width/2+(y_goal)*scale,screen_height/2+(x_goal)*scale)
            self.x_goal_pre=x_goal
            self.y_goal_pre=y_goal
        self.Lidar__1trans.set_scale(1,Lidar1_length/10)
        self.Lidar__1trans.set_translation(droney,dronex)
        self.Lidar__1trans.set_rotation(-x[16]+self.Lidar1_joint_angle)
        self.Lidar__2trans.set_scale(1,Lidar2_length/10)
        self.Lidar__2trans.set_translation(droney,dronex)
        self.Lidar__2trans.set_rotation(-x[16]+self.Lidar2_joint_angle)
        self.Lidar__3trans.set_scale(1,Lidar3_length/10)
        self.Lidar__3trans.set_translation(droney,dronex)
        self.Lidar__3trans.set_rotation(-x[16]+self.Lidar3_joint_angle)
        self.Lidar__4trans.set_scale(1,Lidar4_length/10)
        self.Lidar__4trans.set_translation(droney,dronex)
        self.Lidar__4trans.set_rotation(-x[16]+self.Lidar4_joint_angle)
        
        self.dronetrans.set_translation(droney, dronex)
        self.dronetrans.set_rotation(-x[16])
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None