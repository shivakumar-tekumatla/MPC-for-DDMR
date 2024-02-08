# see this - https://medium.com/@eng_elias/numba-unleashing-the-power-of-python-for-high-performance-computing-fdec2c778b10 
from scipy.optimize import Bounds, minimize 
import cvxpy as cp
import numpy as np
import json
import time
from robot import Robot
import plotly.express as px
import GPy 

class MPC(Robot):
    current_inputs = np.array([0,0])
    def __init__(self,parameters,num_obstacles=3):
        super().__init__()
        self.robot_parameters = parameters["Robot"]
        
        # these are all the states of the robot at a given instance 
        self.robot_position = np.array(self.robot_parameters["Start"])
        self.robot_velocity = np.array([0,0])
        self.theta = self.robot_parameters["Theta"] # initial heading angle is zero 
        self.theta_d = self.robot_parameters["Theta_d"] # inital heading angular velocity is zero 
        self.wheel_angles = self.robot_parameters["Wheel_angles"]
        self.wheel_velocities = self.robot_parameters["Wheel_velocities"]
        
        # config parameters 
        self.Torque_MIN = self.robot_parameters["MIN Torque"]
        self.Torque_MAX = self.robot_parameters["MAX Torque"]
        
        # mpc parameters 
        self.mpc_parameters = parameters["MPC"]
        self.dt = self.mpc_parameters["dt"]
        self.sim_time = self.mpc_parameters["sim_time"]
        self.tracking_cost_weight = self.mpc_parameters["tracking_cost_weight"]
        self.tracking_range_residual = self.mpc_parameters["tracking_range_residual"] # for the exact tracking, keep it zero , if you want offset , then other than zero
        self.heading_cost_weight = self.mpc_parameters["heading_cost_weight"]
        self.desired_heading_angle_to_trajectory = self.mpc_parameters["desired_heading_angle_to_trajectory"]
        self.obstacle_cost_weight = self.mpc_parameters["obstacle_cost_weight"]
        self.obstacle_bounding_size = self.mpc_parameters["obstacle_bounding_size"] # assuming the overall shape of obstacle as square 
        self.HORIZON  = self.mpc_parameters["Horizon"]
        self.STEP = self.mpc_parameters["Step"]
        
        self.robot_bound = self.robot_parameters["Bounding_Size"]
        self.num_obstacles = num_obstacles
        self.bounds = [(self.Torque_MIN,self.Torque_MAX) for a in range(2*self.HORIZON)]
        return
    
    def cost(self,robot_positions,trajectory_positions,obstacles_positions,obs_optim=False):
        # the total cost is computed by first computing the tracking error, heading angle error, and collision error 
        # tracking error is low when the robot is close to the trajectory - position tracking and velocity tracking
        # collision error is low when the robot is as far as possible form the obstacles

        # tracking error
        diff = np.linalg.norm(robot_positions - trajectory_positions,axis=1)
        tracking_range_residual = self.tracking_range_residual - diff 
        tracking_error = 0.5 * self.dt* self.tracking_cost_weight  * tracking_range_residual.dot(tracking_range_residual) #

        # collision error 
        range_to_obstacle = np.mean(np.linalg.norm(robot_positions - obstacles_positions, axis=1))
        # print("Range to obstacle" ,range_to_obstacle)
        if obs_optim :
            collision_error = self.obstacle_cost_weight/(1 + np.exp(self.obstacle_bounding_size * (range_to_obstacle - self.robot_bound)))#)**2
        else:
            collision_error = 0
        return tracking_error+collision_error #collision_error#
    
    def object_function(self,control_inputs):
        # unpacking states  
        x,x_dot,y,y_dot,theta,theta_dot,phi1,phi1_dot,phi2,phi2_dot = self.states 
        # creating empty arrays for robot positions, trajectory positions and obstacle positions
        robot_positions      = np.empty((0,2)) 
        trajectory_positions = np.empty((0,2))
        obstacles_positions  = np.empty((0,2))
        
         
        # current robot position and velocity 
        robot_position = np.array([x,y]) #self.robot_position
        robot_velocity = np.array([x_dot,y_dot]) #self.robot_velocity
        # current wheels' velocities and angles 
        wheels_velocities = np.array([phi1_dot,phi2_dot])#self.wheel_velocities
        wheels_angles = np.array([phi1,phi2])#self.wheel_angles
        
        #current trajectory details 
        trajectory_position = self.trajectory_position  # current desired position 
        trajectory_velocity = self.trajectory_velocity  # current desired velocity
        # current obstacle details         
        obstacles_position = self.obstacles_position 
        obstacles_velocity = self.obstacles_velocity
        
        
        
        robot_positions = np.append(robot_positions, [robot_position], axis=0)
        trajectory_positions = np.append(trajectory_positions, [trajectory_position], axis=0)
        obstacles_positions = np.append(obstacles_positions, [obstacles_position], axis=0)
        
        for i in range(self.HORIZON): 
            t = np.linspace(0,self.STEP,2)  
            states = self.simulate_dynamics(t,robot_position,robot_velocity,theta,theta_dot,wheels_angles,wheels_velocities,control_inputs[i:i+2]) # these are the robot_states in the future 

            # future robot states 
            x,x_dot,y,y_dot,theta,theta_dot,phi1,phi1_dot,phi2,phi2_dot = states.T 
            robot_position = np.array([x[-1],y[-1]]) # last state from the simulation is going to be the next state 
            robot_velocity = np.array([x_dot[-1],y_dot[-1]]) 
            theta = theta[-1]
            theta_dot = theta_dot[-1]
            wheels_velocities = np.array([phi1_dot[-1],phi2_dot[-1]])#self.wheel_velocities
            wheels_angles = np.array([phi1[-1],phi2[-1]])#self.wheel_angles
                        
            robot_positions = np.append(robot_positions, [robot_position], axis=0)
            # future trajectory and obstacle predictions 
            #TODO : use nonlinear model to predict the future trajectory positions and obstacle positions 
            trajectory_position = trajectory_position + trajectory_velocity*self.STEP
            trajectory_positions = np.append(trajectory_positions, [trajectory_position], axis=0)
            obstacles_position = obstacles_position + obstacles_velocity*self.STEP
            obstacles_positions = np.append(obstacles_positions, [obstacles_position], axis=0)
    
        cost_ = self.cost(robot_positions,trajectory_positions,obstacles_positions)
        # print("Cost...",cost,end="\r")
        return cost_

    def control(self,states,trajectory_position,trajectory_velocity,obstacles_position,obstacles_velocity):
        #t,dt,robot_position,robot_velocity,theta,theta_d,trajectory_position,trajectory_velocity,obstacles_position,obstacles_velocity):
        self.states = states 
        self.trajectory_position = trajectory_position
        self.trajectory_velocity = trajectory_velocity
        self.obstacles_position = obstacles_position
        self.obstacles_velocity = obstacles_velocity
        inputs = []
        [inputs.extend([MPC.current_inputs[0],MPC.current_inputs[1]]) for i in range(self.HORIZON)]  # Defining initial inputs same as the current inputs 
        inputs = np.array(inputs)

        res = minimize(self.object_function, inputs, method='SLSQP', bounds= self.bounds)
        # print("Optimizer result ",res,end="\r")
        best_inputs = np.array([res.x[0],res.x[1]])
        MPC.current_inputs = best_inputs
        return best_inputs
          
def trajectory_function(t,r=10,offset = np.array([0, 1])):
    t_angle = t / 30 * (2 * np.pi)
    x = r * np.sin(t_angle)
    y = np.cos(t_angle)  
    return np.array([x, y]) + offset

def obstacle_position_function(t,obstacle_period):
    return np.array([11, -1]) + (t/25) * np.array([-21.1, 0]) + np.cos(t * 2 * np.pi / obstacle_period) * np.array([0, 5])

def plot(sim_time,robot_positions,trajectory_positions,obstacle_positions,range_dims = (10,6),title="Simulation"):
    dt = sim_time[1] - sim_time[0]
    drawing_dt = 0.5
    step = int(drawing_dt / dt)
    indices = np.array(list(range(len(sim_time))[::step]) + [len(sim_time) - 1])
    # Assemble the data frame by stacking subject, drone, obstacle data
    data = dict()
    data['t'] = np.hstack([sim_time[indices], sim_time[indices], sim_time[indices]])
    data['x'] = np.hstack([trajectory_positions[indices, 0],
                           robot_positions[indices, 0],
                           obstacle_positions[indices, 0]])
    data['y'] = np.hstack([trajectory_positions[indices, 1],
                           robot_positions[indices, 1],
                           obstacle_positions[indices, 1]])
    data['type'] = ['Trajectory'] * len(indices) + ['Robot'] * len(indices) + ['Obstacle'] * len(indices)
    obstacle_bounding_size = 1
    data['size'] = [5] * len(indices) + [5] * len(indices) + [53 * obstacle_bounding_size] * len(indices)
    # Make the animated trace
    fig = px.scatter(
        data,
        x='x',
        y='y',
        animation_frame='t',
        animation_group='type',
        color='type',
        category_orders={'type': ['obstacle', 'subject', 'drone']},
        color_discrete_sequence=('#FF5555', '#CCCC00', '#5555FF'),
        size='size',
        size_max=data['size'][-1],
        hover_name='type',
        # template='plotly_dark',
        range_x=(-range_dims[0], range_dims[0]),
        range_y=(-range_dims[1], range_dims[1]),
        height=700,
        title=title
      )
    # Make equal one meter grid
    fig.update_xaxes(dtick=1.0,showline=False)
    fig.update_yaxes(scaleanchor = "x",scaleratio = 1,showline=False,dtick=1.0)
    
    # Draw full curve of the subject's path
    subject_line = px.line(x=trajectory_positions[:, 0],y=trajectory_positions[:, 1]).data[0]
    subject_line.line['color'] = '#FFFF55'
    subject_line.line['width'] = 1
    fig.add_trace(subject_line)

    # Draw full curve of the drone's path
    robot_line = px.line(x=robot_positions[:, 0],y=robot_positions[:, 1]).data[0]
    robot_line.line['color'] = '#AAAAFF'
    robot_line.line['width'] = 1
    fig.add_trace(robot_line)
  
    # Draw full curve of the obstacle's path
    obs_line = px.line(x=obstacle_positions[:, 0],y=obstacle_positions[:, 1]).data[0]
    obs_line.line['color'] = '#FFAAAA'
    obs_line.line['width'] = 1
    fig.add_trace(obs_line)
    fig.show()
    return None 

def main():
    
    # read parameters
    with open("parameters.json","r") as file:
        parameters = json.load(file)
    mpc = MPC(parameters)
    sim_time = np.arange(0,mpc.sim_time, mpc.dt)
    trajectory = np.array([trajectory_function(t) for t in sim_time])
    # trajectory_velocities = np.array([trajectory_velocity_function(t) for t in sim_time])

    obstacle_positions = np.array([obstacle_position_function(t,5) for t in sim_time])

    # Compute the robot trajectory
    start_time = time.time()

    # Robot initial state
    robot_position = np.array(parameters["Robot"]["Start"])
    robot_velocity = np.array(parameters["Robot"]["Vel"])
    theta = parameters["Robot"]["Theta"]
    theta_dot = parameters["Robot"]["Theta_d"]
    wheel_angles = parameters["Robot"]["Wheel_angles"]
    wheel_velocities = parameters["Robot"]["Wheel_velocities"]

    x,y = robot_position.flatten()
    x_dot,y_dot = robot_velocity.flatten()
    phi1,phi2 = wheel_angles#.flatten()
    phi1_dot,phi2_dot = wheel_velocities#.flatten()
    states = np.array([x,x_dot,y,y_dot,theta,theta_dot,phi1,phi1_dot,phi2,phi2_dot])
    robot_positions = [mpc.robot_position]
    robot_velocities = [robot_velocity]

    for i in range(len(sim_time)-1):
        t = np.linspace(sim_time[i], sim_time[i + 1],2)

        trajectory_position =  trajectory[i]
        trajectory_velocity =  (trajectory[i+1]-trajectory_position)/mpc.dt  #trajectory_velocities[i]
        obstacles_position = obstacle_positions[i]
        obstacles_velocity = (obstacle_positions[i+1]-obstacles_position)/mpc.dt #np.array([obstacle_velocities[i]])
        inputs = mpc.control(states,trajectory_position,trajectory_velocity,obstacles_position,obstacles_velocity)#,obstacle_cost_optim)
        print(f"Time {sim_time[i]} Control Inputs {inputs}", end="\r")

        states_ = mpc.simulate_dynamics(t,robot_position,robot_velocity,theta,theta_dot,wheel_angles,wheel_velocities,inputs) # these are the robot_states in the future 

        x,x_dot,y,y_dot,theta,theta_dot,phi1,phi1_dot,phi2,phi2_dot = states_[-1,:].flatten() # last state of the previous sim is next state

        states = np.array([x,x_dot,y,y_dot,theta,theta_dot,phi1,phi1_dot,phi2,phi2_dot ])

        robot_position = np.array([x,y])
        robot_velocity = np.array([x_dot,y_dot])
        wheel_angles = np.array([phi1,phi2])
        wheel_velocities = np.array([phi1_dot,phi2_dot])

        robot_positions.append(robot_position)
        robot_velocities.append(robot_velocity)

    robot_positions = np.vstack(robot_positions)
    robot_velocities = np.vstack(robot_velocities)
    
    plot(sim_time,robot_positions,trajectory,obstacle_positions)
    cost = mpc.cost(robot_positions,trajectory,obstacle_positions)
    print("Total Cost....",cost)
    end_time = time.time()
    run_time = end_time-start_time
    print("Total time it took....",run_time)
    return None

if __name__ == "__main__":
    main()
    


    
