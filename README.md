# MPC-for-DDMR
This repository contains the program for Model Predictive Control of a differential drive robot. Dynamics model for DDMR contains the control of 10 states. They are: Position, Velocity, Heading angle, Heading angle vleocity, and each wheel's angular velocity. 

To verify the dynamics , run the `robot.py` file with appropriate initial conditions and simulation time. It will plot the robot trajectory for those conditions. 


``` 
python robot.py
```
Example trajectory of the robot is shown as the following: 

Initial conditions: 

```
    position = np.array([0,0])
    velocity = np.array([0,0])
    wheels_angles=np.array([0,0])
    wheels_velocities = np.array([0,0])
    theta = 0 
    theta_dot = 0 
    control_inputs = np.array([[1],[2]])
    sim_time = 10 #seconds
```

<img src='https://github.com/shivakumar-tekumatla/MPC-for-DDMR/blob/main/example_path.png'/>
