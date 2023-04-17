import numpy as np 
import scipy.integrate as integrate
import matplotlib.pyplot as plt 

class Robot:
    def __init__(self,mw=0.2,mB=3,d=0.1,W=0.15,ro=0.5,t=0.01,):
        self.mw = mw # mass of wheels
        self.mB = mB # mass of the chassis
        self.d= d   # distance from the center of the wheels to the center of mass of chassis
        self.W = W  # half of wheel-to-wheel distance 
        self.ro = ro # radius of the wheels
        self.t = t # thickness of the wheels
        self.IB = 0.5*self.mB*(self.W**2)  # moment of inertia of the chassis
        self.Izz=self.mw*(3*(self.ro**2)+(self.t**2))/12 #  moment of inertia of wheels about the z-axis
        self.Iyy=self.mw*(self.ro**2)/2 #moment of inertia of wheels about the y-axis
        self.mT=self.mB+2*self.mw # total mass
        self.IT=self.IB+self.mB*(self.d**2)+2*self.mw*(self.W**2)+2*self.Izz # total moment of inertia
        self.mod_2_pi = lambda x:np.mod(x + np.pi, 2 * np.pi) - np.pi #Convert an angle in radians into range [-pi, pi]

        pass

    def dynamics(self,states,t,control_inputs):
        dzdt = np.zeros_like(states)
        x,x_dot,y,y_dot,theta,theta_dot,phi1,phi1_dot,phi2,phi2_dot = states.flatten()
        theta = self.mod_2_pi(theta)
        phi1  = self.mod_2_pi(phi1)
        phi2 = self.mod_2_pi(phi2)
        q_dot = np.array([[x_dot],
                          [y_dot],
                          [theta_dot],
                          [phi1_dot],
                          [phi2_dot]])

        M = np.array([[self.mT,0,-self.mB*self.d*np.sin(theta),0,0],
                      [0,self.mT,self.mB*self.d*np.cos(theta),0,0],
                      [-self.mB*self.d*np.sin(theta),self.mB*self.d*np.cos(theta),self.IT,0,0],
                      [0,0,0,self.Iyy,0],
                      [0,0,0,0,self.Iyy]])

        B=-self.mB*self.d*(theta_dot**2)*np.array([[np.cos(theta)],
                                                   [np.sin(theta)],
                                                   [0],
                                                   [0],
                                                   [0]]) 

        C = np.array([[np.cos(theta),np.sin(theta),0,self.ro/2,-self.ro/2],
                      [-np.sin(theta),np.cos(theta),0,0,0],
                      [0,0,1,0.5*self.ro/self.W,0.5*self.ro/self.W]])
        
        C_dot=np.array([[-np.sin(theta)*theta_dot, np.cos(theta)*theta_dot, 0, 0, 0],
                       [-np.cos(theta)*theta_dot, -np.sin(theta)*theta_dot, 0, 0, 0],
                       [0,0,0,0,0]])

        
        T1,T2 = control_inputs.flatten()
        
        T=np.array([[0],[0],[0],[T1],[T2]])
        lambdas=-np.linalg.inv(C@np.linalg.inv(M)@C.T)@(C@np.linalg.inv(M)@(T-B)+C_dot@q_dot)
        temp=np.linalg.inv(M)@(T-B+C.T@lambdas)

        dzdt[0] = states[1]

        dzdt[1]=temp[0]
        dzdt[2]=states[3]

        dzdt[3]=temp[1]
        dzdt[4]=states[5]
        dzdt[5]=temp[2]
        dzdt[6]=states[7]
        dzdt[7]=temp[3]
        dzdt[8]=states[9]
        dzdt[9]=temp[4]

        return dzdt
    
    def simulate_dynamics(self,t,position,velocity,theta,theta_dot,wheels_angles,wheels_velocities,control_inputs,dt=0.01):
        x,y = position.flatten()
        x_dot,y_dot = velocity.flatten()
        phi1,phi2 = wheels_angles#.flatten()
        phi1_dot,phi2_dot = wheels_velocities#.flatten()
        states = np.array([x,x_dot,y,y_dot,theta,theta_dot,phi1,phi1_dot,phi2,phi2_dot])
        states = integrate.odeint( self.dynamics, states, t ,args=(control_inputs,))
        return np.array(states)


if __name__ == "__main__":
    robot = Robot()
    position = np.array([0,0])
    velocity = np.array([0,0])
    wheels_angles=np.array([0,0])
    wheels_velocities = np.array([0,0])
    theta = 0 
    theta_dot = 0 
    control_inputs = np.array([[1],[2]])
    t = 0
    t_next = 10
    t = np.linspace(t,t_next,1000)
    y = robot.simulate_dynamics(t,position,velocity,theta,theta_dot,wheels_angles,wheels_velocities,control_inputs)
    pos = y[:,0:3]
    x = pos[:,0]
    y = pos[:,2]
    plt.plot(x,y)
    plt.show()

