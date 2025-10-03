#!/usr/bin/env python

"""LQR controller for one cf, use to config the cf and mocap data"""

from pycrazyswarm import Crazyswarm
import time
import math
from simple_pid import PID
import matplotlib.pyplot as plt
import numpy as np
import rospy
import scipy
from crazyswarm.msg import GenericLogData
import threading
import ctypes
import inspect
import tf
import random

TAKEOFF_DURATION = 0.05
HOVER_DURATION = 0.05

# # Backstepping para
k_1 = 1
k_2 = 18
k_3 = 2

# # Adaptive control para
gama = 0.002
# Chanal x
a_x_hat = 1e-7
rho_x_hat = 1e-7
z_3 = 0
alpha_3 = 0
# Chanal y
a_y_hat = 1e-7
rho_y_hat = 1e-7
e_3 = 0
tao_3 = 0

# # Log data
log_size_ad = 10000
a_x_hat_log = np.zeros([log_size_ad, 1], dtype = float)
rho_x_hat_log = np.zeros([log_size_ad, 1], dtype = float)

class kalman_filter:
    def __init__(self,Q,R):
        self.Q = Q
        self.R = R
        
        self.P_k_k1 = 2
        self.Kg = 0
        self.P_k1_k1 = 2
        self.x_k_k1 = 0
        self.ADC_OLD_Value = 0
        self.Z_k = 0
        self.kalman_adc_old=0
        
    def kalman(self,ADC_Value):
       
        self.Z_k = ADC_Value
        self.x_k1_k1 = self.kalman_adc_old
    
        self.x_k_k1 = self.x_k1_k1
        self.P_k_k1 = self.P_k1_k1 + self.Q
    
        self.Kg = self.P_k_k1/(self.P_k_k1 + self.R)
    
        kalman_adc = self.x_k_k1 + self.Kg * (self.Z_k - self.kalman_adc_old)
        self.P_k1_k1 = (1 - self.Kg)*self.P_k_k1
        self.P_k_k1 = self.P_k1_k1
    
        self.kalman_adc_old = kalman_adc
        
        return kalman_adc

class RC_filter:
    def __init__(self,sampleFrq,CutFrq):
        self.sampleFrq = sampleFrq
        self.CutFrq = CutFrq
        self.adc_old=0
       
        
    def LowPassFilter_RC_1order(self,Vi):
        RC = 1.0/2.0/math.pi/self.CutFrq 
        Cof1 = 1/(1+RC * self.sampleFrq)
        Cof2 = RC* self.sampleFrq/(1+RC* self.sampleFrq)
        Vo = Cof1 * Vi + Cof2 * self.adc_old	 	
        self.adc_old = Vo
        return Vo  

def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

def resetAdaptiveControl():
    global a_x_hat, rho_x_hat, a_y_hat, rho_y_hat
    # Chanal x
    a_x_hat = 1e-7
    rho_x_hat = 1e-7
    # Chanal y
    a_y_hat = 1e-7
    rho_y_hat = 1e-7

    # # Chanal x
    # a_x_hat = random.random()
    # rho_x_hat = random.random()
    # # Chanal y
    # a_y_hat = random.random()
    # rho_y_hat = random.random()
    print("Reset adaptive control parameters!")

def thrust2pwm(thrust):
    # input:thrust(float)
    # output:thrust_pwm(int)

    # # Thurst_pwm must be an integer in 0~65535
    # # Thurst must be in 0~0.14 N
    # if thrust > 65535 or thrust < 0:
    #     print("Thurst must be an integer in 0~65535")
    #     break
    c1 = 2.130295e-11
    c2 = 1.032633e-6
    c3 = 5.484560e-4

    pwm = (-c2 + math.sqrt(c2**2 - 4*c1*(c3-thrust)))/(2*c1)
    pwm = int(pwm)
    return pwm

def pwm2thrust(pwm):
    # input:thrust_pwm(int)
    # output:thrust(float)
    c1 = 2.130295e-11
    c2 = 1.032633e-6
    c3 = 5.484560e-4

    thrust = c1*pwm**2 + c2*pwm + c3
    return thrust

def thrust_acc_sat(thrust_acc):
    # Thurst_acc must be in -0.03~0.03 N
    
    max_thrust_acc = 0.03
    min_thrust_acc = -0.03

    if thrust_acc >= max_thrust_acc:
        thrust_acc = max_thrust_acc
    elif thrust_acc <= min_thrust_acc:
        thrust_acc = min_thrust_acc

    return thrust_acc

def thrust_sat(thrust):
    # Thurst must be in 0.02~0.12 N
    max_thrust = 0.12
    min_thrust = 0.02

    if thrust >= max_thrust:
        thrust = max_thrust
    elif thrust <= min_thrust:
        thrust = min_thrust

    return thrust


def pitch_sat(pitch):
    max_pitch = 20
    min_pitch = -20

    if pitch >= max_pitch:
        pitch = max_pitch
    elif pitch <= min_pitch:
        pitch = min_pitch

    return pitch

def roll_sat(roll):
    max_roll = 20
    min_roll = -20

    if roll >= max_roll:
        roll = max_roll
    elif roll <= min_roll:
        roll = min_roll

    return roll

def yawrate_sat(yawrate):
    max_yawrate = 5
    min_yawrate = -5

    if yawrate >= max_yawrate:
        yawrate = max_yawrate
    elif yawrate <= min_yawrate:
        yawrate = min_yawrate

    return yawrate


def getVelocity(cur_pos, last_pos, time):
    v = np.empty([3,1], dtype=float)
    v[0] = (cur_pos[0]-last_pos[0])/time
    v[1] = (cur_pos[1]-last_pos[1])/time
    v[2] = (cur_pos[2]-last_pos[2])/time

    return v


def gainLQR():
    A = np.zeros([9,9], dtype = float)
    A[0][1] = 1
    A[2][3] = 1
    A[4][5] = 1
    A[1][7] = 9.8
    A[3][6] = -9.8
    A[6][6] = -5.018
    A[7][7] = -5.083

    B = np.zeros([9, 4], dtype= float)
    B[5][0] = 1
    B[6][1] = 5.233
    B[7][2] = 5.516
    B[8][3] = 1

    C = np.zeros([4, 9], dtype= float)
    C[0][0] = 1
    C[1][2] = 1
    C[2][4] = 1
    C[3][8] = 1

    # Most important part
    # Decide the state gain Q and input gain R
    Q = np.diag([5, 1, 5, 1, 60, 1, \
                0.5, 0.5, 0.5, \
                0.01, 0.01, 0.01, 0.01])
    R = np.diag([6, 30, 30, 1])

    A_bar1 = np.concatenate((A, np.zeros([9, 4], dtype=float)), axis = 1)
    A_bar2 = np.concatenate((C, np.zeros([4, 4], dtype=float)), axis = 1)
    A_bar = np.concatenate((A_bar1, A_bar2), axis=0)

    B_bar = np.concatenate((B, np.zeros([4, 4], dtype=float)), axis = 0)

    P = scipy.linalg.solve_continuous_are(A_bar, B_bar, Q, R)

    K = np.matrix(np.linalg.inv(R)) * np.matrix(B_bar.transpose()) * np.matrix(P) 

    return K


def xBackstepping(x_1d, x_1d_dot, x_1d_ddot, x_1d_dddot,\
    x_1, x_2, theta, k_1, k_2, k_3):
    global a_x_hat, rho_x_hat, z_3, alpha_3

    g = 9.8
    # tao_theta = 0.1967
    # k_theta = 5.085


    z_1 = x_1 - x_1d
    alpha_1 = x_1d_dot - k_1 * z_1
    alpha_1_dot = x_1d_ddot - k_1 * (x_2 - x_1d_dot)

    z_2 = x_2 - alpha_1
    alpha_2 = 1/g*(alpha_1_dot - k_2*z_2 - z_1)
    alpha_1_ddot = x_1d_dddot - k_1 * (g*math.tan(theta) - x_1d_ddot)

    z_3 = math.tan(theta) - alpha_2
    alpha_2_dot = 1/g*(alpha_1_ddot - k_2*(g*math.tan(theta) - alpha_1_dot) \
        - (x_2 - x_1d_dot))

    alpha_3 = -a_x_hat * theta + (math.cos(theta))**2 \
        * (alpha_2_dot - g*z_2 - k_3*z_3)
    theta_c = rho_x_hat * alpha_3

    return theta_c

def yBackstepping(y_1d, y_1d_dot, y_1d_ddot, y_1d_dddot,\
    y_1, y_2, phi, k_1, k_2, k_3):
    global a_y_hat, rho_y_hat, e_3, tao_3

    g = 9.8
    # tao_phi = 0.1993
    # k_phi = 5.043


    e_1 = y_1 - y_1d
    tao_1 = y_1d_dot - k_1 * e_1
    tao_1_dot = y_1d_ddot - k_1 * (y_2 - y_1d_dot)

    e_2 = y_2 - tao_1
    tao_2 = -1/g*(tao_1_dot - k_2*e_2 - e_1)
    tao_1_ddot = y_1d_dddot - k_1 * (-g*math.tan(phi) - y_1d_ddot)

    e_3 = math.tan(phi) - tao_2
    tao_2_dot = -1/g*(tao_1_ddot - k_2*(-g*math.tan(phi) - tao_1_dot) \
        - (y_2 - y_1d_dot))

    tao_3 = -a_y_hat*phi + (math.cos(phi))**2 \
        * (tao_2_dot + g*e_2 - k_3 * e_3)
    phi_c = rho_y_hat * tao_3

    return phi_c

def adaptiveControl():
    global gama, a_x_hat, rho_x_hat, z_3, alpha_3, \
        a_y_hat, rho_y_hat, e_3, tao_3, a_x_hat_log, rho_x_hat_log
    
    rate = rospy.Rate(50)
    listener = tf.TransformListener()
    # time.sleep(1)

    start_time = time.time()
    last_time = start_time
    count = 0

    while not rospy.is_shutdown():
        now_time = time.time() - start_time
        dt = now_time - last_time 
        last_time = now_time
        if(abs(dt) > 1):
            continue

        try:
            listener.waitForTransform("/world", "/cf5", rospy.Time(0), rospy.Duration(1))
            position, quaternion = listener.lookupTransform("/world", "/cf5", rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
        (phi, theta, y) = tf.transformations.euler_from_quaternion([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])     
        # print("thread1:", alpha_2 + p, "time:", dt)

        a_x_hat_dot = 1/(math.cos(theta))**2 * theta * z_3
        rho_x_hat_dot = -gama * z_3 / (math.cos(theta))**2 * alpha_3
        a_x_hat += a_x_hat_dot * dt
        rho_x_hat += rho_x_hat_dot * dt

        a_y_hat_dot = 1/(math.cos(phi))**2 * phi * e_3
        rho_y_hat_dot = -gama * e_3 / (math.cos(phi))**2 * tao_3
        a_y_hat += a_y_hat_dot * dt
        rho_y_hat += rho_y_hat_dot * dt

        a_x_hat_log[count] = a_x_hat
        rho_x_hat_log[count] = rho_x_hat
        count += 1

        print("time:", now_time, "dt:", dt)
        print("a_x_hat:", a_x_hat, "rho_x_hat:", rho_x_hat, "a_y_hat:", a_y_hat, "rho_y_hat:", rho_y_hat)

        rate.sleep()



def main():
    # Control one cf to fly
    global k_1, k_2, k_3, a_x_hat_log, rho_x_hat_log
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    # # cf physics para
    M = 0.05

    # # Initial KF para
    pos_KF_Q = 0.0001
    pos_KF_R = 0.001
    pos_x_KF =  kalman_filter(pos_KF_Q,pos_KF_R)
    pos_y_KF =  kalman_filter(pos_KF_Q,pos_KF_R)
    pos_z_KF =  kalman_filter(pos_KF_Q,pos_KF_R)

    vel_KF_Q = 0.00001
    vel_KF_R = 0.0001
    vel_vx_KF =  kalman_filter(vel_KF_Q,vel_KF_R)
    vel_vy_KF =  kalman_filter(vel_KF_Q,vel_KF_R)
    vel_vz_KF =  kalman_filter(vel_KF_Q,vel_KF_R)  

    cf.takeoff(targetHeight=0., duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)
    cf.land(targetHeight=0.0, duration=0.05)
    timeHelper.sleep(2)

    # # Initial first desired location with yaw is 0
    des_x = 0
    des_y = 0
    des_z = 0.5

    des_x_dot = 0
    des_x_ddot = 0
    des_x_dddot = 0

    des_y_dot = 0
    des_y_ddot = 0
    des_y_dddot = 0

    flag = 1

    # Firstly, it must be sent some zero command
    # to unlock the on-board controller
    unlock_conut = 0

    start_time = time.time()
    last_time = time.time()

    while unlock_conut <= 5:
        cf.cmdVel(0,0,0,0)
        time.sleep(0.05)
        unlock_conut += 1

    # # Intial logging part
    log_size = 2000
    position_x_log = np.zeros([log_size, 1], dtype = float)
    position_x_fil_log = np.zeros([log_size, 1], dtype = float)
    position_y_log = np.zeros([log_size, 1], dtype = float)
    position_y_fil_log = np.zeros([log_size, 1], dtype = float)
    position_z_log = np.zeros([log_size, 1], dtype = float)
    position_z_fil_log = np.zeros([log_size, 1], dtype = float)
    position_vx_log = np.zeros([log_size, 1], dtype = float)
    position_vx_fil_log = np.zeros([log_size, 1], dtype = float)
    position_vy_log = np.zeros([log_size, 1], dtype = float)
    position_vy_fil_log = np.zeros([log_size, 1], dtype = float)
    position_vz_log = np.zeros([log_size, 1], dtype = float)
    position_vz_fil_log = np.zeros([log_size, 1], dtype = float)
    count = 0

    # # Adaptive control part, in a new threading with high frequency
    add_thread = threading.Thread(target = adaptiveControl)
    add_thread.start()



    frequency = 30
    last_position = np.empty([3,1], dtype = float)

    # # Initial gain LQR
    K = gainLQR()
    e_bar = np.zeros([13,1], dtype = float)

    # # Main loop
    while True:
        # Protect the cf's running time
        now_time = time.time()
        dt = now_time - last_time
        last_time = now_time
        #print("dt:", dt)


        mission = 4
        

        # mission 2 para and each mission para
        test_time = 8
        stable_time = 6
        if mission == 1:
            if now_time - start_time > 5 and now_time - start_time <= 10:
                des_x = 0.4
                des_y = -0.3
                des_z = 0.8

                des_x_dot = 0
                des_x_ddot = 0
                des_x_dddot = 0

                des_y_dot = 0
                des_y_ddot = 0
                des_y_dddot = 0

                if(flag == 1):
                    resetAdaptiveControl()
                    flag = 2
                    print("flag:", flag)

            elif now_time - start_time > 10 and now_time - start_time <= 18:
                des_x = -0.3
                des_y = 0.4
                des_z = 0.6

                des_x_dot = 0
                des_x_ddot = 0
                des_x_dddot = 0

                des_y_dot = 0
                des_y_ddot = 0
                des_y_dddot = 0

                if(flag == 2):
                    resetAdaptiveControl()
                    flag = 3
                    print("flag:", flag)

            elif now_time - start_time > 18 and now_time - start_time <= 20:
                des_x = 0
                des_y = 0
                des_z = 0.2

                des_x_dot = 0
                des_x_ddot = 0
                des_x_dddot = 0

                des_y_dot = 0
                des_y_ddot = 0
                des_y_dddot = 0

                if(flag == 3):
                    resetAdaptiveControl()
                    flag = 4
                    print("flag:", flag)

            elif now_time - start_time > 20:
                break
        elif mission == 2:
            if now_time - start_time > test_time and now_time - start_time <= test_time+stable_time:
                des_x = 0
                des_y = 0
                des_z = 0.15

                des_x_dot = 0
                des_x_ddot = 0
                des_x_dddot = 0

                des_y_dot = 0
                des_y_ddot = 0
                des_y_dddot = 0

                if(flag == 1):
                    resetAdaptiveControl()
                    flag = 2
                    print("flag:", flag)

            elif now_time - start_time > test_time+stable_time: 
                break
        elif mission == 3:
            if now_time - start_time > 5 and now_time - start_time <= 15:
                des_x = 0.3
                des_y = 0
                des_z = 0.5

                des_x_dot = 0
                des_x_ddot = 0
                des_x_dddot = 0

                des_y_dot = 0
                des_y_ddot = 0
                des_y_dddot = 0

                if(flag == 1):
                    resetAdaptiveControl()
                    flag = 2
                    print("flag:", flag)

            elif now_time - start_time > 15: 
                break
        elif mission == 4:
            if now_time - start_time > 3 and now_time - start_time <= 6:
                des_x = 1
                des_y = 0
                des_z = 0.5

                des_x_dot = 0
                des_x_ddot = 0
                des_x_dddot = 0

                des_y_dot = 0
                des_y_ddot = 0
                des_y_dddot = 0

                if(flag == 1):
                    resetAdaptiveControl()
                    flag = 2
                    print("flag:", flag)

            elif now_time - start_time > 6 and now_time - start_time <= 40:
                t = now_time - start_time - 6
                R = 1
                omega = 0.7
                des_x = R*math.cos(omega*t)
                des_y = R*math.sin(omega*t)
                des_z = 0.5

                des_x_dot = -R*omega*math.sin(omega*t)
                des_x_ddot = -R*omega*omega*math.cos(omega*t)
                des_x_dddot = R*omega*omega*omega*math.sin(omega*t)

                des_y_dot = R*omega*math.cos(omega*t)
                des_y_ddot = -R*omega*omega*math.sin(omega*t)
                des_y_dddot = -R*omega*omega*omega*math.cos(omega*t)

                if(flag == 2):
                    resetAdaptiveControl()
                    flag = 3
                    print("flag:", flag)

            elif now_time - start_time > 40 and now_time - start_time <= 43:
                des_x = 0
                des_y = 0
                des_z = 0.2

                des_x_dot = 0
                des_x_ddot = 0
                des_x_dddot = 0

                des_y_dot = 0
                des_y_ddot = 0
                des_y_dddot = 0

                if(flag == 3):
                    resetAdaptiveControl()
                    flag = 4
                    print("flag:", flag)

            elif now_time - start_time > 43: 
                break            

        # # LQR Controller Part
        # Mind that the attitude angle's units are rad
        position, rotation = cf.position()
        # Quaternion to euler angle:roll, pitch , yaw
        (r, p, y) = tf.transformations.euler_from_quaternion([rotation[0], rotation[1], rotation[2], rotation[3]])     

        # Log the position and velocity data
        position_x_log[count] = position[0]
        position_x_fil_log[count] = pos_x_KF.kalman(position[0])
        position[0] = position_x_fil_log[count]
        position_y_log[count] = position[1]
        position_y_fil_log[count] = pos_y_KF.kalman(position[1])
        position[1] = position_y_fil_log[count]
        position_z_log[count] = position[2]
        position_z_fil_log[count] = pos_z_KF.kalman(position[2])
        position[2] = position_z_fil_log[count]  
           
        vel = getVelocity(position, last_position, dt)

        position_vx_log[count] = vel[0]
        vel_vx_F = vel_vx_KF.kalman(vel[0])
        position_vx_fil_log[count] = vel_vx_F

        position_vy_log[count] = vel[1]
        vel_vy_F = vel_vy_KF.kalman(vel[1])
        position_vy_fil_log[count] = vel_vy_F

        position_vz_log[count] = vel[2]
        vel_vz_F = vel_vz_KF.kalman(vel[2])
        position_vz_fil_log[count] = vel_vz_F
        
        # Calculate e_bar:x, x_dot, y, y_dot, z, z_dot, r, p, y, 
        # integral(e_x), integral(e_y), integral(e_z), integral(e_kesi), 
        e_bar[0] = position[0] - des_x
        e_bar[1] = vel_vx_F
        e_bar[2] = position[1] - des_y
        e_bar[3] = vel_vy_F
        e_bar[4] = position[2] - des_z
        e_bar[5] = vel[2] # if use vel_vz_F, it will not stable

        e_bar[6] = r
        e_bar[7] = p
        e_bar[8] = y

        e_bar[9] += e_bar[0]*dt
        e_bar[10] += e_bar[2]*dt
        e_bar[11] += e_bar[4]*dt
        e_bar[12] += e_bar[8]*dt

        # Calculate input LQR part
        balance_thrust = 0.065 # equilibrium thrust divided by 4
        deg = 180/math.pi
        des_command = - np.matrix(K) * np.matrix(e_bar) # mind that it is negative feedback control

        des_command[0] = thrust_sat(thrust_acc_sat(des_command[0]*M/4) + balance_thrust)
        des_command[3] = yawrate_sat(des_command[3]*deg)        
        
        last_position = position

        # # BackStepping Part
        des_command[1] = roll_sat(deg*yBackstepping(des_y, des_y_dot, des_y_ddot, des_y_dddot, position[1], vel[1], r, k_1, k_2, k_3))
        des_command[2] = pitch_sat(deg*xBackstepping(des_x, des_x_dot, des_x_ddot, des_x_dddot, position[0], vel[0], p, k_1, k_2, k_3))

        # # Send command to cf: roll_des, pitch_des, yawrate_des, thrust_des
        cf.cmdVel(des_command[1], des_command[2], -des_command[3], thrust2pwm(des_command[0]))
        # f.cmdVel(0, 0, 0, 0)
        print("thrust:", des_command[0], "des_pitch:", des_command[2], "pitch:", p*deg)
        print("thrust:", des_command[0], "des_roll:", des_command[1], "roll:", r*deg)
        #print("thrust:", des_command[0], "des_yaw:", des_command[3], "yaw:", y*deg)
        time.sleep(1/frequency)
        count += 1

    # Lock the cf
    cf.cmdVel(0,0,0,0)

    # Close the other thread
    _async_raise(add_thread.ident, SystemExit)

    # # plot the log data
    plt.plot(position_x_log,'r')
    plt.plot(position_x_fil_log,'b')   
    plt.xlabel('t [s]')
    plt.ylabel('x [m]')
    plt.show()
    plt.figure(1)

    plt.plot(position_y_log,'r')
    plt.plot(position_y_fil_log,'b')   
    plt.xlabel('t [s]')
    plt.ylabel('y [m]')
    plt.show()
    plt.figure(3)

    plt.plot(position_z_log,'r')
    plt.plot(position_z_fil_log,'b')   
    plt.xlabel('t [s]')
    plt.ylabel('z [m]')
    plt.show()
    plt.figure(3)

    plt.plot(position_vx_log,'r')
    plt.plot(position_vx_fil_log,'b')   
    plt.xlabel('t [s]')
    plt.ylabel('vx [m/s]')
    plt.show()
    plt.figure(4)

    plt.plot(position_vy_log,'r')
    plt.plot(position_vy_fil_log,'b')   
    plt.xlabel('t [s]')
    plt.ylabel('vy [m/s]')
    plt.show()
    plt.figure(5)

    plt.plot(position_vz_log,'r')
    plt.plot(position_vz_fil_log,'b')   
    plt.xlabel('t [s]')
    plt.ylabel('vz [m/s]')
    plt.show()
    plt.figure(6)

    plt.plot(a_x_hat_log,'r')
    plt.xlabel('t [s]')
    plt.ylabel('a_x_hat')
    plt.show()
    plt.figure(7)

    plt.plot(rho_x_hat_log,'r')
    plt.xlabel('t [s]')
    plt.ylabel('rho_x_hat')
    plt.show()
    plt.figure(8) 

if __name__ == "__main__":

    main()
