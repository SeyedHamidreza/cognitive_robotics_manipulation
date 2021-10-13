from grasp_generator import GraspGenerator
from environment.utilities import Camera
from environment.env import Environment
from utils import YcbObjects, PackPileData, IsolatedObjData, summarize
import numpy as np
import pybullet as p
import argparse
import os
import sys
import cv2
import math
import matplotlib.pyplot as plt
import time


class GrasppingScenarios():

    def __init__(self,network_model="GGCNN"):
        
        self.network_model = network_model

        if (network_model == "GR_ConvNet"):
            ##### GR-ConvNet #####
            self.IMG_SIZE = 224
            self.network_path = 'trained_models/GR_ConvNet/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
            sys.path.append('trained_models/GR_ConvNet')
        else:
            # you need to add your network here!
            print("The selected network has not been implemented yet!")
            exit() 
        
        
        self.CAM_Z = 1.9
        self.depth_radius = 1
        self.ATTEMPTS = 3
        self.fig = plt.figure(figsize=(10, 10))
       
                
    def draw_predicted_grasp(self,grasps,color = [0,0,1],lineIDs = []):
        x, y, z, yaw, opening_len, obj_height = grasps

        gripper_size = opening_len + 0.02 
        finger_size = 0.075
        # lineIDs = []
        lineIDs.append(p.addUserDebugLine([x, y, z], [x, y, z+0.15],color, lineWidth=6))

        lineIDs.append(p.addUserDebugLine([x - gripper_size*math.sin(yaw), y - gripper_size*math.cos(yaw), z], 
                                    [x + gripper_size*math.sin(yaw), y + gripper_size*math.cos(yaw), z], 
                                    color, lineWidth=6))

        lineIDs.append(p.addUserDebugLine([x - gripper_size*math.sin(yaw), y - gripper_size*math.cos(yaw), z], 
                                    [x - gripper_size*math.sin(yaw), y - gripper_size*math.cos(yaw), z-finger_size], 
                                    color, lineWidth=6))
        lineIDs.append(p.addUserDebugLine([x + gripper_size*math.sin(yaw), y + gripper_size*math.cos(yaw), z], 
                                    [x + gripper_size*math.sin(yaw), y + gripper_size*math.cos(yaw), z-finger_size], 
                                    color, lineWidth=6))
        
        return lineIDs
    
    def remove_drawing(self,lineIDs):
        for line in lineIDs:
            p.removeUserDebugItem(line)
    
    def dummy_simulation_steps(self,n):
        for _ in range(n):
            p.stepSimulation()

    def is_there_any_object(self,camera):
        self.dummy_simulation_steps(10)
        rgb, depth, _ = camera.get_cam_img()
        #print ("min RGB = ", rgb.min(), "max RGB = ", rgb.max(), "rgb.avg() = ", np.average(rgb))
        #print ("min depth = ", depth.min(), "max depth = ", depth.max())
        if (np.average(rgb) > 238) :
            return False
        else:
            return True
                    
    def isolated_obj_scenario(self,runs, device, vis, output, debug):

        objects = YcbObjects('objects/ycb_objects',
                            mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
                            mod_stiffness=['Strawberry'])

        
        ## reporting the results at the end of experiments in the results folder
        data = IsolatedObjData(objects.obj_names, runs, 'results')

        ## camera settings: cam_pos, cam_target, near, far, size, fov
        center_x, center_y, center_z = 0.05, -0.52, self.CAM_Z
        camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (self.IMG_SIZE, self.IMG_SIZE), 40)
        env = Environment(camera, vis=vis, debug=debug, finger_length=0.06)
        
        generator = GraspGenerator(self.network_path, camera, self.depth_radius, self.fig, self.IMG_SIZE, self.network_model, device)
        
        objects.shuffle_objects()
        for i in range(runs):
            print("----------- run ", i+1, " -----------")
            print ("network model = ", self.network_model)
            print ("size of input image (W, H) = (", self.IMG_SIZE," ," ,self.IMG_SIZE, ")")

            for obj_name in objects.obj_names:
                print(obj_name)

                env.reset_robot()          
                env.remove_all_obj()                        
               
                path, mod_orn, mod_stiffness = objects.get_obj_info(obj_name)
                env.load_isolated_obj(path, mod_orn, mod_stiffness)
                self.dummy_simulation_steps(20)

                number_of_attempts = self.ATTEMPTS
                number_of_failures = 0
                idx = 0 ## select the best grasp configuration
                failed_grasp_counter = 0
                while self.is_there_any_object(camera) and number_of_failures < number_of_attempts:     
                    
                    bgr, depth, _ = camera.get_cam_img()

                    ##convert BGR to RGB
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                                      
                    grasps, save_name = generator.predict_grasp( rgb, depth, n_grasps=number_of_attempts, show_output=output)
                    if (grasps == []):
                        self.dummy_simulation_steps(10)
                        print ("could not find a grasp point!")
                        if failed_grasp_counter > 3:
                            print("Failed to find a grasp points > 3 times. Skipping.")
                            break
                        
                        failed_grasp_counter += 1
                      
                        continue
                        continue


                    if vis:
                        LID =[]
                        for g in grasps:
                            LID = self.draw_predicted_grasp(g,color=[1,0,1],lineIDs=LID)
                        time.sleep(0.5)
                        self.remove_drawing(LID)
                        self.dummy_simulation_steps(10)
                        

                    #print ("grasps.length = ", len(grasps))
                    if (idx > len(grasps)-1):                            
                        continue  

                    lineIDs = self.draw_predicted_grasp(grasps[idx])

                    x, y, z, yaw, opening_len, obj_height = grasps[idx]
                    succes_grasp, succes_target = env.grasp((x, y, z), yaw, opening_len, obj_height)

                    data.add_try(obj_name)
                   
                    if succes_grasp:
                        data.add_succes_grasp(obj_name)
                    if succes_target:
                        data.add_succes_target(obj_name)

                    ## remove visualized grasp configuration 
                    if vis:
                        self.remove_drawing(lineIDs)

                    env.reset_robot()
                    
                    if succes_target:
                        number_of_failures = 0
                        if vis:
                            debugID = p.addUserDebugText("success", [-0.0, -0.9, 0.8], [0,0.50,0], textSize=2)
                            time.sleep(0.25)
                            p.removeUserDebugItem(debugID)
                        
                        if save_name is not None:
                            os.rename(save_name + '.png', save_name + f'_SUCCESS_grasp{i}.png')
                        

                    else:
                        number_of_failures += 1
                        idx +=1        
                        #env.reset_robot() 
                        # env.remove_all_obj()                        
                
                        if vis:
                            debugID = p.addUserDebugText("failed", [-0.0, -0.9, 0.8], [0.5,0,0], textSize=2)
                            time.sleep(0.25)
                            p.removeUserDebugItem(debugID)

        data.write_json(self.network_model)
        summarize(data.save_dir, runs, self.network_model)


    def packed_or_pile_scenario(self,runs, scenario, device, vis, output, debug):
        
        ## reporting the results at the end of experiments in the results folder
        number_of_objects = 5
        if scenario=='packed':
            objects = YcbObjects('objects/ycb_objects',
                            mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
                            mod_stiffness=['Strawberry'],
                            exclude=['CrackerBox'])
            
            data = PackPileData(number_of_objects, runs, 'results', 'packed')

        elif scenario=='pile':
            objects = YcbObjects('objects/ycb_objects',
                            mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
                            mod_stiffness=['Strawberry'],
                            exclude=['CrackerBox'])

            data = PackPileData(number_of_objects, runs, 'results', 'pile')


        center_x, center_y, center_z = 0.05, -0.52, self.CAM_Z
        camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (self.IMG_SIZE, self.IMG_SIZE), 40)
        env = Environment(camera, vis=vis, debug=debug, finger_length=0.06)
        
        generator = GraspGenerator(self.network_path, camera, self.depth_radius, self.fig, self.IMG_SIZE, self.network_model, device)

        
        for i in range(runs):
            env.remove_all_obj()
            env.reset_robot()              
            print("----------- run ", i+1, " -----------")
            print ("network model = ", self.network_model)
            print ("size of input image (W, H) = (", self.IMG_SIZE," ," ,self.IMG_SIZE, ")")

            
            if vis:
                debugID = p.addUserDebugText(f'Experiment {i+1}', [-0.0, -0.9, 0.8], [0,0,255], textSize=2)
                time.sleep(0.5)
                p.removeUserDebugItem(debugID)

            number_of_failures = 0
            objects.shuffle_objects()

            info = objects.get_n_first_obj_info(number_of_objects)

            if scenario=='packed':
                env.create_packed(info)
            elif scenario=='pile':
                env.create_pile(info)
                
            #self.dummy_simulation_steps(50)

            number_of_failures = 0
            ATTEMPTS = 4
            number_of_attempts = ATTEMPTS
            
            while self.is_there_any_object(camera) and number_of_failures < number_of_attempts:                
                #env.move_arm_away()
                try: 
                    idx = 0 ## select the best grasp configuration
                    for i in range(number_of_attempts):
                        data.add_try()  
                        rgb, depth, _ = camera.get_cam_img()
                        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                        
                        grasps, save_name = generator.predict_grasp( rgb, depth, n_grasps=number_of_attempts, show_output=output)
                        if (grasps == []):
                                self.dummy_simulation_steps(30)
                                print ("could not find a grasp point!")                            
                                continue
                        
                        if vis:
                            LID =[]
                            for g in grasps:
                                LID = self.draw_predicted_grasp(g,color=[1,0,1],lineIDs=LID)
                            time.sleep(0.5)
                            self.remove_drawing(LID)
                            self.dummy_simulation_steps(10)
                            
                        #print ("grasps.length = ", len(grasps))
                        if (idx > len(grasps)-1):  
                            print ("idx = ", idx)
                            if len(grasps)-1 == 0 :
                                idx=0
                            else:
                                continue  

                        lineIDs = self.draw_predicted_grasp(grasps[idx])
                        
                        ## perform object grasping and manipulation : 
                        #### succes_grasp means if the grasp was successful, 
                        #### succes_target means if the target object placed in the target basket successfully
                        x, y, z, yaw, opening_len, obj_height = grasps[idx]
                        succes_grasp, succes_target = env.grasp((x, y, z), yaw, opening_len, obj_height)
                        
                        if succes_grasp:
                            data.add_succes_grasp()                     

                        ## remove visualized grasp configuration 
                        if vis:
                            self.remove_drawing(lineIDs)

                        env.reset_robot()
                            
                        if succes_target:
                            data.add_succes_target()
                            number_of_failures = 0

                            if vis:
                                debugID = p.addUserDebugText("success", [-0.0, -0.9, 0.8], [0,0.50,0], textSize=2)
                                time.sleep(0.25)
                                p.removeUserDebugItem(debugID)
                                
                            if save_name is not None:
                                    os.rename(save_name + '.png', save_name + f'_SUCCESS_grasp{i}.png')
                                
                        else:
                            #env.reset_robot()   
                            number_of_failures += 1
                            idx +=1     
                                                    
                            if vis:
                                debugID = p.addUserDebugText("failed", [-0.0, -0.9, 0.8], [0.5,0,0], textSize=2)
                                time.sleep(0.25)
                                p.removeUserDebugItem(debugID)

                            if number_of_failures == number_of_attempts:
                                if vis:
                                    debugID = p.addUserDebugText("breaking point",  [-0.0, -0.9, 0.8], [0.5,0,0], textSize=2)
                                    time.sleep(0.25)
                                    p.removeUserDebugItem(debugID)
                                
                                break

                        #env.reset_all_obj()
        
                except:
                    print("An exception occurred during the experiment!!!")
                    env.reset_robot()
                    #print ("#objects = ", len(env.obj_ids), "#failed = ", number_of_failures , "#attempts =", number_of_attempts)
        
        data.summarize() 
        
def parse_args():
    parser = argparse.ArgumentParser(description='Grasping demo')

    parser.add_argument('--scenario', type=str, default='isolated', help='Grasping scenario (isolated/packed/pile)')
    parser.add_argument('--network', type=str, default='GR_ConvNet', help='Network model (GR_ConvNet/...)')

    parser.add_argument('--runs', type=int, default=1, help='Number of runs the scenario is executed')
    parser.add_argument('--attempts', type=int, default=3, help='Number of attempts in case grasping failed')

    parser.add_argument('--save-network-output', dest='output', type=bool, default=False,
                        help='Save network output (True/False)')

    parser.add_argument('--device', type=str, default='cpu', help='device (cpu/gpu)')
    parser.add_argument('--vis', type=bool, default=True, help='vis (True/False)')
    parser.add_argument('--report', type=bool, default=True, help='report (True/False)')

                        
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    output = args.output
    runs = args.runs
    ATTEMPTS = args.attempts
    device=args.device
    vis=args.vis
    report=args.report
    
    grasp = GrasppingScenarios(args.network)

    if args.scenario == 'isolated':
        grasp.isolated_obj_scenario(runs, device, vis, output=output, debug=False)
    elif args.scenario == 'packed':
        grasp.packed_or_pile_scenario(runs, args.scenario, device, vis, output=output, debug=False)
    elif args.scenario == 'pile':
        grasp.packed_or_pile_scenario(runs, args.scenario, device, vis, output=output, debug=False)

