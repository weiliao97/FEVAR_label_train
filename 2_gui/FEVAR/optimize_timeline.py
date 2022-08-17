# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 10:10:07 2022

@author: 320190618
"""

# import libraries 
import argparse
import os
import numpy as np 
import pandas as pd 
import torch
import torchvision.transforms as transforms
from torch.utils import data
from sklearn.metrics import accuracy_score
from model import *
from prepare_data import *
from utils import *
import tkinter as tk 
import tkinter.messagebox
import customtkinter
from tkinter import *
from PIL import ImageTk, Image
import cv2 
from tkinter import filedialog

# ============ user inputs ============
weights_name = 'cnn_encoder_acc_0.90_epoch148.pth'
# validation set data directory 
datapath = 'C:/Users/320190618/Documents/code_compile/2_gui/FEVAR_only'
full_name = ['BIDMC 20211104_Pat5005', 'Hamburg UKE 20200218_Pat1003', 'Utrecht-UMCU 20210331_Pat2019']
# which index to eval first 
eval_index = 0
# whether perform prediction correction 
use_correction = True
label_dict = {0: 'Navigation', 1: 'Sheath delivery', 2: 'Unsheathed', 3:'Cannulation', 4: 'Final deployment'}
# ============ user inputs ============
 
# creat model 
model = ResCNNEncoder(fc_hidden1=256, fc_hidden2=128, drop_p=0.2, CNN_embed_dim=64)
model.load_state_dict(torch.load(os.path.join(datapath, weights_name), map_location=torch.device('cpu')))
# get validation data info
to_eval = [ f.path for f in os.scandir(datapath) if f.is_dir() and 'case' in f.path]
dev_names, dev_length, dev_y, dev_time = get_name_length(datapath, to_eval[eval_index])
# get validation results 
all_t, all_y_pred, X_all = get_eval_results(model, datapath, to_eval, eval_index, use_correction=True)
#%% GUI Initialization 
colormap = {0: '#4e8df2', 1:'#8349b8', 2:'#ad4e74', 3:'#569482', 4:'#43548c'}
customtkinter.set_appearance_mode("Light")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
transform = transforms.ToPILImage()

def format_timepoints(t):
    '''
    Parameters
    ----------
    t : TYPE, int
        DESCRIPTION. e.g. 102011

    Returns
    -------
    formatted : TYPE. str
        DESCRIPTION. formatted timestamps e.g. '10:20:11'

    '''
    hours = int(t/10000)
    minutes = int((t - hours*10000)/100)
    seconds = t - hours*10000 - minutes*100
    formatted = "%d:%02d:%02d" % (hours, minutes, seconds)
    
    return formatted




class App(customtkinter.CTk):

    WIDTH = 780
    HEIGHT = 520

    def __init__(self):
        super().__init__()


        self.title("FEVAR procedure annotation")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # call .on_closing() when app gets closed

        # ============ create 3 frames ============

        # configure grid layout (2x1)
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=6)
        self.grid_rowconfigure(0, weight=1)

        self.frame_left = customtkinter.CTkFrame(master=self, corner_radius=0)
        self.frame_left.grid(row=0, column=0, sticky="nswe")
        self.frame_middle = customtkinter.CTkFrame(master=self,  corner_radius=0)
        self.frame_middle.grid(row=0, column=1, sticky="nswe")
        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=2, sticky="nswe", padx=0, pady=0)
        
        # ============ frame_left ============

        # configure grid layout (1x11)
        
        self.frame_left.grid_rowconfigure(0, weight=1)   # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(1, weight=8)
        self.frame_left.grid_rowconfigure(2, weight=1)  # empty row with minsize as spacing
        #
        self.frame_middle.grid_rowconfigure(0, weight = 1)   # empty row with minsize as 
        self.frame_middle.grid_rowconfigure(1, weight = 8)
        self.frame_middle.grid_rowconfigure(2, weight = 1)
        self.canvas = customtkinter.CTkCanvas(master=self.frame_middle, width = 90, height=500, bd=0, bg="#d6d6d6")
        self.canvas.pack(side = LEFT, fill = BOTH, expand = True)
        # grid(row=1, column=0, sticky=N+S+W)
        self.canvas.bind("<Configure>", self.reposition)

        # ============ frame_right ============

        # configure grid layout (3x7)
        self.frame_right.rowconfigure((0, 1, 2, 3), weight=4)
        self.frame_right.rowconfigure(7, weight=1)
        self.frame_right.columnconfigure((0, 1), weight=5)
        self.frame_right.columnconfigure(2, weight=2)
        self.label_info_1 = customtkinter.CTkLabel(master=self.frame_right,
                                                    text="Analyzing BIDMC 20211104_Pat5005" ,
                                                    height=30,
                                                    corner_radius = 5, 
                                                    fg_color="#b5b4b0",  # <- custom tuple-color
                                                    justify=tkinter.LEFT)
        self.label_info_1.grid(column=0, row=7, columnspan = 2, sticky="nwe", padx=15, pady=5)

        # ============ frame_left ============
        self.canvas_l = customtkinter.CTkCanvas(master=self.frame_left, width = 150, height=500, bd=0, bg="#d6d6d6")
        self.canvas_l.pack(side = RIGHT, fill = BOTH, expand = True)


        # ============ frame_right ============
        self.image_canvas = customtkinter.CTkCanvas(master=self.frame_right, height=512, width=512, bd=0)
        self.image_canvas.grid(row=0, column=0, rowspan = 4, columnspan=2, pady=20, padx=(20, 10), sticky=N+S+E+W)
        # legend on row 0, column 2 
        self.legend_canvas = customtkinter.CTkCanvas(master=self.frame_right, height=100, width=50, bd=0, bg="#d6d6d6")
        self.legend_canvas.grid(row=0, column=2, rowspan = 2, columnspan=1, pady=20, padx=(10, 20), sticky=N+S+E+W)
        self.legend_canvas.bind("<Configure>", self.legend_pos)
        # line and text for the legend 
        lc_width = self.legend_canvas.winfo_width()
        lc_height = self.legend_canvas.winfo_height()
        
        text_gap = [0 + 0.7/4*i for i in range(5)]
        self.legend_l = []
        self.legend_t = []
        for i in range(5):
            self.legend_l.append(self.legend_canvas.create_line(190*0.1, 204*(0.15+text_gap[i]), 190*0.6,  204*(0.15+text_gap[i]),  width=5, fill = colormap[i],))
            self.legend_t.append(self.legend_canvas.create_text(190*0.65, 204*(0.15+text_gap[i]), text = label_dict[i], anchor=W, font=('Helvetica','10','bold')))
            
        self.entry = customtkinter.CTkEntry(master=self.frame_right,
                                            width=120,
                                            placeholder_text="users can enter frame rate for display etc.")
        self.entry.grid(row=9, column=0, columnspan=2, pady=15, padx=20, sticky="we")

        self.button_5 = customtkinter.CTkButton(master=self.frame_right,
                                                text="Enter sth",
                                                border_width=2,  # <- custom border_width
                                                fg_color=None,  # <- no fg_color
                                                command=self.button_event)
        self.button_5.grid(row=9, column=2, columnspan=1, pady=15, padx=20, sticky="we")
        
        self.entry_2 = customtkinter.CTkEntry(master=self.frame_right,
                                            width=120,
                                            placeholder_text="Open a folder to analyze")
        self.entry_2.grid(row=8, column=0, columnspan=2, pady=5, padx=20, sticky="we")

        self.button_7 = customtkinter.CTkButton(master=self.frame_right,
                                                text="Open",
                                                border_width=2,  # <- custom border_width
                                                fg_color=None,  # <- no fg_color
                                                command=self.open_dir)
        self.button_7.grid(row=8, column=2, columnspan=1, pady=5, padx=20, sticky="we")
        
        
        self.button_6 = customtkinter.CTkButton(master=self.frame_right,
                                                text="Analyze next record",
                                                border_width=2,  # <- custom border_width
                                                fg_color=None,  # <- no fg_color
                                                command=self.refresh_timeline)
        self.button_6.grid(row=7, column=2, columnspan=1, pady=5, padx=20, sticky="we")
        self.eval_ind = 0 
        
        # ============ organize model predictions and time stamps ============
        ## the first evaluation case done before calling the app
        self.classification_results = all_y_pred
        self.timepoints = [all_t[i][0][0] for i in range(len(all_t))]
        Diff = [self.timepoints[i] - self.timepoints[i-1] for i in range(1, len(self.timepoints))]
        # x00, y00, x10, y10 = 3, 58, 15, 70
        gap = [Diff[i]*(621*0.78)/sum(Diff) for i in range(len(Diff))]
        self.button_width = [min(20, gap[i]) for i in range(len(gap))]
        self.button_width.append(20)
        # x01, y01, x11, y11 = 3, 58+gap[0], 15, 70+gap[0]
        # creat a series of points on where the 
        # startx, starty = 3, 58
        self.points = [621*0.12]
        self.button_loc = [60]
        for i in range(len(Diff)):
            self.points.append(self.points[-1]+ gap[i])
            self.button_loc.append(self.button_loc[-1] + gap[i]/1.25)
            
        self.totalimage = []
        for X in X_all:
            curr_l = []
            for f in range(X.shape[1]): #(1, 7, 1, 512, 512 )
                # numpy array to PIL then to tkinter photoimage
                curr_l.append(ImageTk.PhotoImage(transform(X[0, f, 0])))
            self.totalimage.append(curr_l)
        self.listimg = self.totalimage[1]
    
        
        self.line = self.canvas.create_line(9, 621*0.1, 9, 621*0.95, arrow=tk.LAST, width=3)
        self.oval_l = []
        self.text_l = []
        for i in range(len(self.points)):
            self.oval_l.append(self.canvas.create_oval(3, self.points[i], 15, self.points[i] + 12, fill=colormap[self.classification_results[i]]))
            self.text_l.append(self.canvas.create_text(50, self.points[i]+6, text = format_timepoints(self.timepoints[i]), anchor=tk.CENTER))
        #
        
        self.label_1 = customtkinter.CTkLabel(master=self.canvas_l,
                                              text="Labeled Timeline",
                                              text_font=("Roboto Medium", -16))  # font name and size in px
        self.label_1.place(relx=0.95, rely=0.04, anchor=NE)
        self.button = []
        for i in range(len(self.points)):
            self.button.append(customtkinter.CTkButton(master=self.canvas_l,
                                                    text= label_dict[self.classification_results[i]],
                                                     height = self.button_width[i], width =184*0.65, 
                                                    fg_color = colormap[self.classification_results[i]], 
                                                    command=lambda i=i: self.open_frames(i)))
            self.button[i].place(relx=0.9, rely=self.points[i]/621, anchor= NE)
            # 
        self.arrow = None 

    def open_dir(self):
        '''
        Specify a folder and redo evaluation 

        '''
        self.directory = filedialog.askdirectory()
        self.label_info_1.configure(text='Opening folder %s'%self.directory)
        curr_f = self.directory.split('/')[-1]
        all_f = [f.split('\\')[-1] for f in to_eval]
        # do inference
        curr_ind = all_f.index(curr_f)
        # find eval_ind and do refresh_timeline
        self.refresh_timeline(eval_ind=curr_ind)
        
    def legend_pos(self, event):
        '''
        event: window size change event
        If window size changes, re-place legend

        '''
        text_gap = [0 + 0.7/4*i for i in range(5)]
        for i in range(5):
            self.legend_canvas.coords(self.legend_l[i], event.width*0.1, event.height*(0.15+text_gap[i]), event.width*0.3,  event.height*(0.15+text_gap[i]))
            self.legend_canvas.coords(self.legend_t[i], event.width*0.35, event.height*(0.15+text_gap[i]))
            
    def reposition(self, event):
        '''
        event: window size change event
        If window size changes, re-place timeline 

        '''
        self.canvas.coords(self.line, 9, event.height*0.1, 9, event.height*0.95)
        self.points = [event.height*0.12]
        Diff = [self.timepoints[i] - self.timepoints[i-1] for i in range(1, len(self.timepoints))]
        gap = [Diff[i]*(event.height*0.78)/sum(Diff) for i in range(len(Diff))]
        for i in range(len(Diff)):
            self.points.append(self.points[-1]+ gap[i])
        for i, o in enumerate(self.oval_l):
            self.canvas.coords(o, 3, self.points[i], 15, self.points[i] + 12)
        for i, t in enumerate(self.text_l):
            self.canvas.coords(t, 50, self.points[i]+6)

    def refresh_timeline(self, eval_ind=None): # simple version 
        '''
        When analyze next record happens, eval_ind changes

        '''
        if eval_ind == None:
            self.eval_ind +=1 
            self.label_info_1.configure(text='Model inference in progress')
        else: 
            self.eval_ind = eval_ind
        # erase previos line, oval, text, button, label 
        if self.eval_ind > 0:
            self.canvas.delete('all')
            for b in self.button:
                b.destroy()
        ## do model inference and save image data, time, results 
        ##maybe get the canvas real length and then create a line 
        # create these annotations on demand     
        # display Performing model inference.....
        all_t, all_y_pred, X_all =get_eval_results(model, datapath, to_eval, self.eval_ind, use_correction=True)
        self.label_info_1.configure(text='Timeline refreshed! Analyzing %s'%full_name[self.eval_ind])
        # display Timeline refreshed, analyzing xxx 
        self.classification_results = all_y_pred
        self.timepoints = [all_t[i][0][0] for i in range(len(all_t))]
        self.cwidth = self.canvas.winfo_width()
        self.cheight = self.canvas.winfo_height()
        Diff = [self.timepoints[i] - self.timepoints[i-1] for i in range(1, len(self.timepoints))]
        # x00, y00, x10, y10 = 3, 58, 15, 70
        gap = [Diff[i]*(self.cheight*0.78)/sum(Diff) for i in range(len(Diff))]
        self.button_width = [min(20, gap[i]) for i in range(len(gap))]
        self.button_width.append(20)
\
        # creat a series for the time stamps 
        # startx, starty = 3, 58
        self.points = [self.cheight*0.12]
        self.button_loc = [58]
        for i in range(len(Diff)):
            self.points.append(self.points[-1]+ gap[i])
            self.button_loc.append(self.button_loc[-1] + gap[i]/1.4)
            
        self.totalimage = []
        for X in X_all:
            curr_l = []
            for f in range(X.shape[1]): #(1, 7, 1, 512, 512 )
                # numpy array to PIL then to tkinter photoimage
                curr_l.append(ImageTk.PhotoImage(transform(X[0, f, 0])))
            self.totalimage.append(curr_l)
        self.listimg = self.totalimage[1]
    
        
        self.line = self.canvas.create_line(self.cwidth/10, self.cheight*0.1, self.cwidth/10, self.cheight*0.95, arrow=tk.LAST, width=3)
        
        self.oval_l = []
        self.text_l = []
        for i in range(len(self.points)):
            self.oval_l.append(self.canvas.create_oval(3, self.points[i], 15, self.points[i] + 12, fill=colormap[self.classification_results[i]]))
            self.text_l.append(self.canvas.create_text(50, self.points[i]+6, text = format_timepoints(self.timepoints[i]), anchor=tk.CENTER))
        #
        if self.eval_ind == 0:
            self.label_1 = customtkinter.CTkLabel(master=self.frame_left,
                                                  text="Labeled Timeline",
                                                  text_font=("Roboto Medium", -16))  # font name and size in px
            self.label_1.grid(row=1, column=0, pady=10, padx=10)
        self.button = []
        for i in range(len(self.points)):
            self.button.append(customtkinter.CTkButton(master=self.canvas_l,
                                                    text= label_dict[self.classification_results[i]],
                                                     height = self.button_width[i], width =self.cwidth*0.65,
                                                    fg_color = colormap[self.classification_results[i]], 
                                                    command=lambda i=i: self.open_frames(i)))
            self.button[i].place(relx=0.9, rely= self.points[i]/self.cheight, anchor=NE)
            # place(x=25, y=self.button_loc[i]+6)
        
    
    def next_img(self, frame_i):   # takes the current scale position as an argument
        '''
        Parameters
        ----------
        frame_i : TYPE, int
            DESCRIPTION, fram index from the scroller
        '''
        # delete previous image
        self.image_canvas.delete('image')
        # create next image
        # maybe get true canvas size before setting center anchor 
        # also resize frames 
        width = self.image_canvas.winfo_width()
        height = self.image_canvas.winfo_height()
        self.image_canvas.create_image(width/2, height/2, anchor=CENTER, image=self.listimg[int(frame_i) -1], tags='image')
        # self.image_canvas.itemconfig(self.image_on_canvas, image=self.listimg[int(frame_i) -1])

    def button_event(self):
        print("Button pressed")
    
    def open_frames(self, num):
        '''
        Parameters
        ----------
        num : TYPE, int
            DESCRIPTION, button index 
        
        '''
        self.listimg = self.totalimage[num]
        # based on which button is pressed, num identidies in index 
        self.next_img(2)
        # update scroller
        self.slider_1 = customtkinter.CTkSlider(master=self.frame_right,
                                                from_=1,
                                                to=len(self.listimg),
                                                number_of_steps=len(self.listimg) -1,
                                                command=self.next_img)
        self.slider_1.grid(row=4, column=0, columnspan=2, pady=10, padx=20, sticky="we")
        # also put an arrown nearby the button  
        if self.arrow != None:
            self.canvas.delete(self.arrow)
        self.cwidth = self.canvas.winfo_width()
        self.cheight = self.canvas.winfo_height()
        self.arrow = self.canvas.create_line(self.cwidth*9/10, self.points[num]+6, self.cwidth*5/8, self.points[num]+6, arrow=tk.LAST, width=5, fill='#075f91')  
        
        

    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def on_closing(self, event=0):
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()


