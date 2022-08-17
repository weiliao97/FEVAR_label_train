# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:04:10 2022

@author: 320190618
"""
#%%
import tkinter as tk 
import tkinter.messagebox
import customtkinter
from tkinter import *
import os
from PIL import ImageTk, Image
import numpy as np 

classification_results = [0, 0, 1, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 4, 4]
label_dict = {0: 'Guidewire only', 1: 'Sheath delivery', 2: 'Unsheathed', 3:'Branch stent deployed', 4: 'Final deployment'}
colormap = {0: '#4e8df2', 1:'#8349b8', 2:'#ad4e74', 3:'#569482', 4:'#43548c'}
timepoints = [92641, 93541, 94641, 100501, 101221, 102428, 103403, 103624, 111058, 111909, 112935, 113901, 120055, 132550, 133706]
Diff = [timepoints[i] - timepoints[i-1] for i in range(1, 15)]
x00, y00, x10, y10 = 3, 58, 15, 70
gap = [Diff[i]*500/sum(Diff) for i in range(len(Diff))]
button_width = [min(20, gap[i]) for i in range(len(gap))]
button_width.append(20)
x01, y01, x11, y11 = 3, 58+gap[0], 15, 70+gap[0]
# creat a series of points on where the 
startx, starty = 3, 58
points = [58]
button_loc = [58]
for i in range(len(Diff)):
    points.append(points[-1] + gap[i])
    button_loc.append(button_loc[-1] + gap[i]/1.25)

def format_timepoints(t):
    hours = int(t/10000)
    minutes = int((t - hours*10000)/100)
    seconds = t - hours*10000 - minutes*100
    formatted = "%d:%02d:%02d" % (hours, minutes, seconds)
    
    return formatted

import cv2 
def pad_img(img):
    if img.shape[0] > img.shape[1]:
        diff = img.shape[0] - img.shape[1] # 968 - 750 = 218
        ld = int(diff/2)
        rd = diff - ld
        padded_img = cv2.copyMakeBorder(img, 0, 0, ld, rd, borderType= cv2.BORDER_CONSTANT)
    else:
        diff = img.shape[1] - img.shape[0]
        td = int(diff/2)
        bd = diff -td
        padded_img = cv2.copyMakeBorder(img, td, bd, 0, 0, cv2.BORDER_CONSTANT)
    return padded_img

def fit_img(img):
    img_8 = (np.array(Image.open(img))*255/65535).astype('uint8')
    if img_8.shape[0] != img_8.shape[1]:
        img_8= pad_img(img_8)
    img_resize = cv2.resize(img_8, (350, 350))
    return Image.fromarray(img_resize)

a = fit_img('C:/Users/320190618/Documents/Video_annot/0000.png')
b = fit_img('C:/Users/320190618/Documents/Video_annot/0001.png')
c = fit_img('C:/Users/320190618/Documents/Video_annot/0002.png')
d = fit_img('C:/Users/320190618/Documents/Video_annot/0003.png')
e = fit_img('C:/Users/320190618/Documents/Video_annot/0004.png')
f = fit_img('C:/Users/320190618/Documents/Video_annot/0005.png')

customtkinter.set_appearance_mode("Light")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
class App(customtkinter.CTk):

    WIDTH = 780
    HEIGHT = 520

    def __init__(self):
        super().__init__()


        self.title("FORS procedure annotation")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # call .on_closing() when app gets closed

        # ============ create two frames ============

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
        
        self.frame_left.grid_rowconfigure(0, minsize=30)   # empty row with minsize as spacing
        # self.frame_left.grid_rowconfigure(10, minsize=20) 
        # self.frame_left.grid_rowconfigure(5, weight=1)  # empty row as spacing
        # self.frame_left.grid_rowconfigure(8, minsize=10)    # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(16, minsize=10)  # empty row with minsize as spacing
        #
        self.frame_middle.grid_rowconfigure(0, minsize=30)   # empty row with minsize as spacing
        self.frame_middle.grid_rowconfigure(2, minsize=50)
        self.canvas = customtkinter.CTkCanvas(master=self.frame_middle, height=590, width=90, bd=0, bg="#d6d6d6")
        self.canvas.create_line(9, 50, 9, 590,  arrow=tk.LAST, width=3)
        self.canvas.grid(row=1, column=0)
        for i in range(len(points)):
            self.canvas.create_oval(3, points[i], 15, points[i] + 12, fill=colormap[classification_results[i]])
            self.canvas.create_text(50, points[i]+6, text = format_timepoints(timepoints[i]), anchor=tk.CENTER)
        #
        self.label_1 = customtkinter.CTkLabel(master=self.frame_left,
                                              text="Labeled Timeline",
                                              text_font=("Roboto Medium", -16))  # font name and size in px
        self.label_1.grid(row=1, column=0, pady=10, padx=10)
        self.button = []
        for i in range(len(points)):
            self.button.append(customtkinter.CTkButton(master=self.frame_left,
                                                    text= label_dict[classification_results[i]],
                                                     height = button_width[i],
                                                    fg_color = colormap[classification_results[i]], 
                                                    command=lambda i=i: self.open_frames(i)))
            self.button[i].place(x=25, y=button_loc[i]+6)
            # grid(row=i+2, column=0, pady=1, padx=20)

        # self.button_2 = customtkinter.CTkButton(master=self.frame_left,
        #                                         text="CTkButton",
        #                                         command=self.button_event)
        # self.button_2.grid(row=3, column=0, pady=10, padx=20)

        # self.button_3 = customtkinter.CTkButton(master=self.frame_left,
        #                                         text="CTkButton",
        #                                         command=self.button_event)
        # self.button_3.grid(row=4, column=0, pady=10, padx=20)

        # self.label_mode = customtkinter.CTkLabel(master=self.frame_left, text="Appearance Mode:")
        # self.label_mode.grid(row=9, column=0, pady=0, padx=20, sticky="w")

        # self.optionmenu_1 = customtkinter.CTkOptionMenu(master=self.frame_left,
        #                                                 values=["Light", "Dark", "System"],
        #                                                 command=self.change_appearance_mode)
        # self.optionmenu_1.grid(row=10, column=0, pady=10, padx=20, sticky="w")

        # ============ frame_right ============

        # configure grid layout (3x7)
        self.frame_right.rowconfigure((0, 1, 2, 3), weight=4)
        self.frame_right.rowconfigure(7, weight=1)
        self.frame_right.columnconfigure((0, 1), weight=5)
        self.frame_right.columnconfigure(2, weight=1)

        # self.frame_info = customtkinter.CTkFrame(master=self.frame_right)
        # self.frame_info.grid(row=0, column=0, columnspan=2, rowspan=4, pady=20, padx=20, sticky="nsew")
        # # ============ frame_info ============

        # # configure grid layout (1x1)
        # self.frame_info.rowconfigure(0, weight=1)
        # self.frame_info.columnconfigure(0, weight=1)

        self.label_info_1 = customtkinter.CTkLabel(master=self.frame_right,
                                                    text="Analyzing Hamburg UKE 20200211_Pat1001" ,
                                                    height=30,
                                                    corner_radius = 5, 
                                                    fg_color="#b5b4b0",  # <- custom tuple-color
                                                    justify=tkinter.LEFT)
        self.label_info_1.grid(column=0, row=7, columnspan = 2, sticky="nwe", padx=15, pady=5)
        
        # self.scale = Scale(master=root, orient=HORIZONTAL, from_=1, to=len(listimg), resolution=1,
        #               showvalue=False, command=nex_img)
        # scale.pack(side=BOTTOM, fill=X)

        # self.progressbar = customtkinter.CTkProgressBar(master=self.frame_info)
        # self.progressbar.grid(row=1, column=0, sticky="ew", padx=15, pady=15)

        # ============ frame_right ============
        self.image_canvas = customtkinter.CTkCanvas(master=self.frame_right, height=350, width=350, bd=0)
        self.image_canvas.grid(row=0, column=0, rowspan = 4, columnspan=2, pady=5, padx=5, sticky="")

        # self.radio_var = tkinter.IntVar(value=0)

        # self.label_radio_group = customtkinter.CTkLabel(master=self.frame_right,
        #                                                 text="CTkRadioButton Group:")
        # self.label_radio_group.grid(row=0, column=2, columnspan=1, pady=20, padx=10, sticky="")

        # self.radio_button_1 = customtkinter.CTkRadioButton(master=self.frame_right,
        #                                                    variable=self.radio_var,
        #                                                    value=0)
        # self.radio_button_1.grid(row=1, column=2, pady=10, padx=20, sticky="n")

        # self.radio_button_2 = customtkinter.CTkRadioButton(master=self.frame_right,
        #                                                    variable=self.radio_var,
        #                                                    value=1)
        # self.radio_button_2.grid(row=2, column=2, pady=10, padx=20, sticky="n")

        # self.radio_button_3 = customtkinter.CTkRadioButton(master=self.frame_right,
        #                                                    variable=self.radio_var,
        #                                                    value=2)
        # self.radio_button_3.grid(row=3, column=2, pady=10, padx=20, sticky="n")

        self.slider_1 = customtkinter.CTkSlider(master=self.frame_right,
                                                from_=1,
                                                to=3,
                                                number_of_steps=2,
                                                command=self.next_img)
        self.slider_1.grid(row=4, column=0, columnspan=2, pady=10, padx=20, sticky="we")

        # self.slider_2 = customtkinter.CTkSlider(master=self.frame_right,
        #                                         command=self.next_img)
        # self.slider_2.grid(row=5, column=0, columnspan=2, pady=10, padx=20, sticky="we")

        # self.switch_1 = customtkinter.CTkSwitch(master=self.frame_right,
        #                                         text="CTkSwitch")
        # self.switch_1.grid(row=4, column=2, columnspan=1, pady=10, padx=20, sticky="we")

        # self.switch_2 = customtkinter.CTkSwitch(master=self.frame_right,
        #                                         text="CTkSwitch")
        # self.switch_2.grid(row=5, column=2, columnspan=1, pady=10, padx=20, sticky="we")

        # self.combobox_1 = customtkinter.CTkComboBox(master=self.frame_right,
        #                                             values=["Value 1", "Value 2"])
        # self.combobox_1.grid(row=6, column=2, columnspan=1, pady=10, padx=20, sticky="we")

        # self.check_box_1 = customtkinter.CTkCheckBox(master=self.frame_right,
        #                                              text="CTkCheckBox")
        # self.check_box_1.grid(row=6, column=0, pady=10, padx=20, sticky="w")

        # self.check_box_2 = customtkinter.CTkCheckBox(master=self.frame_right,
        #                                              text="CTkCheckBox")
        # self.check_box_2.grid(row=6, column=1, pady=10, padx=20, sticky="w")

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
                                                command=self.button_event)
        self.button_7.grid(row=8, column=2, columnspan=1, pady=5, padx=20, sticky="we")
        
        
        self.button_6 = customtkinter.CTkButton(master=self.frame_right,
                                                text="Analyze next record",
                                                border_width=2,  # <- custom border_width
                                                fg_color=None,  # <- no fg_color
                                                command=self.button_event)
        self.button_6.grid(row=7, column=2, columnspan=1, pady=5, padx=20, sticky="we")

        # set default values
        # self.optionmenu_1.set("Dark")
        # self.button_3.configure(state="disabled", text="Disabled CTkButton")
        # self.combobox_1.set("CTkCombobox")
        # self.radio_button_1.select()
        self.slider_1.set(1)
        # self.slider_2.set(0.7)
        # self.progressbar.set(0.5)
        # self.switch_2.select()
        # self.radio_button_3.configure(state=tkinter.DISABLED)
        # self.check_box_1.configure(state=tkinter.DISABLED, text="CheckBox disabled")
        # self.check_box_2.select()
        image1 = ImageTk.PhotoImage(a)
        image2 = ImageTk.PhotoImage(b)
        image3 = ImageTk.PhotoImage(c)
        image4 = ImageTk.PhotoImage(d)
        image5 = ImageTk.PhotoImage(e)
        image6 = ImageTk.PhotoImage(f)
        l1= [image1, image2, image3]
        l2= [image4, image5, image6]
        self.totalimage = [l1, l2]
        self.listimg = self.totalimage[1]
        # self.image_on_canvas = self.image_canvas.create_image(128, 128, anchor=CENTER, image=self.listimg[0], tags='image')
        
    
    
    def next_img(self, frame_i):   # takes the current scale position as an argument
        # print(frame_i)
        # print(self.listimg[0])
        # delete previous image
        self.image_canvas.delete('image')
        # create next image
        self.image_canvas.create_image(175, 175, anchor=CENTER, image=self.listimg[int(frame_i) -1], tags='image')
        # self.image_canvas.itemconfig(self.image_on_canvas, image=self.listimg[int(frame_i) -1])

    def button_event(self):
        print("Button pressed")
    
    def open_frames(self, num):
        # print(num)
        self.listimg = self.totalimage[num]
        # based on which button is pressed, num identidies in index 
        self.next_img(2)
        

    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def on_closing(self, event=0):
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
