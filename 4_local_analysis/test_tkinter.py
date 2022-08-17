# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 10:55:25 2022

@author: 320190618
"""

from  tkinter import *
from tkinter import filedialog
root = Tk()
root.directory = filedialog.askdirectory()
print (root.directory)
#%%
import tkinter as tk

def redraw(event):
    c.coords(item, 0, 0, event.width, event.height)

root = tk.Tk()
root.geometry("400x400")

c = tk.Canvas(root)
c.pack(fill="both", expand=True)
c.bind("<Configure>", redraw)
item = c.create_line(0,0,400,400)

root.mainloop()
#%%
import tkinter as tk
import random

pressed = False

class Example(tk.Frame):
    def __init__(self, root):
        tk.Frame.__init__(self, root)
        self.canvas = tk.Canvas(self, width=400, height=400, background="bisque")
        self.xsb = tk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.ysb = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.ysb.set, xscrollcommand=self.xsb.set)
        self.canvas.configure(scrollregion=(0,0,1000,1000))

        self.xsb.grid(row=1, column=0, sticky="ew")
        self.ysb.grid(row=0, column=1, sticky="ns")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.fontSize = 10

        #Plot some rectangles
        for n in range(50):
            x0 = random.randint(0, 900)
            y0 = random.randint(50, 900)
            x1 = x0 + random.randint(50, 100)
            y1 = y0 + random.randint(50,100)
            color = ("red", "orange", "yellow", "green", "blue")[random.randint(0,4)]
            self.canvas.create_rectangle(x0,y0,x1,y1, outline="black", fill=color, activefill="black", tags=n)
        self.canvas.create_text(50,10, anchor="nw", text="Click and drag to move the canvas\nScroll to zoom.", font = ("Helvetica", int(self.fontSize)), tags="text")

        # This is what enables using the mouse:
        self.canvas.bind("<ButtonPress-1>", self.move_start)
        self.canvas.bind("<B1-Motion>", self.move_move)

        self.canvas.bind("<ButtonPress-2>", self.pressed2)
        self.canvas.bind("<Motion>", self.move_move2)

        #linux scroll
        self.canvas.bind("<Button-4>", self.zoomerP)
        self.canvas.bind("<Button-5>", self.zoomerM)
        #windows scroll
        self.canvas.bind("<MouseWheel>",self.zoomer)
        # Hack to make zoom work on Windows
        root.bind_all("<MouseWheel>",self.zoomer)

    #move
    def move_start(self, event):
        self.canvas.scan_mark(event.x, event.y)
    def move_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    #move
    def pressed2(self, event):
        global pressed
        pressed = not pressed
        self.canvas.scan_mark(event.x, event.y)
    def move_move2(self, event):
        if pressed:
            self.canvas.scan_dragto(event.x, event.y, gain=1)

    #windows zoom
    def zoomer(self,event):
        if (event.delta > 0):
            self.canvas.scale("all", event.x, event.y, 1.1, 1.1)
            self.fontSize = self.fontSize * 1.1
        elif (event.delta < 0):
            self.canvas.scale("all", event.x, event.y, 0.9, 0.9)
            self.fontSize = self.fontSize * 0.9
        self.canvas.configure(scrollregion = self.canvas.bbox("all"))
        for child_widget in self.canvas.find_withtag("text"):
            self.canvas.itemconfigure(child_widget, font=("Helvetica", int(self.fontSize)))
        print(self.fontSize)

    #linux zoom
    def zoomerP(self,event):
        self.canvas.scale("all", event.x, event.y, 1.1, 1.1)
        self.canvas.configure(scrollregion = self.canvas.bbox("all"))
    def zoomerM(self,event):
        self.canvas.scale("all", event.x, event.y, 0.9, 0.9)
        self.canvas.configure(scrollregion = self.canvas.bbox("all"))

if __name__ == "__main__":
    root = tk.Tk()
    Example(root).pack(fill="both", expand=True)
    root.mainloop()