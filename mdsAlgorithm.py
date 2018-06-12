
import tkinter as tk
import networkx as nx
import matplotlib
import numpy as np
import pandas as pd
import builtins as bl
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.manifold import spectral_embedding as SE
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

class mdsCube(tk.Frame):
    G = nx.cubical_graph()
    #default points for cube
    pos = {0: np.array([ 0.82877618,  0.53211873]), 
       1: np.array([ 0.8059564,  0.       ]), 
       2: np.array([ 0.51148475,  0.37349706]), 
       3: np.array([ 0.54462887,  0.89200482]), 
       4: np.array([ 0.31695909,  0.62593525]), 
      5:np.array([ 0.02260257,  1.        ]), 
      6: np.array([ 0.        ,  0.46707769]), 
       7: np.array([ 0.27222528,  0.10714391])}
    WelcomeMessage = "This MDS Algorithm implementation reduces dimensions of a cube to two." \
    " Enter new point coordinates if you want to change cube dimensions."
    labels = {}
    labels[0] = "A"
    labels[1] = "B"
    labels[2] = "C"
    labels[3] = "D"
    labels[4] = "E"
    labels[5] = "F"
    labels[6] = "G"
    labels[7] = "H"

    def __init__(self, master):
        tk.Frame.__init__(self,master)
        self.master.title("MDS Cube Algorithm")
        self.master.minsize(1000,450)
        self.master.resizable(False,False)
        Header = tk.Label(self.master,text=self.WelcomeMessage,
                          font=("Times BOLD ITALIC", 16) )
        Header.grid(column=0,row=0,padx=10,pady=15,columnspan=5)
        
        CubeP = tk.Label(self.master, text="Cube points", font=("Times BOLD",15))
        CubeP.grid(column=0,row=1)
        PA = tk.Label(self.master, text="Point A:")
        PA.grid(column=0,row=2,pady=5)
        PB = tk.Label(self.master, text="Point B:")
        PB.grid(column=0,row=3,pady=5)
        PC = tk.Label(self.master, text="Point C:")
        PC.grid(column=0,row=4,pady=5)
        PD = tk.Label(self.master, text="Point D:")
        PD.grid(column=0,row=5,pady=5)
        PE = tk.Label(self.master, text="Point E:")
        PE.grid(column=0,row=6,pady=5)
        PF = tk.Label(self.master, text="Point F:")
        PF.grid(column=0,row=7,pady=5)
        PG = tk.Label(self.master, text="Point G:")
        PG.grid(column=0,row=8,pady=5)
        PH = tk.Label(self.master, text="Point H:")
        PH.grid(column=0,row=9,pady=5)
        
        Xpoint = tk.Label(self.master, text="X coordinate", font=("Times BOLD",14))
        Xpoint.grid(column=1,row=1)
        Ypoint = tk.Label(self.master, text="Y coordinate", font=("Times BOLD",14))
        Ypoint.grid(column=2, row=1)
        L = tk.Label(self.master, text="Cube Graph before reduction", font=("Times BOLD",14))
        L.grid(column=3,row=1)
        #getting input for cube points
        self.XA = tk.Entry(self.master,width=10)
        self.XA.grid(column=1,row=2,pady=5)
        self.XA.insert(3,self.pos[0][0])
        self.YA = tk.Entry(self.master,width=10)
        self.YA.grid(column=2,row=2,pady=5)
        self.YA.insert(3,self.pos[0][1])
        
        self.XB = tk.Entry(self.master,width=10)
        self.XB.grid(column=1,row=3,pady=5)
        self.XB.insert(3,self.pos[1][0])
        self.YB = tk.Entry(self.master,width=10)
        self.YB.grid(column=2,row=3,pady=5)
        self.YB.insert(3,self.pos[1][1])
        
        self.XC = tk.Entry(self.master,width=10)
        self.XC.grid(column=1,row=4,pady=5)
        self.XC.insert(3,self.pos[2][0])
        self.YC = tk.Entry(self.master,width=10)
        self.YC.grid(column=2,row=4,pady=5)
        self.YC.insert(3,self.pos[2][1])
        
        self.XD = tk.Entry(self.master,width=10)
        self.XD.grid(column=1,row=5,pady=5)
        self.XD.insert(3,self.pos[3][0])
        self.YD = tk.Entry(self.master,width=10)
        self.YD.grid(column=2,row=5,pady=5)
        self.YD.insert(3,self.pos[3][1])
        
        self.XE = tk.Entry(self.master,width=10)
        self.XE.grid(column=1,row=6,pady=5)
        self.XE.insert(3,self.pos[4][0])
        self.YE = tk.Entry(self.master,width=10)
        self.YE.grid(column=2,row=6,pady=5)
        self.YE.insert(3,self.pos[4][1])
        
        self.XF = tk.Entry(self.master,width=10)
        self.XF.grid(column=1,row=7,pady=5)
        self.XF.insert(3,self.pos[5][0])
        self.YF = tk.Entry(self.master,width=10)
        self.YF.grid(column=2,row=7,pady=5)
        self.YF.insert(3,self.pos[5][1])
        
        self.XG = tk.Entry(self.master,width=10)
        self.XG.grid(column=1,row=8,pady=5)
        self.XG.insert(3,self.pos[6][0])
        self.YG = tk.Entry(self.master,width=10)
        self.YG.grid(column=2,row=8,pady=5)
        self.YG.insert(3,self.pos[6][1])
        
        self.XH = tk.Entry(self.master,width=10)
        self.XH.grid(column=1,row=9,pady=5)
        self.XH.insert(3,self.pos[7][0])
        self.YH = tk.Entry(self.master,width=10)
        self.YH.grid(column=2,row=9,pady=5)
        self.YH.insert(3,self.pos[7][1])
        #button to calculate reduced cube
        tk.Button(self.master, text="Calculate", command=self.fix , bg="blue",
                    padx=15,pady=5).grid(column=3,row=10)
        
        tk.Button(self.master, text="Quit", command=self.quit , bg="blue",
                    padx=15,pady=5).grid(column=4,row=10)
        #draw default cube 
        fig=plt.figure()
        fig.patch.set_facecolor('white')
        nx.draw_networkx_nodes(self.G, self.pos,
                      node_color='b',
                      node_size=500,
                      alpha=1)
        nx.draw_networkx_edges(self.G, self.pos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(self.G,self.pos,self.labels,font_size=12)
        self.Canvas = FigureCanvasTkAgg(fig, self.master)
        self.Canvas.get_tk_widget().grid(column=3,row=2,rowspan=8)
        
        
        MDSmessage= """
        NOTE: You define Cube graph
        in 3D using two values for 
        each point."""
        headerOne = tk.Label(self.master, text=MDSmessage, font=("Times BOLD",14))
        headerOne.grid(column=4,row=2,rowspan=8)
        
        #helper function: calculates distance matrix of the graph using dijkstra
    def getDistanceMatrix(self,graph):
        pathLengths = bl.dict(nx.all_pairs_dijkstra_path_length(graph))
        df = pd.DataFrame(pathLengths)
        distanceMatrix = df.as_matrix()
        return distanceMatrix
    
    def graph_node_position_mds(self,G):
    #if edges of the graph do not have weights, assign 1 as a weight 
        if not hasattr(G.edges(), 'weight'):
            for i in G.edges():
                a = i[0]
                b = i[1]
                G.add_edge(a,b,weight=1)
            DM = self.getDistanceMatrix(self.G) #get distance matrix
            #calculate MDS reduced 
            MDS_ = MDS(n_components=2,dissimilarity='precomputed').fit_transform(DM)
            MDS_dict = {}
            for index, val in enumerate(MDS_):
                MDS_dict[index] = val
        return MDS_dict

    def fix(self):
        new = tk.Toplevel(self.master)
        headerOne = tk.Label(new, text="MDS Reduced Cube", font=("Times BOLD",14))
        headerOne.grid(column=0,row=0)
        self.Canvas.get_tk_widget().destroy()
        #get new cube input
        self.pos[0][0] = self.XA.get()
        self.pos[0][1] = self.YA.get()
        self.pos[1][0] = self.XB.get()
        self.pos[1][1] = self.YB.get()
        self.pos[2][0] = self.XC.get()
        self.pos[2][1] = self.YC.get()
        self.pos[3][0] = self.XD.get()
        self.pos[3][1] = self.YD.get()
        self.pos[4][0] = self.XE.get()
        self.pos[4][1] = self.YE.get()
        self.pos[5][0] = self.XF.get()
        self.pos[5][1] = self.YF.get()
        self.pos[6][0] = self.XG.get()
        self.pos[6][1] = self.YG.get()
        self.pos[7][0] = self.XH.get()
        self.pos[7][1] = self.YH.get()
        #plot new graph
        fig=plt.figure()
        fig.patch.set_facecolor('white')
        nx.draw_networkx_nodes(self.G, self.pos,
                      node_color='b',
                      node_size=500,
                      alpha=1)
        nx.draw_networkx_edges(self.G, self.pos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(self.G,self.pos,self.labels,font_size=12)
        self.Canvas = FigureCanvasTkAgg(fig, self.master)
        self.Canvas.get_tk_widget().grid(column=3,row=2,rowspan=8)
        
        figRed = plt.figure()
        figRed.patch.set_facecolor('white')
        dictPos = self.graph_node_position_mds(self.G)
        nx.draw_networkx_nodes(self.G, dictPos,
                           node_color='r',
                           node_size=500,
                           alpha=1)
        nx.draw_networkx_edges(self.G, dictPos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(self.G,dictPos,self.labels,font_size=12)
        self.CanvasRed = FigureCanvasTkAgg(figRed, new)
        self.CanvasRed.get_tk_widget().grid(column=0, row=1)
        
    def quit(self):
        self.master.destroy()
        
#start the app
if __name__ == '__main__':
    root = tk.Tk()
    window = mdsCube(root)
    window.mainloop()
    
    
    
    
    
    
    
    
    
    
    
    