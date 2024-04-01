import os
import socket
import matplotlib.pyplot as plt
from cycler import cycler

class global_settings:

    def __init__(self):

        self.non_reelection_reform_date = "2015-03-10"
        self.invierte_reform_date = "2017-01-01"
        
        self.non_reelection_reform_year = 2015
        self.invierte_reform_year = 2017

        # For regressions and subsampling:
        self.snip_cuttof  = 1_200_000

        # Default bandwidth:
        self.bandwidth = 150_000 # This is soles

        self.targeted_sub_programs = ['SANEAMIENTO RURAL', 'VÍAS URBANAS', 'VÍAS VECINALES',
                                'SANEAMIENTO URBANO', 
                                'CAMINOS DE HERRADURA', 
                                'VIAS URBANAS',
                                'CAMINOS RURALES',
                                'CONSTRUCCION Y MEJORAMIENTO DE CARRETERAS',
                                'VÍAS NACIONALES',
                                'VÍAS DEPARTAMENTALES', 
                                'REHABILITACION DE CARRETERAS',
                                ]
                                
        self.targeted_programs = ['TRANSPORTE TERRESTRE','TRANSPORTE URBANO','SANEAMIENTO']

        self.ubigeos_flipping = [107, 140137, 140504, 140412, 140410, 140408, 140403, 140402, 140136, 130304, 140135, 140131, 140126, 140124, 140110, 140105, 140505, 140507, 140512, 140513, 140516, 140604, 140606, 140607, 140624, 140714, 140720, 140725, 140731, 140733, 140807, 140811, 140902, 130306, 130203, 140905, 100102, 110113, 110105, 100403, 100211, 100209, 100114, 90410, 130111, 90409, 90208, 90202, 90111, 90109, 90106, 110126, 110203, 110207, 110305, 110306, 110307, 110507, 110605, 110607, 110704, 110902, 120303, 120305, 120605, 120806, 130102, 130110, 140903, 141006, 80704, 200707, 201205, 201103, 201102, 200806, 200803, 200709, 200702, 200115, 200609, 200410, 200406, 200402, 200308, 200305, 201302, 201304, 210106, 210304, 210402, 210602, 210604, 210606, 210803, 210904, 210907, 220109, 220211, 220406, 230105, 230203, 230303, 200302, 200113, 150104, 150602, 180104, 170208, 170204, 170106, 160203, 160103, 150505, 200110, 150502, 150407, 150404, 150303, 150205, 150105, 180302, 190106, 190109, 190114, 190115, 190305, 190402, 190406, 190803, 190804, 200102, 200103, 200104, 200105, 200106, 200107, 200109, 80706, 80616, 202, 30216, 30604, 30602, 30506, 30505, 30312, 30304, 30212, 21308, 30204, 30106, 30102, 21904, 21902, 21606, 30614, 30702, 30708, 40103, 40107, 40111, 40116, 40117, 40118, 40120, 40121, 40122, 40124, 40127, 40214, 40506, 40507, 21605, 21306, 40803, 709, 1301, 1206, 1108, 1005, 1001, 805, 609, 21005, 604, 506, 503, 502, 304, 302, 10105, 10111, 10114, 10403, 10422, 20102, 20103, 20108, 20110, 20111, 20304, 20406, 20411, 20603, 20812, 20814, 20907, 40604, 40806, 80608, 70708, 71204, 71106, 71009, 71005, 71004, 70807, 70704, 70104, 70605, 70603, 70505, 70309, 70303, 70108, 71209, 71211, 71304, 71305, 71307, 80103, 80106, 80113, 80114, 80302, 80402, 80427, 80428, 80512, 80525, 80606, 80607, 70106, 61304, 50207, 51111, 60303, 60111, 60108, 60105, 60103, 60102, 51105, 61104, 51103, 50905, 50714, 50707, 50514, 50310, 60308, 60406, 60506, 60507, 60512, 60604, 60606, 60619, 60703, 60803, 60812, 60904, 60906, 60907, 60910, 60911, 61102, 240105]


        # Settings for plots:
        self.color1 =  "#E69F00"
        self.color2 =  "#56B4E9"
        self.color3 =  "#009E73"
        self.color4 =  "#0072B2"
        self.color5 =  "#D55E00"
        self.color6 =  "#CC79A7"
        self.color7 =  "#F0E442"

        self.ci_color = "b"
        self.ci_transparency = 0.05

        # Lineplot frequency
        self.weekly = 'w'
        self.quarter = 'q'
        self.semester = '2q'        
        self.year = 'y'        
        
        self.markersize = 7
        self.linewidth = 2

    def time_series_canvas(self):

        # plt.rcParams["font.family"] = "Verdana"
        # csfont = {'fontname':'Comic Sans MS'}
        fig = plt.figure(frameon=False, facecolor='white')
        ax = fig.add_subplot(111)
        # ax.grid(True, color='0.8', ls='-', lw=.2, zorder=0)
        ax.yaxis.grid(True, color='0.8', ls='-', lw=.4, zorder=0)

        # ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)

    def cross_section_canvas(self):

        fig = plt.figure(frameon=False, facecolor='white')
        ax = fig.add_subplot(111)

        # ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)


    
    # line cyclers adapted to colourblind people
    def set_default_plot_specifications(self):
        plt.rc("text", usetex=True)
        plt.rc("text.latex", preamble=r"\usepackage{newpxtext}\usepackage{newpxmath}\usepackage{commath}\usepackage{mathtools}")
        plt.rc("font", family="serif", size=12.)
        plt.rc("savefig", dpi=300, pad_inches = 0)
        plt.rc("legend", loc="best", fontsize="medium", fancybox=True, framealpha=0.5)
        plt.rc("lines", linewidth=2.5, markersize=10, markeredgewidth=2.5)


    def get_data_path(self):
        # Get the computer's network name
        hostname = socket.gethostname()
        
        # Define data paths for different hostnames
        if hostname == 'DESKTOP-PF6QSJO':  # Replace with your actual computer name
            data_path = 'J:/My Drive/PovertyPredictionRealTime/data'
        elif hostname == 'DESKTOP-Q5GQULQ':
            data_path = "L:/.shortcut-targets-by-id/12-fuK40uOBz3FM-OXbZtk_OSlYcGpzpa/PovertyPredictionRealTime/data"
        else:
            data_path = '/home/fcalle0/datasets/WorldBankPovertyPrediction/'

        # Check if the data path exists
        if not os.path.exists(data_path):
            raise Exception(f"Data path does not exist: {data_path}")

        return data_path