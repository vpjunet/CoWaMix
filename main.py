    # CoWaMix: A tool to help you mix waters and create brewing water recipes for your coffee
    # Copyright (C) 2022  Valentin Junet

    # This program is free software: you can redistribute it and/or modify
    # it under the terms of the GNU General Public License as published by
    # the Free Software Foundation, either version 3 of the License, or
    # (at your option) any later version.

    # This program is distributed in the hope that it will be useful,
    # but WITHOUT ANY WARRANTY; without even the implied warranty of
    # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    # GNU General Public License for more details.

    # You should have received a copy of the GNU General Public License
    # along with this program.  If not, see <http://www.gnu.org/licenses/>.

#version 0.1.3
from kivy.app import App
from kivy.properties import ListProperty, ObjectProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.uix.popup import Popup
from kivy.uix.dropdown import DropDown
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.spinner import Spinner,SpinnerOption
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.metrics import sp
import os
import json
import numpy as np
from trust_constr import minimize, LinearConstraint, Bounds
import itertools
import sys
sys.tracebacklimit = 0
import warnings
warnings.filterwarnings("ignore")

"""
The whole code is contained in this file. There are two parts, the first one contains the optimization algorithm
to find optimal ratios of waters and the second one contains the GUI for the app, written with Kivy.
"""

#comment/uncomment to play with the size
# Window.size = (280, 600)
# Window.size = (600,330)
# Window.size = (330, 600)

#The height/font size of the interface is determined by the height of the window.
#If the width of the window is higher than the height, reverse this by multiplying the values by w_size.
if Window.width>Window.height:
    w_size = 1*(Window.width/Window.height)
else:
    w_size = 1

"""
This is the first part. The optimization uses the trust-constr (from the trust_constr package which is itself an isolated part of Scipy) method.
The objective will be to find ratio of 2 to 3 waters that optimizes the given objective (general hardness and alkalinity).
"""
#read the input file
def parse_input():
    file = './input.json'
    if os.path.exists(file):
        with open(file, "r") as f:
            input = json.loads(f.read())
        for k in input.keys():
            input[k] = np.array(input[k])
        return input
    else:
        return {'error': 'no input'}

#the core algorithm for the optimization
def find_best_ratio(A,Ahat,H,Hhat):
    #A is the alkalinity of 2 to 3 waters
    #Ahat is the objective alkalinity
    #H is the general harndess of 2 to 3 waters
    #Hhat is the objective general hardness
    n = A.shape[0]
    f = lambda p,A,Ahat,H,Hhat: (A.dot(p)-Ahat)**2+(H.dot(p)-Hhat)**2
    J = lambda p,A,Ahat,H,Hhat: np.array([2*A[i]*(A.dot(p)-Ahat)+2*H[i]*(H.dot(p)-Hhat) for i in np.arange(p.shape[0])]) 
    He = lambda p,A,Ahat,H,Hhat: 2*(A.reshape((p.shape[0],1)).dot(A.reshape(1,p.shape[0])) + 2*(H.reshape((p.shape[0],1)).dot(H.reshape(1,p.shape[0]))))
    p0 = np.random.random(n)
    p0=p0/p0.sum()
    constr = LinearConstraint(np.ones((1,n)),1,1)
    b=Bounds(np.repeat(0,n),np.repeat(1,n))
    res = minimize(f,p0,args=(A,Ahat,H,Hhat),method='trust-constr',jac=J,hess=He,constraints=constr,bounds=b)
    return res

#check some requirements that are necessary in order to have a solution (for example at least one water need to have lower alkalinity level than the objective)
def check_requirements(A,Ahat,H,Hhat,A_b,H_b):
    eps = 1e-7
    A[A==0]=eps
    H[H==0]=eps
    m_H = max([Hhat-H_b,eps])
    M_H = Hhat+H_b
    m_A = max([Ahat-A_b,eps])
    M_A = Ahat+A_b
    req1 = any(H<M_H)
    req2 = any(A<M_A)
    req3 = any(H>m_H)
    req4 = any(A>m_A)
    req5 = any((H/A)<(M_H/m_A))
    req6 = any((H/A)>(m_H/M_A))
    return np.array([req1,req2,req3,req4,req5,req6])

#this is a function to sort the results. The outliers values are placed at the end.
def find_outliers_ind(X,scale=1.5,method='IQR'):
    if method=='IQR':
        Q1 = np.quantile(X,0.25)
        Q3 = np.quantile(X,0.75)
        IQR = Q3-Q1
        thr = Q3+scale*IQR
    if method=='std':
        thr = X.mean()+scale*X.std()
    return np.nonzero(X>thr)[0]

#the function that run the optimization process for the given inputs
def optimize(Ca,Mg,HCO3,Hhat,Ahat,A_b=10,H_b=20,cost=None,name=None):
    #the inputs contains the corresponding values for each waters to combine (calcium, magnesium, bicarbonates,cost,name)
    # and the objective with their tolerance (general hardness, tolerance for GH, alkalinity, tolerance for KH)

    #the constant to change from mg/l to ppm as CaCO3
    c_Ca = 2.5
    c_Mg = 4.12
    c_HCO3 = 0.82
    no_water = '-'
    #the GH and KH of the waters
    H = c_Ca*Ca+c_Mg*Mg
    A = c_HCO3*HCO3
    n = A.shape[0]
    if cost is None:
        cost = np.zeros(n)
    if name is None:
        name = np.array(['Water_%i' % i for i in np.arange(n)])
    req_count = np.zeros(6)
    range_err_count = np.zeros(4)
    res_all = []
    score_all = []
    cost_all = []
    k_comb = [2]
    if n>2:
        k_comb+=[3]
    #optimize over all possible combinations of 2 to 3 waters
    for k in k_comb:
        for ind in itertools.combinations(np.arange(n), k):
            ind = list(ind)
            Ai = A[ind]
            Hi = H[ind]
            req = check_requirements(Ai,Ahat,Hi,Hhat,A_b,H_b)
            req_count += req
            if all(req):
                #call the optimization algorithm
                res = find_best_ratio(Ai,Ahat,Hi,Hhat)
                A_o = Ai.dot(res.x)
                H_o = Hi.dot(res.x)
                range_err = np.array([H_o>=(Hhat-H_b),H_o<=(Hhat+H_b),A_o>=(Ahat-A_b),A_o<=(Ahat+A_b)])
                range_err_count += range_err
                if all(range_err):
                    #prepare the output if there is a solution
                    res.H = H_o
                    res.A = A_o
                    res.ind = ind
                    res.name = name[ind]
                    res.p = res.x.round(3)*100
                    if res.p.sum() != 100:
                        diff = res.p.sum() - 100
                        closest_ind_to_approx = np.argsort(abs(res.p-res.x*100))
                        to_add = .1
                        if diff>0:
                            to_add = -0.1
                        for i in np.arange(int(round(abs(diff)*10,0))):
                            res.p[closest_ind_to_approx[i]] += to_add
                    H_o_round = Hi.dot(res.p)/100
                    A_o_round = Ai.dot(res.p)/100
                    ind_water_s = np.flip(np.argsort(res.p))
                    if sum(res.p==0)>0:
                        ind_water_s = ind_water_s[0:-(sum(res.p==0))]
                    if ind_water_s.shape[0]>1:
                        res.table_row = ['%g%s %s' % (res.p[i],'%',res.name[i]) for i in ind_water_s]
                        if ind_water_s.shape[0]<3:
                            res.table_row += [no_water]
                        cost_i = cost[ind].dot(res.p/100)
                        res.cost = cost_i
                        res.Ca = Ca[ind].dot(res.p/100)*c_Ca
                        res.Mg = Mg[ind].dot(res.p/100)*c_Mg
                        res.table_row += [np.round(H_o_round,1),np.round(A_o_round,1),np.round(H_o_round/A_o_round,1),np.round(res.Ca,1),np.round(res.Mg,1),np.round(cost_i,2)]
                        res_all+=[res]
                        score_all+=[res.fun]
                        cost_all+=[cost_i]
    err_file = './error.txt'
    ind_non_duplicate = []
    results_table = {}
    if any(req_count==0): #if some requirements were not fullfilled, make an error file
        warn_count = 0
        req_type = req_count==0
        mess = '\nNotations for the given inputs:\nGH for the objective general hardness,\nGH_tol for the tolerance of GH,\nKH for the objective alkalinity and\nKH_tol for the tolerance of KH.\n'
        if req_type[0]:
            warn_count+=1
            mess+='\nWarning %i: The general hardness of all the\nwaters is higher than GH+GH_tol. Consider\nadding waters with lower Calcium and/or\nMagnesium levels. If this is not possible,\nincrease GH and/or GH_tol so that\nGH+GH_tol is bigger than %g.\n' % (warn_count,H.min())
        if req_type[1]:
            warn_count+=1
            mess+='\nWarning %i: The alkalinities of all the waters\nare higher than KH+KH_tol. Consider adding\nwaters with a lower Bicarbonates level. If\nthis is not possible, increase KH and/or\nKH_tol so that KH+KH_tol is bigger than\n%g.\n' % (warn_count,A.min())
        if req_type[2]:
            warn_count+=1
            mess+='\nWarning %i: The general hardness of all the\nwaters is lower than GH-GH_tol. Consider\nadding waters with higher Calcium and/or\nMagnesium levels. If this is not possible,\ndecrease GH and/or increase GH_tol so that\nGH-GH_tol is lower than %g.\n' % (warn_count,H.max())
        if req_type[3]:
            warn_count+=1
            mess+='\nWarning %i: The alkalinities of all the waters\nare lower than KH-KH_tol. Consider adding\nwaters with a higher Bicarbonates level. If\nthis is not possible, decrease KH and/or\nincrease KH_tol so that KH-KH_tol is lower\nthan %g.\n' % (warn_count,A.max())
        if req_type[4]:
            warn_count+=1
            mess+='\nWarning %i: The ratios of general hardness\nover alkalinity in all the waters are higher\nthan the corresponding ratio with the\nobjective values (considering the tolerance).\nThis means that the objective water\ncomposition has relatively too few general\nhardness and/or too high alkalinity\ncompared to the ones from the given waters.\nConsider adding waters with less Calcium\nand/or less Magnesium and more\nBicarbonates. If this is not possible, increase\nGH and/or increase GH_tol and/or decrease\nKH and/or increase KH_tol so that the ratio\n(GH+GH_tol)/(KH-KH_tol) is higher than\n%g.\n' % (warn_count,np.min(H[A!=0]/A[A!=0]).round(2))
        if req_type[5]:
            warn_count+=1
            mess+='\nWarning %i: The ratios of general hardness\nover alkalinity in all the waters are lower\nthan the corresponding ratio with the\nobjective values (considering the tolerance).\nThis means that the objective water\ncomposition has relatively too much general\nhardness and/or too few alkalinity compared\nto the ones from the given waters. Consider\nadding waters with more Calcium and/or\nmore Magnesium and less Bicarbonates. If\nthis is not possible, decrease GH and/or\nincrease GH_tol and/or increase KH and/or\nincrease KH_tol so that the ratio\n(GH-GH_tol)/(KH+KH_tol) is lower than\n%g.\n' % (warn_count,np.max(H[A!=0]/A[A!=0]).round(2))
        mess = ('Some requirements were not fullfilled.\nRead %i warning message(s) below.\n' % warn_count) + mess
        with open(err_file,'w') as f:
            f.write(mess)
            f.close()
    else:
        if any(range_err_count==0): #if no solutions were found, make an error file
            range_err_type = range_err_count==0
            warn_count = 0
            mess = ''
            if range_err_type[0]:
                warn_count+=1
                mess+='\nWarning %i: All solutions had general\nhardness lower than the permitted\nminimum value. Consider changing the\ntolerance for the general hardness to a value\nhigher than %g\n' % (warn_count,H_b)
            if range_err_type[1]:
                warn_count+=1
                mess+='\nWarning %i: All solutions had general\nhardness higher than the permitted\nmaximum value. Consider changing the\ntolerance for the general hardness to a value\nhigher than %g\n' % (warn_count,H_b)
            if range_err_type[2]:
                warn_count+=1
                mess+='\nWarning %i: All solutions had alkalinity lower\nthan the permitted minimum value.\nConsider changing the tolerance for the\nalkalinity to a value higher than %g\n' % (warn_count,A_b)
            if range_err_type[3]:
                warn_count+=1
                mess+='\nWarning %i: All solutions had alkalinity\nhigher than the permitted minimum value.\nConsider changing the tolerance for the\nalkalinity to a value higher than %g\n' % (warn_count,A_b)
            mess = ('Could not find solutions within the given range\nof acceptable solutions.\nRead %i warning message(s) below.\n' % warn_count) + mess
            with open(err_file,'w') as f:
                f.write(mess)
                f.close()
        else: #prepare the results table (sorted by cost except for the high outliers in score and cost which are placed at the end)
            cost_all = np.array(cost_all)
            n_solutions = cost_all.shape[0]
            if n_solutions>0:
                score_all = np.array(score_all)
                ind_s = np.argsort(score_all)
                ind_s = ind_s[np.argsort(cost_all[ind_s])]
                cost_all = cost_all[ind_s]
                score_all = score_all[ind_s]
                res_all = [res_all[i] for i in ind_s]
                ind_cost_out = find_outliers_ind(cost_all,method='IQR',scale=1.5)
                ind_score_out = find_outliers_ind(score_all,method='IQR',scale=1.5)
                ind_out = np.unique(np.concatenate((ind_cost_out,ind_score_out)))
                ind_s_out = np.concatenate((np.setdiff1d(np.arange(n_solutions),ind_out),ind_out))
                cost_all = cost_all[ind_s_out]
                score_all = score_all[ind_s_out]
                res_all = [res_all[i] for i in ind_s_out]
                seen_comb = []
                for i in range(len(res_all)):
                    comb_water = ",".join((res_all[i].table_row[0:3]))
                    if comb_water not in seen_comb:
                        seen_comb += [comb_water]
                        ind_non_duplicate+=[i]
            waters_info = np.array([["".join(('100% ',na)) for na in name],[no_water for _ in np.arange(n)],[no_water for _ in np.arange(n)],np.round(H,1),np.round(A,1),np.round(H/A,1),np.round(c_Ca*Ca,1),np.round(c_Mg*Mg,1),np.round(cost,2)]).T
            col_name = ['% Water #1','% Water #2','% Water #3','General Hardness [ppm as CaCO3]','Alkalinity [ppm as CaCO3]','GH:KH','Calcium [ppm as CaCO3]','Magnesium [ppm as CaCO3]','Cost']
            ind_water = np.argsort(((np.array([H,A]).T-np.array([Hhat,Ahat]))**2).sum(axis=1))
            for c_i in range(len(col_name)):
                dict_curr = {}
                for i in range(len(ind_non_duplicate)):
                    dict_curr[str(i+1)] = res_all[ind_non_duplicate[i]].table_row[c_i]
                for i2 in range(len(ind_water)):
                    dict_curr[str(i2+1+len(ind_non_duplicate))] = waters_info[ind_water[i2],c_i]
                results_table[col_name[c_i]] = dict_curr
    return results_table

#the function to get the input and call the optimization
def main():
    input = parse_input()
    results_table = optimize(input['Ca'],input['Mg'],input['HCO3'],input['Hhat'],input['Ahat'],A_b=input['tol_Ahat'],H_b=input['tol_Hhat'],name=input['name'],cost=input['cost'])
    if not results_table == {}:
        results_table = json.dumps(results_table,indent=2)
        file = './prov_table.json'
        with open(file, "w") as f:
            f.write(results_table)
            f.close()

"""
The second part starts here. The GUI is determined there. First a few useful functions are defined.
Then the different screens are defined. Finally the screen manager is defined.
The screen manager also contains all "inter-screen" functions (for example used to add buttons in pre-defined screens) and
binds all buttons.
"""

#if anything else than a number is given, error message popup
def convert_to_float_or_popup(val,popup_window=True):
    try:
        val = float(val.replace(',','.'))
        return val
    except:
        if popup_window:
            f_size = 0.3*0.3*Window.height*0.2*w_size
            content = BoxLayout()
            content.orientation = 'vertical'
            content.add_widget(Label(text='All values must be numerical',font_size=f_size))
            content.close=Button(text='Close',font_size=f_size)
            content.add_widget(content.close)
            popup = Popup(title='Warning',content=content, auto_dismiss=False,size_hint = (0.6,0.3),pos_hint = {"x":0.2,"top":0.9})
            content.close.bind(on_press=popup.dismiss)
            popup.open()
        return 'False'

#function to determine where to set new lines in long text
def add_new_lines(text,f_size_sp):
    f_size = sp(f_size_sp)
    text2 = '\n'
    #remove new lines already there and possible double space
    n_char_max = int(1.4*sp(Window.width)/f_size)
    text = text+'\n\n'
    n_text = len(text)
    ind_paragraph = 0
    n_lines = 2
    while ind_paragraph<n_text: #first cut the text into paragraphs ('\n\n' in the text input)
        n_lines+=1
        ind_paragraph2 = text[ind_paragraph:None].find('\n\n')+ind_paragraph
        text_curr = text[ind_paragraph:ind_paragraph2]
        ind_paragraph = ind_paragraph2+2
        text_curr = text_curr.replace('\n',' ').replace('  ',' ')+' '
        n_text_curr = len(text_curr)
        if n_text_curr>n_char_max:
            ind_prev = 0
            while ind_prev<n_text_curr: #add the new lines
                ind = text_curr[ind_prev:(ind_prev+n_char_max)].rfind(' ')+ind_prev
                text_curr = text_curr[0:ind] + '\n' + text_curr[(ind+1):None]
                ind_prev = ind+1
                n_lines+=1
        text2 += text_curr + '\n'
        h = (n_lines+int(0.2*n_lines))*f_size_sp
    return text2,h

#the screen to add new water to the water dictionary and the main water screen
class AddWaterScreen(Screen):
    def __init__(self,keys,**kwargs):
        super(AddWaterScreen,self).__init__(**kwargs)

        layout_ext = GridLayout()
        layout_ext.cols=1
        h = 0.8*Window.height/9
        f_size = 0.5*h/2*w_size
        p = 1.2

        layout = GridLayout()
        layout.cols=2
        layout.add_widget(Label(text='Name: ',font_size=f_size,height=h))
        layout.name = TextInput(font_size=p*f_size,multiline=False,height=h)
        layout.add_widget(layout.name)
        for key in keys:
            layout.add_widget(Label(text='%s :' % key,font_size=f_size,height=h))
            setattr(layout,key,TextInput(font_size=p*f_size,multiline=False,height=h))
            layout.add_widget(getattr(layout,key))

        layout_ext.layout = layout
        layout_ext.add_widget(layout_ext.layout)
        layout_ext.add = Button(text='Add',font_size=f_size,size_hint_y=None,width=layout_ext.width/2,height=h)
        layout_ext.add_widget(layout_ext.add)
        layout_ext.back = Button(text='Back',font_size=f_size,size_hint_y=None,width=layout_ext.width/2,height=h)
        layout_ext.add_widget(layout_ext.back)
        layout_ext.add_widget(Label(text='',font_size=f_size,size_hint_y=None,width=layout_ext.width/2,height=h))
        layout_ext.add_widget(Label(text='',font_size=f_size,size_hint_y=None,width=layout_ext.width/2,height=h))

        self.layout_ext = layout_ext
        self.add_widget(self.layout_ext)
        self.name = 'add_water'
        self.keys = keys

#the screen which shows the details of a given water
class WaterDetailsScreen(Screen):
    def __init__(self,name,dict,**kwargs):
        super(WaterDetailsScreen,self).__init__(**kwargs)

        layout_ext = GridLayout()
        layout_ext.cols=1
        h = 0.8*Window.height/9
        f_size = 0.5*h/2*w_size
        p = 1.2
        layout_ext.name = Label(text=name,font_size=f_size,size_hint_y=None,width=layout_ext.width/2,height=h)
        layout_ext.add_widget(layout_ext.name)

        layout = GridLayout()
        layout.cols=2
        for key in dict.keys():
            layout.add_widget(Label(text='%s :' % key,font_size=f_size,height=h))
            setattr(layout,key,TextInput(text=str(float(dict[key])),font_size=p*f_size,multiline=False,height=h))
            layout.add_widget(getattr(layout,key))

        layout_ext.layout = layout
        layout_ext.add_widget(layout_ext.layout)
        layout_ext.modify = Button(text='Modify',font_size=f_size,size_hint_y=None,width=layout_ext.width/2,height=h)
        layout_ext.add_widget(layout_ext.modify)
        layout_ext.delete = Button(text='Delete',font_size=f_size,size_hint_y=None,width=layout_ext.width/2,height=h)
        layout_ext.add_widget(layout_ext.delete)
        layout_ext.back = Button(text='Back',font_size=f_size,size_hint_y=None,width=layout_ext.width/2,height=h)
        layout_ext.add_widget(layout_ext.back)
        layout_ext.add_widget(Label(text='',font_size=f_size,size_hint_y=None,width=layout_ext.width/2,height=h))

        self.layout_ext = layout_ext
        self.add_widget(self.layout_ext)
        self.name = 'water_' + name
        
#the screen with all the waters in the water dictionary
class MainWaterScreen(Screen):
    def __init__(self,water_dict,**kwargs):
        super(MainWaterScreen,self).__init__(**kwargs)
        layout_ext = BoxLayout()
        layout_ext.bind(minimum_height=layout_ext.setter('height'))
        layout_ext.orientation = 'vertical'
        layout_ext.spacing= 10
        layout_ext.padding= 10
        layout_ext.size_hint_y=None
        h = (0.8*Window.height/10)*w_size
        f_size = 0.5*h
        self.h = h
        self.f_size = f_size

        layout = BoxLayout()
        layout.bind(minimum_height=layout.setter('height'))
        
        layout.orientation = 'vertical'
        layout.spacing= 10
        layout.padding= 20
        layout.size_hint_y=None
        if water_dict != {}:
            for name_curr in water_dict.keys():
                setattr(layout,name_curr,Button(text=name_curr,font_size=f_size,size_hint_y=None, height=h))
                layout.add_widget(getattr(layout,name_curr))

        root = ScrollView(size_hint=(1, None),size=(Window.width,0.7*Window.height),do_scroll_x=False,do_scroll_y=True)
        root.layout = layout
        root.add_widget(root.layout)

        layout_ext.root = root
        layout_ext.add_widget(layout_ext.root)
        layout_ext.add_water = Button(text='Add Water',font_size=f_size,size_hint_y=None, height=h)
        layout_ext.add_widget(layout_ext.add_water)
        layout_ext.back = Button(text='Back',font_size=f_size,size_hint_y=None, height=h)
        layout_ext.add_widget(layout_ext.back)

        self.name = 'main_water_screen'
        self.layout_ext = layout_ext
        self.add_widget(self.layout_ext)

#a spinner to select and unselect water in the input screen
class SelectWaterSpinner(Button):
    dropdown = ObjectProperty(None)
    waters = ListProperty([])
    selected_waters = ListProperty([])

    def __init__(self,**kwargs):
        self.bind(dropdown=self.update)
        self.bind(waters=self.update)
        self.bind(selected_waters=self.change_text)
        super(SelectWaterSpinner, self).__init__(**kwargs)
        self.bind(on_release=self.dropdown_with_toogle)
        self.f_size = self.font_size
        self.size_hint_y=None
        self.text_size_width = Window.width/2

    def dropdown_with_toogle(self, instance):
        if self.dropdown is not None:
            if self.dropdown.parent:
                self.dropdown.dismiss()
            else:
                self.dropdown.open(self)

    def update(self, *args):
        if self.dropdown is None:
            self.dropdown = DropDown()
        waters = self.waters
        if len(waters)>0:
            if self.dropdown.children:
                self.dropdown.clear_widgets()
            for water in waters:
                btn = ToggleButton(text=water,font_size=(0.5*0.8*Window.height/20)*w_size,size_hint_y=None, height=(0.8*Window.height/20)*w_size,background_color = (160/255, 160/255, 160/255, 1))
                btn.bind(state=self.select_value)
                self.dropdown.add_widget(btn)

    def select_value(self, instance, value):
        if value == 'down':
            if instance.text not in self.selected_waters:
                self.selected_waters.append(instance.text)
        else:
            if instance.text in self.selected_waters:
                self.selected_waters.remove(instance.text)

    def change_text(self, instance, value):
        self.font_size = self.f_size
        if len(value)>0:
            self.text = value[0]
            r = 1
            for v in value[1:None]:
                text = self.text+', '+v
                if len(text[text.rfind('\n')+1:None])>30:
                    self.text+=',\n'+v
                    r+=1
                else:
                    self.text = text
            if r>6:
                self.font_size = 0.8*self.f_size
        else:
            self.text = ''

#the screen to select the input
class SelectInputScreen(Screen):
    def __init__(self,water_list,**kwargs):
        super(SelectInputScreen,self).__init__(**kwargs)
        h = 0.8*Window.height/9
        h_ext = h
        f_size = 0.3*h*w_size
        f_size2 = 0.7*f_size
        layout_ext = GridLayout()
        layout_ext.cols=1

        layout = GridLayout()
        layout.cols = 2
        layout.add_widget(Label(text='Select Waters\n(minimum 2, no maximum)',font_size=f_size2,size_hint_y=None, height=2*h))
        layout.selected_water = SelectWaterSpinner(waters=water_list,font_size=0.6*f_size,size_hint_y=None,height=2*h)
        layout.add_widget(layout.selected_water)

        layout.add_widget(Label(text='Objective General Hardness\n(GH) [ppm as CaCO3]',font_size=f_size2,size_hint_y=None, height=h))
        layout.Hhat = TextInput(font_size=f_size,multiline=False)
        layout.add_widget(layout.Hhat)
        layout.add_widget(Label(text='Objective Alkalinity\n(KH) [ppm as CaCO3]',font_size=f_size2,size_hint_y=None, height=h))
        layout.Ahat = TextInput(font_size=f_size,multiline=False)
        layout.add_widget(layout.Ahat)
        layout.add_widget(Label(text='Tolerance GH\n(i.e. GH-tolerance to\nGH+tolerance is accepted)',font_size=f_size2,size_hint_y=None, height=h))
        layout.tol_Hhat = TextInput(text='20',font_size=f_size,multiline=False)
        layout.add_widget(layout.tol_Hhat)
        layout.add_widget(Label(text='Tolerance KH\n(i.e. KH-tolerance to\nKH+tolerance is accepted)',font_size=f_size2,size_hint_y=None, height=h))
        layout.tol_Ahat = TextInput(text='10',font_size=f_size,multiline=False)
        layout.add_widget(layout.tol_Ahat)

        layout_ext.layout = layout
        layout_ext.add_widget(layout_ext.layout)
        layout_ext.submit = Button(text='Submit',font_size=f_size,size_hint_y=None,width=layout_ext.width/2,height=h_ext)
        layout_ext.add_widget(layout_ext.submit)
        layout_ext.back = Button(text='Back',font_size=f_size,size_hint_y=None,width=layout_ext.width/2,height=h_ext)
        layout_ext.add_widget(layout_ext.back)
        layout_ext.add_widget(Label(text='',font_size=f_size,size_hint_y=None,width=layout_ext.width/2,height=h_ext))
        layout_ext.add_widget(Label(text='',font_size=f_size,size_hint_y=None,width=layout_ext.width/2,height=h_ext))

        self.layout_ext = layout_ext
        self.add_widget(layout_ext)
        self.name = 'select_input_screen'
    
    def update_list(self,new_waters):
        self.layout_ext.layout.selected_water.waters = new_waters

#the table for the results
class TableLayout(BoxLayout):
    def __init__(self,data='', **kwargs):
        super(TableLayout,self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint_y = None
        layout = GridLayout()
        layout.size_hint_y=None
        layout.bind(minimum_height=layout.setter('height'))
        rows = list(data.keys())
        columns = data[rows[0]].keys()
        layout.cols = len(columns)+1
        layout.size_hint_x=0.7*layout.cols*(1/w_size) #increase here for wider columns
        h = (0.8*0.9*Window.height/10)*w_size
        f_size = 0.4*h
        layout.add_widget(Label(text='  Click to\n  Add to Recipes',font_size=f_size,size_hint_y=None,height=h))
        layout.btns = {}
        for ind in columns:
            layout.btns[ind] = Button(text=ind,font_size=f_size,size_hint_y=None,height=h)
            layout.add_widget(layout.btns[ind])
            layout.btns[ind].data = {'id':ind}

        for k in rows:
            layout.add_widget(Label(text=k.replace(' [','\n['),font_size=f_size,size_hint_y=None,height=h))
            for ind in columns:
                layout.btns[ind].data[k] = str(data[k][ind])
                layout.add_widget(Label(text=str(data[k][ind]),font_size=f_size,size_hint_y=None,height=h))

        root = ScrollView(size_hint=(None, None),size=(Window.width, 0.8*Window.height),do_scroll_x=True,do_scroll_y=True)
        root.layout = layout
        root.add_widget(root.layout)

        self.root = root
        self.add_widget(root)
        
#the screen with the optimization results
class MixResultsScreen(Screen):
    def __init__(self,**kwargs):
        super(MixResultsScreen,self).__init__(**kwargs)

        with open('./prov_table.json','r') as f:
            data = json.loads(f.read())
        layout = TableLayout(data)
        h = 0.05*Window.height*w_size
        f_size = 0.6*h
        layout.add_widget(Label(text='<-- scroll -->',font_size=0.8*f_size,size_hint_y=None, height=h))
        layout.back = Button(text='Leave',font_size=f_size,size_hint_y=None, height=h)
        layout.add_widget(layout.back)
        self.layout = layout
        self.add_widget(self.layout)

        self.name = 'mix_results_screen'

#the screen to add recipe to the recipe dictionary
class AddRecipeScreen(Screen):
    def __init__(self,data,**kwargs):
        super(AddRecipeScreen,self).__init__(**kwargs)
        layout_ext = GridLayout()
        h = (0.8*Window.height/10)*w_size
        f_size = 0.7*0.5*h
        layout_ext.size_hint_y = None
        layout_ext.bind(minimum_height=layout_ext.setter('height'))
        layout_ext.cols=1
        layout_ext.add_widget(Label(text='Add Recipe: '+data['id'],font_size=f_size,size_hint_y=None,width=layout_ext.width/2,height=h))
        data_copy = data.copy()
        data_copy.pop('id')

        layout = GridLayout()
        layout.cols=2
        layout.data = data_copy
        layout.size_hint_y = None
        layout.bind(minimum_height=layout.setter('height'))
        layout.add_widget(Label(text='Name',font_size=f_size,size_hint_y=None,height=h))
        layout.name = TextInput(font_size=f_size,size_hint_y=None,multiline=False,height=h)
        layout.add_widget(layout.name)
        for key in data_copy.keys():
            layout.add_widget(Label(text=key.replace(' [','\n['),font_size=f_size,size_hint_y=None,height=h))
            layout.add_widget(Label(text=data_copy[key],font_size=f_size,size_hint_y=None,height=h))

        layout_ext.layout = layout
        layout_ext.add_widget(layout_ext.layout)
        layout_ext.add = Button(text='Add',font_size=f_size,size_hint_y=None,width=layout_ext.width/2,height=h)
        layout_ext.add_widget(layout_ext.add)
        layout_ext.back = Button(text='Back',font_size=f_size,size_hint_y=None,width=layout_ext.width/2,height=h)
        layout_ext.add_widget(layout_ext.back)

        root = ScrollView(size_hint=(1, None),size=(Window.width, Window.height),do_scroll_x=False,do_scroll_y=True)
        root.layout_ext = layout_ext
        root.add_widget(root.layout_ext)
        self.root = root
        self.add_widget(self.root)
        self.name = 'add_recipe_screen'

#the screen with all the saved recipe from the recipe dictionary
class MainRecipeScreen(Screen):
    def __init__(self,recipe_dict,**kwargs):
        super(MainRecipeScreen,self).__init__(**kwargs)
        layout_ext = BoxLayout()
        layout_ext.bind(minimum_height=layout_ext.setter('height'))
        layout_ext.orientation = 'vertical'
        layout_ext.size = (layout_ext.width,layout_ext.height)
        layout_ext.spacing= 10
        layout_ext.padding= 10
        layout_ext.size_hint_y=None
        h = (0.8*Window.height/10)*w_size
        f_size = 0.5*h
        self.h = h
        self.f_size = f_size

        layout = BoxLayout()
        layout.bind(minimum_height=layout.setter('height'))
        
        layout.orientation = 'vertical'
        layout.size = (layout.width,layout.height)
        layout.spacing= 10
        layout.padding= 20
        layout.size_hint_y=None
        if recipe_dict != {}:
            for name_curr in recipe_dict.keys():
                setattr(layout,name_curr,Button(text=name_curr,font_size=f_size,size_hint_y=None, height=h))
                layout.add_widget(getattr(layout,name_curr))

        root = ScrollView(size_hint=(1, None),size=(Window.width, 0.7*Window.height),do_scroll_x=False,do_scroll_y=True)
        root.layout = layout
        root.add_widget(root.layout)

        layout_ext.root = root
        layout_ext.add_widget(layout_ext.root)
        layout_ext.add_recipe = Button(text='Add Recipe',font_size=f_size,size_hint_y=None, height=h)
        layout_ext.add_widget(layout_ext.add_recipe)
        layout_ext.back = Button(text='Back',font_size=f_size,size_hint_y=None, height=h)
        layout_ext.add_widget(layout_ext.back)

        self.layout_ext = layout_ext
        self.add_widget(self.layout_ext)
        self.name = 'main_recipe_screen'

#the screen with the details of a given recipe and with the option to get the dosage of the waters in the recipe for a desired quantity of water
class RecipeDetailsScreen(Screen):
    def __init__(self,name,data,**kwargs):
        super(RecipeDetailsScreen,self).__init__(**kwargs)
        layout_ext = GridLayout()
        h = (0.8*Window.height/10)*w_size
        f_size = 0.7*0.5*h
        self.f_size=f_size
        self.h = h
        layout_ext.size_hint_y = None
        layout_ext.bind(minimum_height=layout_ext.setter('height'))
        layout_ext.cols=1
        layout_ext.add_widget(Label(text=name,font_size=f_size,size_hint_y=None,width=layout_ext.width/2,height=h))

        layout = GridLayout()
        layout.cols=2
        layout.size_hint_y = None
        layout.bind(minimum_height=layout.setter('height'))
        self.p = []
        self.waters = []
        for key in data.keys():
            if ('% Water' in key) and data[key]!='-':
                a = data[key].split('% ')
                self.p+=[float(a[0])/100]
                self.waters+=[a[1]]
            layout.add_widget(Label(text=key.replace(' [','\n['),font_size=f_size,size_hint_y=None,height=h))
            layout.add_widget(Label(text=data[key],font_size=f_size,size_hint_y=None,height=h))
        class MySpinnerOption(SpinnerOption):
            def __init__(self,**kwargs):
                super(MySpinnerOption,self).__init__(**kwargs)
                self.font_size = f_size
                self.background_color = (160/255, 160/255, 160/255, 1)
        layout.quantity = Spinner(text='Quantity [liter]',values=('Quantity [liter]', 'Quantity [gallon]'),size_hint_y=None,height = h,font_size = f_size,option_cls=MySpinnerOption)
        layout.add_widget(layout.quantity)
        layout.quantity_input = TextInput(font_size=f_size,size_hint_y=None,multiline=False,height=h)
        layout.add_widget(layout.quantity_input)
        layout.dosage_g = {}
        for n in self.waters:
            layout.add_widget(Label(text=n,font_size=f_size,size_hint_y=None,height=h))
            layout.dosage_g[n] = Label(text='X g',font_size=f_size,size_hint_y=None,height=h)
            layout.add_widget(layout.dosage_g[n])

        layout_ext.layout = layout
        layout_ext.add_widget(layout_ext.layout)
        layout_ext.get_dosage = Button(text='Get Dosage',font_size=f_size,size_hint_y=None,width=layout_ext.width/2,height=h)
        layout_ext.get_dosage.bind(on_release=self.get_dosage)
        layout_ext.add_widget(layout_ext.get_dosage)
        layout_ext.delete = Button(text='Delete',font_size=f_size,size_hint_y=None,width=layout_ext.width/2,height=h)
        layout_ext.add_widget(layout_ext.delete)
        layout_ext.back = Button(text='Back',font_size=f_size,size_hint_y=None,width=layout_ext.width/2,height=h)
        layout_ext.add_widget(layout_ext.back)

        root = ScrollView(size_hint=(1, None),size=(Window.width, Window.height),do_scroll_x=False,do_scroll_y=True)
        root.layout_ext = layout_ext
        root.add_widget(root.layout_ext)
        self.root = root
        self.add_widget(self.root)
        self.name = 'recipe_'+name

    #the fuction to get the dosage
    def get_dosage(self,instance):
        if self.root.layout_ext.layout.quantity_input.text=='':
            for n in self.waters:
                self.root.layout_ext.layout.dosage_g[n].text = 'X g'
        else:
            input = convert_to_float_or_popup(self.root.layout_ext.layout.quantity_input.text)
            if input!='False':
                if self.root.layout_ext.layout.quantity.text=='Quantity [liter]':
                    c = 1
                elif self.root.layout_ext.layout.quantity.text=='Quantity [gallon]':
                    c = 3.8
                q = c*input
                for (i,n) in enumerate(self.waters):
                    self.root.layout_ext.layout.dosage_g[n].text = str(round(1000*q*self.p[i],1)) + ' g'

#the screen with the details of a given recipe and with the option to get the dosage of the waters in the recipe for a desired quantity of water
class AddManualRecipeScreen(Screen):
    def __init__(self,water_dict,**kwargs):
        super(AddManualRecipeScreen,self).__init__(**kwargs)
        keys_water_p = ['% Water #1','% Water #2','% Water #3']
        self.keys_water_p = keys_water_p
        keys = ['General Hardness [ppm as CaCO3]','Alkalinity [ppm as CaCO3]','GH:KH','Calcium [ppm as CaCO3]','Magnesium [ppm as CaCO3]','Cost']
        layout_ext = GridLayout()
        h = (0.8*Window.height/10)*w_size
        f_size = 0.7*0.5*h
        self.f_size=f_size
        self.h = h
        layout_ext.size_hint_y = None
        layout_ext.bind(minimum_height=layout_ext.setter('height'))
        layout_ext.cols=1

        layout = GridLayout()
        layout.cols=2
        layout.size_hint_y = None
        layout.bind(minimum_height=layout.setter('height'))
        layout.add_widget(Label(text='Name',font_size=f_size,size_hint_y=None,height=h))
        layout.name = TextInput(text='',font_size=f_size,size_hint_y=None,height=h,multiline=False)
        layout.add_widget(layout.name)
        self.water_dict = water_dict
        water_list = ['-']+['%'+k for k in self.water_dict.keys()]
        class MySpinnerOption(SpinnerOption):
            def __init__(self,**kwargs):
                super(MySpinnerOption,self).__init__(**kwargs)
                self.font_size = f_size
                self.background_color = (160/255, 160/255, 160/255, 1)
        layout.water_spinner1 = Spinner(text=keys_water_p[0],values=water_list,size_hint_y=None,height = h,font_size = f_size,option_cls=MySpinnerOption)
        layout.water_p1 = TextInput(text='',font_size=f_size,size_hint_y=None,height=h,multiline=False)
        layout.add_widget(layout.water_spinner1)
        layout.add_widget(layout.water_p1)
        layout.water_spinner2 = Spinner(text=keys_water_p[1],values=water_list,size_hint_y=None,height = h,font_size = f_size,option_cls=MySpinnerOption)
        layout.water_p2 = TextInput(text='',font_size=f_size,size_hint_y=None,height=h,multiline=False)
        layout.add_widget(layout.water_spinner2)
        layout.add_widget(layout.water_p2)
        layout.water_spinner3 = Spinner(text=keys_water_p[2],values=water_list,size_hint_y=None,height = h,font_size = f_size,option_cls=MySpinnerOption)
        layout.water_p3 = Label(text='X',font_size=f_size,size_hint_y=None,height=h)
        layout.add_widget(layout.water_spinner3)
        layout.add_widget(layout.water_p3)
        self.keys = keys
        for key in keys:
            layout.add_widget(Label(text=key.replace(' [','\n['),font_size=f_size,size_hint_y=None,height=h))
            setattr(layout,key,Label(text='X',font_size=f_size,size_hint_y=None,height=h))
            layout.add_widget(getattr(layout,key))

        layout_ext.layout = layout
        layout_ext.add_widget(layout_ext.layout)
        layout_ext.view = Button(text='View Recipe',font_size=f_size,size_hint_y=None,width=layout_ext.width/2,height=h)
        layout_ext.view.bind(on_release=self.view_recipe)
        layout_ext.add_widget(layout_ext.view)
        layout_ext.add = Button(text='Add Recipe',font_size=f_size,size_hint_y=None,width=layout_ext.width/2,height=h)
        layout_ext.add_widget(layout_ext.add)
        layout_ext.back = Button(text='Back',font_size=f_size,size_hint_y=None,width=layout_ext.width/2,height=h)
        layout_ext.add_widget(layout_ext.back)

        root = ScrollView(size_hint=(1, None),size=(Window.width, Window.height),do_scroll_x=False,do_scroll_y=True)
        root.layout_ext = layout_ext
        root.add_widget(root.layout_ext)
        self.root = root
        self.add_widget(self.root)
        self.name = 'add_manual_recipe'

    def update_list(self,new_water_dict):
        self.water_dict = new_water_dict
        water_list = ['-']+['%'+k for k in self.water_dict.keys()]
        self.root.layout_ext.layout.water_spinner1.values = water_list
        self.root.layout_ext.layout.water_spinner2.values = water_list
        self.root.layout_ext.layout.water_spinner3.values = water_list

    def view_recipe(self,instance):
        eps = 1e-4
        waters = [instance.parent.layout.water_spinner1.text.replace('%',''),instance.parent.layout.water_spinner2.text.replace('%',''),instance.parent.layout.water_spinner3.text.replace('%','')]
        p1 = convert_to_float_or_popup(instance.parent.layout.water_p1.text)
        if p1=='False':
            return False
        p1 = round(p1,1)
        p2 = convert_to_float_or_popup(instance.parent.layout.water_p2.text,popup_window=False)
        if p2=='False' and (waters[1] in self.water_dict.keys()):
            if (waters[2] not in self.water_dict.keys()):
                p2 = round(100-p1,1)
                p3 = 0.0
            else:
                convert_to_float_or_popup(instance.parent.layout.water_p2.text)
                return False
        elif p2=='False':
            p2 = 0.0
            p3 = 0.0
        else:
            p2 = round(p2,1)
            p3 = round(100-p1-p2,1)
        p = [p1,p2,p3]
        waters_c = waters.copy()
        p_c = p.copy()
        ind_keep = [0,1,2]
        for i,w in enumerate(waters_c):
            if (w not in self.water_dict.keys()) or (p_c[i]<eps):
                waters.remove(w)
                p.remove(p_c[i])
                ind_keep.remove(i)
        popup = False
        tt = ''
        if abs(100-sum(p))>eps:
            popup = True
            tt += 'The percentages must sum to 100\n'
        if len(waters)==0:
            popup = True
            tt += 'Select at least 1 water\n'
        if popup:
            f_size = 0.3*0.3*Window.height*0.2*w_size
            content = BoxLayout()
            content.orientation = 'vertical'
            content.add_widget(Label(text=tt,font_size=f_size))
            content.close=Button(text='Close',font_size=f_size)
            content.add_widget(content.close)
            popup = Popup(title='Warning',content=content, auto_dismiss=False,size_hint = (0.6,0.3),pos_hint = {"x":0.2,"top":0.9})
            content.close.bind(on_press=popup.dismiss)
            popup.open()
            if len(p)<3:
                instance.parent.layout.water_p3.text = 'X'
            return False
        data = self.get_values(waters,p)
        instance.parent.layout.data = data
        for k in self.keys:
            getattr(instance.parent.layout,k).text = str(data[k])
        for i,ind in enumerate(ind_keep):
            getattr(instance.parent.layout,'water_p%i' % (ind+1)).text = str(p[i])
        if len(p)<3:
            instance.parent.layout.water_p3.text = 'X'
        return True
    
    def get_values(self,waters,p):
        c_Ca = 2.5
        c_Mg = 4.12
        c_HCO3 = 0.82
        data = {}
        keys = self.keys #keys=['General Hardness [ppm as CaCO3]','Alkalinity [ppm as CaCO3]','GH:KH','Calcium [ppm as CaCO3]','Magnesium [ppm as CaCO3]','Cost']
        keys_water = ["Calcium [mg/l]","Magnesium [mg/l]","Bicarbonates [mg/l]","cost"]
        for i,k in enumerate(self.keys_water_p):
            if i<len(p):
                data[k] = '%g%s %s' % (p[i],'%',waters[i])
            else:
                data[k] = '-'
        for k in keys:
            data[k] = 0
        for i,w in enumerate(waters):
            data[keys[3]] += p[i]*c_Ca*self.water_dict[w][keys_water[0]]/100
            data[keys[4]] += p[i]*c_Mg*self.water_dict[w][keys_water[1]]/100
            data[keys[5]] += p[i]*self.water_dict[w][keys_water[3]]/100
            data[keys[1]] += p[i]*c_HCO3*self.water_dict[w][keys_water[2]]/100
        for i in [1,3,4]:
            data[keys[i]] = round(data[keys[i]],1)
        data[keys[5]] = round(data[keys[5]],2)
        data[keys[0]] = round(data[keys[3]]+data[keys[4]],1)
        if data[keys[1]]==0:
            data[keys[2]] = 'nan'
        else:
            data[keys[2]] = round(data[keys[0]]/data[keys[1]],1)
        for k in keys:
            data[k] = str(data[k])
        return data

#the screen showing the error messages
class ErrorScreen(Screen):
    def __init__(self,error_file,**kwargs):
        super(ErrorScreen,self).__init__(**kwargs)
        layout = GridLayout()
        layout.cols=1
        layout.bind(minimum_height=layout.setter('height'))
        layout.size_hint_y=None
        h2 = 0.1*Window.height
        f_size = 0.02*Window.height*w_size
        layout.error_file = error_file
        with open(error_file,'r') as f:
            text = f.read()
            f.close()
        error_message = GridLayout()
        error_message.cols=1
        error_message.bind(minimum_height=error_message.setter('height'))
        error_message.size_hint_y=None
        t,h = add_new_lines('\n\nWarning\n\n\n\n'+text,f_size)
        error_message.add_widget(Label(text=t,font_size=f_size,size_hint_y=None,height=h))
        root = ScrollView(size_hint=(1, None),size=(Window.width, 0.9*Window.height),do_scroll_x=False,do_scroll_y=True)
        root.error_message = error_message
        root.add_widget(root.error_message)
        layout.root = root
        layout.add_widget(layout.root)

        layout.close = Button(text='Close',font_size=f_size,size_hint_y=None,height=h2)
        layout.add_widget(layout.close)
        self.layout = layout
        self.add_widget(self.layout)
        self.name = 'error_screen'

#the screen with the information on the app
class InfoScreen(Screen):
    def __init__(self,**kwargs):
        super(InfoScreen,self).__init__(**kwargs)
        layout_ext = GridLayout()
        layout_ext.cols = 1
        h_ext = 0.8*Window.height/8
        p=0.7
        h2 = h_ext/1000
        f_size = 0.2*h_ext*w_size
        layout_ext.size_hint_y = None
        layout_ext.bind(minimum_height=layout_ext.setter('height'))

        layout = GridLayout()
        layout.cols = 1
        layout.size_hint_y = None
        layout.bind(minimum_height=layout.setter('height'))
        layout.add_widget(Label(text='Info',font_size=1.5*f_size,size_hint_y=None,height=h_ext))

        layout.descr = Button(text='Description',font_size=1.3*f_size,size_hint_y=None,height=p*h_ext,background_color =(1, 1, 1, 0.5))
        layout.add_widget(layout.descr)
        layout.descr.isopen = 0
        tt,hh = add_new_lines('Brewing water interacts with ground coffee to extract incredible\nand delicious flavors. Different bottled waters will lead to\ndifferent results and mixing them can optimize the mineral\ncontent for coffee extraction and tasting experience.\n\nCoWaMix stands for Coffee Water Mix and was designed to help\nyou combine various bottled waters to get an optimal -or\nexperimental- brewing water recipe for your coffee. CoWaMix\noptimizes your brewing water through two parameters:\nthe general hardness (GH) and the alkalinity (KH). You can add\nmultiple waters that you have available, set your own GH and KH\nobjectives and the app will return possible combinations of 2 to 3\nwaters that optimize your objectives. Once you have a desired\nmix, you can add it to your recipe, select a target quantity of\nbrewing water and the app will tell you how many grams of each\nwater from the recipe to mix.\n\nAre you ready to taste your next brew? Add your waters, make\nyour mixes, select your recipes, and get your brewing water ready\nwith CoWaMix!\n\nCoWaMix is freely available for use and to explore how water can\naffect your daily brews.',f_size)
        layout.descr.text_on_click = ['',tt]
        layout.descr.h_on_click = [h2,hh]
        layout.descr.label = Label(text=layout.descr.text_on_click[layout.descr.isopen],font_size=f_size,size_hint_y=None,height=layout.descr.h_on_click[layout.descr.isopen])
        layout.add_widget(layout.descr.label)
        layout.descr.bind(on_release=self.open_close)
    
        layout.water = Button(text='My Waters Button',font_size=1.3*f_size,size_hint_y=None,height=p*h_ext,background_color =(1, 1, 1, 0.5))
        layout.add_widget(layout.water)
        layout.water.isopen = 0
        tt,hh = add_new_lines('Click on the “My Waters” button to access your water dictionary.\nYou can add and save there all the required information from\nthe waters that you have available. The required information\nare: a name, the levels of calcium, magnesium and bicarbonates\n(in mg/l, it is generally written on the bottle of the water or can\nbe found online) and the cost (typically price per liter). The cost is\nuseful to select between different water recipes;  for two\nequivalent recipes, you can choose the one with the least cost.\nAdditionally, you could set a lower/higher cost to a water you\nwould like to favor/disfavor since the resulting mixes will be\nsorted by cost.\n\nOnce a water is added, you can view, modify, or delete it by\nclicking on the button with its name.',f_size)
        layout.water.text_on_click = ['',tt]
        layout.water.h_on_click = [h2,hh]
        layout.water.label = Label(text=layout.water.text_on_click[layout.water.isopen],font_size=f_size,size_hint_y=None,height=layout.water.h_on_click[layout.water.isopen])
        layout.add_widget(layout.water.label)
        layout.water.bind(on_release=self.open_close)
    
        layout.mix = Button(text='Make a Mix Button',font_size=(1.3*f_size),size_hint_y=None,height=p*h_ext,background_color =(1, 1, 1, 0.5))
        layout.add_widget(layout.mix)
        layout.mix.isopen = 0
        tt,hh = add_new_lines('After adding some waters into your dictionary, you can start\nmixing them by clicking on the “Make a Mix” button. You will\nhave to select at least 2 waters from your water dictionary. You\ncan then set the objective general hardness (GH) and alkalinity\n(KH) that you want your mix of waters to reach. There is also the\noption of changing the tolerance for accepted solutions. In this\ncase, mixes with a resulting GH and KH that differs from the\nobjectives by the corresponding tolerance will also be accepted\nas solutions. For example,  if your objective is 80 ppm for GH\nand the tolerance is 20, then mixes with resulting GH between 60\nand 100 will be acceptable solutions.\n\nClick on “Submit” to start finding your mixes! Once the\ncomputations are finished, a table with the mixes will appear.\nYou can click on the first row of each column (each mix) to see all\nthe details of this mix and add it to your recipe (do not forget to\ngive it a name). Aside from the mixes, the result table will also\ncontain the information about the individual waters in case their\nvalues are already close enough to your objectives.\n\nNote that, while there are no maximum number of waters to\nmix, a high number of waters will lead to longer computation\ntime. Try to keep this number below 12.\n\nDepending on the available waters and the objectives set, it is\npossible that no solution are found. For example, if all\nwaters have GH higher than the objective GH, it will not be\npossible to mix them and obtain a lower GH. In such cases, a\nwarning message will appear suggesting possible changes.\n\nThe mixes of given waters will have the displayed GH and KH\nlevels, however, since they are made of mineralized water, they\nwill also contain other minerals which are not considered. The\nmixes only detail calcium, magnesium, and bicarbonates\ncontents. Be also aware that, when used in a machine, some\nwaters might be damaging for your equipment.',f_size)
        layout.mix.text_on_click = ['',tt]
        layout.mix.h_on_click = [h2,hh]
        layout.mix.label = Label(text=layout.mix.text_on_click[layout.mix.isopen],font_size=f_size,size_hint_y=None,height=layout.mix.h_on_click[layout.mix.isopen])
        layout.add_widget(layout.mix.label)
        layout.mix.bind(on_release=self.open_close)

        layout.recipe = Button(text='My Recipes Button',font_size=(1.3*f_size),size_hint_y=None,height=p*h_ext,background_color =(1, 1, 1, 0.5))
        layout.add_widget(layout.recipe)
        layout.recipe.isopen = 0
        tt,hh = add_new_lines('After adding mixes of water to your recipes, you can click on the “My Recipes” button to access your recipe dictionary. You will find there all your recipes and you can see their details by clicking on them. You can also manually create a recipe by clicking on “Add Recipe” and choosing a percentage for the waters you would like to mix.\n\nIn your recipe, there is the possibility of setting a target quantity (in liters or gallons) of water to mix and, after clicking on “Get Dosage”, the required amount in grams of each water in the recipe will appear. You can then mix these waters according to the given dosages, get your desired quantity of water, start brewing your coffee and enjoy!',f_size)
        layout.recipe.text_on_click = ['',tt]
        layout.recipe.h_on_click = [h2,hh]
        layout.recipe.label = Label(text=layout.recipe.text_on_click[layout.recipe.isopen],font_size=f_size,size_hint_y=None,height=layout.recipe.h_on_click[layout.recipe.isopen])
        layout.add_widget(layout.recipe.label)
        layout.recipe.bind(on_release=self.open_close)

        layout.about = Button(text='About',font_size=(1.3*f_size),size_hint_y=None,height=p*h_ext,background_color =(1, 1, 1, 0.5))
        layout.add_widget(layout.about)
        layout.about.isopen = 0
        tt,hh = add_new_lines('CoWaMix is free of charge and also available on GitHub:',f_size)
        tt+='[ref=https://github.com/vpjunet/CoWaMix][u]github.com/vpjunet/CoWaMix[/u][/ref]\n'
        tt2,hh2 = add_new_lines('It is written in Python with libraries Kivy, Numpy, trust-constr from Scipy and more-itertools, packaged for Android with Buildozer from Kivy and its dependencies. Related open-source licenses details, terms and conditions and privacy policy can be found in the above-mentioned repository.',f_size)
        layout.about.text_on_click = ['',tt+tt2]
        layout.about.h_on_click = [h2,hh+hh2]
        layout.about.label = Label(text=layout.about.text_on_click[layout.about.isopen],font_size=f_size,size_hint_y=None,height=layout.about.h_on_click[layout.about.isopen], markup=True)
        layout.about.label.bind(on_ref_press=self.hyperlink)
        layout.add_widget(layout.about.label)
        layout.about.bind(on_release=self.open_close)

        root = ScrollView(size_hint=(1, None),size=(Window.width, 0.9*Window.height),do_scroll_x=False,do_scroll_y=True)
        root.layout = layout
        root.add_widget(root.layout)
        layout_ext.root = root
        layout_ext.add_widget(layout_ext.root)
        layout_ext.back = Button(text='Back',font_size=1.5*f_size,size_hint_y=None,height=h_ext)
        layout_ext.add_widget(layout_ext.back)
        self.layout_ext = layout_ext
        self.add_widget(self.layout_ext)
        self.name = 'info_screen'

    def open_close(self,instance):
        isopen = ~instance.isopen
        instance.label.height = instance.h_on_click[isopen]
        instance.label.text = instance.text_on_click[isopen]
        instance.isopen = isopen
    
    def hyperlink(self, instance, value):
        import webbrowser
        webbrowser.open(value)

#the screen manager where everythin is put together
class MenuScreenManager(ScreenManager):
    def __init__(self,user_path,**kwargs):
        super(MenuScreenManager,self).__init__(**kwargs)

        #read the waters and recipes dictionary
        self.file_water = "/".join((user_path,'waters.json'))
        self.file_recipe = "/".join((user_path,'recipes.json'))
        if not os.path.exists(self.file_water) and not os.path.exists(self.file_recipe):
            self.generate_first_time_data()
        if os.path.exists(self.file_water):
            with open(self.file_water, "r") as f:
                self.water_dict = json.loads(f.read())
        else:
            self.water_dict = {}
        if os.path.exists(self.file_recipe):
            with open(self.file_recipe, "r") as f:
                self.recipe_dict = json.loads(f.read())
        else:
            self.recipe_dict = {}

        layout = BoxLayout()
        layout.bind(minimum_height=layout.setter('height'))
        layout.orientation = 'vertical'
        layout.spacing= 10
        layout.padding= 50
        layout.size_hint_y=None
        h = Window.height*w_size
        p1 = 0.9
        p2 = 0.6
        n_button = 5
        font1 = (0.2*h/4)
        font2 = font1
        #define the main/menu screen from which all the other screens can be found
        layout.main_im = Image(source='logo.png',allow_stretch=True,keep_ratio=True,size_hint_y=None,height=p1*p2*h)
        layout.add_widget(layout.main_im)

        layout.add_widget(Label(text='Coffee Water Mix',font_size=font2,size_hint_y=None, height=p1*p2*h/n_button))
        
        #add buttons to open the other screens
        layout.info = Button(text='Info',font_size=font1,size_hint_y=None, height=p1*p2*h/n_button)
        layout.add_widget(layout.info)

        layout.water_window = Button(text='My Waters',font_size=font1,size_hint_y=None, height=p1*p2*h/n_button)
        layout.add_widget(layout.water_window)

        layout.recipe_window = Button(text='My Recipes',font_size=font1,size_hint_y=None, height=p1*p2*h/n_button)
        layout.add_widget(layout.recipe_window)

        layout.mix_window = Button(text='Make a Mix',font_size=font1,size_hint_y=None, height=p1*p2*h/n_button)
        layout.add_widget(layout.mix_window)

        root = ScrollView(size_hint=(1, None),size=(Window.width, Window.height),do_scroll_x=False,do_scroll_y=True)
        root.layout = layout
        root.add_widget(root.layout)

        menu_screen = Screen(name='menu')
        menu_screen.root = root
        menu_screen.add_widget(menu_screen.root)
        self.menu_screen = menu_screen
        self.add_widget(self.menu_screen) #add menu to screen manager
        self.info_screen = InfoScreen()
        self.add_widget(self.info_screen) #add info screen and bind some buttons
        self.menu_screen.root.layout.info.bind(on_release=self.go2info)
        self.info_screen.layout_ext.back.bind(on_release=self.go2menu)
        self.main_water_screen = MainWaterScreen(self.water_dict) #add the main water screen
        self.add_widget(self.main_water_screen)
        self.menu_screen.root.layout.water_window.bind(on_release=self.go2water)
        self.main_recipe_screen = MainRecipeScreen(self.recipe_dict) #add the main recipe screen
        self.add_widget(self.main_recipe_screen)
        self.menu_screen.root.layout.recipe_window.bind(on_release=self.go2recipe)

        if self.water_dict != {}: #create the water details screen for the waters in the dictionary and bind buttons
            for name_curr in self.water_dict.keys():
                curr_screen = WaterDetailsScreen(name_curr,self.water_dict[name_curr])
                curr_screen.layout_ext.back.bind(on_release=self.go2waterAndSetDefault)
                curr_screen.layout_ext.modify.bind(on_release=self.modify_water)
                curr_screen.layout_ext.delete.bind(on_release=self.delete_water_popup)
                getattr(self.main_water_screen.layout_ext.root.layout,name_curr).bind(on_release=self.change_water_screen)
                self.add_widget(curr_screen)

        self.add_water_screen = AddWaterScreen(["Calcium [mg/l]","Magnesium [mg/l]","Bicarbonates [mg/l]","cost"]) #add the screen to add a new water to the dictionary and bind buttons
        self.add_water_screen.layout_ext.back.bind(on_release=self.go2waterAndClear)
        self.add_water_screen.layout_ext.add.bind(on_release=self.add_water)
        self.main_water_screen.layout_ext.add_water.bind(on_release=self.go_to_add_water_screen)
        self.main_water_screen.layout_ext.back.bind(on_release=self.go2menu)
        self.add_widget(self.add_water_screen)

        if self.recipe_dict != {}: #create the recipe details screen for the recipes in the dictionary and bind buttons
            for name_curr in self.recipe_dict.keys():
                curr_screen = RecipeDetailsScreen(name_curr,self.recipe_dict[name_curr])
                curr_screen.root.layout_ext.back.bind(on_release=self.go2recipeAndClearDosage)
                curr_screen.root.layout_ext.delete.bind(on_release=self.delete_recipe_popup)
                getattr(self.main_recipe_screen.layout_ext.root.layout,name_curr).bind(on_release=self.change_recipe_screen)
                self.add_widget(curr_screen)
        self.main_recipe_screen.layout_ext.back.bind(on_release=self.go2menu)
        self.main_recipe_screen.layout_ext.add_recipe.bind(on_release=self.go_to_add_manual_recipe_screen)
        self.add_manual_recipe_screen = AddManualRecipeScreen(self.water_dict)
        self.add_manual_recipe_screen.root.layout_ext.add.bind(on_release=self.add_manual_recipe)
        self.add_manual_recipe_screen.root.layout_ext.back.bind(on_release=self.go2recipeAndClear)
        self.add_widget(self.add_manual_recipe_screen)

        self.select_input_screen = SelectInputScreen(self.water_dict.keys()) #add the screen to select the inputs and bind buttons
        self.select_input_screen.layout_ext.back.bind(on_release=self.go2menuAndClearInput)
        self.select_input_screen.layout_ext.submit.bind(on_release=self.popup_and_launch_analysis)
        self.add_widget(self.select_input_screen)
        self.menu_screen.root.layout.mix_window.bind(on_release=self.go2mix)


    def go2menu(self,instance): #go back to the menu screen
        setattr(self,'current','menu')
    
    def go2info(self,instance): #go to the info screen
        setattr(self,'current','info_screen')
    
    def clear_input(self): #clear the given inputs when leaving the screen to select inputs
        self.remove_widget(self.select_input_screen)
        self.select_input_screen = SelectInputScreen(self.water_dict.keys())
        self.select_input_screen.layout_ext.back.bind(on_release=self.go2menuAndClearInput)
        self.select_input_screen.layout_ext.submit.bind(on_release=self.popup_and_launch_analysis)
        self.add_widget(self.select_input_screen)

    def go2menuAndClearInput(self,instance): #go to the menu and clear the inputs on the select input screen
        setattr(self,'current','menu')
        self.clear_input()

    def go2water(self,instance): #go to the main water screen
        setattr(self,'current','main_water_screen')

    def go2waterAndSetDefault(self,instance): #go to the main water screen and set the values of the water details screen to the ones from the dictionary (in case they are changed without being saved)
        setattr(self,'current','main_water_screen')
        name_curr = instance.parent.parent.name.replace('water_','')
        for k in self.water_dict[name_curr].keys():
            getattr(instance.parent.layout,k).text = str(self.water_dict[name_curr][k])
    
    def go2waterAndClear(self,instance): #go to the main water screen clear the given values in the screen to add a water
        setattr(self,'current','main_water_screen')
        self.add_water_screen.layout_ext.layout.name.text = ''
        for k in self.add_water_screen.keys:
            getattr(self.add_water_screen.layout_ext.layout,k).text = ''

    def go2recipeAndClear(self,instance): #go to the main recipe screen clear the given values in the screen to manually add a recipe
        setattr(self,'current','main_recipe_screen')
        self.add_manual_recipe_screen.root.layout_ext.layout.name.text = ''
        self.add_manual_recipe_screen.root.layout_ext.layout.water_spinner1.text = self.add_manual_recipe_screen.keys_water_p[0]
        self.add_manual_recipe_screen.root.layout_ext.layout.water_p1.text = ''
        self.add_manual_recipe_screen.root.layout_ext.layout.water_spinner2.text = self.add_manual_recipe_screen.keys_water_p[1]
        self.add_manual_recipe_screen.root.layout_ext.layout.water_p2.text = ''
        self.add_manual_recipe_screen.root.layout_ext.layout.water_spinner3.text = self.add_manual_recipe_screen.keys_water_p[2]
        self.add_manual_recipe_screen.root.layout_ext.layout.water_p3.text = 'X'
        for k in self.add_manual_recipe_screen.keys:
            getattr(self.add_manual_recipe_screen.root.layout_ext.layout,k).text = 'X'

    def go2recipe(self,instance): #go to the main recipe screen
        setattr(self,'current','main_recipe_screen')
    
    def go2recipeAndClearDosage(self,instance): #go to the main recipe screen from the recipe details and clear the value for the dosage
        setattr(self,'current','main_recipe_screen')
        keys = list(instance.parent.layout.dosage_g.keys())
        for name in keys:
            instance.parent.layout.dosage_g[name].text = 'X g'
        instance.parent.layout.quantity_input.text = ''

    def go2mix(self,instance): #go to the input selection screen
        self.select_input_screen.update_list(self.water_dict.keys())
        setattr(self,'current','select_input_screen')

    def change_water_screen(self,instance): #go to the water details screen of a given water
        setattr(self,'current','water_'+instance.text)

    def change_recipe_screen(self,instance): #go to the recipe details screen of a given recipe
        setattr(self,'current','recipe_'+instance.text)

    def modify_water(self,instance): #modify the water details screen of a given water
        name_curr = instance.parent.parent.name.replace('water_','')
        empty_val = []
        for k in self.water_dict[name_curr].keys():
            val = getattr(instance.parent.layout,k).text
            if val=='':
                empty_val += [k]
        if len(empty_val)>0:
                mess = "Missing value for %s"% ",\n".join([s for s in empty_val])
                self.popup_message(mess)
                return
        new_dict = {}
        for k in self.water_dict[name_curr].keys():
            val = convert_to_float_or_popup(getattr(instance.parent.layout,k).text)
            if val=='False':
                return
            new_dict[k]=val
        self.water_dict[name_curr] = new_dict
        with open(self.file_water,"w") as f:
            f.write(json.dumps(self.water_dict, indent = 2))
            f.close()
        setattr(self,'current','main_water_screen')

    def delete_water_popup(self,instance): #popup when pressing the delete button
        content = BoxLayout()
        content.orientation = 'vertical'
        f_size = 0.3*0.2*Window.height*0.4*w_size
        content.add_widget(Label(text='Are you sure you want to delete it?',font_size=f_size))
        content.answer = GridLayout()
        content.answer.cols=2
        content.answer.yes=Button(text='Yes',font_size=f_size)
        content.add_widget(content.answer.yes)
        content.answer.no=Button(text='No',font_size=f_size)
        content.add_widget(content.answer.no)
        popup = Popup(title='',content=content, auto_dismiss=False,size_hint = (0.8,0.3),pos_hint = {"x":0.1,"top":0.9})
        content.answer.no.bind(on_release=popup.dismiss)
        content.answer.yes.delete_info = instance.parent.parent
        content.answer.yes.popup = popup
        content.answer.yes.bind(on_release=self.delete_water)
        popup.open()

    def delete_water(self,instance): #delete the water, i.e. remove it from the main water screen, remove its details screen and remove it from dictionary
        instance.popup.dismiss()
        setattr(self,'current','main_water_screen')
        name_curr = instance.delete_info.name.replace("water_",'')
        self.remove_widget(instance.delete_info)
        self.main_water_screen.layout_ext.root.layout.remove_widget(getattr(self.main_water_screen.layout_ext.root.layout,name_curr))
        self.water_dict.pop(name_curr)
        with open(self.file_water,"w") as f:
            f.write(json.dumps(self.water_dict, indent = 2))
            f.close()

    def delete_recipe_popup(self,instance): #popup when pressing the delete button
        content = BoxLayout()
        content.orientation = 'vertical'
        f_size = 0.3*0.2*Window.height*0.4*w_size
        content.add_widget(Label(text='Are you sure you want to delete it?',font_size=f_size))
        content.answer = GridLayout()
        content.answer.cols=2
        content.answer.yes=Button(text='Yes',font_size=f_size)
        content.add_widget(content.answer.yes)
        content.answer.no=Button(text='No',font_size=f_size)
        content.add_widget(content.answer.no)
        popup = Popup(title='',content=content, auto_dismiss=False,size_hint = (0.8,0.3),pos_hint = {"x":0.1,"top":0.9})
        content.answer.no.bind(on_release=popup.dismiss)
        content.answer.yes.delete_info = instance.parent.parent.parent
        content.answer.yes.popup = popup
        content.answer.yes.bind(on_release=self.delete_recipe)
        popup.open()

    def delete_recipe(self,instance): #delete the recipe, i.e. remove it from the main recipe screen, remove its details screen and remove it from dictionary
        instance.popup.dismiss()
        setattr(self,'current','main_recipe_screen')
        name_curr = instance.delete_info.name.replace("recipe_",'')
        self.remove_widget(instance.delete_info)
        self.main_recipe_screen.layout_ext.root.layout.remove_widget(getattr(self.main_recipe_screen.layout_ext.root.layout,name_curr))
        self.recipe_dict.pop(name_curr)
        with open(self.file_recipe,"w") as f:
            f.write(json.dumps(self.recipe_dict, indent = 2))
            f.close()

    def go_to_add_water_screen(self,instance): #go to the add water screen
        setattr(self,'current','add_water')
    
    def go_to_add_manual_recipe_screen(self,instance): #go to the add manual recipe screen
        self.add_manual_recipe_screen.update_list(self.water_dict)
        setattr(self,'current','add_manual_recipe')

    def add_water(self,instance): #add a new water, i.e. add it to the main water screen, create a water details screen and add it to the dictionary
        keys = instance.parent.parent.keys
        name = instance.parent.layout.name.text
        empty_val = []
        if name=='':
            empty_val+=['Name']
        for k in keys:
            val = getattr(instance.parent.layout,k).text
            if val=='':
                empty_val+=[k]
        if len(empty_val)>0:
            mess = "Missing value for %s"% ",\n".join([s for s in empty_val])
            self.popup_message(mess)
            return
        if name in self.water_dict.keys():
            mess = "The name '%s'\nhas already been used" % name
            self.popup_message(mess)
            return
        max_char = 11
        if len(name)>max_char:
            mess = "The name can have\n%i characters maximum" % max_char
            self.popup_message(mess)
            return
        new_dict = {}
        for k in keys:
            val = convert_to_float_or_popup(getattr(instance.parent.layout,k).text)
            if val=='False':
                return
            new_dict[k]=val
        setattr(self.main_water_screen.layout_ext.root.layout,name,Button(text=name,font_size=self.main_water_screen.f_size,size_hint_y=None, height=self.main_water_screen.h))
        self.main_water_screen.layout_ext.root.layout.add_widget(getattr(self.main_water_screen.layout_ext.root.layout,name))
        instance.parent.layout.name.text = ''
        for k in keys:
            getattr(instance.parent.layout,k).text = ''
        self.water_dict[name] = new_dict
        new_screen = WaterDetailsScreen(name,self.water_dict[name])
        new_screen.layout_ext.back.bind(on_release=self.go2waterAndSetDefault)
        new_screen.layout_ext.modify.bind(on_release=self.modify_water)
        new_screen.layout_ext.delete.bind(on_release=self.delete_water_popup)
        self.add_widget(new_screen)
        getattr(self.main_water_screen.layout_ext.root.layout,name).bind(on_release=self.change_water_screen)
        with open(self.file_water,"w") as f:
            f.write(json.dumps(self.water_dict, indent = 2))
            f.close()
        setattr(self,'current','main_water_screen')

    def popup_message(self,mess,title='Warning',popup_while_computing=False): #general popup message
        content = BoxLayout()
        content.orientation = 'vertical'
        f_size = 0.3*0.3*Window.height*0.2*w_size
        content.add_widget(Label(text=mess,font_size=f_size))
        if popup_while_computing:
            self.popup = Popup(title=title,content=content, auto_dismiss=False,size_hint = (0.6,0.5),pos_hint = {"x":0.2,"top":0.9})
            return
        content.close=Button(text='Close',font_size=f_size,size_hint_y=None,height= 0.1*0.5*Window.height)
        content.add_widget(content.close)
        popup = Popup(title=title,content=content, auto_dismiss=False,size_hint = (0.7,0.5),pos_hint = {"x":0.2,"top":0.9})
        content.close.bind(on_press=popup.dismiss)
        popup.open()

    def popup_and_launch_analysis(self,instance): #check inputs and popup to inform that the analysis is in process and call the function to launch the analysis
        empty_val = []
        Hhat = self.select_input_screen.layout_ext.layout.Hhat.text
        if Hhat=='':
            empty_val += ['General Hardness']
        tol_Hhat = self.select_input_screen.layout_ext.layout.tol_Hhat.text
        if tol_Hhat=='':
            empty_val += ['Tolerance GH']
        Ahat = self.select_input_screen.layout_ext.layout.Ahat.text
        if Ahat=='':
            empty_val += ['Alkalinity']
        tol_Ahat =self.select_input_screen.layout_ext.layout.tol_Ahat.text
        if tol_Ahat=='':
            empty_val += ['Tolerance KH']
        name = self.select_input_screen.layout_ext.layout.selected_water.selected_waters
        mess_1 = ""
        mess_2 = ""
        if len(empty_val)>0:
            mess_1 = "Missing value for %s."% ",\n".join([s for s in empty_val])
        if len(name)<2:
            mess_2 = "At least 2 waters need to be selected."
        if mess_1!="" and mess_2!="":
            mess_1 += "\n"
        mess = mess_1 + mess_2
        if mess!="":
            self.popup_message(mess)
            return
        self.popup_message('Mixing in Progress...',title='Please Wait',popup_while_computing=True)
        self.popup.open()
        Clock.schedule_once(self.launch_analysis,0.5)


    def launch_analysis(self,instance): #launch the analysis with the given inputs (calling main() from the first part). Create the results screen (or error screen) and the recipe details to possibly add to the recipe dictionary.
        input = {}
        Hhat = convert_to_float_or_popup(self.select_input_screen.layout_ext.layout.Hhat.text)
        if Hhat=='False':
            self.popup.dismiss()
            delattr(self,'popup')
            return
        tol_Hhat = convert_to_float_or_popup(self.select_input_screen.layout_ext.layout.tol_Hhat.text)
        if tol_Hhat=='False':
            self.popup.dismiss()
            delattr(self,'popup')
            return
        Ahat = convert_to_float_or_popup(self.select_input_screen.layout_ext.layout.Ahat.text)
        if Ahat=='False':
            self.popup.dismiss()
            delattr(self,'popup')
            return
        tol_Ahat = convert_to_float_or_popup(self.select_input_screen.layout_ext.layout.tol_Ahat.text)
        if tol_Ahat=='False':
            self.popup.dismiss()
            delattr(self,'popup')
            return
        name = self.select_input_screen.layout_ext.layout.selected_water.selected_waters
        ca = []
        mg = []
        hco3 = []
        cost = []
        for n in name:
            ca+=[self.water_dict[n]['Calcium [mg/l]']]
            mg+=[self.water_dict[n]['Magnesium [mg/l]']]
            hco3+=[self.water_dict[n]['Bicarbonates [mg/l]']]
            cost+=[self.water_dict[n]['cost']]
        input['name'] = name
        input['Mg'] = mg
        input['Ca'] = ca
        input['HCO3'] = hco3
        input['cost'] = cost
        input['Hhat'] = Hhat
        input['Ahat'] = Ahat
        input['tol_Hhat'] = tol_Hhat
        input['tol_Ahat'] = tol_Ahat
        input_file = './input.json'
        with open(input_file,'w') as f:
            f.write(json.dumps(input, indent = 2))
            f.close()
        error_file = './error.txt'
        if os.path.exists(error_file):
            os.remove(error_file)
        main()
        os.remove(input_file)
        self.popup.dismiss()
        delattr(self,'popup')
        if os.path.exists(error_file):
            self.error_screen = ErrorScreen(error_file)
            self.error_screen.layout.close.bind(on_release=self.go2selectInputScreenAndClearError)
            self.add_widget(self.error_screen)
            setattr(self,'current','error_screen')
        else:
            self.mix_results_screen = MixResultsScreen()
            self.add_widget(self.mix_results_screen)
            self.add_recipe_screens = {}
            for k in self.mix_results_screen.layout.root.layout.btns.keys():
                self.mix_results_screen.layout.root.layout.btns[k].bind(on_release=self.make_add_recipe_screen)
            self.mix_results_screen.layout.back.bind(on_release=self.go2menuAndClearMixResultsScreen_popup)
            setattr(self,'current','mix_results_screen')
            self.clear_input()

    def make_add_recipe_screen(self,instance): #create the screen to add a recipe to the recipe dictionary
        self.add_recipe_screen = AddRecipeScreen(instance.data)
        self.add_widget(self.add_recipe_screen)
        self.add_recipe_screen.root.layout_ext.add.bind(on_release=self.add_recipe)
        self.add_recipe_screen.root.layout_ext.back.bind(on_release=self.go2resultsAndClearAddRecipeScreen)
        setattr(self,'current','add_recipe_screen')
    
    def go2selectInputScreenAndClearError(self,instance): #leave the error message and go back to the screen to select input
        setattr(self,'current','select_input_screen')
        if hasattr(self,'error_screen'):
            os.remove(instance.parent.error_file)
            self.remove_widget(self.error_screen)
            delattr(self,'error_screen')
    
    def go2resultsAndClearAddRecipeScreen(self,instance): #go back to the results screen from the screen to add a recipe
        setattr(self,'current','mix_results_screen')
        if hasattr(self,'add_recipe_screen'):
            self.remove_widget(self.add_recipe_screen)
            delattr(self,'add_recipe_screen')

    def go2menuAndClearMixResultsScreen_popup(self,instance): #popup before leaving results screen
        content = BoxLayout()
        content.orientation = 'vertical'
        f_size = 0.3*0.2*Window.height*0.3*w_size
        content.add_widget(Label(text="Are you sure you want to leave?",font_size=f_size))
        content.answer = GridLayout()
        content.answer.cols=2
        content.answer.yes=Button(text='Yes',font_size=f_size)
        content.add_widget(content.answer.yes)
        content.answer.no=Button(text='No',font_size=f_size)
        content.add_widget(content.answer.no)
        popup = Popup(title="The recipes which weren't added will be lost",content=content, auto_dismiss=False,size_hint = (0.8,0.3),pos_hint = {"x":0.1,"top":0.9})
        content.answer.no.bind(on_release=popup.dismiss)
        content.answer.yes.popup = popup
        content.answer.yes.bind(on_release=self.go2menuAndClearMixResultsScreen)
        popup.open()

    def go2menuAndClearMixResultsScreen(self,instance): #go to the menu and leave the results screen (remove corresponding temp file)
        instance.popup.dismiss()
        setattr(self,'current','menu')
        if hasattr(self,'mix_results_screen'):
            self.remove_widget(self.mix_results_screen)
            delattr(self,'mix_results_screen')
        if os.path.exists('./prov_table.json'):
            os.remove('./prov_table.json')
    
    def add_manual_recipe(self,instance): #add manually a new recipe, i.e. without the optimization step
        cont = instance.parent.parent.parent.view_recipe(instance.parent.view)
        if cont:
            cont = self.add_recipe(instance)
            if cont:
                self.go2recipeAndClear(instance)

    def add_recipe(self,instance):  #add a new recipe, i.e. add it to the main recipe screen, create a recipe details screen and add it to the dictionary
        name = instance.parent.layout.name.text
        data = instance.parent.layout.data
        if name in self.recipe_dict.keys():
            mess = "The name '%s'\nhas already been used" % name
            self.popup_message(mess)
            return False
        if name=='':
            mess = "Please enter a name for the recipe"
            self.popup_message(mess)
            return False
        max_char = 17
        if len(name)>max_char:
            mess = "The name can have\n%i characters maximum" % max_char
            self.popup_message(mess)
            return False
        setattr(self.main_recipe_screen.layout_ext.root.layout,name,Button(text=name,font_size=self.main_recipe_screen.f_size,size_hint_y=None, height=self.main_recipe_screen.h))
        self.main_recipe_screen.layout_ext.root.layout.add_widget(getattr(self.main_recipe_screen.layout_ext.root.layout,name))

        self.recipe_dict[name] = data
        new_screen = RecipeDetailsScreen(name,data)
        new_screen.root.layout_ext.back.bind(on_release=self.go2recipeAndClearDosage)
        new_screen.root.layout_ext.delete.bind(on_release=self.delete_recipe_popup)
        self.add_widget(new_screen)
        getattr(self.main_recipe_screen.layout_ext.root.layout,name).bind(on_release=self.change_recipe_screen)
        with open(self.file_recipe,"w") as f:
            f.write(json.dumps(self.recipe_dict, indent = 2))
            f.close()
        if hasattr(self,'add_recipe_screen'):
            setattr(self,'current','mix_results_screen')
            self.remove_widget(self.add_recipe_screen)
            delattr(self,'add_recipe_screen')
        return True
    
    def generate_first_time_data(self): #create example dictionary in first installation
        waters = {"Distilled": {"Calcium [mg/l]": 0.0,"Magnesium [mg/l]": 0.0,"Bicarbonates [mg/l]": 0.0,"cost": 0.45},
                    "Evian": {"Calcium [mg/l]": 80.0,"Magnesium [mg/l]": 26.0,"Bicarbonates [mg/l]": 360.0,"cost": 0.66},
                    "Volvic": {"Calcium [mg/l]": 12.0,"Magnesium [mg/l]": 8.0,"Bicarbonates [mg/l]": 74.0,"cost": 0.77},
                    "Contrex": {"Calcium [mg/l]": 468.0,"Magnesium [mg/l]": 74.5,"Bicarbonates [mg/l]": 372.0,"cost": 0.8}}
        recipes = {"My Mix 87:40 #1": {"% Water #1": "86.6% Distilled","% Water #2": "9.5% Evian","% Water #3": "3.9% Contrex","General Hardness [ppm as CaCO3]": "86.8","Alkalinity [ppm as CaCO3]": "39.9","GH:KH": "2.2","Calcium [ppm as CaCO3]": "64.6","Magnesium [ppm as CaCO3]": "22.1","Cost": "0.48"},
                    "My Mix 87:40 #2": {"% Water #1": "49.9% Distilled","% Water #2": "46.2% Volvic","% Water #3": "3.9% Contrex","General Hardness [ppm as CaCO3]": "86.7","Alkalinity [ppm as CaCO3]": "39.9","GH:KH": "2.2","Calcium [ppm as CaCO3]": "59.5","Magnesium [ppm as CaCO3]": "27.2","Cost": "0.61"},
                    "My Mix 94:40": {"% Water #1": "86.6% Distilled","% Water #2": "8.9% Evian","% Water #3": "4.5% Contrex","General Hardness [ppm as CaCO3]": "93.8","Alkalinity [ppm as CaCO3]": "40.0","GH:KH": "2.3","Calcium [ppm as CaCO3]": "70.4","Magnesium [ppm as CaCO3]": "23.3","Cost": "0.48"},
                    "My Mix 118:50": {"% Water #1": "83.2% Distilled","% Water #2": "11.1% Evian","% Water #3": "5.7% Contrex","General Hardness [ppm as CaCO3]": "118.3","Alkalinity [ppm as CaCO3]": "50.2","GH:KH": "2.4","Calcium [ppm as CaCO3]": "88.9","Magnesium [ppm as CaCO3]": "29.4","Cost": "0.49"}}
        with open(self.file_water,'w') as f:
            f.write(json.dumps(waters, indent = 2))
            f.close()
        with open(self.file_recipe,'w') as f:
            f.write(json.dumps(recipes, indent = 2))
            f.close()


class cowamix(App): #the app
    def on_start(self): #unable the back button from the screen
        from kivy.base import EventLoop
        EventLoop.window.bind(on_keyboard=self.block_exit)
  
    def block_exit(self, window, key, *largs):
        if key == 27:
              return True

    def build(self):
        return MenuScreenManager(user_path=self.user_data_dir)


if __name__=='__main__':
    cowamix().run()
