# CoWaMix
A tool to help you mix waters and create brewing water recipes for your coffee.

## Installation
To install CoWaMix, clone the repository
'''bash
git clone https://github.com/vpjunet/CoWaMix.git
'''
create a virtual environment
'''bash
python3 -m virtualenv cowamix_venv
'''
activate it on linux/macOS
'''bash
source cowamix_venv/bin/activate
'''
or Windows
'''bash
./cowamix_venv/Scripts/activate
'''
and install the dependencies
'''bash
python3 -m pip install kivy numpy trust-constr more-itertools
'''

## Run
Run the app
'''bash
python3 main.py
'''
Note that the data are stored in the 'App().user_data_dir' folder.
Make sure that you have access to this path.
You can also change this path in the class 'cowamix' with the input argument 'user_path'
of the class 'MenuScreenManager'.

