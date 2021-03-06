# CoWaMix
## Coffee Water Mix 
A tool to help you mix waters and create brewing water recipes for your coffee.

The app is available on [Google Play Store](https://play.google.com/store/apps/details?id=ch.vpjunet.cowamix). Read the [Privacy Policy](../main/app_info/privacy_policy/privacy_policy.md) and the [Terms and Conditions](../main/app_info/terms_and_conditions/terms_and_conditions.md).
![alt text](../main/icon.png)
## Installation
To install CoWaMix, clone the repository
```bash
git clone https://github.com/vpjunet/CoWaMix.git
```
create a virtual environment
```bash
python3 -m virtualenv cowamix_venv
```
activate it on linux/macOS
```bash
source cowamix_venv/bin/activate
```
or Windows
```bash
./cowamix_venv/Scripts/activate
```
and install the requirements.
```bash
python3 -m pip install kivy numpy trust-constr more-itertools
```
To deactivate the environment, use
```bash
deactivate
```

## Run
Run the app
```bash
python3 main.py
``` 
Note that the data are stored in the `App().user_data_dir` folder.
Make sure that you have access to this path.
You can also change this path in the class `cowamix` with the input argument `user_path`
of the class `MenuScreenManager`.

## License
CoWaMix is licensed under [GPLv3](../main/LICENSE).

