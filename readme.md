# Environment Set Up

## Clone the Repository
Open the terminal and navigate to a folder you want to clone this repository to. Run `git clone https://github.com/CWRUbotixROV/rov-bootcamp.git`. Then, `cd rov-bootcamp` to enter the repository.

## Install Libraries
### Create a Virtual Environment
When installing libraries for python, it is good practice to use a "virtual environment" which keeps the libraries installed separate for each project you work on.
To create a new virtual environment for this project, run `python -m venv .venv`. This command will create a directory `.venv` to put the virtual environment in.

### Activating the Virtual Environment
The command to activate the virtual environment depends on your OS.

- Windows: `.venv\Scripts\activate.bat`
- Linux/Mac: `source .venv/bin/activate`

After this command is run, you should see `(.venv)` before each line in the command prompt.

### Install Libraries
Before installing libraries, make sure your virtual environment is active by checking for `(.venv)` in the terminal. Now, when libraries are installed, they will only be installed for this virtual environment.

The file `requirements.txt` contains all the libraries that are needed. To install all of these libraries at once, run `pip install -r requirements.txt`.

## Verify Environment Set Up
The python script `env_test.py` will try to import all the libraries needed to verify everything is installed correctly. To run it, run `python env_test.py`. If it prints "All libraries installed" then the environment is correctly set up.