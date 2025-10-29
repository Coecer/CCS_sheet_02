Test 1, 2, 3


### How to activate virtual environment, here called ".venv", via terminal inside the project folder ###

cd .\CCS_sheet_02\
.\.venv\Scripts\Activate  

# ensure that venv is then really active by checking if it shows (.venv) 


# checking which packages are already installed
pip list

# create requirements.txt file
pip freeze > requirements.txt

# upgrade pip
python.exe -m pip install --upgrade pip

# install packages
pip install numpy matplotlib #pandas ...


