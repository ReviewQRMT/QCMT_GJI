# Q_CMT
## An automated procedure for Regional Moment Tensor inversion
### _Enhancing Solutions and Uncertainty Estimations for Small to Moderate Earthquakes by Accounting for Data and Theoretical Errors_ 
#### by @eqhalauwet

(Static version for GJI publication)


### Requirements

- Python 3.6+
- Required libraries are listed in **prerequisites/cmt_env.yml**


### Installation (tested using anaconda v2022.05 on Ubuntu 20 & 22)

- Download and extract this code on your machine 
- Open terminal and execute the following commands to install Anaconda (change version with: "_2022.05_":
  - **_`sudo apt install curl`_**
  - **_`curl -O https://repo.anaconda.com/archive/Anaconda3-<INSTALLER_VERSION>-Linux-x86_64.sh`_**
  - **_`bash Anaconda3-<INSTALLER_VERSION>-Linux-x86_64.sh`_**
  - Follow the Anaconda installation procedure
- Restart the terminal and navigate to the **Q_CMT_GJI** directory to install all required libraries:
  - _**`conda env create -f prerequisites/cmt_env.yml`**_
- Manualy copy and replace several built-in modules with those available in the **prerequisites** directory:
  - _`basic_client.py`_ to **<ANACONDA_DIRECTORY>/envs/QCMT/lib/<PYTHON VERSION>/site-packages/obspy/clients/seedlink**
  - _`rmt_client.py`_ to **<ANACONDA_DIRECTORY>/envs/QCMT/lib/<PYTHON VERSION>/site-packages/obspy/clients/seedlink**
  - _`response.py`_ to **<ANACONDA_DIRECTORY>/envs/QCMT/lib/<PYTHON VERSION>/site-packages/obspy/core/inventory**
- Add your **Q_CMT** directory to the **_PYTHONPATH_**:
  - **_`export PYTHONPATH="${PYTHONPATH}:<YOUR_Q_CMT_PATH>"`_**


### Usages

- Download **gf_stores** from https://bit.ly/Auto_RMT_Outputs
- Open the terminal in the **Analysis** directory then activate the _**QCMT**_ environment
- Run _**`python RUN_RMT_KATALOG.py`**_ in the specific analysis directory (**north_banda** for real earthquake or **synthetic_test**)
  - _The script will start by reading the data directory and performing the analysis._
  - _Result will be stored at a new output directory within each analysis directory_
