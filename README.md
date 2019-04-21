Directions on installing gym and such on a TOTALLY fresh (gcloud) computer...

```
#install miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh

#install packages
sudo apt update
sudo apt install -y libsm6 libxrender1 libfontconfig1
sudo apt-get install python-opengl xvfb

git clone https://github.com/xunil17/ACRL-Car-Final.git
cd ACRL-Car-Final/
pip install tensorflow
pip install opencv-python
pip install gym[box2d]


#when we run, create a virtual display to render in
xvfb-run -s "-screen 0 1400x900x24" python car_agent.py savefolder
```