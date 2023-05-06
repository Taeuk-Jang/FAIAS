This is experimental code for the paper Adversarial Fairness Network for ICLR2022

Please download VGG16 data from the link below and save in ./examples/data/:

https://drive.google.com/file/d/1g4t73LCL2fhTIde1lsgXsXdTRHMadwbC/view?usp=sharing

./example/FAIAS_VGG16.ipynb : example implementation of FAIAS on CelebA data with VGG16 feature extraction.
                              It sweeps learning rate for selector and predictor to find the optimal hyper-parameter.
./examples/fair_main_update1.py : the main code that contains FAIAS network.