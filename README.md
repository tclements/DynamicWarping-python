# DynamicWarping-python

Python code for Dynamic Time Warping

![Dynamic Time Warping](SINEdistance.png "Dynamic Time Warping")

This is a python translation of [Dylan Mikesell's DynamicWarping repo](https://github.com/dylanmikesell/DynamicWarping) in MATLAB. 

You can run the run_example.py with two example shifts.
1) A step function of shifts.
2) A sine function of shitts.

In (1) we see that we do not well match the shift in the area the shift 
occurs. This is because the single shift in the step function is larger 
than dt. Therefore is has to spread this shift out over a number of samples.

In (2) we see that we recover the fits well with b=1. If we go to larger b 
values we see that we no longer recover the shifts, because we do not allow
large enough steps.

Read the comments in the script run_example.py to see other things like forward and backward accumulation operations.

Cite the paper: [Mikesell et al., 2015, A comparison of methods to estimate seismic phase delays: numerical examples for coda wave interferometry, Geophysical Journal International](https://academic.oup.com/gji/article/202/1/347/587747)
