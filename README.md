# SystemID-and-PID-Tuning (C++)

This code was developed for SystemID and automatic PID tuning of a radian laser tracker. 
When the laser tracker is mounted on a tripod, the coupled dynamics of the tracker and the tripod is of fifth order. By injecting a PRBS signal (To excite all the modes) to the laser tracker the azimuth and elevation output of the laser tracker are read. Then the code usese the data and fits an optimized model to the fifth order dynamic. Using the modeled dynamic of the laser tacker - tripod, the PID controller gains on the tracker are tuned. 
