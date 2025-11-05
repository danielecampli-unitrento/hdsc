The MATLAB files are named from R10 to R450, where R indicates the radius of the circular trap in micrometers.

Each file contains two MATLAB variables: 
	1) signal --> having dimension 100x5401x10, where:
		dimension 1 (100) represents the spatial points (along lateral dimension) where the transducer record the signal
		dimension 2 (5401) represents the temporal points at which the transducer records the signal
		dimension 3 (10) represents the center frequencies of the transmitted pulse (from 1 to 10 MHz with a bandwidth of 1 MHz at - 10 dB)
	2) medium_mat --> having dimension 1x10, where each element contains the ground truth map for each center frequency (the maps should coincide for each center frequency, given a fixed radius, but they were saved to check, as a confirm, that the maps are the same)