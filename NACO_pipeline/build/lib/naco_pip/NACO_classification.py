#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:48:04 2020

@author: lewis
"""
__author__ = 'Lewis Picker'
__all__ = ['input_dataset','find_agpm_list']
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from photutils import CircularAperture, aperture_photometry
from vip_hci.fits import open_fits, write_fits
from vip_hci.preproc import frame_fix_badpix_isolated
from vip_hci.var import frame_filter_lowpass
from hciplot import plot_frames
import pdb
import naco_pip.fits_info as fits_info
#no module fits_ifo
#create a file that user will input values for observations, then read those values into the script.
#test = input_dataset('/home/lewis/Documents/Exoplanets/data_sets/HD179218/Tests/','/home/lewis/Documents/Exoplanets/data_sets/HD179218/Corrected/')
#create a real_ndit file 
#discard bad cubes those that are <2/3 of the median size do that check in mk_dico

def find_agpm_list(self, file_list, coro = True, threshold = 0):

        cube = open_fits(self.outpath + file_list[0])
        nz, ny, nx = cube.shape
        median_frame = np.median(cube, axis = 0)
        median_frame = frame_filter_lowpass(median_frame, median_size = 7, mode = 'median')
        write_fits(self.outpath +'median_frame',median_frame)        
        median_frame = frame_filter_lowpass(median_frame, mode = 'gauss',fwhm_size = 5)
        write_fits(self.outpath +'median_frame_1',median_frame)
        cy,cx = np.unravel_index(np.argmax(median_frame), median_frame.shape)
        return cy, cx
    

class input_dataset():
    def __init__(self, inpath, outpath,coro= True):
        # dit_sci, ndit_sci,ndit_sky, dit_unsat, ndit_unsat, dit_flat,wavelegnth, size_telescope, pixel_scale, 
        
        self.inpath = inpath
        self.outpath = outpath
        #creates a list of all files in the path
        old_list = os.listdir(self.inpath)
        #removes non .fits files
        self.file_list = [file for file in  old_list if file.endswith('.fits')]        
        self.dit_sci = fits_info.dit_sci
        self.ndit_sci = fits_info.ndit_sci
        self.ndit_sky = fits_info.ndit_sky
        self.dit_unsat = fits_info.dit_unsat
        self.ndit_unsat = fits_info.ndit_unsat
        self.dit_flat = fits_info.dit_flat
        self.wavelength = fits_info.wavelength
        self.size_telescope = fits_info.size_telescope
        self.pixel_scale = fits_info.pixel_scale
        #calculate the resoluton element.
        self.resel = (self.wavelength*180*3600)/(self.size_telescope *np.pi*
                                                 self.pixel_scale)
        
    def bad_columns(self, verbose = True, debug = True):

        bcm = np.zeros((1026, 1024) ,dtype=np.float64)
        #creating bad pixel map
        for i in range(3, 509, 8):
            for j in range(512):
                bcm[j,i] = 1

        if verbose: 
            print(self.file_list)
        for fname in self.file_list:
            if verbose:
                print('about to fix', fname)
            cube_fname, header_fname = open_fits(self.inpath + fname,
                                                header = True, verbose = debug)
            test_fname = cube_fname.copy()
            #crop the bad pixcel map to the same dimentions of the frames

            if len(cube_fname.shape) == 3:
                nz, ny, nx = cube_fname.shape
                cy, cx = ny/2 , nx/2
                ini_y, fin_y = int(512-cy), int(512+cy)
                ini_x, fin_x = int(512-cx), int(512+cx)
                bcm_crop = bcm[ini_y:fin_y,ini_x:fin_x]
                for j in range(nz):
                    #replace bad columns in each frame of the cubes
                    test_fname[j] = frame_fix_badpix_isolated(test_fname[j],
                                    bpm_mask= bcm_crop, sigma_clip=3,
                                    num_neig=5, size=5, protect_mask=False,
                                    radius=30, verbose=debug, debug=False)
                write_fits(self.outpath + fname, test_fname,
                           header_fname, output_verify = 'fix')
                
            else:
                ny, nx = cube_fname.shape
                cy, cx = ny/2 , nx/2
                ini_y, fin_y = int(512-cy), int(512+cy)
                ini_x, fin_x = int(512-cx), int(512+cx)
                bcm_crop = bcm[ini_y:fin_y,ini_x:fin_x]
                test_fname = frame_fix_badpix_isolated(test_fname,
                             bpm_mask= bcm_crop, sigma_clip=3, num_neig=5,
                             size=5, protect_mask=False, radius=30,
                             verbose=debug, debug=False)
                write_fits(self.outpath + fname, test_fname,
                           header_fname, output_verify = 'fix')
            if verbose:
                    print('done fixing',fname)

    def mk_dico(self, coro = True, verbose = True, debug = False):
        if coro:
           #creacting a dictionary
           #add a list of the legnth of frames in the cubes add that to the dico
           file_list = [f for f in listdir(self.outpath) if
                        isfile(join(self.outpath, f))]
           fits_list = []
           sci_list = []
           sci_list_mjd = []
           sky_list = []
           unsat_list = []
           unsat_list_mjd = []
           flat_list = []
           X_sci_list = []
           X_unsat_list = []
           flat_dark_list = []
           sci_dark_list = []
           unsat_dark_list = []
           sci_frames = []
           sky_frames = []
           unsat_frames = []
           flat_frames = []

           for fname in file_list:
               if fname.endswith('.fits') and fname.startswith('NACO'):
                   fits_list.append(fname)
                   cube, header = open_fits(self.outpath +fname, header=True,
                                            verbose=debug)
                   if header['HIERARCH ESO DPR CATG'] == 'SCIENCE'and \
                       header['HIERARCH ESO DPR TYPE'] == 'OBJECT' and \
                       header['HIERARCH ESO DET DIT'] == self.dit_sci and \
                           header['HIERARCH ESO DET NDIT'] in self.ndit_sci and\
                        cube.shape[0] > 2/3*self.ndit_sci:
                            
                        sci_list.append(fname)
                        sci_list_mjd.append(header['MJD-OBS'])
                        X_sci_list.append(header['AIRMASS'])
                        sci_frames.append(cube.shape[0])
                   elif (header['HIERARCH ESO DPR CATG'] == 'SCIENCE' and \
                         header['HIERARCH ESO DPR TYPE'] == 'SKY' and \
                        header['HIERARCH ESO DET DIT'] == self.dit_sci and\
                        header['HIERARCH ESO DET NDIT'] in self.ndit_sky) and\
                       cube.shape[0] > 2/3*self.ndit_sky:
                           
                       sky_list.append(fname)
                       sky_frames.append(cube.shape[0])
                   elif header['HIERARCH ESO DPR CATG'] == 'SCIENCE' and \
                       header['HIERARCH ESO DET DIT'] == self.dit_unsat and \
                           header['HIERARCH ESO DET NDIT'] in self.ndit_unsat:
                       unsat_list.append(fname)
                       unsat_list_mjd.append(header['MJD-OBS'])
                       X_unsat_list.append(header['AIRMASS'])
                       unsat_frames.append(cube.shape[0])
                   elif 'FLAT,SKY' in header['HIERARCH ESO DPR TYPE']:
                       flat_list.append(fname)
                       flat_frames.append(cube.shape[0])
                   elif 'DARK' in header['HIERARCH ESO DPR TYPE']:
                       if header['HIERARCH ESO DET DIT'] == self.dit_flat:
                           flat_dark_list.append(fname)
                       if header['HIERARCH ESO DET DIT'] == self.dit_sci:
                           sci_dark_list.append(fname)
                       if header['HIERARCH ESO DET DIT'] == self.dit_unsat:
                           unsat_dark_list.append(fname)
           with open(self.outpath+"sci_list.txt", "w") as f:
                for sci in sci_list:
                    f.write(sci+'\n')
           with open(self.outpath+"sky_list.txt", "w") as f:
                for sci in sky_list:
                    f.write(sci+'\n')
           with open(self.outpath+"unsat_list.txt", "w") as f:
                for sci in unsat_list:
                    f.write(sci+'\n')
           with open(self.outpath+"unsat_dark_list.txt", "w") as f:
                for sci in unsat_dark_list:
                    f.write(sci+'\n')
           with open(self.outpath+"flat_dark_list.txt", "w") as f:
                for sci in flat_dark_list:
                    f.write(sci+'\n')
           with open(self.outpath+"sci_dark_list.txt", "w") as f:
                for sci in sci_dark_list:
                    f.write(sci+'\n')
           with open(self.outpath+"flat_list.txt", "w") as f:
                for sci in flat_list:
                    f.write(sci+'\n')
           
           open(self.outpath+"resel.txt", "w").write(str(self.resel))


#The sky cubes should be identical to the science cubes without the object
#in the middle if the skylist is empty it could be caused by misclasification
#of the data (or non-existant) we can look for the sky cubes from the science
#and measure the flux at the centre of the coronagraph
#the flux at the centre of the sky cubes should be significantly lower


    def find_sky_in_sci_cube(self, nres = 3, coro = True, verbose = True, debug = True):
       flux_list = []
       fname_list = []
       sci_list = []
       with open(self.outpath+"sci_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sci_list.append(line.split('\n')[0])

       sky_list = []
       with open(self.outpath+"sky_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sky_list.append(line.split('\n')[0])

       #draw sci list from dico
       #find corono centre of the cube.
                
       cy,cx = find_agpm_list(self, sci_list)
       if debug:
           print(cy,cx)
           #plot_frames(open_fits(self.outpath + sci_list[0])[-1], circle = (cy,cx))

       #create the aperture
       circ_aper = CircularAperture((cx,cy), round(nres*self.resel))
       #total flux through the aperture
       for fname in sci_list:
           cube_fname = open_fits(self.outpath + fname)
           median_frame = np.median(cube_fname, axis = 0)
           circ_aper_phot = aperture_photometry(median_frame,
                                                    circ_aper, method='exact')
       #append it to the flux list.
           circ_flux = np.array(circ_aper_phot['aperture_sum'])
           flux_list.append(circ_flux[0])
           fname_list.append(fname)

       median_flux = np.median(flux_list)
       sd_flux = np.std(flux_list)

       for i in range(len(flux_list)):
           if flux_list[i] < median_flux - 2*sd_flux:
               sky_list.append(fname_list[i])
               sci_list.remove(fname_list[i])
       with open(self.outpath+"sci_list.txt", "w") as f:
                for sci in sci_list:
                    f.write(sci+'\n')
       with open(self.outpath+"sky_list.txt", "w") as f:
                for sci in sky_list:
                    f.write(sci+'\n')


       #plot of centre flux
       if debug:
           plt.figure(1)
           plt.title('Flux of median sci frames at the centre \
                     of the coronagraph')
           plt.plot(flux_list, 'bo', label = 'centre flux')
           plt.ylabel('Flux')
           plt.legend()
           plt.savefig(self.outpath + 'flux_plot')
           plt.show()
