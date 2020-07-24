#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:48:04 2020

@author: lewis
"""
__author__ = 'Lewis Picker'
__all__ = ['input_dataset','find_AGPM_list']
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from photutils import CircularAperture, aperture_photometry
from vip_hci.fits import open_fits, write_fits
from vip_hci.preproc import frame_fix_badpix_isolated
from vip_hci.var import frame_filter_lowpass
import naco_pip.fits_info as fits_info
import pdb

#test = input_dataset('/home/lewis/Documents/Exoplanets/data_sets/HD179218/Tests/','/home/lewis/Documents/Exoplanets/data_sets/HD179218/Debug/')


def find_AGPM_list(self, file_list, verbose = True, debug = False):
        """
        This method will find the location of the AGPM
        (roughly the location of the star) 
        """
        cube = open_fits(self.outpath + file_list[0])
        nz, ny, nx = cube.shape
        median_frame = np.median(cube, axis = 0)
        median_frame = frame_filter_lowpass(median_frame, median_size = 7, mode = 'median')       
        median_frame = frame_filter_lowpass(median_frame, mode = 'gauss',fwhm_size = 5)
        ycom,xcom = np.unravel_index(np.argmax(median_frame), median_frame.shape)
        if verbose:
            print('The location of the AGPM is','ycom =',ycom,'xcom =', xcom)
        if debug:
            pdb.set_trace()
        return [ycom, xcom]

class input_dataset():
    def __init__(self, inpath, outpath,coro= True): 
        self.inpath = inpath
        self.outpath = outpath
        old_list = os.listdir(self.inpath)
        self.file_list = [file for file in  old_list if file.endswith('.fits')]        
        self.dit_sci = fits_info.dit_sci
        self.ndit_sci = fits_info.ndit_sci
        self.ndit_sky = fits_info.ndit_sky
        self.dit_unsat = fits_info.dit_unsat
        self.ndit_unsat = fits_info.ndit_unsat
        self.dit_flat = fits_info.dit_flat

        
    def bad_columns(self, verbose = True, debug = False):
        """
        In NACO data there are systematic bad columns in the lower left quadrant
        This method will correct those bad colums with the median of the neighbouring pixels
        """
        #creating bad pixel map
        bcm = np.zeros((1026, 1024) ,dtype=np.float64)
        for i in range(3, 509, 8):
            for j in range(512):
                bcm[j,i] = 1

        for fname in self.file_list:
            if verbose:
                print('Fixing', fname)
            tmp, header_fname = open_fits(self.inpath + fname,
                                                header = True, verbose = debug)
            if verbose:
                print(tmp.shape)
            #crop the bad pixel map to the same dimentions of the frames
            if len(tmp.shape) == 3:
                nz, ny, nx = tmp.shape
                cy, cx = ny/2 , nx/2
                ini_y, fin_y = int(512-cy), int(512+cy)
                ini_x, fin_x = int(512-cx), int(512+cx)
                bcm_crop = bcm[ini_y:fin_y,ini_x:fin_x]
                for j in range(nz):
                    #replace bad columns in each frame of the cubes
                    tmp[j] = frame_fix_badpix_isolated(tmp[j],
                                    bpm_mask= bcm_crop, sigma_clip=3,
                                    num_neig=5, size=5, protect_mask=False,
                                    radius=30, verbose=debug, debug=False)
                write_fits(self.outpath + fname, tmp,
                           header_fname, output_verify = 'fix')
                
            else:
                ny, nx = tmp.shape
                cy, cx = ny/2 , nx/2
                ini_y, fin_y = int(512-cy), int(512+cy)
                ini_x, fin_x = int(512-cx), int(512+cx)
                bcm_crop = bcm[ini_y:fin_y,ini_x:fin_x]
                tmp = frame_fix_badpix_isolated(tmp,
                             bpm_mask= bcm_crop, sigma_clip=3, num_neig=5,
                             size=5, protect_mask=False, radius=30,
                             verbose=debug, debug=False)
                write_fits(self.outpath + fname, tmp,
                           header_fname, output_verify = 'fix')
            if verbose:
                    print('done fixing',fname)

    def mk_dico(self, coro = True, verbose = True, debug = False):
        if coro:
           #creacting a dictionary
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
           
           if verbose: 
               print('Creating dictionary')
           for fname in file_list:
               if fname.endswith('.fits') and fname.startswith('NACO'):
                   fits_list.append(fname)
                   cube, header = open_fits(self.outpath +fname, header=True,
                                            verbose=debug)
                   if header['HIERARCH ESO DPR CATG'] == 'SCIENCE'and \
                       header['HIERARCH ESO DPR TYPE'] == 'OBJECT' and \
                       header['HIERARCH ESO DET DIT'] == self.dit_sci and \
                           header['HIERARCH ESO DET NDIT'] in self.ndit_sci and\
                        cube.shape[0] > 2/3*min(self.ndit_sci): #avoid bad cubes
                            
                        sci_list.append(fname)
                        sci_list_mjd.append(header['MJD-OBS'])
                        X_sci_list.append(header['AIRMASS'])
                        sci_frames.append(cube.shape[0])
                        
                   elif (header['HIERARCH ESO DPR CATG'] == 'SCIENCE' and \
                         header['HIERARCH ESO DPR TYPE'] == 'SKY' and \
                        header['HIERARCH ESO DET DIT'] == self.dit_sci and\
                        header['HIERARCH ESO DET NDIT'] in self.ndit_sky) and\
                       cube.shape[0] > 2/3*min(self.ndit_sky): #avoid bad cubes
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
           if verbose: 
               print('Done :)')


    def find_sky_in_sci_cube(self, nres = 3, coro = True, verbose = True, plot = None, debug = False):
       """
       Empty SKY list could be caused by a misclasification of the header in NACO data
       This method will check the flux of the SCI cubes around the location of the AGPM 
       A SKY cube should be less bright at that location allowing the seperation of cubes
       
       """

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
                
       self.resel = (fits_info.wavelength*180*3600)/(fits_info.size_telescope *np.pi*
                                                 fits_info.pixel_scale)
                
       agpm_pos = find_AGPM_list(self, sci_list)
       if verbose: 
           print('The rougth location of the star is','y  = ', agpm_pos[0] , 'x =', agpm_pos[1])

       #create the aperture
       circ_aper = CircularAperture((agpm_pos[1],agpm_pos[0]), round(nres*self.resel))
       #total flux through the aperture
       for fname in sci_list:
           cube_fname = open_fits(self.outpath + fname, verbose = debug)
           median_frame = np.median(cube_fname, axis = 0)
           circ_aper_phot = aperture_photometry(median_frame,
                                                    circ_aper, method='exact')
       #append it to the flux list.
           circ_flux = np.array(circ_aper_phot['aperture_sum'])
           flux_list.append(circ_flux[0])
           fname_list.append(fname)
           if verbose: 
               print('centre flux has been measured for', fname)

       median_flux = np.median(flux_list)
       sd_flux = np.std(flux_list)
       if verbose:
           print('Sorting Sky from Sci')

       for i in range(len(flux_list)):
           if flux_list[i] < median_flux - 2*sd_flux:
               sky_list.append(fname_list[i])
               sci_list.remove(fname_list[i])
               symbol = 'bo'
           if plot: 
               if flux_list[i] > median_flux - 2*sd_flux:
                   symbol = 'go'
               else:
                   symbol = 'ro'
               plt.plot(i, flux_list[i]/median_flux , symbol)
       if plot:         
           plt.title('Normalised flux around star')
           plt.ylabel('normalised flux')
           if plot == 'save':
               plt.savefig(self.outpath + 'flux_plot')
           if plot == 'show':
               plt.show()
                         
       with open(self.outpath+"sci_list.txt", "w") as f:
                for sci in sci_list:
                    f.write(sci+'\n')
       with open(self.outpath+"sky_list.txt", "w") as f:
                for sci in sky_list:
                    f.write(sci+'\n')
       if verbose:
           print('done :)')

       
